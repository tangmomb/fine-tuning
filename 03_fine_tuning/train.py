"""Fine-tuning LoRA BF16 des checkpoints Qwen Text-to-SQL sur une H100.

Le script apprend exclusivement sur train.jsonl. Le split test Spider reste
réservé à 02_evaluation_brut_models afin d'éviter toute fuite de données.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "train.jsonl"
VALIDATION_DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "validation.jsonl"
DEFAULT_MODELS = ("Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class SqlDataset(Dataset):
    """Tokenise et masque le prompt : la loss ne porte que sur le SQL assistant."""

    def __init__(self, rows: list[dict[str, Any]], processor: Any, max_length: int) -> None:
        self.rows = rows
        self.processor = processor
        self.max_length = max_length
        self.eos_token_id = processor.tokenizer.eos_token_id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        messages = self.rows[index]["messages"]
        if len(messages) != 3 or messages[-1].get("role") != "assistant":
            raise ValueError(f"Exemple {index} invalide : trois messages system/user/assistant attendus.")
        prompt = self.processor.apply_chat_template(
            messages[:-1], add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", enable_thinking=False,
        )["input_ids"][0]
        answer = self.processor.tokenizer(
            messages[-1]["content"], add_special_tokens=False, return_tensors="pt"
        )["input_ids"][0]
        ids = torch.cat((prompt, answer, torch.tensor([self.eos_token_id], dtype=torch.long)))[: self.max_length]
        labels = ids.clone()
        labels[: min(len(prompt), len(ids))] = -100
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}


@dataclass
class SqlCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return {
            "input_ids": pad_sequence([x["input_ids"] for x in features], batch_first=True, padding_value=self.pad_token_id),
            "attention_mask": pad_sequence([x["attention_mask"] for x in features], batch_first=True, padding_value=0),
            "labels": pad_sequence([x["labels"] for x in features], batch_first=True, padding_value=-100),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=DEFAULT_MODELS,
                        help="Modèle(s) à entraîner. Sans cette option, un choix interactif est proposé.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--validation-dataset", type=Path, default=VALIDATION_DATASET)
    parser.add_argument("--output-root", type=Path, default=ROOT / "03_fine_tuning" / "artifacts" / "adapters")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32,
                        help="2 x 32 = batch effectif 64 sur une H100 mono-GPU.")
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler", choices=("cosine", "linear"), default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume-from-checkpoint", type=str)
    return parser.parse_args()


def choose_models() -> list[str]:
    print("Choisissez le(s) modèle(s) à entraîner :")
    for index, model_name in enumerate(DEFAULT_MODELS, start=1):
        print(f"  {index}) {model_name}")
    print("  5) Les quatre modèles")
    choice = input("Choix [5] : ").strip() or "5"
    choices = {str(index): [model_name] for index, model_name in enumerate(DEFAULT_MODELS, start=1)}
    choices["5"] = list(DEFAULT_MODELS)
    if choice not in choices:
        raise ValueError("Choix invalide. Entrez un nombre entre 1 et 5.")
    return choices[choice]


def restore_fp32_checkpoint_parameters(model: Any, model_path: Path) -> int:
    """Préserve les rares paramètres volontairement publiés en FP32.

    Les 0.8B/2B Qwen contiennent notamment ``linear_attn.A_log`` et des
    normalisations en FP32. ``torch_dtype=bfloat16`` est souhaité pour les
    couches principales mais convertir ces tenseurs de stabilité serait une
    perte de précision inutile. Ils ne représentent qu'une fraction minime de
    la VRAM et ne sont pas des cibles LoRA.
    """
    parameters = dict(model.named_parameters())
    restored = 0
    for weight_file in model_path.glob("*.safetensors"):
        with safe_open(str(weight_file), framework="pt", device="cpu") as archive:
            for name in archive.keys():
                if archive.get_slice(name).get_dtype() != "F32":
                    continue
                # Les checkpoints Qwen multimodaux préfixent les poids texte
                # par ``model.language_model``. AutoModelForCausalLM charge
                # toutefois le sous-modèle texte directement sous ``model``.
                # Traduire ce préfixe permet de restaurer les rares poids de
                # stabilité FP32 après le chargement BF16.
                parameter_name = name.replace("model.language_model.", "model.", 1)
                if parameter_name not in parameters:
                    raise KeyError(f"Paramètre FP32 introuvable dans le modèle : {name}")
                parameter = parameters[parameter_name]
                parameter.data = archive.get_tensor(name).to(dtype=torch.float32)
                restored += 1
    return restored


def train_model(
    model_name: str, args: argparse.Namespace, rows: list[dict[str, Any]], validation_rows: list[dict[str, Any]]
) -> None:
    model_path = ROOT / "models" / model_name
    if not model_path.is_dir():
        raise FileNotFoundError(f"Checkpoint absent : {model_path}")
    output_dir = args.output_root / model_name
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, local_files_only=True, attn_implementation="sdpa"
    )
    fp32_parameters = restore_fp32_checkpoint_parameters(model, model_path)
    if fp32_parameters:
        print(f"{model_name} : {fp32_parameters} paramètres de stabilité conservés en FP32.")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.CAUSAL_LM, r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules="all-linear", bias="none",
    ))
    model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir=str(output_dir), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler, optim="adamw_torch_fused", max_grad_norm=1.0,
        bf16=True, tf32=True, logging_steps=10, save_strategy="epoch", save_total_limit=2,
        eval_strategy="epoch", per_device_eval_batch_size=args.per_device_batch_size,
        load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
        report_to="none", remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=SqlDataset(rows, processor, args.max_seq_length),
        eval_dataset=SqlDataset(validation_rows, processor, args.max_seq_length),
        data_collator=SqlCollator(tokenizer.pad_token_id),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model()
    processor.save_pretrained(output_dir)
    (output_dir / "run_config.json").write_text(json.dumps({
        "base_model": str(model_path), "dataset": str(args.dataset), "examples": len(rows),
        "validation_dataset": str(args.validation_dataset), "validation_examples": len(validation_rows),
        "bf16": True, "restored_fp32_base_parameters": fp32_parameters,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "epochs": args.epochs, "learning_rate": args.learning_rate, "warmup_ratio": args.warmup_ratio,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length, "optimizer": "AdamW", "lr_scheduler": args.lr_scheduler,
        "gradient_clipping": 1.0,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("GPU CUDA requise : ce lanceur est prévu pour la H100.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 non pris en charge par cette GPU.")
    rows = read_jsonl(args.dataset)
    validation_rows = read_jsonl(args.validation_dataset)
    if not validation_rows:
        raise ValueError("Le dataset de validation est vide.")
    random.Random(args.seed).shuffle(rows)
    set_seed(args.seed)
    for model_name in args.models or choose_models():
        train_model(model_name, args, rows, validation_rows)


if __name__ == "__main__":
    main()
