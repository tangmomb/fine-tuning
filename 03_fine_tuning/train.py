"""Fine-tuning LoRA BF16 des checkpoints Qwen Text-to-SQL sur une H100.

Le script apprend exclusivement sur train.jsonl. Le split test Spider reste
réservé à 02_evaluation_brut_models afin d'éviter toute fuite de données.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from peft import LoraConfig, TaskType, get_peft_model
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "train.jsonl"
VALIDATION_DATASET = ROOT / "01_data" / "06_training_dataset" / "03_production" / "validation.jsonl"
VALIDATION_TRANSLATIONS = ROOT / "01_data" / "04_merge_translations" / "03_production" / "dev.jsonl"
VALIDATION_CHECKS = ROOT / "01_data" / "05_quality_control" / "03_production" / "01_deterministic_checks" / "dev_deterministic_checks.jsonl"
VALIDATION_JUDGMENTS = ROOT / "01_data" / "05_quality_control" / "03_production" / "09_judgments" / "sol_judgments.jsonl"
VALIDATION_DATABASES = ROOT / "BRUT_spider-original" / "data" / "spider_data" / "database"
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
    parser.add_argument("--output-root", type=Path, default=ROOT / "03_fine_tuning" / "artifacts")
    parser.add_argument("--run-name", type=str,
                        help="Nom du dossier de run. Par défaut : qwen3.5-<taille>-<date_heure_utc>.")
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


def make_run_directory(model_name: str, args: argparse.Namespace) -> Path:
    """Crée un dossier de run autonome et horodaté, sans écraser un run existant."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%SZ")
    default_name = f"{model_name.lower()}-{timestamp}"
    run_name = args.run_name or default_name
    if len(args.models or ()) > 1 and args.run_name:
        run_name = f"{args.run_name}-{model_name.lower()}"
    run_dir = args.output_root / run_name
    if run_dir.exists():
        raise FileExistsError(f"Le dossier de run existe déjà : {run_dir}. Choisissez --run-name différent.")
    for directory in ("checkpoints", "tokenizer", "training", "eval", "logs"):
        (run_dir / directory).mkdir(parents=True, exist_ok=False)
    return run_dir


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def validation_execution_rows(validation_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Retrouve les IDs, SQL gold et bases SQLite du split dev conservé pour le SFT."""
    translations = read_jsonl(VALIDATION_TRANSLATIONS)
    checks = {row["id"]: row for row in read_jsonl(VALIDATION_CHECKS)}
    judgments = {row["id"]: row for row in read_jsonl(VALIDATION_JUDGMENTS)}
    selected: list[tuple[str, dict[str, Any]]] = []
    for index, translation in enumerate(translations):
        identifier = f"dev:{index}"
        judgment = judgments.get(identifier)
        if judgment and judgment.get("verdict") != "pass":
            continue
        if not judgment and checks[identifier].get("status") != "pass":
            continue
        selected.append((identifier, translation))
    if len(selected) != len(validation_rows):
        raise ValueError(
            f"Validation SFT ({len(validation_rows)}) et métadonnées exécutables ({len(selected)}) divergent."
        )
    rows = []
    for sft_row, (identifier, translation) in zip(validation_rows, selected):
        messages = sft_row["messages"]
        prompt = json.loads(messages[1]["content"])
        if prompt != {"question": translation["question"], "schema": translation["schema"]}:
            raise ValueError(f"Validation SFT non alignée avec la traduction {identifier}.")
        rows.append({
            "id": identifier,
            "db_id": translation["db_id"],
            "sql": messages[2]["content"],
            "messages": messages[:2],
        })
    return rows


def evaluate_validation_execution(model: Any, processor: Any, validation_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Génère et score le SQL sur le split validation sans jamais consulter le split test."""
    assets_dir = str(ROOT / "02_evaluation_brut_models" / "assets")
    if assets_dir not in sys.path:
        sys.path.insert(0, assets_dir)
    evaluator = importlib.import_module("run_evaluation")
    evaluator.DATABASES = VALIDATION_DATABASES
    records = validation_execution_rows(validation_rows)
    device = next(model.parameters()).device
    processor.tokenizer.padding_side = "left"
    predictions: list[dict[str, Any]] = []
    was_training = model.training
    use_cache = model.config.use_cache
    model.eval()
    model.config.use_cache = True
    try:
        for start in range(0, len(records), 8):
            batch = records[start:start + 8]
            message_batches = [row["messages"] for row in batch]
            inputs = processor.apply_chat_template(
                message_batches, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", enable_thinking=False, processor_kwargs={"padding": True},
            )
            inputs.pop("mm_token_type_ids", None)
            inputs = {key: value.to(device) for key, value in inputs.items()}
            with torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, do_sample=False)
            generated = output[:, inputs["input_ids"].shape[1]:]
            for row, text in zip(batch, processor.batch_decode(generated, skip_special_tokens=True)):
                predictions.append({"id": row["id"], "raw_output": text})
    finally:
        model.config.use_cache = use_cache
        if was_training:
            model.train()
    gold = [{key: row[key] for key in ("id", "db_id", "sql")} for row in records]
    return evaluator.score(predictions, gold)


class RunArtifactCallback(TrainerCallback):
    """Archive l'adaptateur et les métriques à chaque validation d'époque."""

    def __init__(self, run_dir: Path, validation_examples: int) -> None:
        self.run_dir = run_dir
        self.validation_examples = validation_examples

    def on_evaluate(self, args: TrainingArguments, state: Any, control: Any,
                    metrics: dict[str, float] | None = None, **kwargs: Any) -> Any:
        model = kwargs["model"]
        epoch = round(float(state.epoch or 0))
        checkpoint_dir = self.run_dir / "checkpoints" / f"epoch-{epoch}"
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        report = {
            "epoch": epoch,
            "global_step": state.global_step,
            "validation_examples": self.validation_examples,
            "selection_metric": "eval_loss",
            **(metrics or {}),
        }
        eval_dir = self.run_dir / "eval" / f"epoch-{epoch}"
        eval_dir.mkdir(parents=True, exist_ok=True)
        write_json(eval_dir / f"epoch-{epoch}-validation.json", report)
        return control


class JsonlLogCallback(TrainerCallback):
    """Écrit les métriques brutes du Trainer au fil de l'eau, y compris en cas d'arrêt."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path

    def on_log(self, args: TrainingArguments, state: Any, control: Any,
               logs: dict[str, float] | None = None, **kwargs: Any) -> Any:
        record = {"step": state.global_step, "epoch": state.epoch, **(logs or {})}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        return control


def write_run_readme(run_dir: Path, model_name: str) -> None:
    (run_dir / "README.md").write_text(
        f"# Fine-tuning LoRA — {model_name}\n\n"
        "- `checkpoints/epoch-N/` : adaptateur LoRA sauvegardé après la validation de l'époque N.\n"
        "- `tokenizer/` : tokenizer et template de chat requis au rechargement.\n"
        "- `training/` : arguments et configuration reproductible du run.\n"
        "- `eval/epoch-N/epoch-N-validation.json` : loss de validation du Trainer pour l'époque N.\n"
        "- `eval/epoch-N/val_metrics.json` et `val_predictions.jsonl` : génération et exécution SQL "
        "du checkpoint N sur le split validation.\n"
        "- `eval/test_metrics.json` et `eval/test_predictions.jsonl` : évaluation finale sur le split test.\n"
        "- `logs/training_log.jsonl` : métriques brutes émises pendant l'entraînement.\n",
        encoding="utf-8",
    )


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
    run_dir = make_run_directory(model_name, args)
    write_run_readme(run_dir, model_name)
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
    # Transformers 5 utilise warmup_steps (warmup_ratio a été retiré).
    # Le lanceur vise une H100 mono-GPU ; le batch effectif détermine donc le
    # nombre de mises à jour par époque.
    updates_per_epoch = math.ceil(
        len(rows) / (args.per_device_batch_size * args.gradient_accumulation_steps)
    )
    warmup_steps = math.ceil(args.warmup_ratio * updates_per_epoch * args.epochs)
    training_args = TrainingArguments(
        # Les artefacts utiles sont archivés par RunArtifactCallback. Le
        # Trainer ne crée donc pas ses checkpoints internes redondants.
        output_dir=str(run_dir / "training" / "trainer_state"), num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate, warmup_steps=warmup_steps,
        lr_scheduler_type=args.lr_scheduler, optim="adamw_torch_fused", max_grad_norm=1.0,
        bf16=True, tf32=True, logging_steps=10, save_strategy="no",
        eval_strategy="epoch", per_device_eval_batch_size=args.per_device_batch_size,
        load_best_model_at_end=False,
        # Regroupe des séquences de longueurs proches : moins de padding et
        # des pics de VRAM plus prévisibles pour les schémas SQL les plus longs.
        train_sampling_strategy="group_by_length",
        report_to="none", remove_unused_columns=False, seed=args.seed,
    )
    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=SqlDataset(rows, processor, args.max_seq_length),
        eval_dataset=SqlDataset(validation_rows, processor, args.max_seq_length),
        data_collator=SqlCollator(tokenizer.pad_token_id),
        callbacks=[
            RunArtifactCallback(run_dir, len(validation_rows)),
            JsonlLogCallback(run_dir / "logs" / "training_log.jsonl"),
        ],
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    processor.save_pretrained(run_dir / "tokenizer")
    torch.save(training_args, run_dir / "training" / "training_args.bin")
    validation_reports = sorted((run_dir / "eval").glob("epoch-*/epoch-*-validation.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in validation_reports]
    best = min(reports, key=lambda report: report["eval_loss"]) if reports else None
    execution_reports = []
    # Après la fin des époques, recharger chaque adaptateur gelé et l'évaluer
    # séparément : l'inférence ne perturbe jamais l'entraînement ni son état.
    for report in reports:
        epoch = report["epoch"]
        checkpoint = run_dir / "checkpoints" / f"epoch-{epoch}"
        adapter_name = f"epoch-{epoch}"
        model.load_adapter(str(checkpoint), adapter_name=adapter_name)
        model.set_adapter(adapter_name)
        predictions, execution_metrics = evaluate_validation_execution(model, processor, validation_rows)
        execution_report = {
            "checkpoint": f"checkpoints/epoch-{epoch}",
            "validation_examples": len(validation_rows),
            **execution_metrics,
        }
        eval_dir = run_dir / "eval" / f"epoch-{epoch}"
        write_json(eval_dir / "val_metrics.json", execution_report)
        write_jsonl(eval_dir / "val_predictions.jsonl", predictions)
        execution_reports.append(execution_report)
    write_json(run_dir / "training" / "run_config.json", {
        "base_model": str(model_path), "dataset": str(args.dataset), "examples": len(rows),
        "validation_dataset": str(args.validation_dataset), "validation_examples": len(validation_rows),
        "bf16": True, "restored_fp32_base_parameters": fp32_parameters,
        "lora": {"r": 16, "alpha": 32, "dropout": 0.05, "target_modules": "all-linear"},
        "epochs": args.epochs, "learning_rate": args.learning_rate, "warmup_ratio": args.warmup_ratio,
        "effective_batch_size": args.per_device_batch_size * args.gradient_accumulation_steps,
        "max_seq_length": args.max_seq_length, "optimizer": "AdamW", "lr_scheduler": args.lr_scheduler,
        "gradient_clipping": 1.0,
        "run_directory": str(run_dir),
        "lowest_eval_loss_checkpoint": (f"checkpoints/epoch-{best['epoch']}" if best else None),
        "lowest_eval_loss": (best["eval_loss"] if best else None),
        "validation_execution_by_epoch": execution_reports,
    })
    print(f"Run terminé : {run_dir}")
    if best:
        print(f"Meilleure eval_loss : checkpoints/epoch-{best['epoch']} ({best['eval_loss']:.4f})")


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
