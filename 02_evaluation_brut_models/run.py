"""Point d'entrée interactif indépendant de la machine d'exécution."""

from assets.run_benchmark import main


MODEL_NAMES = ("Qwen3.5-0.8B", "Qwen3.5-2B", "Qwen3.5-4B", "Qwen3.5-9B")


if __name__ == "__main__":
    main(
        default_model_names=MODEL_NAMES,
        default_device="auto",
        environment="auto",
        default_batch_size=1,
    )
