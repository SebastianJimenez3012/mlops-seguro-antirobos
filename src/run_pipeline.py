"""Orquestador secuencial simple de las etapas principales."""
import subprocess
import sys

STEPS = [
    ("Validación de datos", [sys.executable, "-m", "src.validate_data"]),
    ("Preparación de datos", [sys.executable, "-m", "src.prepare_data"]),
    ("Entrenamiento", [sys.executable, "-m", "src.train_pipeline"]),
    ("Quality gate", [sys.executable, "-m", "src.validate_model"]),
]


def main() -> None:
    for name, command in STEPS:
        print(f"\n=== {name} ===")
        subprocess.run(command, check=True)
    print("\nPipeline end-to-end completado correctamente.")


if __name__ == "__main__":
    main()
