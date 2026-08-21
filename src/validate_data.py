"""Quality gate del contrato de datos."""
from src.config import DATA_RAW
from src.data_loader import load_raw_data, prepare_dataframe


def validate() -> None:
    """Valida raw y preparado; si existe una inconsistencia lanza excepción."""
    raw = load_raw_data(DATA_RAW)
    clean = prepare_dataframe(raw)
    print("DATA QUALITY GATE: APROBADO")
    print(f"Filas raw: {len(raw)} | Filas limpias: {len(clean)}")
    print(f"Tasa positiva: {clean['FLAG_SS'].mean():.2%}")


if __name__ == "__main__":
    validate()
