"""Etapa de ingestión/preparación reproducible."""
import json
from hashlib import sha256

from src.config import DATA_PROCESSED, DATA_RAW
from src.data_loader import load_raw_data, prepare_dataframe


def file_hash(path) -> str:
    """Retorna SHA256 del archivo para trazabilidad."""
    digest = sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_and_save() -> dict:
    """Carga raw, valida, limpia y persiste el dataset procesado."""
    raw = load_raw_data(DATA_RAW)
    clean = prepare_dataframe(raw)
    DATA_PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(DATA_PROCESSED, index=False)
    metadata = {
        "source": str(DATA_RAW.relative_to(DATA_RAW.parents[2])),
        "raw_rows": int(len(raw)),
        "processed_rows": int(len(clean)),
        "duplicates_removed": int(len(raw) - len(clean)),
        "positive_rate": round(float(clean["FLAG_SS"].mean()), 6),
        "raw_sha256": file_hash(DATA_RAW),
    }
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return metadata


if __name__ == "__main__":
    prepare_and_save()
