import json
from pathlib import Path

CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

DATA_CKPT_FILE = CKPT_DIR / "data_pipeline_state.json"


def load_data_checkpoint():
    if not DATA_CKPT_FILE.exists():
        print("🔹 No data pipeline checkpoint found. Starting fresh.")
        return {"sequences_seen": 0}

    try:
        with open(DATA_CKPT_FILE, "r") as f:
            return json.load(f)
    except Exception:
        print("⚠️ Corrupt data checkpoint. Resetting.")
        return {"sequences_seen": 0}


def save_data_checkpoint(sequences_seen: int):
    with open(DATA_CKPT_FILE, "w") as f:
        json.dump(
            {"sequences_seen": sequences_seen},
            f,
            indent=2
        )
