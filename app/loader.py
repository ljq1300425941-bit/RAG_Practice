from pathlib import Path


def load_text_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists() or not input_dir.is_dir():
        return []

    return sorted(input_dir.rglob("*.txt"))

