from pathlib import Path
from app.models import DocumentChunk
from app.loader import load_text_files

def build_chunks_from_dir(input_dir: Path, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    all_chunks = []
    files = load_text_files(input_dir)

    for file_path in files:
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            print(f"读取文件失败: {file_path}, error={e}")
            continue

        chunks = split_text_to_chunks(
            text=text,
            source_file=file_path.name,
            chunk_size=chunk_size,
            overlap=overlap,
        )
        all_chunks.extend(chunks)

    return all_chunks


def split_text_to_chunks(text: str, source_file: str, chunk_size: int, overlap: int) -> list[DocumentChunk]:
    chunks = []

    text = text.strip()
    if not text or chunk_size <= 0:
        return chunks

    step = chunk_size - overlap
    if step <= 0:
        return chunks

    chunk_index = 0

    for start in range(0, len(text), step):
        end = start + chunk_size
        chunk_text = text[start:end].strip()

        if chunk_text:
            chunk = DocumentChunk(
                chunk_id=f"{Path(source_file).stem}_chunk_{chunk_index}",
                source_file=source_file,
                text=chunk_text,
                start_pos=start,
                end_pos=min(end, len(text)),
            )
            chunks.append(chunk)
            chunk_index += 1

        if end >= len(text):
            break

    return chunks
