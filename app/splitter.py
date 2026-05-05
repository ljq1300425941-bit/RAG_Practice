import re
from pathlib import Path

from app.models import DocumentChunk
from app.loader import load_text_files


def build_chunks_from_dir(
    input_dir: Path,
    chunk_size: int,
    overlap: int,
    splitter_type: str = "fixed",
) -> list[DocumentChunk]:
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
            splitter_type=splitter_type,
        )
        all_chunks.extend(chunks)

    return all_chunks


def split_text_to_chunks(
    text: str,
    source_file: str,
    chunk_size: int,
    overlap: int,
    splitter_type: str = "fixed",
) -> list[DocumentChunk]:
    if splitter_type == "fixed":
        return split_text_to_chunks_fixed(
            text=text,
            source_file=source_file,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    if splitter_type == "paragraph":
        return split_text_to_chunks_by_paragraph(
            text=text,
            source_file=source_file,
            chunk_size=chunk_size,
        )

    if splitter_type == "sentence":
        return split_text_to_chunks_by_sentence(
            text=text,
            source_file=source_file,
            chunk_size=chunk_size,
            overlap=overlap,
        )

    raise ValueError(f"未知 splitter_type: {splitter_type}")


def split_text_to_chunks_fixed(
    text: str,
    source_file: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
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
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{Path(source_file).stem}_chunk_{chunk_index}",
                    source_file=source_file,
                    text=chunk_text,
                    start_pos=start,
                    end_pos=min(end, len(text)),
                )
            )
            chunk_index += 1

        if end >= len(text):
            break

    return chunks


def split_text_to_chunks_by_paragraph(
    text: str,
    source_file: str,
    chunk_size: int,
) -> list[DocumentChunk]:
    text = text.strip()
    if not text or chunk_size <= 0:
        return []

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n+", text)
        if p.strip()
    ]

    segments = []
    for paragraph in paragraphs:
        if len(paragraph) <= chunk_size:
            segments.append(paragraph)
        else:
            segments.extend(_split_long_text(paragraph, chunk_size))

    return _build_chunks_from_segments(
        segments=segments,
        source_file=source_file,
        original_text=text,
    )


def split_text_to_chunks_by_sentence(
    text: str,
    source_file: str,
    chunk_size: int,
    overlap: int,
) -> list[DocumentChunk]:
    text = text.strip()
    if not text or chunk_size <= 0:
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    segments = []
    current_sentences = []
    current_length = 0

    for sentence in sentences:
        if not current_sentences:
            current_sentences.append(sentence)
            current_length = len(sentence)
            continue

        candidate_length = current_length + len(sentence)

        if candidate_length <= chunk_size:
            current_sentences.append(sentence)
            current_length = candidate_length
        else:
            segments.append("".join(current_sentences))

            if overlap > 0 and current_sentences:
                kept = current_sentences[-overlap:]
                current_sentences = kept + [sentence]
                current_length = sum(len(s) for s in current_sentences)
            else:
                current_sentences = [sentence]
                current_length = len(sentence)

    if current_sentences:
        segments.append("".join(current_sentences))

    final_segments = []
    for segment in segments:
        if len(segment) <= chunk_size:
            final_segments.append(segment)
        else:
            final_segments.extend(_split_long_text(segment, chunk_size))

    return _build_chunks_from_segments(
        segments=final_segments,
        source_file=source_file,
        original_text=text,
    )


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？；.!?;])\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_long_text(text: str, chunk_size: int) -> list[str]:
    return [
        text[start:start + chunk_size].strip()
        for start in range(0, len(text), chunk_size)
        if text[start:start + chunk_size].strip()
    ]


def _build_chunks_from_segments(
    segments: list[str],
    source_file: str,
    original_text: str,
) -> list[DocumentChunk]:
    chunks = []
    search_start = 0

    for index, segment in enumerate(segments):
        start_pos = original_text.find(segment, search_start)

        if start_pos == -1:
            start_pos = search_start

        end_pos = start_pos + len(segment)
        search_start = end_pos

        chunks.append(
            DocumentChunk(
                chunk_id=f"{Path(source_file).stem}_chunk_{index}",
                source_file=source_file,
                text=segment,
                start_pos=start_pos,
                end_pos=end_pos,
            )
        )

    return chunks