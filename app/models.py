from dataclasses import dataclass

@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    text: str
    start_pos: int
    end_pos: int