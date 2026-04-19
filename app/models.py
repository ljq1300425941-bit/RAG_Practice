from dataclasses import dataclass
import numpy as np

@dataclass
class DocumentChunk:
    chunk_id: str
    source_file: str
    text: str
    start_pos: int
    end_pos: int

@dataclass
class ChunkEmbedding:
    chunk:DocumentChunk
    embedding:np.ndarray
