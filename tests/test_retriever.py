from app.splitter import split_text_to_chunks


def test_split_text_to_chunks_basic():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = split_text_to_chunks(
        text=text,
        source_file="test.txt",
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) > 0
    assert chunks[0].source_file == "test.txt"
    assert chunks[0].chunk_id.startswith("test_chunk_")
    assert chunks[0].start_pos == 0
    assert chunks[0].end_pos == 10


def test_split_text_to_chunks_empty_text():
    chunks = split_text_to_chunks(
        text="",
        source_file="test.txt",
        chunk_size=10,
        overlap=2,
    )
    assert chunks == []


def test_split_text_to_chunks_invalid_step():
    chunks = split_text_to_chunks(
        text="abcdefg",
        source_file="test.txt",
        chunk_size=4,
        overlap=4,
    )
    assert chunks == []