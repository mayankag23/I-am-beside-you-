from llmstudy.ingest import split_text


def test_split_empty():
    assert split_text("") == []


def test_split_sizes():
    text = "a" * 2500
    chunks = split_text(text, chunk_size=1000, overlap=200)
    # Expect at least 3 chunks
    assert len(chunks) >= 3
    # No chunk should be empty
    assert all(len(c) > 0 for c in chunks)
