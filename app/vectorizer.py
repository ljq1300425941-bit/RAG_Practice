import numpy as np

def text_to_vector(text: str, vocab: list[str]) -> np.ndarray:
    text = text.lower()
    values = []

    for word in vocab:
        values.append(text.count(word.lower()))

    return np.array(values, dtype=float)