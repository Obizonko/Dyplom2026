"""Мова повідомлення (uk / ru / en) через langid. Евристика: на коротких текстах можливі помилки."""
from __future__ import annotations

import numpy as np


def detect_language(texts, langs: tuple[str, ...] = ("uk", "ru", "en")) -> np.ndarray:
    import langid

    langid.set_languages(list(langs))
    return np.array([langid.classify(str(t))[0] for t in texts])


def _detect_chunk(texts: list[str]) -> list[str]:
    return list(detect_language(texts))


def detect_language_parallel(texts, n_jobs: int = 2, min_per_job: int = 2000) -> np.ndarray:
    """Те саме, що detect_language, але частинами в кількох процесах. Порядок збережено.
    Для малих вхідних масивів або n_jobs <= 1 працює послідовно."""
    texts = [str(t) for t in texts]
    if n_jobs <= 1 or len(texts) < n_jobs * min_per_job:
        return detect_language(texts)
    from multiprocessing import get_all_start_methods, get_context

    # fork є лише в Linux/macOS; на Windows працює spawn (функція верхнього рівня, тож це безпечно)
    method = "fork" if "fork" in get_all_start_methods() else "spawn"
    chunks = [c.tolist() for c in np.array_split(np.array(texts, dtype=object), n_jobs * 8) if len(c)]
    with get_context(method).Pool(n_jobs) as pool:
        parts = pool.map(_detect_chunk, chunks)
    return np.array([x for part in parts for x in part])
