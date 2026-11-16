"""Мова повідомлення (uk / ru / en) через langid. Евристика: на коротких текстах можливі помилки."""
from __future__ import annotations

import numpy as np


def detect_language(texts, langs: tuple[str, ...] = ("uk", "ru", "en")) -> np.ndarray:
    import langid

    langid.set_languages(list(langs))
    return np.array([langid.classify(str(t))[0] for t in texts])
