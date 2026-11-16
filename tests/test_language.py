import pytest

pytest.importorskip("langid")

from narr.language import detect_language  # noqa: E402


def test_detects_ukrainian_and_russian():
    texts = ["Сьогодні в місті було оголошено повітряну тривогу, залишайтесь в укриттях.",
             "Сегодня в городе была объявлена воздушная тревога, оставайтесь в укрытиях."]
    assert list(detect_language(texts)) == ["uk", "ru"]


def test_parallel_matches_sequential():
    from narr.language import detect_language_parallel

    base = ["Сьогодні в місті було оголошено повітряну тривогу.", "Сегодня в городе была объявлена воздушная тревога.",
            "Today the city announced an air raid alert."]
    texts = base * 40
    seq = detect_language(texts)
    par = detect_language_parallel(texts, n_jobs=2, min_per_job=10)
    assert list(par) == list(seq)
    assert list(detect_language_parallel(texts, n_jobs=1)) == list(seq)
