import pytest

pytest.importorskip("langid")

from narr.language import detect_language  # noqa: E402


def test_detects_ukrainian_and_russian():
    texts = ["Сьогодні в місті було оголошено повітряну тривогу, залишайтесь в укриттях.",
             "Сегодня в городе была объявлена воздушная тревога, оставайтесь в укрытиях."]
    assert list(detect_language(texts)) == ["uk", "ru"]
