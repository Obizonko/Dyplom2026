"""Підсумок ручної розмітки «явний / неявний» і сліпа вибірка для перевірки."""
import pandas as pd

from narr import splits as sp


def _lab():
    rows = []
    for i in range(1, 41):
        e = "explicit" if i <= 15 else ("implicit" if i <= 35 else "unclear")
        c = {"explicit": "direct", "implicit": "framing", "unclear": "unrelated"}[e] + (";borderline" if i in (3, 20) else "")
        rows.append({"n": i, "channel_id": 1, "message_id": i, "lang": "ru" if i % 5 else "uk", "narrative_label": "x", "text": f"t{i}", "explicitness": e, "comment": c})
    return pd.DataFrame(rows)


def test_summary_counts_and_sensitivity():
    lab = _lab()
    m = sp.implicit_manual_summary(lab, min_implicit=10)
    assert m["n_rows"] == 40 and m["n_filled"] == 40 and not m["bad_values"] and not m["bad_codes"]
    assert (m["explicit"]["all"], m["implicit"]["all"], m["unclear"]["all"]) == (15, 20, 5)
    assert m["implicit"]["all"] == m["implicit"]["ru"] + m["implicit"]["uk"]
    assert m["n_borderline"] == 2
    assert m["implicit_narrow"]["all"] == 19      # №20 неявний borderline
    assert m["implicit_wide"]["all"] == 21        # №3 явний borderline
    assert m["by_code"]["unrelated"] == 5


def test_summary_flags_invalid_values_and_enough_flag():
    lab = _lab()
    lab.loc[0, "explicitness"] = "Explicit "
    lab.loc[1, "comment"] = "weird"
    lab.loc[2, "explicitness"] = ""
    m = sp.implicit_manual_summary(lab)
    assert "Explicit" in m["bad_values"] or "" in m["bad_values"]
    assert "weird" in m["bad_codes"]
    assert m["n_filled"] == 39
    assert not m["enough_ru"]["base"]                     # 20 неявних усього, менше 30


def test_spotcheck_is_blind_balanced_and_reproducible():
    lab = _lab()
    s1, s2 = sp.spotcheck_sample(lab, n_per_class=5, seed=42), sp.spotcheck_sample(lab, n_per_class=5, seed=42)
    assert s1.equals(s2) and len(s1) == 10
    assert "explicitness" not in s1.columns and "comment" not in s1.columns
    got = lab[lab["n"].isin(s1["n"])]["explicitness"].value_counts().to_dict()
    assert got == {"explicit": 5, "implicit": 5}


def test_spotcheck_agreement():
    lab = _lab()
    u = sp.spotcheck_sample(lab, n_per_class=5, seed=1)
    assert sp.spotcheck_agreement(lab, u) is None
    truth = lab.set_index("n")["explicitness"]
    u["explicitness_user"] = [truth[n] for n in u["n"]]
    u.loc[0, "explicitness_user"] = "unclear"
    u.loc[1, "explicitness_user"] = "explicit" if truth[u.loc[1, "n"]] == "implicit" else "implicit"
    a = sp.spotcheck_agreement(lab, u)
    assert a == {"n_checked": 10, "n_binary": 9, "n_agree": 8, "n_user_unclear": 1}
