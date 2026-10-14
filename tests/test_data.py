"""Mechanical checks on the experiment's input data (spec section 4)."""

import json
import re
from collections import Counter
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parent.parent / "data"
TOPICS = ["food", "travel", "products", "work", "weather", "films"]
ARABIC = re.compile(r"[؀-ۿ]")
LATIN = re.compile(r"[A-Za-z]")


def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def words(text: str) -> list[str]:
    return re.findall(r"[\w']+", text.lower())


def arabic_share(text: str) -> float:
    letters = [c for c in text if ARABIC.match(c) or LATIN.match(c)]
    return sum(1 for c in letters if ARABIC.match(c)) / len(letters)


def test_pairs_count_ids_and_topic_balance() -> None:
    pairs = load("pairs")
    assert len(pairs) == 60
    assert [p["id"] for p in pairs] == [f"p{i:02d}" for i in range(1, 61)]
    assert Counter(p["topic"] for p in pairs) == {t: 10 for t in TOPICS}


@pytest.mark.parametrize("pair", load("pairs"), ids=lambda p: p["id"])
def test_pair_is_minimal(pair: dict) -> None:
    pos, neg = words(pair["positive"]), words(pair["negative"])
    assert pos != neg
    # Same frame: at most 2 words swapped per side, length within 1 word.
    assert abs(len(pos) - len(neg)) <= 1
    assert len(set(pos) ^ set(neg)) <= 4
    assert not ARABIC.search(pair["positive"] + pair["negative"])


def test_prompts_are_aligned_translations() -> None:
    prompts = load("prompts")
    assert len(prompts) == 60
    by_lang = {lang: [p for p in prompts if p["lang"] == lang] for lang in ("en", "ar")}
    expected = [f"q{i:02d}" for i in range(1, 31)]
    assert [p["id"] for p in by_lang["en"]] == expected
    assert [p["id"] for p in by_lang["ar"]] == expected
    assert len({p["text"] for p in by_lang["en"]}) == 30
    for p in by_lang["en"]:
        assert not ARABIC.search(p["text"])
    for p in by_lang["ar"]:
        assert arabic_share(p["text"]) > 0.95, p


def test_reference_texts_are_aligned_translations() -> None:
    refs = load("reference")
    assert len(refs) == 40
    for lang in ("en", "ar"):
        ids = [r["id"] for r in refs if r["lang"] == lang]
        assert ids == [f"r{i:02d}" for i in range(1, 21)]
    for r in refs:
        if r["lang"] == "ar":
            assert arabic_share(r["text"]) > 0.95, r


def test_instructions_cover_both_tones_in_both_languages() -> None:
    instr = load("instructions")
    assert {(i["id"], i["lang"]) for i in instr} == {
        ("pos", "en"), ("neg", "en"), ("pos", "ar"), ("neg", "ar"),
    }
    for i in instr:
        if i["lang"] == "ar":
            assert arabic_share(i["text"]) > 0.95, i
