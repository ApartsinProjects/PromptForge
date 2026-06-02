"""Unit tests for the Mode Hunter's deterministic helpers.

The Mode Hunter's main hunt() method calls an LLM. Two helpers
(_count_substring and top_ngrams) are deterministic and used to verify
candidate banned patterns. We test them here without any API key.
"""
from __future__ import annotations

from attrforge.critics.mode_hunter import ModeHunter, ModeHunterConfig


def test_count_substring_counts_corpus_occurrences():
    """_count_substring returns the number of corpus entries containing the substring."""
    corpus = [
        "Hello team, hope you are well.",
        "Hello team, just checking in.",
        "Hello team, quick question.",
        "Different opener entirely.",
    ]
    assert ModeHunter._count_substring("Hello team,", corpus) == 3


def test_count_substring_zero_for_missing_pattern():
    """An LLM-returned candidate that does not actually appear, count = 0."""
    corpus = ["Hello team, hope you are well.", "Hello team, just checking in."]
    assert ModeHunter._count_substring("absolutely never said", corpus) == 0


def test_top_ngrams_returns_most_frequent_synth_ngrams():
    """top_ngrams surfaces frequent n-grams from a text list as (ngram, count) pairs."""
    texts = [
        "Hello team, can I help you?",
        "Hello team, what can I do?",
        "Hello team, is there an issue?",
    ]
    pairs = ModeHunter.top_ngrams(texts, n=2, top_k=5)
    joined = " | ".join(f"{ng}:{ct}" for ng, ct in pairs)
    assert "hello team" in joined.lower()
    # Counts are positive ints.
    assert all(isinstance(ct, int) and ct > 0 for _, ct in pairs)


def test_library_property_starts_empty():
    """A fresh ModeHunter has no remembered findings."""
    hunter = ModeHunter(client=None, config=ModeHunterConfig())
    assert hunter.library == []


def test_is_domain_canonical_vetoes_overlapping_content_words():
    """A pattern whose content words have >=50% overlap with the real
    seed is judged domain-canonical and should NOT be banned, even when
    the exact phrasing is absent from the seed (which is the dominant
    case at small N).
    """
    real = [
        "the visuals are stunning; the performances are uneven, and the score lingers",
        "an arthritic attempt at directing; pacing collapses in the third act",
        "long after the credits roll you will remember the lead performance",
        "cinematography that breathes; a quietly devastating piece of cinema",
    ]
    # All content words present (or 4-char-prefix match) in real -> VETO ban.
    assert ModeHunter._is_domain_canonical("the visuals are", real)
    assert ModeHunter._is_domain_canonical("the performances are", real)
    assert ModeHunter._is_domain_canonical("long after the credits roll", real)


def test_is_domain_canonical_allows_genuine_artifacts():
    """A pattern whose content words have no real-seed overlap is allowed
    to be banned -- this is a genuine synthesis artifact."""
    real = [
        "the visuals are stunning; the performances are uneven",
        "an arthritic attempt at directing; pacing collapses in the third act",
    ]
    # Nonsense / foreign vocabulary not in real seed -> allow ban.
    assert not ModeHunter._is_domain_canonical("xyzzy plover", real)
    # Customer-support style phrases on a film-criticism real seed -> allow ban.
    assert not ModeHunter._is_domain_canonical("I understand your frustration", real)


def test_is_domain_canonical_handles_morphological_variation():
    """Stem-style matching via 4-char prefix catches the dominant
    morphological cases: ``visuals`` matches ``visual``, ``credits``
    matches ``credit``, ``performances`` matches ``performance``.

    A known limitation: short-stem variants whose morpheme splits before
    the 4th character (e.g., ``pace`` -> ``pacing`` differ at char 4)
    are not detected by 4-char prefix matching. This is accepted as a
    conservative-side false negative; the Updater's positive
    coverage-hole guidance compensates."""
    real = [
        "a visual feast that lingers",
        "a credit roll over a black screen",
        "the performance carries the film",
    ]
    assert ModeHunter._is_domain_canonical("stunning visuals", real)  # visuals~visual
    assert ModeHunter._is_domain_canonical("the credits roll", real)  # credits~credit
    assert ModeHunter._is_domain_canonical("the performances are", real)  # performances~performance


def test_veto_can_be_disabled_via_config():
    """When ``veto_domain_canonical=False`` the soft veto is bypassed
    and the original strict ``n_real_obs > 0`` check is the only filter."""
    cfg = ModeHunterConfig(veto_domain_canonical=False)
    hunter = ModeHunter(client=None, config=cfg)
    assert hunter.config.veto_domain_canonical is False
