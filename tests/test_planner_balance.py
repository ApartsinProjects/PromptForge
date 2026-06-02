"""Unit tests for Fix A (v2.9.6): balanced planner enforces per-class
ceil(n/K) samples and uses across-iter ``existing`` to compensate for
prior-iter imbalances.
"""
from __future__ import annotations

from collections import Counter

from attrforge.planner import AttributePlanner, PlannerConfig
from attrforge.schema import AttributeSchema, SyntheticSample


def _schema_10way() -> AttributeSchema:
    return AttributeSchema(
        label_attribute="intent",
        attributes={
            "intent": [f"class_{i}" for i in range(10)],
            "style": ["formal", "informal"],
        },
    )


def test_class_balanced_distributes_evenly_within_single_iter():
    """16 samples / 10 classes -> ceil(16/10)=2 per class; with the round-
    robin and shuffle, every class should appear at least once.
    """
    planner = AttributePlanner(_schema_10way(), PlannerConfig(seed=17, class_balanced=True))
    targets = planner.plan(n=16)
    counts = Counter(t.values["intent"] for t in targets)
    # 16 targets, 10 classes -> per_class_target=2; queue length = sum(deficits)=20
    # truncated to 16; every class gets at least 1.
    assert len(targets) == 16
    assert min(counts.values()) >= 1
    assert max(counts.values()) <= 2


def test_class_balanced_compensates_for_prior_iter_underfill():
    """If existing samples are heavily skewed toward class_0, the next
    iter's plan should preferentially fill the under-represented classes.
    """
    schema = _schema_10way()
    planner = AttributePlanner(schema, PlannerConfig(seed=17, class_balanced=True))
    # Pretend iter 1 produced 10 samples, all of class_0 (extreme skew).
    existing = [
        SyntheticSample(
            sample_id=f"prior_{i}",
            text=f"text {i}",
            requested_attributes={"intent": "class_0", "style": "formal"},
        )
        for i in range(10)
    ]
    targets = planner.plan(n=16, existing=existing)
    counts = Counter(t.values["intent"] for t in targets)
    # class_0 already has 10; per_class_target=ceil(26/10)=3; class_0 deficit=0
    # other classes have deficit=3 each. Queue=27 truncated to 16.
    # class_0 should appear LESS than the others.
    if "class_0" in counts:
        assert counts["class_0"] <= 1, f"class_0 should be down-weighted; got {counts}"
    # The 9 other classes share 16 slots -> at least most appear.
    non_zero_classes = [c for c in [f"class_{i}" for i in range(1, 10)] if counts.get(c, 0) >= 1]
    assert len(non_zero_classes) >= 8, f"most non-class-0 should be filled; counts={counts}"


def test_class_balanced_handles_single_class_schema_gracefully():
    """Degenerate case: only one class value -> falls back to stratified."""
    schema = AttributeSchema(
        label_attribute="intent",
        attributes={"intent": ["only"], "style": ["formal", "informal"]},
    )
    planner = AttributePlanner(schema, PlannerConfig(seed=17, class_balanced=True))
    targets = planner.plan(n=8)
    assert len(targets) == 8
    # All targets should have intent='only'
    assert all(t.values["intent"] == "only" for t in targets)


def test_class_balanced_disabled_falls_back_to_stratified():
    """When class_balanced=False, the planner uses the legacy stratified
    sampling which may produce imbalanced class distributions."""
    planner = AttributePlanner(
        _schema_10way(),
        PlannerConfig(seed=17, class_balanced=False, strategy="stratified"),
    )
    targets = planner.plan(n=16)
    assert len(targets) == 16
    # Stratified is random; can't assert exact balance.
