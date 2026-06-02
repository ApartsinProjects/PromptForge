"""Attribute Planner.

Decides which attribute vectors to ask the generator to produce next.

Two strategies are provided:

* ``stratified``: marginal balance across each attribute, sampled jointly.
  Cheap, no LLM call, useful as a default.

* ``coverage_gap``: targets attribute combinations that have low coverage
  in the dataset so far. The auditor's missing-mode hints can be folded
  in directly via ``targeted_combinations``.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass

from attrforge.schema import AttributeSchema, AttributeVector, SyntheticSample


@dataclass
class PlannerConfig:
    strategy: str = "stratified"
    batch_size: int = 16
    seed: int | None = None
    # When True (v2.9.6 default), the planner enforces explicit per-class
    # balance: each value of the schema's label_attribute receives at least
    # ceil(n / K) targets per batch, with leftover n % K randomized. Without
    # this, random sampling can leave a class with 0 or 1 synth examples
    # (the Banking77 seed-17 case: card_not_working got 1 sample of 48 ->
    # per-class accuracy 0.325 vs 0.95 real-only).
    class_balanced: bool = True


class AttributePlanner:
    """Produce target attribute vectors for the next generation batch."""

    def __init__(self, schema: AttributeSchema, config: PlannerConfig | None = None) -> None:
        self.schema = schema
        self.config = config or PlannerConfig()
        self._rng = random.Random(self.config.seed)
        self._counter = 0

    def plan(
        self,
        n: int | None = None,
        *,
        existing: list[SyntheticSample] | None = None,
        targeted_combinations: list[dict[str, str]] | None = None,
    ) -> list[AttributeVector]:
        """Return ``n`` target vectors using the configured strategy.

        ``targeted_combinations`` lets the diversity auditor seed the
        planner with under-represented modes. These vectors are emitted
        first; the remainder is filled by the configured strategy.
        """
        n = n if n is not None else self.config.batch_size
        out: list[AttributeVector] = []

        if targeted_combinations:
            for partial in targeted_combinations[:n]:
                full = self._fill_partial(partial)
                if full is not None:
                    out.append(self._wrap(full))

        if self.config.class_balanced and self.schema.label_attribute:
            out.extend(self._class_balanced(n - len(out), out, existing or []))
        elif self.config.strategy == "stratified":
            out.extend(self._stratified(n - len(out)))
        elif self.config.strategy == "coverage_gap":
            out.extend(self._coverage_gap(n - len(out), existing or []))
        else:
            raise ValueError(f"Unknown planner strategy: {self.config.strategy!r}")

        return out[:n]

    def _class_balanced(
        self,
        n: int,
        already_picked: list[AttributeVector],
        existing_samples: list[SyntheticSample],
    ) -> list[AttributeVector]:
        """Explicit per-class balance: every label value gets ceil(n/K) targets.

        The non-class attributes are randomized as in the stratified
        strategy. This is the dominant fix for the Banking77 seed-17
        failure mode (one class got 1 sample of 48 by chance). When
        targeted_combinations have already populated some class slots,
        the remaining quota is adjusted accordingly so each class
        finishes at >=ceil(n_total/K) targets.

        Falls back gracefully if the label_attribute is missing or has
        only one allowed value (degenerate case).
        """
        if n <= 0:
            return []
        label_attr = self.schema.label_attribute
        if not label_attr:
            return self._stratified(n)
        allowed_labels = list(self.schema.values(label_attr) or [])
        if len(allowed_labels) <= 1:
            return self._stratified(n)
        # Cross-iter balance: also count prior-iter accepted samples so a
        # class that was under-filled in iter 1 receives more in iter 2.
        already_per_class: dict[str, int] = {v: 0 for v in allowed_labels}
        for av in already_picked:
            lbl = av.values.get(label_attr)
            if lbl in already_per_class:
                already_per_class[lbl] += 1
        for sample in existing_samples:
            lbl = sample.requested_attributes.get(label_attr)
            if lbl in already_per_class:
                already_per_class[lbl] += 1
        # Total target per class is ceil((total batch over all iters) / K).
        K = len(allowed_labels)
        total_batch = len(existing_samples) + len(already_picked) + n
        per_class_target = -(-total_batch // K)  # ceil(total / K)
        # Build the queue of class labels that need samples via ROUND-ROBIN
        # over the deficits. This distributes n slots fairly across labels
        # with equal deficit, rather than greedily filling the first label
        # to its full quota before moving on (which left some classes at 0
        # when len(deficits-queue) > n and many classes had equal deficit).
        per_label_remaining: dict[str, int] = {
            label: max(0, per_class_target - already_per_class[label])
            for label in allowed_labels
        }
        queue: list[str] = []
        # Round-robin: in each pass, give one slot to every label that
        # still has deficit, in order of largest remaining deficit first.
        while len(queue) < n and any(v > 0 for v in per_label_remaining.values()):
            # Order labels by remaining deficit (largest first); tie-break
            # via the RNG so ordering is not deterministic across calls.
            labels_with_deficit = [
                lab for lab, rem in per_label_remaining.items() if rem > 0
            ]
            self._rng.shuffle(labels_with_deficit)
            labels_with_deficit.sort(
                key=lambda lab: per_label_remaining[lab], reverse=True,
            )
            for label in labels_with_deficit:
                if len(queue) >= n:
                    break
                queue.append(label)
                per_label_remaining[label] -= 1
        # Pad with randomly-selected labels if queue is still shorter than
        # n (e.g. all deficits were 0 because the prior iters already over-
        # filled every class).
        while len(queue) < n:
            queue.append(self._rng.choice(allowed_labels))
        # Final shuffle so the order of generation is randomized.
        self._rng.shuffle(queue)
        # Now build the AttributeVector for each queued label by
        # randomizing the other attributes (stratified per non-class
        # attribute) and respecting schema constraints.
        out: list[AttributeVector] = []
        for label in queue[:n]:
            values: dict[str, str] | None = None
            for _ in range(32):  # max retries per slot for constraint sat
                cand = {label_attr: label}
                for name in self.schema.names():
                    if name == label_attr:
                        continue
                    cand[name] = self._rng.choice(self.schema.values(name))
                if self.schema.is_valid(cand):
                    values = cand
                    break
            if values is not None:
                out.append(self._wrap(values))
        return out

    def _wrap(self, values: dict[str, str]) -> AttributeVector:
        self._counter += 1
        return AttributeVector(
            sample_id=f"target_{self._counter:05d}", values=values
        )

    def _fill_partial(self, partial: dict[str, str]) -> dict[str, str] | None:
        """Fill missing attributes randomly while respecting constraints.

        Tries up to ``max_tries`` random completions; returns None if it
        cannot find a valid one (e.g. partial conflicts with every fill).
        """
        max_tries = 32
        for _ in range(max_tries):
            full = dict(partial)
            for name in self.schema.names():
                if name not in full:
                    full[name] = self._rng.choice(self.schema.values(name))
            if self.schema.is_valid(full):
                return full
        return None

    def _stratified(self, n: int) -> list[AttributeVector]:
        if n <= 0:
            return []
        out: list[AttributeVector] = []
        attempts = 0
        max_attempts = n * 10
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            values = {
                name: self._rng.choice(vals)
                for name, vals in self.schema.attributes.items()
            }
            if self.schema.is_valid(values):
                out.append(self._wrap(values))
        return out

    def _coverage_gap(
        self, n: int, existing: list[SyntheticSample]
    ) -> list[AttributeVector]:
        """Score every two-attribute combination by inverse coverage, then sample."""
        if n <= 0:
            return []
        pair_counts: dict[tuple[str, str, str, str], int] = {}
        names = self.schema.names()
        for sample in existing:
            attrs = sample.requested_attributes
            for a, b in itertools.combinations(names, 2):
                if a in attrs and b in attrs:
                    key = (a, attrs[a], b, attrs[b])
                    pair_counts[key] = pair_counts.get(key, 0) + 1

        # Inverse-count weights, normalized; never zero so every pair stays possible.
        weights: list[tuple[tuple[str, str, str, str], float]] = []
        for a, b in itertools.combinations(names, 2):
            for va in self.schema.values(a):
                for vb in self.schema.values(b):
                    c = pair_counts.get((a, va, b, vb), 0)
                    weights.append(((a, va, b, vb), 1.0 / (1 + c)))

        out: list[AttributeVector] = []
        attempts = 0
        max_attempts = n * 20
        while len(out) < n and attempts < max_attempts:
            attempts += 1
            (a, va, b, vb), _ = self._weighted_pick(weights)
            values = self._fill_partial({a: va, b: vb})
            if values is not None:
                out.append(self._wrap(values))
        return out

    def _weighted_pick(
        self, weights: list[tuple[tuple[str, str, str, str], float]]
    ) -> tuple[tuple[str, str, str, str], float]:
        total = sum(w for _, w in weights)
        r = self._rng.random() * total
        acc = 0.0
        for key, w in weights:
            acc += w
            if acc >= r:
                return key, w
        return weights[-1]
