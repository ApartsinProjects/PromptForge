"""Attribute Verifier.

For each synthetic sample, asks an LLM judge whether the text actually
reflects every requested attribute. Returns per-sample verdicts that the
prompt updater consumes verbatim.

Empirical calibration (v2.9.3): the verifier is shown k=3 real-seed
examples of each labeled attribute value as in-context anchors. The
LLM judges 'does this synth sample look like the real examples of
intent=positive?' rather than the dataset-agnostic 'is this positive
in generic English?'. Only the class attribute (and any other attribute
that is labeled in the real seed) gets calibration; un-labeled schema
attributes (style, difficulty, scenario_type on most datasets) fall
back to schema-name interpretation.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass

import yaml

from synsmith.llm import LLMClient, json_chat
from synsmith.prompts import VERIFIER_SYSTEM, VERIFIER_USER_TEMPLATE
from synsmith.schema import (
    AttributeSchema,
    AttributeVerdict,
    RealExample,
    SyntheticSample,
)


@dataclass
class VerifierConfig:
    """Knobs for empirical-anchor calibration."""

    k_real_per_value: int = 3
    """How many real examples to show per labeled attribute value."""

    enable_real_anchors: bool = True
    """When False, falls back to schema-name-only verification (v2.9.2 behaviour)."""

    seed: int | None = None


class AttributeVerifier:
    def __init__(
        self,
        client: LLMClient,
        schema: AttributeSchema,
        real_examples: list[RealExample] | None = None,
        config: VerifierConfig | None = None,
    ) -> None:
        self.client = client
        self.schema = schema
        self.config = config or VerifierConfig()
        self._rng = random.Random(self.config.seed)
        # Bucket real examples by (attribute_name, attribute_value) where
        # the attribute value is present on the RealExample.
        self._anchors_by_attr: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        if real_examples:
            for ex in real_examples:
                # The 'label' field is the canonical class anchor.
                if ex.label is not None:
                    self._anchors_by_attr[schema.label_attribute][str(ex.label)].append(ex.text)
                # If the example carries optional attribute annotations, anchor those too.
                extra = getattr(ex, "attributes", None) or {}
                for k, v in extra.items():
                    if v is not None:
                        self._anchors_by_attr[k][str(v)].append(ex.text)

    def batch_verify(
        self,
        samples: list[SyntheticSample],
        *,
        batch_client: "BatchLLMClient",  # noqa: F821
    ) -> list[AttributeVerdict]:
        """Verify N samples via OpenAI Batch API: buffer, flush once, parse.

        Mirrors verify() output but ~50% the cost on N>=2 samples.
        Empirical anchors are still injected per call.
        """
        from synsmith.llm import parse_json
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        # Step 1: buffer
        deferreds = []
        for sample in samples:
            anchor_block = self._format_anchors(sample.requested_attributes)
            user_msg = VERIFIER_USER_TEMPLATE.format(
                attribute_schema=schema_str,
                sample_id=sample.sample_id,
                requested_attributes=json.dumps(sample.requested_attributes),
                text=sample.text,
                real_anchor_block=anchor_block,
            )
            d = batch_client.chat(
                VERIFIER_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=400,
            )
            deferreds.append((sample, d))
        # Step 2: flush
        batch_client.flush()
        # Step 3: parse + coerce; fall back to real-time on parse error.
        verdicts: list[AttributeVerdict] = []
        for sample, d in deferreds:
            try:
                obj = parse_json(d.text())
            except Exception:
                # Fall back: synchronous repair retry
                anchor_block = self._format_anchors(sample.requested_attributes)
                obj = json_chat(
                    self.client,
                    VERIFIER_SYSTEM,
                    [{"role": "user", "content": VERIFIER_USER_TEMPLATE.format(
                        attribute_schema=schema_str,
                        sample_id=sample.sample_id,
                        requested_attributes=json.dumps(sample.requested_attributes),
                        text=sample.text,
                        real_anchor_block=anchor_block,
                    )}],
                    temperature=0.0,
                    max_tokens=400,
                    retries=1,
                )
            verdicts.append(self._coerce(obj, sample))
        return verdicts

    def verify(self, samples: list[SyntheticSample]) -> list[AttributeVerdict]:
        schema_str = yaml.safe_dump(self.schema.attributes, sort_keys=False)
        verdicts: list[AttributeVerdict] = []
        for sample in samples:
            anchor_block = self._format_anchors(sample.requested_attributes)
            user_msg = VERIFIER_USER_TEMPLATE.format(
                attribute_schema=schema_str,
                sample_id=sample.sample_id,
                requested_attributes=json.dumps(sample.requested_attributes),
                text=sample.text,
                real_anchor_block=anchor_block,
            )
            obj = json_chat(
                self.client,
                VERIFIER_SYSTEM,
                [{"role": "user", "content": user_msg}],
                temperature=0.0,
                max_tokens=400,
                retries=1,
            )
            verdicts.append(self._coerce(obj, sample))
        return verdicts

    def _format_anchors(self, requested: dict) -> str:
        """Render the real-distribution anchors for the requested attribute values.

        For each requested (attribute, value) pair where real-seed labels are
        available, show k=k_real_per_value sampled real-seed texts. The judge
        uses these as empirical referents: 'does this synth sample look like
        these real samples of intent=positive?' rather than 'is this positive
        in generic English?'.
        """
        if not self.config.enable_real_anchors or not self._anchors_by_attr:
            return "(no real-distribution anchors available; judge by schema description)"
        chunks: list[str] = []
        for attr, val in requested.items():
            pool = self._anchors_by_attr.get(attr, {}).get(str(val))
            if not pool:
                continue
            k = min(self.config.k_real_per_value, len(pool))
            picks = self._rng.sample(pool, k)
            lines = [f"Real examples of {attr}={val!r} (use as empirical referent):"]
            for p in picks:
                lines.append(f"  - {p[:240]}")
            chunks.append("\n".join(lines))
        if not chunks:
            return "(no real-distribution anchors for the requested attribute values)"
        return "\n\n".join(chunks)

    def _coerce(self, obj: dict, sample: SyntheticSample) -> AttributeVerdict:
        """Coerce LLM verifier response into AttributeVerdict.

        Class-primary rule (v2.9.5): a sample passes verification iff the
        SCHEMA's label_attribute (typically 'intent', the class) is NOT in
        failed_attributes. Mismatches on auxiliary schema attributes
        (style, difficulty, scenario_type, noise, ambiguity, etc.) that
        do not affect the downstream classifier are still surfaced in
        failed_attributes for updater feedback, but do not by themselves
        cause attribute_match=False.

        Rationale: schemas often include un-anchored auxiliary attributes
        (no real-seed labels available) that the LLM verifier interprets
        over-strictly. On TREC, the un-anchored auxiliary attributes
        caused attr_pass=0/16 across all iterations, starving the
        updater of meaningful per-iter feedback. The downstream
        classifier only uses text->label, so class-attribute fidelity
        IS the operationally relevant pass criterion.
        """
        failed = [str(a) for a in obj.get("failed_attributes", [])]
        class_attr = self.schema.label_attribute
        class_match = class_attr not in failed
        # Backward-compat: if the LLM said attribute_match=True explicitly
        # AND the class attribute isn't in the failed list, honour the True.
        # If the LLM said False but the failure was only on auxiliary
        # attributes, override to True (class-primary semantics).
        llm_match = bool(obj.get("attribute_match", False))
        attribute_match = class_match if class_attr else llm_match
        return AttributeVerdict(
            sample_id=str(obj.get("sample_id") or sample.sample_id),
            attribute_match=attribute_match,
            failed_attributes=failed,
            reason=str(obj.get("reason", "")),
        )
