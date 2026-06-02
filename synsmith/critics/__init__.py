"""Critics that drive the prompt update loop.

Three baseline critics (verifier, discriminator, auditor) plus four
GAN-style adversaries (pack discriminator, mode-seeking, mode hunter,
coverage hole finder) for mode-collapse defense.
"""
from synsmith.critics.auditor import DiversityAuditor
from synsmith.critics.coverage_hole import CoverageHoleFinder
from synsmith.critics.discriminator import RealismDiscriminator
from synsmith.critics.mode_hunter import ModeHunter
from synsmith.critics.mode_seeking import ModeSeeking
from synsmith.critics.pack_discriminator import PackDiscriminator
from synsmith.critics.verifier import AttributeVerifier

__all__ = [
    "AttributeVerifier",
    "CoverageHoleFinder",
    "DiversityAuditor",
    "ModeHunter",
    "ModeSeeking",
    "PackDiscriminator",
    "RealismDiscriminator",
]
