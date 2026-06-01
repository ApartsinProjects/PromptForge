"""Critics that drive the prompt update loop.

Three baseline critics (verifier, discriminator, auditor) plus four
GAN-style adversaries (pack discriminator, mode-seeking, mode hunter,
coverage hole finder) for mode-collapse defense.
"""
from attrforge.critics.auditor import DiversityAuditor
from attrforge.critics.coverage_hole import CoverageHoleFinder
from attrforge.critics.discriminator import RealismDiscriminator
from attrforge.critics.mode_hunter import ModeHunter
from attrforge.critics.mode_seeking import ModeSeeking
from attrforge.critics.pack_discriminator import PackDiscriminator
from attrforge.critics.verifier import AttributeVerifier

__all__ = [
    "AttributeVerifier",
    "CoverageHoleFinder",
    "DiversityAuditor",
    "ModeHunter",
    "ModeSeeking",
    "PackDiscriminator",
    "RealismDiscriminator",
]
