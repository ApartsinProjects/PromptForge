"""AttrForge: multi-objective prompt debugging for synthetic data generation.

Public API:

    >>> from attrforge import AttrForge, AttributeSchema
    >>> forge = AttrForge.from_config("examples/customer_support/config.yaml")
    >>> result = forge.run(iterations=5)

See the README for the full pipeline walkthrough.
"""
# Best-effort .env autoload (no-op if no file). Must run before any
# subpackage reads os.environ. Existing env vars always win.
from attrforge._dotenv import load_dotenv as _load_dotenv

_load_dotenv()

from attrforge.schema import (
    AttributeSchema,
    AttributeVector,
    RealExample,
    SyntheticSample,
)
from attrforge.loop import AttrForge, IterationResult, RunResult

__all__ = [
    "AttrForge",
    "AttributeSchema",
    "AttributeVector",
    "IterationResult",
    "RealExample",
    "RunResult",
    "SyntheticSample",
]

__version__ = "0.1.0"
