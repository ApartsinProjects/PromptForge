"""SynSmith: multi-objective prompt debugging for synthetic data generation.

Public API:

    >>> from synsmith import SynSmith, AttributeSchema
    >>> forge = SynSmith.from_config("examples/customer_support/config.yaml")
    >>> result = forge.run(iterations=5)

See the README for the full pipeline walkthrough.
"""
# Best-effort .env autoload (no-op if no file). Must run before any
# subpackage reads os.environ. Existing env vars always win.
from synsmith._dotenv import load_dotenv as _load_dotenv

_load_dotenv()

from synsmith.schema import (
    AttributeSchema,
    AttributeVector,
    RealExample,
    SyntheticSample,
)
from synsmith.loop import SynSmith, IterationResult, RunResult

__all__ = [
    "SynSmith",
    "AttributeSchema",
    "AttributeVector",
    "IterationResult",
    "RealExample",
    "RunResult",
    "SyntheticSample",
]

__version__ = "0.1.0"
