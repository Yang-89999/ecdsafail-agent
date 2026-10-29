from .base import CompetitionAdapter, ScoreResult
from .ecdsafail import EcdsaFailAdapter

ADAPTERS = {"ecdsafail": EcdsaFailAdapter}


def load_adapter(cfg) -> CompetitionAdapter:
    return ADAPTERS[cfg.adapter](cfg)


__all__ = ["CompetitionAdapter", "ScoreResult", "EcdsaFailAdapter", "load_adapter"]
