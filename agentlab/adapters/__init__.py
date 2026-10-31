from .base import CompetitionAdapter, ScoreResult
from .cleanroom import CleanroomAdapter
from .ecdsafail import EcdsaFailAdapter

ADAPTERS = {"ecdsafail": EcdsaFailAdapter, "ecdsafail-cleanroom": CleanroomAdapter}


def load_adapter(cfg) -> CompetitionAdapter:
    return ADAPTERS[cfg.adapter](cfg)


__all__ = ["CompetitionAdapter", "ScoreResult", "EcdsaFailAdapter", "load_adapter"]
