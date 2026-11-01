from .base import CompetitionAdapter, ScoreResult
from .cleanroom import CleanroomAdapter
from .ecdsafail import EcdsaFailAdapter
from .win import EcdsaFailWinAdapter

ADAPTERS = {"ecdsafail": EcdsaFailAdapter, "ecdsafail-cleanroom": CleanroomAdapter, "ecdsafail-win": EcdsaFailWinAdapter}


def load_adapter(cfg) -> CompetitionAdapter:
    return ADAPTERS[cfg.adapter](cfg)


__all__ = ["CompetitionAdapter", "ScoreResult", "EcdsaFailAdapter", "load_adapter"]
