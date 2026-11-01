"""chia.simulators — simulator build/run nodes for CHIA loops."""

try:
    from chia.simulators.gem5 import (
        Gem5Node,
        Gem5ToolServer,
        Gem5BuildArtifact,
        Gem5RunResult,
        Gem5SourceState,
        Gem5Isa,
        Gem5Variant,
    )
except ImportError:  # ray / MCP not installed — gem5 node unavailable
    pass

from chia.simulators.champsim import (
    CachePrefetchStats,
    CacheStats,
    ChampSimBuildResult,
    ChampSimRunResult,
    ChampSimSourceState,
)

try:
    from chia.simulators.champsim import ChampSimNode
except ImportError:  # ray not installed — ChampSimNode unavailable
    pass

__all__ = [
    "Gem5Node",
    "Gem5ToolServer",
    "Gem5BuildArtifact",
    "Gem5RunResult",
    "Gem5SourceState",
    "Gem5Isa",
    "Gem5Variant",
    "ChampSimNode",
    "CachePrefetchStats",
    "CacheStats",
    "ChampSimBuildResult",
    "ChampSimRunResult",
    "ChampSimSourceState",
]
