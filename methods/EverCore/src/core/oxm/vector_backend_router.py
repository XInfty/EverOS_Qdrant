"""
Vector-Backend-Router — env-gated factory for vector repositories.

The service layer (``agentic_layer/*``, ``memory_layer/*``, ``biz_layer/*``)
historically instantiated ``*MilvusRepository()`` classes directly. After
the Qdrant adapter was added (see [[Qdrant]] in the EWM docs), this
hard-coded write/read path bypassed ``VECTOR_STORE_BACKEND`` entirely —
the lifespan layer correctly skipped Milvus, but every service still
constructed Milvus repos and crashed at first use.

This router resolves the env flag once per construction call and returns
the right repository instance. Both backends expose the same public
surface (``vector_search``, ``create_and_save_*``, ``delete_by_*``) so
callers don't need to know which backend they got.

Usage::

    from core.oxm.vector_backend_router import get_episodic_repo
    self.episodic_repo = get_episodic_repo()
    # caller-facing methods are identical across backends:
    results = await self.episodic_repo.vector_search(query_vector=v, ...)

Adding a new memory type: write a ``get_<name>_repo()`` thin factory that
follows the same pattern. Lazy imports keep startup cheap and avoid
pulling Qdrant client packages into Milvus-only deployments.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_VALID_BACKENDS = {"qdrant", "milvus"}


def _backend() -> str:
    """Return the active backend name in normalised form (``"qdrant"`` or
    ``"milvus"``). Mirrors the case-insensitive gate used by
    ``QdrantLifespanProvider`` and ``MilvusLifespanProvider``. An
    unrecognised value (e.g. a typo like ``VECTOR_STORE_BACKEND=qdarnt``)
    falls back to ``"milvus"`` but emits a warning so the operator
    notices instead of silently routing to the wrong backend.
    """
    raw = os.getenv("VECTOR_STORE_BACKEND", "milvus").strip().lower()
    if raw not in _VALID_BACKENDS:
        logger.warning(
            "VECTOR_STORE_BACKEND=%r is not a known backend (expected one of "
            "%s); falling back to 'milvus'",
            raw, sorted(_VALID_BACKENDS),
        )
        return "milvus"
    return raw


def _is_qdrant() -> bool:
    return _backend() == "qdrant"


def _resolve(qdrant_cls: type[Any], milvus_cls: type[Any]) -> Any:
    """Resolve the right backend bean via the DI container so existing
    singleton scope (and any constructor wiring done by the DI scanner)
    is preserved. Falls back to direct instantiation only if no bean is
    registered — that should not happen in production, but the safety
    net keeps stand-alone unit tests with no DI scan from crashing.
    """
    cls = qdrant_cls if _is_qdrant() else milvus_cls
    try:
        from core.di import get_bean_by_type
        return get_bean_by_type(cls)
    except Exception:
        return cls()


def get_episodic_repo() -> Any:
    from infra_layer.adapters.out.search.repository.episodic_memory_qdrant_repository import (
        EpisodicMemoryQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.episodic_memory_milvus_repository import (
        EpisodicMemoryMilvusRepository,
    )
    return _resolve(EpisodicMemoryQdrantRepository, EpisodicMemoryMilvusRepository)


def get_atomic_fact_repo() -> Any:
    from infra_layer.adapters.out.search.repository.atomic_fact_qdrant_repository import (
        AtomicFactQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.atomic_fact_milvus_repository import (
        AtomicFactMilvusRepository,
    )
    return _resolve(AtomicFactQdrantRepository, AtomicFactMilvusRepository)


def get_foresight_repo() -> Any:
    from infra_layer.adapters.out.search.repository.foresight_qdrant_repository import (
        ForesightQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.foresight_milvus_repository import (
        ForesightMilvusRepository,
    )
    return _resolve(ForesightQdrantRepository, ForesightMilvusRepository)


def get_agent_case_repo() -> Any:
    from infra_layer.adapters.out.search.repository.agent_case_qdrant_repository import (
        AgentCaseQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.agent_case_milvus_repository import (
        AgentCaseMilvusRepository,
    )
    return _resolve(AgentCaseQdrantRepository, AgentCaseMilvusRepository)


def get_agent_skill_repo() -> Any:
    from infra_layer.adapters.out.search.repository.agent_skill_qdrant_repository import (
        AgentSkillQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.agent_skill_milvus_repository import (
        AgentSkillMilvusRepository,
    )
    return _resolve(AgentSkillQdrantRepository, AgentSkillMilvusRepository)


def get_user_profile_repo() -> Any:
    from infra_layer.adapters.out.search.repository.user_profile_qdrant_repository import (
        UserProfileQdrantRepository,
    )
    from infra_layer.adapters.out.search.repository.user_profile_milvus_repository import (
        UserProfileMilvusRepository,
    )
    return _resolve(UserProfileQdrantRepository, UserProfileMilvusRepository)
