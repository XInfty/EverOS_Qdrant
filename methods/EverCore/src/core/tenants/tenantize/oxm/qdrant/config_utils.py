"""
Tenant-Aware Qdrant Configuration Utilities.

Analog zu ``core.tenants.tenantize.oxm.milvus.config_utils``, aber deutlich
schlanker — Qdrant braucht keine pymilvus-Connection-Cache-Keys, weil ein
einzelner ``QdrantClient`` alle Collections eines Endpoints bedient.

Hauptaufgabe: Aufloesen des **tenant-aware Collection-Namens** anhand des
Tenant-Context.

Resolution-Reihenfolge (analog Milvus):
    1. Tenant-Context vorhanden + ``storage_info["qdrant"]["collection_prefix"]`` gesetzt
       -> ``f"{collection_prefix}_{original_name}"``
    2. Tenant-Context vorhanden + nur ``storage_info["milvus"]["collection_prefix"]`` gesetzt
       -> ``f"{milvus_prefix}_{original_name}"`` (Migrations-Bruecke: gleiche Tenant-Namen
          fuer Qdrant wie fuer Milvus, bis pro-Tenant Qdrant-Config explizit gesetzt wird)
    3. Kein Tenant-Context -> Base-Resource-Prefix + ``original_name`` (z.B. ``s0001_v1_episodic_memory``)
"""

import os
from hashlib import sha256
from typing import Any, Dict, Optional

from core.observation.logger import get_logger

logger = get_logger(__name__)


def get_tenant_qdrant_config() -> Optional[Dict[str, Any]]:
    """
    Hole das Qdrant-Storage-Dict des aktiven Tenant-Context.

    Returns:
        Storage-Info-Dict (``collection_prefix`` und ggf. ``host``/``port``/``api_key``)
        oder ``None`` falls kein Tenant aktiv.
    """
    # Lazy import vermeidet Circular-Dependency bei Adapter-Discovery-Time.
    from core.tenants.tenantize.tenant_context import get_current_tenant

    try:
        tenant_info = get_current_tenant()
        if not tenant_info:
            return None

        qdrant_cfg = tenant_info.get_storage_info("qdrant")
        if qdrant_cfg:
            return qdrant_cfg

        # Fallback: Falls noch kein dediziertes Qdrant-Config-Dict im
        # Storage-Info, nutze den Milvus-Eintrag (gleicher collection_prefix
        # ist sinnvolle Migrations-Bruecke).
        return tenant_info.get_storage_info("milvus") or tenant_info.get_storage_info(
            "milvus_config"
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to resolve tenant qdrant config: %s", e)
        return None


def _base_prefixed_collection_name(original_name: str) -> str:
    """``{base_resource_prefix}_{original_name}`` (no-tenant Fallback)."""
    # Lazy import — same circular-avoidance reason as above.
    from core.tenants.tenant_constants import get_base_resource_prefix

    return f"{get_base_resource_prefix()}_{original_name}"


def get_tenant_aware_collection_name(original_name: str) -> str:
    """
    Resolve a tenant-aware Qdrant collection name.

    Args:
        original_name: Bare collection name (e.g., ``"v1_episodic_memory"``).

    Returns:
        Tenant-prefixed name (e.g., ``"acme_v1_episodic_memory"``,
        ``"s0001_v1_episodic_memory"``, etc.).
    """
    try:
        cfg = get_tenant_qdrant_config()
        if cfg and cfg.get("collection_prefix"):
            return f"{cfg['collection_prefix']}_{original_name}"

        # Kein expliziter Prefix im Tenant-Context — Fall back to base resource.
        return _base_prefixed_collection_name(original_name)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "Failed to resolve tenant-aware Qdrant collection name for '%s': %s",
            original_name,
            e,
        )
        return _base_prefixed_collection_name(original_name)


def get_qdrant_connection_cache_key(config: Dict[str, Any]) -> str:
    """
    Build a deterministic cache key for a Qdrant connection.

    Used by ``QdrantClientFactory.get_named_client`` when callers route via
    tenant-specific endpoints (each unique ``(host, port, api_key_hash)``
    triple becomes one cached client). For the common case of one shared
    Qdrant endpoint across tenants, this returns a stable single key.

    Args:
        config: Dict containing at least ``host``/``port`` or ``url``.
                ``api_key`` is hashed (not the raw value) when included.

    Returns:
        A short stable string suitable as factory alias.
    """
    if "url" in config and config["url"]:
        endpoint = str(config["url"])
    else:
        endpoint = f"{config.get('host', 'localhost')}:{config.get('port', 6333)}"

    api_key = config.get("api_key")
    if api_key:
        # Hash the api_key fingerprint, not the raw value. Tolerate bytes,
        # str, or other types — coerce safely before hashing.
        if isinstance(api_key, bytes):
            key_bytes = api_key
        else:
            key_bytes = str(api_key).encode("utf-8")
        endpoint += f"#{sha256(key_bytes).hexdigest()[:8]}"

    # Transport flags must participate in the cache key — two tenants that
    # share host:port but disagree on ``https`` or ``prefer_grpc`` need
    # *different* cached clients. Without these in the key, the first
    # tenant's client config would be reused for the second tenant.
    https = config.get("https")
    if https is not None:
        endpoint += f"#https={bool(https)}"
    prefer_grpc = config.get("prefer_grpc")
    if prefer_grpc is not None:
        endpoint += f"#grpc={bool(prefer_grpc)}"

    return endpoint


def _load_qdrant_env(prefix: str = "") -> Dict[str, Any]:
    """
    Read Qdrant connection settings from environment variables. Used as a
    fallback when no tenant-storage-info is present.

    Currently this helper is staged for the tenant-aware connection routing
    that will be wired in alongside the per-tenant ``QdrantClientFactory``
    flow (see TenantAwareQdrantCollectionWithSuffix and the factory). It is
    deliberately exported as module-private (``_load_qdrant_env``) until the
    routing layer consumes it; do not flag as dead code in the meantime.

    Args:
        prefix: Optional env prefix (e.g., ``"A"`` reads ``A_QDRANT_HOST``).

    Returns:
        Dict mit ``host``, ``port``, ``api_key``, ``https``, ``prefer_grpc``.
    """
    def _env(name: str, default: Optional[str] = None) -> str:
        key = f"{prefix.upper()}_{name}" if prefix else name
        if default is None:
            return os.getenv(key, "")
        return os.getenv(key, default)

    def _safe_port(raw: str, default: int) -> int:
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid QDRANT_PORT value %r — falling back to %d", raw, default
            )
            return default
        if not (1 <= value <= 65535):
            logger.warning(
                "QDRANT_PORT %d out of TCP range — falling back to %d", value, default
            )
            return default
        return value

    return {
        "host": _env("QDRANT_HOST", "localhost"),
        "port": _safe_port(_env("QDRANT_PORT", "6333"), 6333),
        "api_key": _env("QDRANT_API_KEY") or None,
        "https": _env("QDRANT_HTTPS", "").strip().lower() in {"1", "true", "yes", "on"},
        "prefer_grpc": _env("QDRANT_PREFER_GRPC", "").strip().lower()
        in {"1", "true", "yes", "on"},
    }
