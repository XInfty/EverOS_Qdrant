"""
Qdrant Client Factory

Analoge Implementierung zu ``core.component.milvus_client_factory.MilvusClientFactory``
fuer die Milvus->Qdrant-Migration.

Provides Qdrant client connection functionality based on environment variables.
"""

import os
from typing import Dict, Optional

from qdrant_client import QdrantClient

from core.di.decorators import component
from core.observation.logger import get_logger

logger = get_logger(__name__)


def _truthy(value: Optional[str]) -> bool:
    """Konsistentes Env-Boolean-Parsing analog zu anderen EverOS-Configs."""
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_qdrant_config(prefix: str = "") -> dict:
    """
    Get Qdrant configuration from environment variables.

    Args:
        prefix: Environment variable prefix, e.g., prefix="A" reads
                "A_QDRANT_HOST"; if empty reads "QDRANT_HOST" etc.

    Environment variables:
        - ``{PREFIX_}QDRANT_HOST`` (default ``localhost``)
        - ``{PREFIX_}QDRANT_PORT`` (default ``6333``, HTTP)
        - ``{PREFIX_}QDRANT_GRPC_PORT`` (default ``6334``)
        - ``{PREFIX_}QDRANT_API_KEY`` (optional)
        - ``{PREFIX_}QDRANT_HTTPS`` (default ``false``)
        - ``{PREFIX_}QDRANT_PREFER_GRPC`` (default ``false``)
        - ``{PREFIX_}QDRANT_TIMEOUT`` (default ``30`` Sekunden)

    Returns:
        dict mit Schluesseln ``host``, ``port``, ``grpc_port``, ``api_key``,
        ``https``, ``prefer_grpc``, ``timeout``, ``url`` (assembled).
    """

    def _env(name: str, default: Optional[str] = None) -> str:
        if prefix:
            key = f"{prefix.upper()}_{name}"
        else:
            key = name
        if default is None:
            return os.getenv(key, "")
        return os.getenv(key, default)

    host = _env("QDRANT_HOST", "localhost")
    port = int(_env("QDRANT_PORT", "6333"))
    grpc_port = int(_env("QDRANT_GRPC_PORT", "6334"))
    # api_key / https sind explizit None wenn env nicht gesetzt — so kann
    # qdrant-client die Defaults / URL-Scheme-Detection selbst uebernehmen.
    api_key_raw = _env("QDRANT_API_KEY")
    api_key: Optional[str] = api_key_raw or None
    https_raw = os.getenv(f"{prefix.upper()}_QDRANT_HTTPS" if prefix else "QDRANT_HTTPS")
    https: Optional[bool] = _truthy(https_raw) if https_raw is not None else None
    prefer_grpc = _truthy(_env("QDRANT_PREFER_GRPC", "false"))
    timeout = int(_env("QDRANT_TIMEOUT", "30"))

    # URL-Assembly: wenn https explizit gesetzt, halte die Praeferenz. Sonst http.
    scheme = "https" if https else "http"
    if host.startswith("http://") or host.startswith("https://"):
        url = f"{host}:{port}"
    else:
        url = f"{scheme}://{host}:{port}"

    config = {
        "host": host,
        "port": port,
        "grpc_port": grpc_port,
        "api_key": api_key,
        "https": https,
        "prefer_grpc": prefer_grpc,
        "timeout": timeout,
        "url": url,
    }

    logger.info("Getting Qdrant config [prefix=%s]:", prefix or "default")
    logger.info(
        "  URL: %s (prefer_grpc=%s, grpc_port=%s, https=%s)",
        url, prefer_grpc, grpc_port, https,
    )
    logger.info("  Auth: %s", "ApiKey" if api_key else "None")
    logger.info("  Timeout: %ss", timeout)

    return config


@component(name="qdrant_client_factory", primary=False)
class QdrantClientFactory:
    """
    Qdrant Client Factory.

    Bietet Caching/Management fuer ``QdrantClient``-Instanzen, ein Client pro
    benannter Konfiguration (analog ``MilvusClientFactory``).

    ``primary=False``: Wenn ``VECTOR_STORE_BACKEND=qdrant`` gesetzt ist, wird
    diese Factory ueber den Bean-Namen ``qdrant_client_factory`` explizit
    gerouted (siehe Phase 1.2 / Repository-Layer-Refactor). So bleibt
    Milvus-Factory in der Setup-Phase Default und nichts veraendert sich
    bis zum Cutover.
    """

    def __init__(self) -> None:
        self._clients: Dict[str, QdrantClient] = {}
        self._default_config: Optional[dict] = None
        # Note: typischer use-case ist single-init in lifespan-startup, daher
        # kein Lock noetig. Bei concurrent access aus FastAPI-Coroutines auf
        # verschiedene named clients kann theoretisch eine Race entstehen
        # (beide passen den cache-miss-check, beide erstellen Client, einer
        # ueberschreibt den anderen im dict). Fix in Phase 2 via threading.Lock
        # falls Concurrent-Pattern auftritt.
        logger.info("QdrantClientFactory initialized")

    def get_client(
        self,
        url: str = "",
        host: str = "",
        port: int = 6333,
        grpc_port: int = 6334,
        api_key: Optional[str] = None,
        https: Optional[bool] = None,
        prefer_grpc: bool = False,
        timeout: int = 30,
        alias: Optional[str] = None,
        **kwargs,
    ) -> QdrantClient:
        """
        Get oder erzeuge cached Qdrant-Client.

        Args:
            url: Voll-qualifizierte URL (z.B. ``http://localhost:6333``). Wenn
                 angegeben, ueberschreibt sie ``host``/``port``.
            host: Hostname (Default ``localhost`` falls weder ``url`` noch
                  ``host`` gesetzt).
            port: HTTP/REST-Port (Default ``6333``).
            grpc_port: gRPC-Port (Default ``6334``, nur wenn ``prefer_grpc``).
            api_key: Optionaler Qdrant Cloud API-Key. ``None`` = anonymous.
            https: TLS-Praeferenz. ``None`` (Default) ueberlaesst qdrant-client
                   die Auto-Detection ueber das URL-Schema. Explizit ``True``/
                   ``False`` ueberschreibt das.
            prefer_grpc: gRPC statt HTTP fuer Datenwege.
            timeout: Request-Timeout in Sekunden.
            alias: Cache-Key (Default ``default``).

        Returns:
            ``QdrantClient`` (gecached pro ``alias``).
        """
        cache_key = alias or "default"
        if cache_key in self._clients:
            return self._clients[cache_key]

        client_kwargs: dict = {
            "prefer_grpc": prefer_grpc,
            "grpc_port": grpc_port,
            "timeout": timeout,
        }
        if api_key:
            client_kwargs["api_key"] = api_key
        if https is not None:
            client_kwargs["https"] = https
        if url:
            client_kwargs["url"] = url
        else:
            client_kwargs["host"] = host or "localhost"
            client_kwargs["port"] = port

        client_kwargs.update(kwargs)

        client = QdrantClient(**client_kwargs)
        self._clients[cache_key] = client
        logger.info(
            "Qdrant client created and cached: %s (alias=%s, prefer_grpc=%s, https=%s)",
            url or f"{client_kwargs.get('host')}:{port}",
            cache_key,
            prefer_grpc,
            https,
        )
        return client

    def get_default_client(self) -> QdrantClient:
        """Get default Qdrant client basierend auf Env-Konfiguration."""
        if self._default_config is None:
            self._default_config = get_qdrant_config()

        cfg = self._default_config
        return self.get_client(
            url=cfg["url"],
            api_key=cfg["api_key"],
            https=cfg["https"],
            prefer_grpc=cfg["prefer_grpc"],
            grpc_port=cfg["grpc_port"],
            timeout=cfg["timeout"],
            alias="default",
        )

    def get_named_client(self, name: str) -> QdrantClient:
        """
        Get Qdrant client by name. ``name`` wird als Env-Praefix verwendet,
        z.B. ``name="A"`` liest ``A_QDRANT_HOST``, ``A_QDRANT_PORT``, ...

        Args:
            name: Praefix-Name (Env-Var-Praefix). ``default`` -> default client.

        Returns:
            ``QdrantClient`` (gecached unter ``name``).
        """
        if name.lower() == "default":
            return self.get_default_client()

        cfg = get_qdrant_config(prefix=name)
        logger.info("Loading named Qdrant config [name=%s]: %s", name, cfg["url"])

        return self.get_client(
            url=cfg["url"],
            api_key=cfg["api_key"],
            https=cfg["https"],
            prefer_grpc=cfg["prefer_grpc"],
            grpc_port=cfg["grpc_port"],
            timeout=cfg["timeout"],
            alias=name,
        )

    def close_all_clients(self) -> None:
        """Schliesst alle gecachten Qdrant-Clients."""
        for alias, client in self._clients.items():
            try:
                client.close()
            except Exception as e:  # noqa: BLE001
                logger.error("Error closing Qdrant client [alias=%s]: %s", alias, e)
        self._clients.clear()
        logger.info("All Qdrant clients closed")
