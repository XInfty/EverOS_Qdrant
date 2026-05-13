"""
Qdrant lifespan provider — Analog zu ``core.lifespan.milvus_lifespan``.

Wird vom DI-Container automatisch entdeckt und in die FastAPI-Lifespan-Kette
eingehaengt. Initialisierung ist **gated** durch das Env-Flag
``VECTOR_STORE_BACKEND``:

    VECTOR_STORE_BACKEND=qdrant   -> Qdrant wird initialisiert
    VECTOR_STORE_BACKEND=milvus   -> No-Op (Milvus-Lifespan uebernimmt; Default)
    VECTOR_STORE_BACKEND unset    -> No-Op (= Default ``milvus``)

So kann der Adapter-Layer im Repo liegen, ohne dass er aktiv eingreift bis
zum Cutover.
"""

import os
from collections import defaultdict
from typing import Any, Dict, List, Type

from fastapi import FastAPI

from core.di.decorators import component
from core.di.utils import get_all_subclasses, get_bean
from core.lifespan.lifespan_interface import LifespanProvider
from core.observation.logger import get_logger
from core.oxm.qdrant.qdrant_collection_base import QdrantCollectionBase

logger = get_logger(__name__)


# Order 19: zwischen milvus_lifespan (18) und business_lifespan (20). So
# laufen beide Vector-Backends initialisiert (im Cutover-Fall), und
# business-Logik startet erst danach.
_QDRANT_LIFESPAN_ORDER = 19

# Env-Flag, das den aktiven Vector-Store waehlt. Default ``milvus`` damit
# nichts an der bestehenden Deployment-Topologie aendert bis zum Cutover.
_ENV_BACKEND_FLAG = "VECTOR_STORE_BACKEND"
_BACKEND_QDRANT = "qdrant"


def _backend_is_qdrant() -> bool:
    return os.getenv(_ENV_BACKEND_FLAG, "milvus").strip().lower() == _BACKEND_QDRANT


@component(name="qdrant_lifespan_provider")
class QdrantLifespanProvider(LifespanProvider):
    """Qdrant lifespan provider (feature-gated)."""

    def __init__(self, name: str = "qdrant", order: int = _QDRANT_LIFESPAN_ORDER):
        super().__init__(name, order)
        self._qdrant_factory = None
        self._qdrant_clients: dict = {}

    async def startup(self, app: FastAPI) -> Any:
        """
        Start Qdrant connection und Collection-Initialisierung.

        Skipped wenn ``VECTOR_STORE_BACKEND != qdrant``.
        """
        if not _backend_is_qdrant():
            logger.info(
                "Qdrant lifespan skipped (%s='%s', Qdrant inactive)",
                _ENV_BACKEND_FLAG,
                os.getenv(_ENV_BACKEND_FLAG, "milvus"),
            )
            return

        logger.info("Initializing Qdrant connection...")

        try:
            self._qdrant_factory = get_bean("qdrant_client_factory")

            # Alle konkreten QdrantCollectionBase-Subklassen sammeln.
            all_collection_classes = [
                cls
                for cls in get_all_subclasses(QdrantCollectionBase)
                if cls._COLLECTION_NAME is not None
            ]

            # Gruppieren nach _DB_USING (analog Milvus).
            using_collections: Dict[str, List[Type[QdrantCollectionBase]]] = defaultdict(list)
            for collection_class in all_collection_classes:
                using = collection_class._DB_USING
                using_collections[using].append(collection_class)
                logger.info(
                    "Discovered Qdrant Collection class: %s [using=%s]",
                    collection_class.__name__,
                    using,
                )

            # Pro using: Client holen + Collections initialisieren.
            for using, collection_classes in using_collections.items():
                client = self._qdrant_factory.get_named_client(using)
                self._qdrant_clients[using] = client

                for collection_class in collection_classes:
                    try:
                        collection = collection_class()
                        collection.ensure_all()
                        logger.info(
                            "Qdrant Collection '%s' initialized [using=%s]",
                            collection.name,
                            using,
                        )
                    except Exception as e:
                        logger.error(
                            "Failed to initialize Qdrant Collection '%s' [using=%s]: %s",
                            collection_class._COLLECTION_NAME,
                            using,
                            e,
                        )
                        raise

            logger.info("Qdrant connection initialization completed")

        except Exception as e:
            logger.error("Error during Qdrant initialization: %s", str(e))
            raise

    async def shutdown(self, app: FastAPI) -> None:
        """Close Qdrant connections (No-Op wenn nicht initialisiert)."""
        if not _backend_is_qdrant() or self._qdrant_factory is None:
            return

        logger.info("Closing Qdrant connections...")
        try:
            self._qdrant_factory.close_all_clients()
            logger.info("Qdrant connections closed")
        except Exception as e:
            logger.error("Error while closing Qdrant connections: %s", str(e))

        # State-Cleanup analog Milvus.
        for attr in ("qdrant_clients", "qdrant_factory"):
            if hasattr(app.state, attr):
                delattr(app.state, attr)
