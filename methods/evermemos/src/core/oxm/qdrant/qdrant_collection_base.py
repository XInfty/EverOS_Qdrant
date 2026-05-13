"""
Qdrant Collection Base — Stub fuer Phase 1 der Milvus->Qdrant-Migration.

Konzept-Mapping (laut qdrant.tech/documentation/migrate-to-qdrant/from-milvus):

    Milvus                  Qdrant
    -----------------------------------------------------------
    Collection              Collection (1:1)
    FieldSchema(vector)     VectorParams(size, distance)
    FieldSchema(scalar)     Payload field (schema-flexible)
    Index(HNSW, COSINE)     HnswConfig + Distance.COSINE
    Partition               Payload-Field ODER separate Collection
    COSINE                  Cosine
    L2                      Euclid
    IP                      Dot

Diese Klasse ist absichtlich minimal. Voll-Implementierung erfolgt in
nachfolgenden Commits auf ``feature/qdrant-adapter``.

Wichtig: alle Methoden hier sind so ausgelegt, dass sie ohne aktive
qdrant-Verbindung importierbar sind — sodass das Modul auch geladen werden
kann, wenn ``VECTOR_STORE_BACKEND != "qdrant"``.
"""

from typing import Any, ClassVar, List, Optional

from core.observation.logger import get_logger

logger = get_logger(__name__)


class IndexConfig:
    """
    Konfiguration fuer Qdrant-Vector-Index. Analog zu
    ``core.oxm.milvus.milvus_collection_base.IndexConfig`` aber mit Qdrant-
    nativen Feldern.

    TODO Phase 1.2: ``hnsw_config`` (m, ef_construct, full_scan_threshold),
    ``quantization_config`` (scalar/PQ/BQ), ``on_disk_payload``,
    ``sparse_vectors_config``.
    """

    def __init__(
        self,
        size: int = 1024,
        distance: str = "Cosine",
        on_disk: bool = False,
        hnsw_m: int = 16,
        hnsw_ef_construct: int = 100,
    ) -> None:
        self.size = size
        self.distance = distance
        self.on_disk = on_disk
        self.hnsw_m = hnsw_m
        self.hnsw_ef_construct = hnsw_ef_construct


class QdrantCollectionBase:
    """
    Qdrant-Collection-Management-Basisklasse (analog MilvusCollectionBase).

    Subclasses MUST define:
        _COLLECTION_NAME: ClassVar[str]
        _VECTOR_PARAMS: ClassVar[IndexConfig]
        _PAYLOAD_INDEXES: ClassVar[list[str]] = []   # field names to index

    Optional:
        _DB_USING: ClassVar[str] = "default"          # client name

    Aktueller Stand (Phase 1.1 Skeleton): ``ensure_all()`` ist No-Op, damit
    ``QdrantLifespanProvider.startup()`` ueber registrierte Subklassen
    iterieren kann ohne Crash. Voll-Logik kommt im Sub-Commit.
    """

    _COLLECTION_NAME: ClassVar[Optional[str]] = None
    _DB_USING: ClassVar[str] = "default"
    _VECTOR_PARAMS: ClassVar[Optional[IndexConfig]] = None
    _PAYLOAD_INDEXES: ClassVar[List[str]] = []

    def __init__(self) -> None:
        if self._COLLECTION_NAME is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} must define '_COLLECTION_NAME'"
            )

    @property
    def name(self) -> str:
        # _COLLECTION_NAME ist nach __init__-Check garantiert nicht None
        return self._COLLECTION_NAME  # type: ignore[return-value]

    @property
    def using(self) -> str:
        return self._DB_USING

    def ensure_all(self) -> None:
        """
        Stellt sicher dass Collection + Payload-Indexes existieren.

        TODO Phase 1.2:
        - ``client.collection_exists(name)`` pruefen
        - falls nicht: ``client.create_collection(name, vectors_config=...)``
        - pro ``_PAYLOAD_INDEXES``: ``client.create_payload_index(name, field)``
          mit korrektem ``PayloadSchemaType`` (Keyword/Integer/Float/Bool).

        Aktuell: No-Op + Debug-Log, damit Lifespan-Provider iterieren kann.
        """
        logger.debug(
            "QdrantCollectionBase.ensure_all() stub for '%s' [using=%s] "
            "— TODO Phase 1.2",
            self.name,
            self.using,
        )

    def upsert(self, points: List[Any]) -> None:
        """TODO Phase 2: ``client.upsert(name, points=points)``."""
        raise NotImplementedError("Phase 2: implement Qdrant upsert")

    def search(self, query_vector: List[float], **kwargs: Any) -> List[Any]:
        """TODO Phase 2: ``client.search(name, query_vector, query_filter, ...)``."""
        raise NotImplementedError("Phase 2: implement Qdrant search")

    def delete(self, point_ids: List[Any]) -> None:
        """TODO Phase 2: ``client.delete(name, points_selector=PointIdsList(points=ids))``."""
        raise NotImplementedError("Phase 2: implement Qdrant delete")
