"""
Voyage AI Rerank Service Implementation

Reranking service using Voyage AI commercial API (rerank-2.5).

Voyage's API differs from DeepInfra/vLLM in two ways:
- POST to a fixed endpoint (no model-suffix path).
- Request shape: ``{"query", "documents", "model"}`` with plain strings
  (no Qwen ``<|im_start|>`` template wrapping).
- Response shape: ``{"data": [{"index", "relevance_score"}], "usage": {...}}``
  (DeepInfra/Cohere return ``results``; Voyage returns ``data``).

The service normalises Voyage's response into the EverOS standard
``{"results": [{"index", "score", "rank"}]}`` shape so callers don't care
which backend produced the scores.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from agentic_layer.rerank_interface import (
    RerankServiceInterface,
    RerankError,
    extract_text_from_hit,
)
from core.di.utils import get_bean_by_type
from core.component.token_usage_collector import TokenUsageCollector

logger = logging.getLogger(__name__)


@dataclass
class VoyageRerankConfig:
    """Voyage rerank service configuration"""

    api_key: str = ""  # skip-sensitive-check
    base_url: str = "https://api.voyageai.com/v1/rerank"
    model: str = "rerank-2.5"
    timeout: int = 30
    max_retries: int = 3
    batch_size: int = 100  # Voyage accepts up to 1000 docs/request
    max_concurrent_requests: int = 5


class VoyageRerankService(RerankServiceInterface):
    """Voyage AI reranking service implementation"""

    def __init__(self, config: Optional[VoyageRerankConfig] = None):
        if config is None:
            config = VoyageRerankConfig()

        self.config = config
        self.session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        self._session_lock = asyncio.Lock()
        logger.info(
            f"Initialized VoyageRerankService | url={config.base_url} | model={config.model}"
        )

    async def _ensure_session(self):
        # Lock guards against concurrent first-touch creating multiple sessions
        # and leaking the loser; the inner re-check covers the racy second waiter.
        if self.session is not None and not self.session.closed:
            return
        async with self._session_lock:
            if self.session is not None and not self.session.closed:
                return
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
            )

    async def close(self):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _send_rerank_request_batch(
        self, query: str, documents: List[str]
    ) -> Dict[str, Any]:
        """POST one batch to the Voyage rerank endpoint."""
        await self._ensure_session()

        # Voyage expects plain strings, NOT Qwen-template wrapped.
        request_data = {
            "query": query,
            "documents": documents,
            "model": self.config.model,
        }

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    async with self.session.post(
                        self.config.base_url, json=request_data
                    ) as response:
                        if response.status == 200:
                            json_body = await response.json()
                            return self._parse_response(json_body, len(documents))
                        error_text = await response.text()
                        logger.error(
                            f"Voyage rerank API error {response.status}: {error_text}"
                        )
                        if attempt < self.config.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)
                            continue
                        raise RerankError(
                            f"API failed: {response.status} - {error_text}"
                        )
                except RerankError:
                    raise
                except Exception as e:
                    logger.error(f"Voyage rerank exception: {e}")
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise RerankError(f"Exception: {e}")

    def _parse_response(
        self, json_body: Dict[str, Any], num_docs: int
    ) -> Dict[str, Any]:
        """Voyage returns ``data: [{index, relevance_score}]`` — translate
        into a dense ``scores`` array aligned to the request order."""
        scores = [0.0] * num_docs
        for item in json_body.get("data", []):
            idx = item.get("index")
            if idx is None or not (0 <= idx < num_docs):
                continue
            scores[idx] = float(item.get("relevance_score", 0.0))

        usage = json_body.get("usage", {}) or {}
        return {
            "scores": scores,
            "input_tokens": int(usage.get("total_tokens", 0) or 0),
            "request_id": json_body.get("id"),
        }

    async def rerank_documents(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        """Low-level reranking.

        Voyage's `/v1/rerank` endpoint accepts only `query` + `documents`; it
        has no separate ``instruction`` field. To preserve the call-site
        contract (search_mem_service passes skill-specific instructions), the
        instruction is prepended to the query when provided, matching the
        behaviour of vLLM/DeepInfra implementations.
        """
        if not documents:
            return {"results": []}

        # Codex R2 P2: validate batch_size before slicing so a misconfigured
        # negative/zero RERANK_BATCH_SIZE fails loudly instead of silently
        # producing zero batches.
        batch_size = self.config.batch_size or 100
        if batch_size <= 0:
            raise RerankError(
                f"Invalid Voyage batch_size={batch_size}; must be > 0"
            )

        # Codex R2 P2: honour instruction (was silently dropped, breaking
        # skill-biased reranking in search_mem_service).
        effective_query = (
            f"{instruction}\n{query}" if instruction else query
        )

        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        batch_tasks = [
            self._send_rerank_request_batch(effective_query, batch) for batch in batches
        ]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        # Codex R2 P1: fail-fast on batch errors. The previous implementation
        # extended scores with synthetic -100.0 sentinels and continued, which
        # silently degraded ordering when Voyage was unreachable. With
        # RERANK_FALLBACK_PROVIDER=none in production this re-introduced the
        # exact silent-fail pattern that took 3 days to detect in the
        # original Qdrant-migration bug. Raise so the rerank_service factory
        # / HybridRerankService can route to a configured fallback or
        # propagate the outage to the caller.
        failures = [
            (i, r) for i, r in enumerate(batch_results) if isinstance(r, Exception)
        ]
        if failures:
            first_i, first_err = failures[0]
            raise RerankError(
                f"Voyage rerank failed for {len(failures)}/{len(batches)} batches; "
                f"first failure: batch {first_i}: {first_err}"
            )

        all_scores: List[float] = []
        total_input_tokens = 0
        last_response = None

        for result in batch_results:
            all_scores.extend(result.get("scores", []))
            total_input_tokens += result.get("input_tokens", 0)
            last_response = result

        try:
            collector = get_bean_by_type(TokenUsageCollector)
            collector.add(self.config.model, total_input_tokens, 0, call_type="rerank")
        except Exception:
            pass

        combined_response = {
            "scores": all_scores,
            "input_tokens": total_input_tokens,
            "request_id": last_response.get("request_id") if last_response else None,
        }
        return self._convert_response_format(combined_response, len(documents))

    def _convert_response_format(
        self, combined_response: Dict[str, Any], num_documents: int
    ) -> Dict[str, Any]:
        scores = combined_response.get("scores", [])
        if len(scores) < num_documents:
            scores.extend([0.0] * (num_documents - len(scores)))
        scores = scores[:num_documents]

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = [
            {"index": original_index, "score": score, "rank": rank}
            for rank, (original_index, score) in enumerate(indexed_scores)
        ]
        return {
            "results": results,
            "input_tokens": combined_response.get("input_tokens", 0),
            "request_id": combined_response.get("request_id"),
        }

    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return []

        all_texts = [extract_text_from_hit(hit) for hit in hits]
        if not all_texts:
            return []

        try:
            # CodeRabbit security: avoid logging raw query (multi-tenant data
            # leakage risk). Log only metadata.
            logger.debug(
                "Voyage reranking: query_len=%d, num_texts=%d",
                len(query),
                len(all_texts),
            )
            rerank_result = await self.rerank_documents(query, all_texts, instruction)
            if "results" not in rerank_result:
                raise RerankError("Invalid rerank API response: missing results field")

            results_meta = rerank_result.get("results", [])
            reranked_hits = []
            for item in results_meta:
                original_idx = item.get("index", 0)
                score = item.get("score", 0.0)
                if 0 <= original_idx < len(hits):
                    hit = hits[original_idx].copy()
                    hit["score"] = score
                    reranked_hits.append(hit)

            if top_k is not None and top_k > 0:
                reranked_hits = reranked_hits[:top_k]

            if reranked_hits:
                top_scores = [f"{h.get('score', 0):.4f}" for h in reranked_hits[:3]]
                logger.info(
                    f"Voyage reranking completed: {len(reranked_hits)} results, top scores: {top_scores}"
                )
            return reranked_hits

        except Exception as e:
            logger.error(f"Voyage reranking failed: {e}")
            sorted_hits = sorted(hits, key=lambda x: x.get("score", 0), reverse=True)
            if top_k is not None and top_k > 0:
                sorted_hits = sorted_hits[:top_k]
            return sorted_hits

    def get_model_name(self) -> str:
        return self.config.model
