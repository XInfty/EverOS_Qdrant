# Fork-Only Patches — Manual Restore Checklist

This fork (`XInfty/EverOS_Qdrant`) carries patches that do **not** exist upstream in `EverMind-AI/EverOS`. Upstream periodically renames or restructures the tree (most recently `methods/evermemos/` → `methods/EverCore/` around 2026-05-13). When that happens, custom files can silently disappear and custom edits inside renamed files can be reverted.

Real incidents this list exists to prevent:

- **2026-05-12** — 14 LLM call-sites had `response_format={"type":"json_object"}` patches; a model swap exposed that this isn't strict enough. Solution was migrating 13/15 sites to `response_format={"type":"json_schema", "strict": true}` with per-site schemas.
- **2026-05-15** — `rerank_voyage.py` plus the corresponding factory branch in `rerank_service.py` disappeared during the rename. Retrieval went silently empty for three days (HTTP 200 + `episodes=[]`, not a hang) before being noticed. Restored in #7.

## After every `git merge upstream/main`

Run the checks below. If any fails, restore the patch before pushing.

### 1. Voyage AI rerank provider

- [ ] File `methods/EverCore/src/agentic_layer/rerank_voyage.py` exists (~252 LOC). Voyage returns `data: [{index, relevance_score}]`; this file normalises to the EverOS-standard `{results: [{index, score, rank}]}`.
- [ ] `methods/EverCore/src/agentic_layer/rerank_service.py` contains a factory branch for `provider.lower() == "voyage"` between the `deepinfra` branch and the `else: raise RerankError`. Defaults `base_url` to `https://api.voyageai.com/v1/rerank` when env override is empty.

Quick grep:

```bash
test -f methods/EverCore/src/agentic_layer/rerank_voyage.py \
  && grep -q '"voyage"' methods/EverCore/src/agentic_layer/rerank_service.py \
  && echo "voyage: OK" || echo "voyage: MISSING"
```

### 2. `timezone` import in episodic Qdrant repo

- [ ] `methods/EverCore/src/infra_layer/adapters/out/search/repository/episodic_memory_qdrant_repository.py` line 15 reads `from datetime import datetime, timezone` (not just `datetime`). Required by `tz=timezone.utc` further down.

Quick grep:

```bash
grep -q '^from datetime import datetime, timezone' \
  methods/EverCore/src/infra_layer/adapters/out/search/repository/episodic_memory_qdrant_repository.py \
  && echo "timezone: OK" || echo "timezone: MISSING"
```

### 3. Strict JSON-Schema response_format

13 of 15 LLM call-sites should use `response_format={"type": "json_schema", "strict": True, "schema": {...}}` instead of `{"type": "json_object"}`. Per-site schema name in the second column:

| File | Schema name |
|---|---|
| `methods/EverCore/src/memory_layer/memory_extractor/episode_memory_extractor.py:270` | `episode_memory` |
| `methods/EverCore/src/memory_layer/memcell_extractor/conv_memcell_extractor.py:409` | `batch_boundary_result` |
| `methods/EverCore/src/agentic_layer/agentic_utils.py:326` | `sufficiency_check` |
| `methods/EverCore/src/agentic_layer/agentic_utils.py:403` | `multi_query_generation` |
| `methods/EverCore/src/agentic_layer/search_mem_service.py:1618` | `skill_relevance` |
| `methods/EverCore/src/memory_layer/cluster_manager/manager.py:658` | `cluster_assignment` |
| `methods/EverCore/src/memory_layer/memory_extractor/agent_skill_extractor.py:301` | `skill_operations` |
| `methods/EverCore/src/memory_layer/memory_extractor/agent_skill_extractor.py:330` | `skill_maturity` |
| `methods/EverCore/src/memory_layer/memory_extractor/foresight_extractor.py:120` | `foresight_associations` (wrapped object, parser accepts both shapes) |
| `methods/EverCore/src/memory_layer/memory_extractor/atomic_fact_extractor.py:190` | `atomic_facts_extraction` |
| `methods/EverCore/src/memory_layer/memory_extractor/agent_case_extractor.py:394` | `filter_decision` |
| `methods/EverCore/src/memory_layer/memory_extractor/agent_case_extractor.py:415` | `experience_record` |
| `methods/EverCore/src/memory_layer/memory_extractor/profile_extractor.py` (2 sites, shared parser) | `_parse_profile_response` type-coercion (see §4) |

Quick count:

```bash
grep -rE 'response_format.*json_schema' methods/EverCore/src/ | wc -l
# expect: >= 13
```

### 4. Profile extractor — defense-in-depth coercion

- [ ] `methods/EverCore/src/memory_layer/memory_extractor/profile_extractor.py` has post-parse coercion in `_parse_profile_response` that runs `json.dumps(...)` on non-string `description / trait / evidence / category` values before downstream code touches them. The profile output shape is too heterogeneous for strict schema; coercion catches LLM drift instead.

Quick grep:

```bash
grep -q 'json.dumps' \
  methods/EverCore/src/memory_layer/memory_extractor/profile_extractor.py \
  && echo "profile coercion: OK" || echo "profile coercion: MISSING"
```

### 5. Intentional opt-outs (do NOT add strict schema)

- `methods/EverCore/src/memory_layer/memory_extractor/agent_case_extractor.py:370` (`tool_pre_compress`) stays on `json_object`. The `compressed_messages` shape is heterogeneous (assistant + tool_calls vs tool message) and breaks strict schema. The caller validates `len(compressed_messages) == len(messages)` as a safety net.

## Suggested CI guard

A `.github/workflows/fork-patch-guard.yml` job that runs the four greps above on every PR and fails if any returns the wrong count. PR welcome.

## How to update this file

When a new patch is added that lives only in this fork:

1. Add it to the relevant section above (or create a new one).
2. Add a quick-grep snippet.
3. Update `.github/workflows/fork-patch-guard.yml` if it exists.
4. Reference the PR that introduced the patch.

## Related history

- PR #7 — Restored Voyage rerank + timezone import after the 2026-05-13 rename.
- XInfty/XInfty_docs#2 — EWM Bug-Cluster doc update for 2026-05-15.
