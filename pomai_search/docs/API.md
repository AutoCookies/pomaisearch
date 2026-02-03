# API

All responses follow JSON format. Errors use:

```json
{ "ok": false, "code": "INVALID_ARGUMENT", "message": "..." }
```

## POST /v1/upsert

Request:

```json
{
  "key": "doc-1",
  "vector": [1.0, 0.0, 0.0],
  "metadata": { "tag": "demo" },
  "text": "hello world"
}
```

Response:

```json
{ "ok": true }
```

## POST /v1/delete

Request:

```json
{ "key": "doc-1" }
```

Response:

```json
{ "ok": true }
```

## POST /v1/search

Request:

```json
{
  "vector": [1.0, 0.0, 0.0],
  "topk": 10,
  "filter": { "tag": "demo" }
}
```

Response:

```json
{
  "ok": true,
  "results": [
    { "key": "doc-1", "score": 0.99, "metadata": { "tag": "demo" } }
  ]
}
```

## POST /v1/search_hybrid

Request:

```json
{
  "text_query": "hello",
  "vector": [1.0, 0.0, 0.0],
  "alpha": 0.7,
  "topk": 10
}
```

Response: same as `/v1/search`.

## POST /v1/search_explain

Request:

```json
{
  "vector": [1.0, 0.0, 0.0],
  "topk": 10,
  "policy": {
    "max_latency_ms": 50,
    "max_candidates": 200,
    "recall_bias": 0.5
  }
}
```

Response:

```json
{
  "ok": true,
  "results": [
    { "key": "doc-1", "score": 0.99, "metadata": { "tag": "demo" } }
  ],
  "explain": {
    "query_id": "123",
    "snapshot_id": "456",
    "global_seed": "7",
    "contract_version": 2,
    "execution_plan": [
      {
        "name": "vector_stage_1",
        "index": "hnsw",
        "ef_search": 50,
        "nprobe": 0,
        "max_candidates": 20,
        "time_ms": 1.2,
        "candidates_out": 20
      }
    ],
    "results": [
      {
        "vector_score_raw": 0.99,
        "vector_score_normed": 0.99,
        "keyword_score_raw": 0.0,
        "keyword_score_normed": 0.0,
        "fusion_method": "weighted_sum",
        "final_score": 0.99,
        "rank_before_fusion": 1,
        "rank_after_fusion": 1
      }
    ]
  }
}
```

## POST /v1/search_hybrid_explain

Request:

```json
{
  "text_query": "hello",
  "vector": [1.0, 0.0, 0.0],
  "alpha": 0.7,
  "topk": 10,
  "policy": {
    "max_latency_ms": 25,
    "max_candidates": 100,
    "recall_bias": 0.5,
    "fusion_method": "rrf"
  }
}
```

Response: same shape as `/v1/search_explain` with keyword scores filled.

## GET /v1/metrics

Response:

```json
{
  "queries_total": 10,
  "upserts_total": 5,
  "deletes_total": 1,
  "query_latency_ms": { "le_1": 2, "le_5": 5, "overflow": 0 }
}
```

## GET /v1/healthz

Response:

```json
{ "ok": true }
```

## GET /v1/readyz

Response:

```json
{ "ok": true }
```
