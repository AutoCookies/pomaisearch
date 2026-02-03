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
