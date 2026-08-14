# PREDICTIONS.md

Before each advanced assignment, write what you expect **before** you measure. **Never edit a past entry** — the value is the calibration record. After six weeks, knowing where your intuition is systematically wrong is the rarest skill in the field.

---

## Stage 0C — Cost model for a Q&A system

*Fill this in before running any measurement.*

**System on paper:** Q&A over 10,000 documents, 1,000 queries/day.

| Design parameter | Choice |
|------------------|--------|
| Model | _TBD_ |
| Quantization | _TBD_ |
| Hardware | _TBD_ |
| Context budget / query | _TBD tokens_ |
| Retrieval top-k | _TBD_ |
| Expected input tokens | _TBD_ |
| Expected output tokens | _TBD_ |

**Predictions (write before measuring):**

| Metric | Predicted | Measured | Gap |
|--------|-----------|----------|-----|
| Cost per query (or sec/query if self-hosted) | _TBD_ | | |
| p50 latency | _TBD_ | | |
| p95 latency | _TBD_ | | |
| Monthly total | _TBD_ | | |

**Crudest build to measure against:** model + fixed 2K-token stuffed context, no retrieval.

**Pass:** state which single term dominates *cost* and which dominates *latency* — they must **not** be the same term. (If they look identical, you've likely conflated TTFT with total time. Re-measure.)

*Gap analysis paragraph: TBD after measurement.*
