# GPT-4o Annotation Cost Analysis

## Summary

The 5-level cascade (page → paragraphs → lines → words → characters) makes
**one API call per node** in the hierarchy tree. Dense PDF pages are expensive
because word-count drives call-count, and every call includes an image.

## API Call Structure Per Page

| Level | What | Calls | Image detail |
|-------|------|-------|-------------|
| 0 | Page region | 1 | low |
| 1 | Paragraphs | 1 | low |
| 2 | Lines (per paragraph) | N_para | high |
| 3 | Words (per line) | N_lines | high |
| 4 | Characters (per word) | N_words | high |

**Total calls = 2 + N_para + N_lines + N_words**

## Per-Call Token Cost (from actual data)

Source: `pipeline_logs/annotate_batch_20260228_002440.log` — one fully completed
dense PDF page (`pdf_000014`, 46 paras, 68 lines, 543 words, 659 calls).

- **Prompt tokens/call**: ~504 (includes ~255 image tokens at detail=high)
- **Completion tokens/call**: ~196
- **Cost/call**: ~$0.00322
  - Input: 504 × $2.50/1M = $0.00126
  - Output: 196 × $10.00/1M = $0.00196

## Cost By Page Type

### Doc pages (non-PDF, typically simpler layouts)
- 66 annotated samples available
- **Avg calls/page**: 63
- **Avg cost/page: ~$0.20**
- Range: $0.01 (near-blank) – $1.76 (dense doc)

### PDF pages (scanned PDFs, typically dense text)
- 32 annotated samples available
- **Avg calls/page**: 564
- **Avg cost/page: ~$1.82**
- Range: $0.01 (sparse) – $4.45 (very dense, 1382 calls)

## Dataset Composition

| | Total | Annotated | Remaining |
|--|-------|-----------|-----------|
| Doc pages | 100 | 66 | 34 |
| PDF pages | 2,230 | 32 | 2,198 |
| **All** | **2,330** | **98** | **2,232** |

## Budget Projections

| Budget | Docs | PDFs | Total pages |
|--------|------|------|-------------|
| $10 | 34 ($7) | 1-2 | ~36 |
| $25 | 34 ($7) | ~10 | ~44 |
| $50 | 34 ($7) | ~23 | ~57 |
| $100 | 34 ($7) | ~51 | ~85 |
| $4,000 | 34 ($7) | ~2,198 | ~2,232 (all) |

## Notes

- Output tokens dominate cost (~61% of per-call cost) due to JSON bbox schemas
- The `rate_limit_delay` default is 0.5s; can be reduced with `--rate-limit-delay 0.1`
- GPT-4o pricing: $2.50/1M input, $10.00/1M output (as of 2026-03)
- Script has early abort on quota errors (`insufficient_quota` / `billing`)
- Token tracker prints cumulative cost in logs; use `tracker.summary()` at end
- Already-annotated pages are NOT automatically skipped — filter the input file list
