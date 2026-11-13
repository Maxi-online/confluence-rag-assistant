# FAQ ingestion and retrieval reference

The FAQ module is part of the wider Confluence/RAG assistant. It converts approved
source material into FAQ records, generates embeddings, stores records in Weaviate,
and retrieves them according to the session's authorization state. The root
[README](../README.md) records the verified offline scope and integration limits.

## Code navigation

| Module | Responsibility |
| --- | --- |
| [csv_to_faq.py](../integrations/csv_to_faq.py) | Offline CSV/Excel conversion and synthetic sample generation |
| [weaviate_faq_loader.py](../integrations/weaviate_faq_loader.py) | Collection setup, embedding calls and insertion |
| [sync_faq_table.py](../data/sync_faq_table.py) | Explicit-file ingestion with private configuration support |
| [load_faq_to_weaviate.py](../load_faq_to_weaviate.py) | Wrapper for the explicit ingestion CLI |
| [test_faq_search.py](../test_faq_search.py) | Connected retrieval probe |
| [graph_base.py](../ai/mas/utils/graph_base.py) | Retrieval/client integration |
| [text_utils.py](../integrations/confluence_api/text_utils.py) | Confluence HTML cleaning |
| [weaviate_upload.py](../integrations/confluence_api/weaviate_upload.py) | Confluence ingestion integration |

The `reset_faq_class.py` utility deletes collection data. It is an administrative
operation, not a prerequisite for the offline sample or routine retrieval tests.

## Record schema

| Field | Meaning |
| --- | --- |
| `question` | FAQ question |
| `answer` | Source-backed answer |
| `keywords` | Search terms as a list |
| `topic` | Subject/category |
| `main_point` | Main point from the source |
| `user_type` | Visibility category |
| `implementation_period` | Optional period supplied by the input record |

The converter's English input labels are `applicant` and `student`. It preserves
the existing collection wire labels so application filters remain compatible.
Unauthenticated applicant retrieval should return applicant-visible records only;
authorized student retrieval can include student records. The presence of a
filter in one helper does not establish end-to-end access control for every route.

## Offline conversion and layout selection

See the [portable ingestion guide](faq-ingestion.md) for complete runnable examples.
CSV conversion needs no account credentials or provider connection. Standard input
uses a header at row 0 and data starting immediately afterward. Legacy external
headers are recognized as data aliases. A private `--column-map` supplies any
other approved header mapping.

No input is discovered by a customer's filename. A nonstandard layout is selected
explicitly through `--header-row` and `--data-start-row`; no old row offset is
silently inherited. The sample generator uses fictional content and no institution,
tuition information, enrollment schedule or customer contact details.

## Connected ingestion

Use a deployment environment with the application's dependencies and explicit
`WEAVIATE_URL`, `WEAVIATE_API_KEY` and `OPENAI_API_KEY` values. Run from the root:

```powershell
python load_faq_to_weaviate.py --csv artifacts/sample_faq.csv
```

The same input can be supplied to the synchronization entrypoint:

```powershell
python data/sync_faq_table.py --csv artifacts/sample_faq.csv
```

Both commands perform real connected ingestion when configured. The sync entrypoint
also accepts private configuration and layout parameters. Supply secrets through
the process environment or ignored private configuration; avoid embedding them in
shell history. Match the embedding model and vector dimension to the collection.

Ingestion creates the collection schema when needed, reads/validates FAQ records,
embeds question/keyword/answer text and inserts records. The legacy insertion
path uses generated UUIDs; repeating an import is not proof of idempotence.
Embedding failure behavior and duplicate handling need separate production review.

## Retrieval probes

With a configured local integration environment, compare applicant and authorized
student retrieval:

```powershell
python test_faq_search.py --query 'What documents are required for admission?'
python test_faq_search.py --query 'How can a student request a certificate?' --authorized
```

Queries are illustrative English examples, not imported institutional knowledge.
Validate the requested record's visibility as well as its text. The Common
Questions Agent can search FAQ content before the broader knowledge base; review
authorization propagation across both paths.

## Evaluation

Use a synthetic page hierarchy and labeled questions to check retrieval coverage,
answer/source agreement, abstention when evidence is missing and exclusion of
inaccessible documents. Separate retrieval relevance from generation quality.
Test applicant/student boundaries on every reachable retrieval route. The offline
suite covers HTML cleaning, portable CSV conversion, explicit offsets/mappings,
visibility validation and English conversation contracts. It does not verify a
live provider account, production ACLs or embedding/retrieval accuracy.
