# Portable FAQ ingestion

The converter accepts an explicitly selected CSV or Excel file. It does not
search for a customer's filename or assume a particular worksheet layout.
Standard CSV conversion uses Python's standard library and needs no provider
credentials. Excel support uses the application's optional pandas/Excel dependencies.

## Offline sample

Run from the repository root in an isolated Python environment:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
python integrations/csv_to_faq.py --create-sample --sample-output artifacts/sample_faq.csv
python integrations/csv_to_faq.py --input artifacts/sample_faq.csv --output artifacts/sample_faq.json
```

The two records are fictional. The default header is input row 0, with data
starting immediately after it. English headers are `question`, `answer`,
`keywords`, `topic`, `main_point`, `user_type` and `implementation_period`.
Question, answer and visibility are required. Empty records are skipped; ambiguous headers
and unknown visibility labels fail instead of silently publishing private data.

Use `applicant` or `student` for the sample's visibility labels. The converter
preserves the collection's existing wire labels so the application's authorization
filters remain compatible. Legacy external-language column aliases are also supported.

## Private layouts

`--header-row` and `--data-start-row` are zero-based input row indices. They can
be selected explicitly for an approved file; no rows are skipped twice.
This runnable example adds four synthetic preamble rows before the same sample:

```powershell
$sampleRows = Get-Content artifacts/sample_faq.csv -Encoding UTF8
@('Synthetic preamble 1', 'Synthetic preamble 2', 'Synthetic preamble 3', 'Synthetic preamble 4') + $sampleRows | Set-Content artifacts/offset_faq.csv -Encoding UTF8
python integrations/csv_to_faq.py --input artifacts/offset_faq.csv --output artifacts/offset_faq.json --header-row 4 --data-start-row 5
```

Those offsets describe the selected input, so use them only with a file that has
that layout. `--column-map` can point to a private JSON object mapping canonical
field names to that file's actual header names. No mapping is preconfigured.

## Connected ingestion

The Weaviate entrypoint requires the full application dependencies and explicit
`WEAVIATE_URL`, `WEAVIATE_API_KEY` and `OPENAI_API_KEY` values in its environment:

```powershell
python -m integrations.weaviate_faq_loader --csv artifacts/sample_faq.csv
```

It also accepts `--header-row`, `--data-start-row` and `--column-map`. Running it
creates/uses a collection and calls configured providers. The offline suite
tests CSV conversion, offsets, mappings and visibility handling; it does not
validate a live provider account, embedding quality or production deployment.
