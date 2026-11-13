"""Convert explicitly selected CSV/Excel FAQ data to the collection schema.

CSV ingestion uses the standard library. Excel support imports pandas lazily.
English headers are the public default; legacy Russian headers remain supported
as external data, with the existing collection visibility labels preserved.
"""
import argparse
import csv
import json
from pathlib import Path

COLUMN_ALIASES = {
    "question": ("question", "Вопрос"),
    "answer": ("answer", "Ответ сотрудника"),
    "keywords": ("keywords", "Ключевые слова"),
    "topic": ("topic", "Тема запроса"),
    "main_point": ("main_point", "Основные моменты"),
    "user_type": ("user_type", "Тип пользователя"),
    "implementation_period": ("implementation_period", "Срок реализации"),
}
USER_TYPES = {"applicant": "Абитуриент", "student": "Студент"}

def _read_rows(path, header_row, data_start_row):
    if header_row < 0 or data_start_row <= header_row:
        raise ValueError("Header and first-data row must be valid, ordered zero-based indices")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        import pandas as pd
        frame = pd.read_excel(path, header=header_row, skiprows=range(header_row + 1, data_start_row)).fillna("")
        return [str(c).strip() for c in frame.columns], frame.values.tolist()
    for encoding in ("utf-8-sig", "cp1251", "latin-1"):
        try:
            text = path.read_text(encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    try:
        dialect = csv.Sniffer().sniff(text[:8192], delimiters=",;\t")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.reader(text.splitlines(keepends=True), dialect))
    if header_row >= len(rows):
        raise ValueError("The configured header row is outside the input file")
    return [c.strip() for c in rows[header_row]], rows[data_start_row:]

def load_faq_from_csv(input_file, header_row=0, data_start_row=None, column_mapping=None):
    """Read a selected file; explicit offsets/mapping support private layouts.

    column_mapping maps canonical field names to the input's actual headers.
    Unknown visibility labels fail rather than silently publishing private data
    as applicant-visible content.
    """
    first_data = header_row + 1 if data_start_row is None else data_start_row
    headers, rows = _read_rows(Path(input_file), header_row, first_data)
    if len(set(headers)) != len(headers):
        raise ValueError("Duplicate FAQ column names are ambiguous")
    mapping = column_mapping or {}
    if set(mapping) - set(COLUMN_ALIASES):
        raise ValueError("Column mapping contains unknown canonical fields")
    indices = {}
    for field, aliases in COLUMN_ALIASES.items():
        choices = (mapping[field],) if field in mapping else aliases
        source = next((name for name in choices if name in headers), None)
        if source is not None:
            indices[field] = headers.index(source)
        elif field in mapping:
            raise ValueError("An explicitly mapped FAQ column is absent")
    if not {"question", "answer", "user_type"} <= indices.keys():
        raise ValueError("FAQ input requires question, answer and visibility columns")
    if len(set(indices.values())) != len(indices):
        raise ValueError("One input column cannot represent multiple FAQ fields")
    records = []
    for row in rows:
        def value(field):
            index = indices.get(field)
            return str(row[index]).strip() if index is not None and index < len(row) else ""
        question, answer = value("question"), value("answer")
        if not question or not answer:
            continue
        user_type = value("user_type")
        user_type = USER_TYPES.get(user_type.casefold(), user_type)
        if user_type not in USER_TYPES.values():
            raise ValueError("Unknown FAQ visibility label")
        records.append({
            "question": question, "answer": answer,
            "keywords": [part.strip() for part in value("keywords").split(",") if part.strip()],
            "topic": value("topic"), "main_point": value("main_point"),
            "user_type": user_type, "implementation_period": value("implementation_period"),
        })
    return records

def convert_csv_to_json(input_file, output_file, header_row=0, data_start_row=None, column_mapping=None):
    """Convert a selected FAQ file without contacting an external service."""
    records = load_faq_from_csv(input_file, header_row, data_start_row, column_mapping)
    Path(output_file).write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return records

def create_sample_csv(output_file):
    """Write two fictional records with standard English headers."""
    with Path(output_file).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(COLUMN_ALIASES))
        writer.writeheader()
        writer.writerow({"question": "What is this example?", "answer": "A fictional applicant FAQ record.", "keywords": "demo, applicant", "user_type": "applicant"})
        writer.writerow({"question": "Who can read student records?", "answer": "An authorized student session in this demonstration.", "keywords": "demo, student", "user_type": "student"})

def main():
    parser = argparse.ArgumentParser(description="Convert an explicitly selected FAQ CSV/Excel file to JSON")
    parser.add_argument("--input", help="Input CSV or Excel path")
    parser.add_argument("--output", help="Output JSON path")
    parser.add_argument("--header-row", type=int, default=0, help="Zero-based header row")
    parser.add_argument("--data-start-row", type=int, help="Zero-based first data row; defaults to the row after the header")
    parser.add_argument("--column-map", help="Private JSON mapping of canonical field names to input headers")
    parser.add_argument("--create-sample", action="store_true")
    parser.add_argument("--sample-output", default="sample_faq.csv")
    args = parser.parse_args()
    if args.create_sample:
        create_sample_csv(args.sample_output)
    elif args.input and args.output:
        mapping = json.loads(Path(args.column_map).read_text(encoding="utf-8")) if args.column_map else None
        records = convert_csv_to_json(args.input, args.output, args.header_row, args.data_start_row, mapping)
        print(json.dumps({"records_written": len(records)}))
    else:
        parser.error("Specify --create-sample, or both --input and --output")

if __name__ == "__main__":
    main()
