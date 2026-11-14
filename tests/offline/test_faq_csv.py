"""Exercise portable FAQ imports without provider clients or account credentials."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from integrations.csv_to_faq import COLUMN_ALIASES, USER_TYPES, create_sample_csv, load_faq_from_csv

class FAQCSVTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='faq-ingestion-test-')
        self.directory = Path(self.temporary.name).resolve()
        self.assertEqual(self.directory.parent, Path(tempfile.gettempdir()).resolve())
        self.addCleanup(self.temporary.cleanup)

    def write(self, name, rows):
        path = self.directory / name
        with path.open('w', encoding='utf-8', newline='') as stream:
            csv.writer(stream).writerows(rows)
        return path

    def test_standard_sample_import_is_independent_of_filename(self):
        for name in ('approved.csv', 'arbitrary-name.csv'):
            path = self.directory / name
            create_sample_csv(path)
            result = load_faq_from_csv(path)
            self.assertEqual(len(result), 2)
            self.assertEqual([r['user_type'] for r in result], [USER_TYPES['applicant'], USER_TYPES['student']])

    def test_legacy_headers_preserve_student_visibility(self):
        fields = ('question', 'answer', 'user_type')
        headers = [COLUMN_ALIASES[field][-1] for field in fields]
        path = self.write('legacy.csv', [headers, ['A student question?', 'A private student answer.', USER_TYPES['student']]])
        self.assertEqual(load_faq_from_csv(path)[0]['user_type'], USER_TYPES['student'])

    def test_explicit_header_and_data_rows_are_not_skipped_twice(self):
        path = self.write('offsets.csv', [['Metadata'], ['Metadata'], ['question', 'answer', 'user_type'], ['Skip this note'], ['A question?', 'An answer.', 'applicant']])
        result = load_faq_from_csv(path, header_row=2, data_start_row=4)
        self.assertEqual([r['question'] for r in result], ['A question?'])

    def test_private_column_mapping_is_explicit(self):
        path = self.write('mapped.csv', [['Prompt', 'Response', 'Audience'], ['A question?', 'An answer.', 'applicant']])
        result = load_faq_from_csv(path, column_mapping={'question': 'Prompt', 'answer': 'Response', 'user_type': 'Audience'})
        self.assertEqual(result[0]['answer'], 'An answer.')

    def test_missing_and_duplicate_columns_fail(self):
        for rows in ([['question'], ['A question?']], [['question', 'question', 'answer'], ['A', 'B', 'C']]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    load_faq_from_csv(self.write('invalid.csv', rows))

    def test_unknown_visibility_never_falls_back_to_public(self):
        path = self.write('visibility.csv', [['question', 'answer', 'user_type'], ['Question?', 'Confidential answer.', 'unknown-private-group']])
        with self.assertRaisesRegex(ValueError, 'visibility'):
            load_faq_from_csv(path)

    def test_absent_visibility_never_defaults_to_public(self):
        path = self.write('unclassified.csv', [['question', 'answer'], ['Question?', 'Unclassified answer.']])
        with self.assertRaisesRegex(ValueError, 'visibility'):
            load_faq_from_csv(path)

    def test_one_source_column_cannot_supply_question_and_answer(self):
        path = self.write('collision.csv', [['Text', 'Audience'], ['Question or answer?', 'applicant']])
        with self.assertRaises(ValueError):
            load_faq_from_csv(path, column_mapping={'question': 'Text', 'answer': 'Text', 'user_type': 'Audience'})

    def test_cli_roundtrip_needs_no_provider_credentials(self):
        root = Path(__file__).resolve().parents[2]
        source, output = self.directory / 'sample.csv', self.directory / 'sample.json'
        script = root / 'integrations/csv_to_faq.py'
        subprocess.run([sys.executable, str(script), '--create-sample', '--sample-output', str(source)], check=True, capture_output=True)
        subprocess.run([sys.executable, str(script), '--input', str(source), '--output', str(output)], check=True, capture_output=True)
        self.assertEqual(len(json.loads(output.read_text(encoding='utf-8'))), 2)

if __name__ == '__main__':
    unittest.main()
