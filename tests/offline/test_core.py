import unittest
from integrations.confluence_api.text_utils import static_clean_html

class IngestionTests(unittest.TestCase):
    def test_script_and_style_removed(self):self.assertEqual(static_clean_html("<script>unsafe()</script><style>secret</style><p>Public</p>"),"Public")
    def test_heading_and_list_preserved(self):
        text=static_clean_html("<h1>Demo</h1><ul><li>First</li><li>Second</li></ul>")
        self.assertIn("Demo",text);self.assertIn("First",text);self.assertIn("Second",text)
    def test_confluence_macro_removed(self):self.assertNotIn("macro-secret",static_clean_html("<ac:structured-macro>macro-secret</ac:structured-macro><p>Visible</p>"))
    def test_unicode(self):self.assertIn("Справочник",static_clean_html("<p>Справочник</p>"))
