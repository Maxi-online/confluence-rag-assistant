"""English clarification messages must remain recognizable by orchestration."""
import ast
from pathlib import Path
import unittest

class ConversationLanguageContracts(unittest.TestCase):
    def test_english_clarification_detection(self):
        path = Path(__file__).resolve().parents[2] / 'ai/mas/agents/orchestrator_agent.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'is_clarification')
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), 'exec'), namespace)
        detect = namespace['is_clarification']
        for message in ('Could you clarify the program?', 'Could you please clarify your goal?', 'Please specify: program area', 'Clarify: education goal'):
            with self.subTest(message=message):
                self.assertTrue(detect(message))
        self.assertFalse(detect('The program starts in September.'))

if __name__ == '__main__':
    unittest.main()
