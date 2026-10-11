"""Interface checks use a controlled backend; they do not evaluate BART."""
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch


class DemoClassifier:
    def classify(self, text, labels, *, multi_label=False):
        return {"sequence": text, "labels": labels, "scores": [0.7, 0.2, 0.08, 0.02],
                "model": "interface-test-double", "multi_label": multi_label}


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Install Streamlit to run interface tests")
class InterfaceTests(unittest.TestCase):
    def setUp(self):
        import streamlit as st
        st.cache_resource.clear()
    def test_form_result_and_download(self):
        from streamlit.testing.v1 import AppTest
        with patch("classifier.NewsClassifier", DemoClassifier):
            app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
            app.button[0].click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Suggested topic: sports")
            self.assertEqual(len(app.dataframe), 1)
            self.assertEqual(len(app.download_button), 1)
    def test_blank_input_shows_error(self):
        from streamlit.testing.v1 import AppTest
        app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py")).run()
        app.text_area[0].set_value(" ")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("non-empty", app.error[0].value)


if __name__ == "__main__":
    unittest.main()
