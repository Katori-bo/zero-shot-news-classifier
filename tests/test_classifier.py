import unittest
from types import SimpleNamespace

from classifier import NewsClassifier, validate_input


class Tokenizer:
    model_max_length = 24
    def __call__(self, text, hypothesis):
        return {"input_ids": list(range(len((text + " " + hypothesis).split()) + 4))}


class Pipeline:
    tokenizer = Tokenizer()
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash="test-revision"))
    def __call__(self, text, **kwargs):
        self.options = kwargs
        return {"sequence": text, "labels": ["sports", "politics"], "scores": [0.8, 0.2]}


class InputTests(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(validate_input(" News ", [" sports ", "politics"]), ("News", ["sports", "politics"]))
    def test_invalid_inputs(self):
        for text, labels in [(" ", ["a", "b"]), ("News", ["a"]), ("News", ["a", " "]),
                             ("News", ["Sports", "sports"]), ("News", ["a", 2]), ("News", "a,b")]:
            with self.subTest(text=text, labels=labels), self.assertRaises(ValueError):
                validate_input(text, labels)


class InferenceContractTests(unittest.TestCase):
    def setUp(self):
        self.classifier = NewsClassifier()
        self.pipeline = Pipeline()
        self.classifier._pipeline = self.pipeline
    def test_options_and_metadata(self):
        result = self.classifier.classify("Cricket match", ["sports", "politics"])
        self.assertEqual(self.pipeline.options["hypothesis_template"], "This text is about {}.")
        self.assertFalse(self.pipeline.options["multi_label"])
        self.assertEqual(result["revision"], "test-revision")
        self.assertEqual(result["labels"][0], "sports")
    def test_multi_label(self):
        self.classifier.classify("Cricket match", ["sports", "politics"], multi_label=True)
        self.assertTrue(self.pipeline.options["multi_label"])
    def test_long_pair_rejected(self):
        with self.assertRaises(ValueError):
            self.classifier.classify("word " * 30, ["sports", "politics"])


if __name__ == "__main__":
    unittest.main()
