# Zero-Shot News Classifier

A college NLP project that classifies short English passages using categories supplied at prediction time. It uses **BART-MNLI** to compare a passage with natural-language topic descriptions. No training examples are collected for the application's labels.

## Features

- Streamlit interface with editable passages and category names.
- Single-topic classification and optional independent topic scoring.
- Ranked results, a score chart, and JSON download.
- Command-line interface with optional JSON export.
- Validation for empty text, duplicate labels, and overlength model inputs.
- Unit tests, interface tests, and an explicit demonstration evaluation script.

## Setup

Python **3.11 or newer** is recommended. Dependencies were checked locally on Python 3.14 with a CPU-only PyTorch installation.

```bash
git clone https://github.com/Katori-bo/zero-shot-news-classifier.git
cd zero-shot-news-classifier
python -m venv .venv
```

Activate the environment:

```bash
# Linux / macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Install the CPU model runtime, then the application dependencies:

```bash
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements.txt
```

The first classification downloads the pretrained model. Internet access and sufficient memory are required. Model weights are downloaded from Hugging Face and are not committed to this repository. A failed download is an execution error, not a prediction.

## Run the interface

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit. Enter a passage, edit the comma-separated topic list, and click **Classify passage**. Keep the labels focused on one aspect, such as topic.

## Run from the command line

```bash
python cli.py "The team won the final cricket match."
python cli.py "Students attended a mathematics workshop." \
  --labels sports technology politics education --output result.json
```

Use quoted arguments for labels containing spaces. Add `--multi-label` to score categories independently. `--revision MODEL_COMMIT_SHA` fixes the checkpoint revision, and `--device 0` selects the first CUDA GPU when a compatible runtime is installed.

## Method

For each category, the application creates a hypothesis such as `This text is about sports.` BART-MNLI evaluates the passage-hypothesis pair. In single-topic mode, the entailment logits are normalized across the candidate set. In multiple-topic mode, entailment and contradiction are compared independently for each category.

The returned JSON includes labels, scores, input, model identity, checkpoint revision, template, mode, and prediction duration. Model loading is excluded from the reported prediction duration.

## Tests and evaluation

```bash
python -m unittest discover -s tests -v
python evaluate.py
```

Unit tests use controlled test doubles to check validation and the inference contract. Interface tests check form submission, error messages, result display, and JSON download; they do not measure model accuracy. Without Streamlit, interface tests are skipped. GitHub Actions runs the dependency-free tests.

`evaluate.py` performs real inference on 12 hand-authored sentences in `examples/evaluation_set.json` and writes `examples/sample_results.json`. It calculates accuracy and macro-F1. This small, straightforward demonstration set is **not a held-out benchmark**, and its results should not be generalized to arbitrary news articles. A fair academic evaluation should use a separate, balanced collection fixed before template selection.

## Recorded demonstration

Real CPU inference was completed on 12 October 2026. BART-MNLI assigned the expected topic to **12/12 fixed examples (100% accuracy; macro-F1 1.00)**. [Recorded predictions](examples/sample_results.json) include all category scores and the checkpoint revision. The sentences are short, hand-authored examples, so this result is not evidence of 100% accuracy on unseen news. No fine-tuning was performed.

## Limitations

- The highest score expresses preference within the supplied label set; it is not a calibrated guarantee of correctness.
- Single-topic mode selects a category even when all candidates are unsuitable.
- Label wording and overlapping categories can change results.
- Inputs are rejected when any passage-hypothesis pair exceeds the model token limit.
- The checkpoint targets English, and mixed-topic passages may need multiple-topic interpretation.

## Project structure

```text
app.py                         Streamlit interface
classifier.py                  Validation and model inference
cli.py                         Command-line interface
evaluate.py                    Demonstration evaluation
examples/evaluation_set.json   Fixed input examples
tests/                         Unit and interface checks
reports/project_report.pdf     Written project report
```

## References

- [Yin, Hay and Roth (2019): Benchmarking Zero-shot Text Classification](https://aclanthology.org/D19-1404/)
- [Lewis et al. (2020): BART](https://aclanthology.org/2020.acl-main.703/)
- [BART-MNLI model card](https://huggingface.co/facebook/bart-large-mnli)
- [Hugging Face pipeline documentation](https://huggingface.co/docs/transformers/main/en/main_classes/pipelines)
