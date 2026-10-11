"""Run the small, fixed demonstration set; this is not a benchmark."""
import json
from pathlib import Path

from classifier import NewsClassifier

ROOT = Path(__file__).parent


def main():
    dataset = json.loads((ROOT / "examples" / "evaluation_set.json").read_text())
    model = NewsClassifier()
    labels = ["sports", "technology", "politics", "education"]
    records = []
    for sample in dataset:
        result = model.classify(sample["text"], labels)
        records.append({**sample, "predicted": result["labels"][0], "result": result})
    correct = sum(item["expected"] == item["predicted"] for item in records)
    f1 = []
    for label in labels:
        tp = sum(r["expected"] == label and r["predicted"] == label for r in records)
        fp = sum(r["expected"] != label and r["predicted"] == label for r in records)
        fn = sum(r["expected"] == label and r["predicted"] != label for r in records)
        f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    report = {"scope": "12 hand-authored demonstration sentences; not a held-out benchmark",
              "count": len(records), "correct": correct, "accuracy": correct / len(records),
              "macro_f1": sum(f1) / len(f1), "records": records}
    path = ROOT / "examples" / "sample_results.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
