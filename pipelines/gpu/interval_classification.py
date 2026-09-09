"""Train downstream classifiers from already trained interval checkpoints."""
import argparse
import json
from pathlib import Path
from ecg_project.training.interval_classification import ClassificationConfig, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/qwen_E_classifier.json')
    args = parser.parse_args()
    run(ClassificationConfig(**json.loads(Path(args.config).read_text(encoding='utf-8'))))


if __name__ == '__main__':
    main()
