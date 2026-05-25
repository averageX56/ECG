"""
evaluation/
Модуль оценки качества моделей ЭКГ-классификации.

Публичный API:
  from evaluation.metrics import (
      compute_macro_auc, compute_fmax, compute_per_class_metrics,
      classification_report_df, full_evaluation_report,
      bootstrap_ci, RIBEIRO_2020_TARGETS, SNOMED_ABBRS,
  )
  from evaluation.reproduce_ribeiro import run_reproduce
  from evaluation.ablation import run_ablation
"""

from evaluation.metrics import (
    compute_macro_auc,
    compute_fmax,
    compute_per_class_metrics,
    compute_challenge_metric,
    classification_report_df,
    full_evaluation_report,
    bootstrap_ci,
    log_summary,
    RIBEIRO_2020_TARGETS,
    SNOMED_ABBRS,
)

__all__ = [
    "compute_macro_auc",
    "compute_fmax",
    "compute_per_class_metrics",
    "compute_challenge_metric",
    "classification_report_df",
    "full_evaluation_report",
    "bootstrap_ci",
    "log_summary",
    "RIBEIRO_2020_TARGETS",
    "SNOMED_ABBRS",
]
