"""Record diagnosis groups for the interval classifier, separate from legacy tasks.

Code descriptions and RBBB/PAC/PVC scoring equivalences:
https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_scored.csv
https://github.com/physionetchallenges/evaluation-2021/blob/main/dx_mapping_unscored.csv
Subtype pooling (AF, PVC, SVEB, AVB, IVCD) is a project grouping, not a
claim that every included SNOMED concept is an exact synonym.
"""
from ecg_project.data.catalog import TARGETS as LEGACY_TARGETS
from ecg_project.data.policy import source_name

LABEL_VERSION = 'common_diagnoses_v1'
TARGETS = {name: list(codes) for name, codes in LEGACY_TARGETS.items()}
TARGETS['PVC'] += ['164884008']  # Ventricular ectopics in CPSC/PTB-XL/INCART.
TARGETS.update({
    'RBBB': ['59118001', '713427006'],
    'STD': ['429622005', '704997005'],  # Includes inferior ST depression.
    'STE': ['164931005'],
})


def diagnosis_labels(frame):
    """Rebuild labels from codes, keeping missing/unsupported labels unknown."""
    frame = frame.copy()
    codes = frame.labels.fillna('').map(
        lambda value: {c.strip() for c in str(value).split(';') if c.strip()})
    supported = frame.split.ne('inference')
    for name, group in TARGETS.items():
        group = set(group)
        frame[name] = [int(bool(value & group)) if value and ok else -1
                       for value, ok in zip(codes, supported)]
    return frame


def select_common_classes(frame, required_sources=None, min_train=20):
    """Require positives in every supported source, then train-only support.

    Source coverage intentionally uses all cohorts to define the requested common
    label vocabulary. It must not be presented as label-blind external evaluation.
    """
    sources = frame.source.map(source_name)
    supported = frame.split.ne('inference')
    required = sorted({source_name(s) for s in required_sources} if required_sources is not None
                      else set(sources[supported]))
    if not required:
        raise ValueError('No supported diagnosis datasets')
    selected, details = [], {}
    for name, codes in TARGETS.items():
        y = frame[name]
        by_source = {s: dict(positive=int((supported & sources.eq(s) & y.eq(1)).sum()),
                             negative=int((supported & sources.eq(s) & y.eq(0)).sum()),
                             unknown=int((supported & sources.eq(s) & y.lt(0)).sum()))
                     for s in required}
        missing = [s for s, counts in by_source.items() if not counts['positive']]
        train = frame.split.eq('train')
        positive, negative = int((train & y.eq(1)).sum()), int((train & y.eq(0)).sum())
        reasons = []
        if missing:
            reasons.append('no_positive_in:' + ','.join(missing))
        if min(positive, negative) < min_train:
            reasons.append(f'train_requires_{min_train}_positive_and_negative')
        if not reasons:
            selected.append(name)
        details[name] = dict(codes=codes, by_source=by_source, train_positive=positive,
                             train_negative=negative, exclusion_reasons=reasons)
    return selected, dict(version=LABEL_VERSION, required_sources=required,
                         selected_classes=selected, candidates=details,
                         selection_policy='Positive in every supported source; train support only. '
                         'All-cohort label coverage defines vocabulary, not model/threshold selection.')
