import pandas as pd

from ecg_project.data.catalog import TARGETS as LEGACY_TARGETS
from ecg_project.data.classifier_labels import diagnosis_labels, select_common_classes


def test_equivalent_codes_subtypes_and_unknown_labels():
    codes = ['59118001', '713427006', '164884008', '427172004;17338001',
             '251182009', '429622005', '704997005', '164931005',
             '426783006', '', '164889003', '713426002', '164909002']
    frame = diagnosis_labels(pd.DataFrame(dict(labels=codes, source='CPSC',
        split=['train'] * 10 + ['inference', 'train', 'train'])))
    assert frame.RBBB.tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 0, -1, -1, 0, 0]
    assert frame.PVC.iloc[2:5].tolist() == [1, 1, 1]
    assert frame.STD.iloc[5:8].tolist() == [1, 1, 0]
    assert frame.STE.iloc[7] == 1
    assert '164884008' not in LEGACY_TARGETS['PVC']


def test_every_source_required_and_aliases_merged():
    frame = diagnosis_labels(pd.DataFrame([
        dict(source='Training_WFDB', split='train', labels='59118001;429622005'),
        dict(source='CPSC', split='train', labels='426783006'),
        dict(source='WFDB_GEORGIA', split='external', labels='713427006'),
        dict(source='LUDB', split='inference', labels=''),
    ]))
    classes, report = select_common_classes(frame, min_train=1)
    assert 'RBBB' in classes and 'STD' not in classes
    assert report['required_sources'] == ['CPSC', 'GEORGIA']
    assert report['candidates']['STD']['exclusion_reasons'] == ['no_positive_in:GEORGIA']
    classes, report = select_common_classes(frame, ['CPSC', 'GEORGIA', 'STP'], min_train=1)
    assert not classes  # A source lost during preparation cannot disappear silently.
    assert report['candidates']['RBBB']['by_source']['STP']['positive'] == 0


def test_unknown_supported_source_and_holdout_counts_cannot_supply_train_support():
    frame = diagnosis_labels(pd.DataFrame([
        dict(source='CPSC', split='train', labels='164889003'),
        dict(source='CPSC', split='train', labels='426783006'),
        *[dict(source='GEORGIA', split='external', labels='164889003') for _ in range(25)],
        dict(source='STP', split='external_long', labels=''),
    ]))
    classes, report = select_common_classes(frame)
    assert not classes
    assert report['candidates']['AF']['exclusion_reasons'] == [
        'no_positive_in:STP', 'train_requires_20_positive_and_negative']
    assert report['candidates']['AF']['by_source']['STP']['unknown'] == 1
