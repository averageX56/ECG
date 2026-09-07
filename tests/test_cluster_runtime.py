import json
from pathlib import Path
from unittest.mock import Mock
import pytest


def test_cluster_preparation_preserves_existing_paths(tmp_path, monkeypatch):
    from pipelines.gpu.cluster import prepare_qwen_pseudo_gpu
    monkeypatch.chdir(tmp_path)
    for name in ('data', 'LUDB', 'artifacts'):
        Path(name).mkdir()
        (Path(name)/'keep').write_text('unchanged')
    Path('artifacts/catalog.csv').write_text('catalog')
    Path('teacher.pt').write_text('teacher')
    prepare=Mock()
    monkeypatch.setattr('ecg_project.data.qwen_pseudo.prepare', prepare)
    prepare_qwen_pseudo_gpu(checkpoint='teacher.pt')
    assert prepare.call_args.kwargs['device']=='cuda'
    assert prepare.call_args.kwargs['batch_size']==64
    assert prepare.call_args.kwargs['workers']==8
    for name in ('data', 'LUDB', 'artifacts'):
        assert not Path(name).is_symlink()
        assert (Path(name)/'keep').read_text()=='unchanged'


def test_cluster_missing_teacher_fails_before_prepare(tmp_path, monkeypatch):
    from pipelines.gpu.cluster import prepare_qwen_pseudo_gpu
    monkeypatch.chdir(tmp_path)
    prepare=Mock()
    monkeypatch.setattr('ecg_project.data.qwen_pseudo.prepare',prepare)
    with pytest.raises(FileNotFoundError):prepare_qwen_pseudo_gpu(checkpoint='missing.pt')
    prepare.assert_not_called()


def test_notebook_is_cluster_only():
    n=json.loads(Path('notebooks/04_all_pipelines_a100.ipynb').read_text(encoding='utf8'))
    code='\n'.join(''.join(c['source']) for c in n['cells'] if c['cell_type']=='code')
    for forbidden in ('google.colab','/content','attach_drive','rmtree','pipelines.gpu.colab','stage_qwen_raw'):
        assert forbidden not in code
    assert 'pipelines.gpu.cluster' in code
    import ast
    for c in n['cells']:
        if c['cell_type']=='code':ast.parse(''.join(c['source']))


def test_cluster_extended_sources_exact():
    from pipelines.gpu.cluster import EXTENDED_QWEN_SOURCES
    assert set(EXTENDED_QWEN_SOURCES)=={'CPSC_EXTRA','PTBXL','CPSC','CHAPMAN'}


def test_cluster_scratch_interruption_never_publishes(tmp_path, monkeypatch):
    from pipelines.gpu.cluster import prepare_qwen_pseudo_gpu
    monkeypatch.chdir(tmp_path)
    Path('artifacts').mkdir();Path('artifacts/catalog.csv').write_text('catalog')
    Path('teacher.pt').write_text('teacher')
    def interrupted(**kwargs):
        path=Path(kwargs['output'])
        assert path.is_relative_to(tmp_path/'scratch')
        path.mkdir(parents=True);(path/'partial').write_text('partial')
        raise KeyboardInterrupt
    monkeypatch.setattr('ecg_project.data.qwen_pseudo.prepare',interrupted)
    with pytest.raises(KeyboardInterrupt):
        prepare_qwen_pseudo_gpu(checkpoint='teacher.pt',scratch='scratch')
    assert not Path('artifacts/qwen_pseudo_extended').exists()
    assert Path('teacher.pt').read_text()=='teacher'


def test_cluster_persistent_partial_resumes_without_replacement(tmp_path, monkeypatch):
    from pipelines.gpu.cluster import prepare_qwen_pseudo_gpu
    monkeypatch.chdir(tmp_path)
    Path('artifacts/qwen_pseudo_extended').mkdir(parents=True)
    Path('artifacts/qwen_pseudo_extended/keep').write_text('partial')
    Path('artifacts/catalog.csv').write_text('catalog');Path('teacher.pt').write_text('teacher')
    prepare=Mock();monkeypatch.setattr('ecg_project.data.qwen_pseudo.prepare',prepare)
    result=prepare_qwen_pseudo_gpu(checkpoint='teacher.pt',scratch='scratch')
    assert result==tmp_path/'artifacts/qwen_pseudo_extended'
    assert Path(prepare.call_args.kwargs['output'])==result
    assert (result/'keep').read_text()=='partial'
    assert not Path('scratch').exists()


def test_cluster_completed_teacher_mismatch(tmp_path, monkeypatch):
    from pipelines.gpu.cluster import prepare_qwen_pseudo_gpu
    monkeypatch.chdir(tmp_path)
    output=Path('artifacts/qwen_pseudo_extended');output.mkdir(parents=True)
    (output/'provenance.json').write_text('original')
    Path('artifacts/catalog.csv').write_text('catalog');Path('teacher.pt').write_text('teacher')
    monkeypatch.setattr('ecg_project.data.cache.validate_cache',lambda p:{'teacher_sha256':'different'})
    prepare=Mock();monkeypatch.setattr('ecg_project.data.qwen_pseudo.prepare',prepare)
    with pytest.raises(ValueError,match='Teacher hash mismatch'):
        prepare_qwen_pseudo_gpu(checkpoint='teacher.pt')
    prepare.assert_not_called()
    assert (output/'provenance.json').read_text()=='original'
