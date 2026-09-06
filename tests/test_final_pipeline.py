import json
import pytest
from ecg_project.workflows.final_pipeline import run


def test_final_pipeline_rejects_missing_or_modified_model(tmp_path):
    weights=tmp_path/'weights.pt';manifest=tmp_path/'pipeline.json'
    manifest.write_text(json.dumps({'models':{'delineator':{'path':str(weights),'sha256':'incorrect'}}}))
    with pytest.raises(FileNotFoundError,match='transfer required artifact'):run('unused',pipeline=manifest)
    weights.write_bytes(b'changed')
    with pytest.raises(ValueError,match='artifact differs'):run('unused',pipeline=manifest)
