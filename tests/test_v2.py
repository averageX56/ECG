import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
import torch
from ecg_project.data.policy import record_split,assign_splits,ssl_train_rows,build_record_manifest
from ecg_project.data.cache import begin_cache,shard,publish,validate_cache,require_checkpoint
from ecg_project.data.delineation_v2 import partial_mask,qt_rows
from ecg_project.data.beat_datasets import AAMI
from ecg_project.processing.hubert import select_window,prepare_signal_v2,VERSION,VERSION_V2,LEADS
from ecg_project.data.protection import assert_unprotected
from ecg_project.training.distillation import distillation_loss


def frame():
    return pd.DataFrame([dict(source=s,record_id=str(i),patient_id=p,fold=f) for i,(s,p,f) in enumerate([
        ('WFDB_PTB-XL','PTB:1',1),('WFDB_PTB-XL','PTB:2',9),('WFDB_PTB-XL','PTB:3',10),
        ('WFDB_GEORGIA','',0),('Training_StPetersburg','',0),('Training_WFDB','',0)])])


def test_shared_split_and_ssl_exclusion():
    f=assign_splits(frame())
    assert f.split.tolist()==['train','valid','test','external','external_long','train']
    selected=ssl_train_rows(frame(),['PTBXL','CPSC','GEORGIA','STP'])
    assert selected.record_id.tolist()==['0','5']
    broken=frame();broken.loc[1,'patient_id']='PTB:1'
    with pytest.raises(ValueError,match='Leakage'):assign_splits(broken)
    with pytest.raises(ValueError):record_split('PTBXL',0)


def test_aami_mapping_preserves_unknown_class():
    assert [AAMI[s] for s in ['N','L','A','a','V','E','F','/','f','Q']]==['N','N','S','S','V','V','F','Q','Q','Q']
    assert '+' not in AAMI


def test_cache_reuse_stale_and_provenance(tmp_path):
    root=begin_cache(tmp_path/'cache',{'version':2,'checkpoint':'abc'})
    calls=[]
    def compute():calls.append(1);return {'x':np.arange(4)}
    p=shard(root,'record',{'signal_hash':'a'},compute)
    shard(root,'record',{'signal_hash':'a'},compute);assert len(calls)==1
    with pytest.raises(ValueError,match='Stale'):shard(root,'record',{'signal_hash':'b'},compute)
    with pytest.raises(ValueError,match='Stale'):begin_cache(root,{'version':3,'checkpoint':'abc'})
    publish(root,{'preprocessing':'v2'},[p]);validate_cache(root,'v2')
    p.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash'):validate_cache(root,'v2')


def test_missing_checkpoint_before_pool(monkeypatch,tmp_path):
    from ecg_project.data import beat_datasets
    monkeypatch.setattr(beat_datasets,'ordered_map',lambda *a,**k:pytest.fail('Pool was started'))
    with pytest.raises(FileNotFoundError,match='Delineator checkpoint missing'):
        beat_datasets.prepare(checkpoint=tmp_path/'absent.pt',sources=['MIT'])


def test_partial_qt_labels_and_manual_only(tmp_path):
    y=partial_mask([dict(wave='T',onset=None,peak=20,offset=30)],100)
    assert (y[:20]==-100).all() and (y[20:31]==3).all() and (y[34:]==-100).all()
    (tmp_path/'split.json').write_text(json.dumps({'records':[dict(record_id=str(i),split='adapt_train',origin='NSR') for i in range(4)]}))
    for i in range(4):(tmp_path/f'{i}.pu').write_text('automatic')
    with pytest.raises(FileNotFoundError,match='Manual q1c'):qt_rows(tmp_path)
    for i in range(4):(tmp_path/f'{i}.q1c').write_text('manual')
    rows=qt_rows(tmp_path)
    assert [r['split'] for r in rows]==['train','train','train','valid']


def test_hubert_duration_and_no_padding():
    rec=SimpleNamespace(signal=np.random.default_rng(0).normal(size=(6000,12)),fs=500,leads=LEADS)
    a,meta=select_window(rec);assert a.signal.shape==(5000,12) and meta['original_duration']==12
    np.testing.assert_array_equal(a.signal,rec.signal[:5000])
    b,_=prepare_signal_v2(rec);c,_=prepare_signal_v2(rec);np.testing.assert_array_equal(b,c)
    rec.signal=rec.signal[:4999]
    with pytest.raises(ValueError,match='shorter'):select_window(rec)
    assert VERSION!=VERSION_V2


@pytest.mark.parametrize('field,value',[('patient_id','p'),('record_id','r'),('signal_hash','h')])
def test_pseudo_protected_identities(field,value):
    registry=[dict(source='PTBXL',patient_id='p',record_id='r',signal_hash='h')]
    row=dict(source='PTBXL',patient_id='other',record_id='other',signal_hash='other');row[field]=value
    with pytest.raises(ValueError):assert_unprotected(row,registry)


def test_distillation_masks_temperature_and_gradient():
    student=torch.randn(2,4,20,requires_grad=True);teacher=torch.randn_like(student);valid=torch.zeros(2,20,dtype=torch.bool);valid[:,5:10]=True
    loss=distillation_loss(student,teacher,valid,2);loss.backward()
    assert torch.isfinite(loss) and student.grad[:,:,5:10].abs().sum()>0
    assert student.grad[:,:,:5].abs().sum()==0
    assert distillation_loss(student,teacher,torch.zeros_like(valid)).item()==0


def manual_cache(root):
    from ecg_project.data.delineation_v2 import VERSION as version
    from ecg_project.data.cache import atomic_json
    root.mkdir();files=[];rows=[]
    for split,n in [('train',2),('valid',1)]:
        for name,value in [('x',np.random.default_rng(1).normal(size=(n,1,128)).astype('float32')),
                           ('y',np.tile(np.arange(128)%4,(n,1)).astype('int64'))]:
            p=root/f'{split}_{name}.npy';np.save(p,value);files.append(p)
        rows.extend(dict(source='LUDB',patient_id=split,record_id=split+str(i),split=split,signal_hash=split+str(i)) for i in range(n))
    atomic_json(root/'manifest.json',rows);files.append(root/'manifest.json')
    publish(root,dict(preprocessing=version,patients={'train':['train'],'valid':['valid']}),files)
    return root


def pseudo_cache(root):
    from ecg_project.data.qwen_pseudo import VERSION as version
    from ecg_project.data.cache import atomic_json
    root.mkdir();logits=np.zeros((4,128),np.float32);logits[2]=9
    np.savez_compressed(root/'r.npz',x=np.zeros((1,128),np.float32),logits=logits,
                        confidence=np.full(128,.999,np.float32),valid=np.ones(128,bool))
    atomic_json(root/'manifest.json',[dict(key='r',split='train',source='CPSC',patient_id='',record_id='r',signal_hash='r')])
    atomic_json(root/'protected_registry.json',[dict(source='LUDB',patient_id='valid',record_id='valid0',signal_hash='valid0')])
    publish(root,dict(preprocessing=version,teacher_sha256='teacher',accepted_pseudo_fraction=1.),list(root.iterdir()))
    return root


def test_mixed_training_resume_and_config_mismatch(tmp_path,monkeypatch):
    from dataclasses import replace
    from ecg_project.training import delineation_v2 as trainer
    monkeypatch.setattr(trainer,'Delineator',lambda:torch.nn.Conv1d(1,4,1))
    cfg=trainer.DelineationConfig(input_root=str(manual_cache(tmp_path/'manual')),pseudo_root=str(pseudo_cache(tmp_path/'pseudo')),
        output=str(tmp_path/'run'),final_checkpoint='',device='cpu',epochs=1,batch_size=2,lambda_kd=.5,consistency_weight=0.)
    result=trainer.train(cfg)
    state=torch.load(result/'latest.pt',weights_only=True)
    assert state['epoch']==1 and state['best_epoch']==1
    assert json.loads((result/'run.json').read_text())['training_regime']=='teacher_student_hard_soft'
    trainer.train(replace(cfg,epochs=2))
    assert torch.load(result/'latest.pt',weights_only=True)['epoch']==2
    before=(result/'latest.pt').read_bytes()
    with pytest.raises(ValueError,match='Resume config'):trainer.train(replace(cfg,lambda_kd=.2))
    assert (result/'latest.pt').read_bytes()==before


def test_pseudo_dataset_rejects_modified_holdout_and_confidence(tmp_path):
    from ecg_project.training.distillation import PseudoDataset
    root=pseudo_cache(tmp_path/'pseudo')
    ds=PseudoDataset(root,tau=1.,kd_threshold=0.)
    _,hard,_,soft=ds[0]
    assert (hard==-100).all() and soft.all()
    rows=json.loads((root/'manifest.json').read_text());rows[0]['patient_id']='valid'
    (root/'manifest.json').write_text(json.dumps(rows))
    with pytest.raises(ValueError,match='hash'):PseudoDataset(root)
    publish(root,dict(preprocessing=ds.meta['preprocessing']),[root/'manifest.json',root/'r.npz',root/'protected_registry.json'])
    with pytest.raises(ValueError,match='Protected'):PseudoDataset(root)


def test_experiment_matrix_and_analysis_bundle(tmp_path):
    from pipelines.gpu.experiments import qwen_experiment
    from scripts.make_analysis_bundle import make_bundle
    import zipfile
    a,b,c,d,e=[qwen_experiment(n) for n in 'ABCDE']
    assert a.pseudo_root is None and b.pseudo_root is None
    assert c.pseudo_root!=d.pseudo_root and d.pseudo_root==e.pseudo_root
    assert d.lambda_kd==0 and e.lambda_kd>0
    assert qwen_experiment('E',vram_gb=80).batch_size==32
    root=tmp_path/'artifacts/cluster/run';root.mkdir(parents=True)
    for name in ['run.json','best_metrics.json','config.json','best.pt','model.safetensors','signals.npy']:(root/name).write_text('{}')
    output=make_bundle(tmp_path/'bundle.zip',tmp_path)
    with zipfile.ZipFile(output) as z:
        assert len(z.namelist())==3 and all(n.endswith('.json') for n in z.namelist())


def test_pseudo_worker_real_tensors_and_resumption(tmp_path,monkeypatch):
    from ecg_project.data import qwen_pseudo as worker
    root=tmp_path/'cache';root.mkdir();checkpoint=tmp_path/'teacher.pt';checkpoint.write_bytes(b'teacher')
    rec=SimpleNamespace(signal=np.random.default_rng(1).normal(size=(2500,1)).astype('float32'),fs=250,leads=['II'])
    monkeypatch.setattr(worker,'load_record',lambda _:rec)
    monkeypatch.setattr(worker,'source_hashes',lambda _:dict(raw='sha256'))
    model=torch.nn.Conv1d(1,4,1)
    monkeypatch.setattr(worker,'cached_predictor',lambda *a:SimpleNamespace(model=model))
    row=dict(path='raw',record_id='a',source='CPSC',patient_id='',split='train')
    result,excluded=worker._one(row,root,str(checkpoint),.95,[])
    assert excluded is None and result['samples']==2500
    with np.load(root/'CPSC_a.npz') as z:
        assert z['logits'].shape==(4,2500) and not z['valid'][:125].any()
    before=(root/'CPSC_a.npz').read_bytes()
    worker._one(row,root,str(checkpoint),.95,[])
    assert before==(root/'CPSC_a.npz').read_bytes()


def test_record_input_metadata_and_short_exclusion(tmp_path,monkeypatch):
    from ecg_project.data import record_inputs as worker
    rec=SimpleNamespace(signal=np.random.default_rng(1).normal(size=(6000,12)),fs=500,leads=LEADS)
    monkeypatch.setattr(worker,'load_record',lambda _:rec)
    monkeypatch.setattr(worker,'source_hashes',lambda _:dict(raw='sha256'))
    row=dict(path='raw',source='CPSC',source_key='CPSC',record_id='a',split='valid',patient_id='',lead_availability='original')
    records,excluded=worker._prepare(row,'founder',tmp_path,3)
    assert len(records)==1 and not excluded and records[0]['original_duration']==12
    rec.signal=rec.signal[:4000]
    records,excluded=worker._prepare(row,'founder',tmp_path,3)
    assert not records and excluded[0]['exclusion_reason']=='shorter_than_10_seconds'


def test_curriculum_rejects_regression_and_requires_same_cohorts():
    from pipelines.gpu.curriculum import accepted
    old=dict(valid_macro_wave_dice=.8,by_source={'LUDB':dict(dice=[1.,.8,.8,.8]),'QTDB':dict(dice=[1.,.8,.8,.8])})
    better=dict(valid_macro_wave_dice=.81,by_source={'LUDB':dict(dice=[1.,.81,.81,.81]),'QTDB':dict(dice=[1.,.81,.81,.81])})
    assert accepted(better,old)
    assert not accepted(old,better)
    regressed=dict(better,by_source={'LUDB':dict(dice=[1.,.9,.9,.9]),'QTDB':dict(dice=[1.,.79,.9,.9])})
    assert not accepted(regressed,old)
    with pytest.raises(ValueError,match='cohorts'):accepted(dict(better,by_source={}),old)


def test_warm_start_keeps_previous_run_immutable(tmp_path,monkeypatch):
    from dataclasses import replace
    from ecg_project.training import delineation_v2 as trainer
    monkeypatch.setattr(trainer,'Delineator',lambda:torch.nn.Conv1d(1,4,1))
    cfg=trainer.DelineationConfig(input_root=str(manual_cache(tmp_path/'manual')),output=str(tmp_path/'first'),
        final_checkpoint='',device='cpu',epochs=1,batch_size=2,consistency_weight=0.)
    first=trainer.train(cfg);before=(first/'best.pt').read_bytes()
    second=trainer.train(replace(cfg,output=str(tmp_path/'second'),warm_start=str(first/'best.pt')))
    assert (first/'best.pt').read_bytes()==before
    assert 'warm_start_sha256' in torch.load(second/'latest.pt',weights_only=True)['identity']


def test_teacher_training_provenance_is_required(tmp_path):
    from ecg_project.data.qwen_pseudo import teacher_training_manifest
    root=manual_cache(tmp_path/'manual');path=tmp_path/'teacher.pt'
    torch.save({'state_dict':{}},path)
    with pytest.raises(ValueError,match='provenance'):teacher_training_manifest(path)
    meta=json.loads((root/'provenance.json').read_text())
    torch.save(dict(identity=dict(config=dict(architecture='unet',input_root=str(root)),cache=meta)),path)
    rows,digest=teacher_training_manifest(path)
    assert len(rows)==2 and all(r['split']=='train' for r in rows)
    (root/'manifest.json').write_text('[]')
    with pytest.raises(ValueError,match='hash mismatch'):teacher_training_manifest(path)


def test_curriculum_warm_starts_best_accepted_cycle_and_resumes(tmp_path,monkeypatch):
    from ecg_project.training.delineation_v2 import DelineationConfig
    from ecg_project.training import delineation_v2 as trainer
    from pipelines.gpu import experiments
    from pipelines.gpu.curriculum import run_curriculum
    monkeypatch.setattr(experiments,'assert_qwen_inputs',lambda cfg:None)
    seen=[];scores=iter([.8,.79,.82])
    def fake_train(cfg):
        seen.append(cfg);score=next(scores);out=Path(cfg.output);out.mkdir(parents=True)
        (out/'best.pt').write_bytes(str(score).encode())
        (out/'best_metrics.json').write_text(json.dumps(dict(valid_macro_wave_dice=score,by_source={'LUDB':dict(dice=[1.,score,score,score])})))
        return out
    monkeypatch.setattr(trainer,'train',fake_train)
    cfg=DelineationConfig(input_root=str(manual_cache(tmp_path/'manual')),pseudo_root=str(pseudo_cache(tmp_path/'pseudo')),
        architecture='qwen',output=str(tmp_path/'experiment'))
    root=run_curriculum(cfg)
    state=json.loads((root/'selection.json').read_text())
    assert [r['accepted'] for r in state['cycles']]==[True,False,True]
    assert seen[1].warm_start==seen[2].warm_start==str(root/'cycle_01/best.pt')
    run_curriculum(cfg);assert len(seen)==3


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_batched_pseudo_inference_and_cached_resume(tmp_path,monkeypatch,device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    from ecg_project.data import qwen_pseudo as worker
    rec=SimpleNamespace(signal=np.random.default_rng(4).normal(size=(2500,1)).astype('float32'),fs=250,leads=['II'])
    monkeypatch.setattr(worker,'load_record',lambda _:rec)
    monkeypatch.setattr(worker,'source_hashes',lambda _:dict(raw='sha256'))
    rows=[dict(path='raw',record_id=str(i),source='CPSC',patient_id='',split='train') for i in range(3)]
    model=torch.nn.Conv1d(1,4,1).to(device);batches=[]
    hook=model.register_forward_pre_hook(lambda model,args:batches.append(len(args[0])))
    items=[worker._input(r,tmp_path,'teacher',.95,[]) for r in rows]
    result=list(worker.batched_results(items,tmp_path,model,device,2,.95))
    assert batches==[2,1] and len(result)==3
    with np.load(tmp_path/'CPSC_0.npz') as z:
        with torch.no_grad():expected=model(torch.from_numpy(z['x'][None]).to(device))[0].cpu().numpy()
        np.testing.assert_allclose(z['logits'],expected,atol=1e-5)
    batches.clear()
    items=[worker._input(r,tmp_path,'teacher',.95,[]) for r in rows]
    list(worker.batched_results(items,tmp_path,model,device,2,.95))
    assert not batches  # A complete cache causes zero teacher forward passes.
    hook.remove()


def test_colab_attach_never_deletes_existing_base(tmp_path):
    from pipelines.gpu.colab import attach_drive
    root=tmp_path/'repo';drive=tmp_path/'drive';(root/'artifacts').mkdir(parents=True)
    base=root/'artifacts/base.pt';base.write_bytes(b'keep')
    with pytest.raises(FileExistsError,match='Nonempty local path'):attach_drive(root,drive)
    assert base.read_bytes()==b'keep'
