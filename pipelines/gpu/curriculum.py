"""Validation-gated pseudo-data curriculum, with explicit finite cycle budgets."""
from dataclasses import asdict,replace
from pathlib import Path
import json
from ecg_project.data.cache import atomic_json,begin_cache
from ecg_project.data.catalog import file_hash


def accepted(candidate,incumbent,min_gain=.001,source_tolerance=.005):
    if incumbent is None:return True
    if set(candidate['by_source'])!=set(incumbent['by_source']):raise ValueError('Curriculum validation cohorts differ')
    if candidate['valid_macro_wave_dice']<incumbent['valid_macro_wave_dice']+min_gain:return False
    for source in incumbent['by_source']:
        old=incumbent['by_source'][source]['dice'][1:];new=candidate['by_source'][source]['dice'][1:]
        if any(n<o-source_tolerance for n,o in zip(new,old)):return False
    return True


def run_curriculum(cfg,ratios=(1,2,4),max_stale_cycles=2,min_gain=.001,source_tolerance=.005):
    from pipelines.gpu.experiments import assert_qwen_inputs
    from ecg_project.training.delineation_v2 import train
    if cfg.architecture!='qwen' or not cfg.pseudo_root:raise ValueError('Curriculum requires Qwen with pseudo data')
    if not ratios or any(int(r)!=r or r<1 for r in ratios) or any(b<=a for a,b in zip(ratios,ratios[1:])):
        raise ValueError('Ratios must be increasing positive integers')
    if max_stale_cycles<1 or min_gain<0 or source_tolerance<0:raise ValueError('Invalid curriculum stopping criteria')
    assert_qwen_inputs(cfg)
    config=asdict(cfg)
    if not cfg.pseudo_full_pass:config.pop('pseudo_full_pass')
    root=begin_cache(cfg.output+'_curriculum',dict(config=config,ratios=list(ratios),max_stale_cycles=max_stale_cycles,
        min_gain=min_gain,source_tolerance=source_tolerance,
        manual_provenance_sha256=file_hash(Path(cfg.input_root)/'provenance.json'),
        pseudo_provenance_sha256=file_hash(Path(cfg.pseudo_root)/'provenance.json')))
    journal=root/'selection.json'
    state=json.loads(journal.read_text()) if journal.exists() else dict(cycles=[],best_checkpoint=cfg.warm_start,best_metrics=None,stale_cycles=0,status='running')
    # Optional incumbent supplies the validation baseline for the first proposed cycle.
    if not state['cycles'] and cfg.warm_start:
        state['best_metrics']=json.loads((Path(cfg.warm_start).parent/'best_metrics.json').read_text())
        initial=json.loads((Path(cfg.warm_start).parent/'config.json').read_text())
        if Path(initial['input_root'])!=Path(cfg.input_root):raise ValueError('Warm-start curriculum must use the same validation cache')
    try:
        for index,ratio in enumerate(ratios):
            if index<len(state['cycles']):continue
            if state['stale_cycles']>=max_stale_cycles:break
            proposal=replace(cfg,output=str(root/f'cycle_{index+1:02d}'),pseudo_per_manual=int(ratio),warm_start=state['best_checkpoint'])
            result=train(proposal)
            report=json.loads((result/'best_metrics.json').read_text())
            keep=accepted(report,state['best_metrics'],min_gain,source_tolerance)
            state['cycles'].append(dict(cycle=index+1,manual_to_pseudo=f'1:{ratio}',run=str(result),accepted=keep,
                validation_dice=report['valid_macro_wave_dice'],best_checkpoint_sha256=file_hash(result/'best.pt')))
            if keep:state.update(best_checkpoint=str(result/'best.pt'),best_metrics=report,stale_cycles=0)
            else:state['stale_cycles']+=1
            atomic_json(journal,state)
        state['status']='plateau' if state['stale_cycles']>=max_stale_cycles else 'planned_cycles_complete'
    except BaseException:
        state['status']='interrupted';raise
    finally:atomic_json(journal,state)
    return root
