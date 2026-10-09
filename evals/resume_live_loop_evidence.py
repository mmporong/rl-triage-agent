"""사전등록된 인프라 중단 사례를 기존 원장에서 이어 간다. 누락은 보존한다."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'evals'))
import live_loop_evidence as live

A, loop = live.A, live.loop


def resume_case(prereg_path, tag, case):
    prereg = live.read(prereg_path)
    if (case, tag) != (prereg['amendment']['case'], prereg['amendment']['tag']):
        raise ValueError('사전등록한 중단 사례·tag만 이어갈 수 있다')
    live.verify_prereg(prereg_path, prereg)
    spec = next(c for c in prereg['cases'] if c['case'] == case)
    folder = ROOT / 'evals/results' / tag / 'cases' / case
    if any((folder/name).exists() for name in ('events.json','audit.json','result.json')):
        raise ValueError('기존 감사 결과는 덮어쓰지 않는다')
    snapshot=prereg['amendment']['prior_artifacts']['state_snapshot']
    if live.sha(loop._dir(case)/'state.json') != snapshot['sha256_lf']:
        raise ValueError('중단 시점 이후 state가 달라졌다')
    validation=SimpleNamespace(loop_dir=loop._dir(case),contract=folder/'contract.json',reference=ROOT/spec['sources']['reference']['path'],before=ROOT/spec['sources']['before']['path'],after=None,abstain=True,events=None,timeline=None)
    A.finalize(validation)
    live.gpu_snapshot()
    sources = {k: ROOT / v['path'] for k, v in spec['sources'].items()}
    prior = live.read(sources['prior_probes'])
    ref = {p: r['measurement'] for p,r in live.read(sources['probe_reference'])['probes'].items()}
    events, runs, agreements = [], [], []
    _, state, ledger = loop._load(case)
    for rid, req in ledger.requests().items():
        if req['state'] == 'rejected':
            events.append({'attempt_id':f'{case}:{rid}','status':'rejected',
                           'costs':{'cpu_s':0,'human_review_s':0,'simulator_steps':0}})
        if req['state'] != 'done':
            continue
        probe = req['probe']
        meta = live.read(loop._execution_path(case,probe,tag))
        report = live.read(loop._probe_path(case,probe,tag))
        agreement = live.measurement_agreement(probe,report['probes'][probe]['measurement'],
                                               prior['probes'][probe]['measurement'],ref.get(probe),prereg)
        agreement.update(receipt_outcome=req['receipt']['outcome'],receipt_agreement=req['receipt']['outcome']==agreement['prior_outcome'])
        agreements.append(agreement)
        events.append({'attempt_id':f'{case}:{rid}','status':'failed',
                       'costs':{'cpu_s':None,'human_review_s':0,'simulator_steps':None}})
        runs.append({'request_id':rid,'probe':probe,'worker_exit':1,'process_to_receipt_s':None,
                     'execution':loop._execution_path(case,probe,tag).relative_to(ROOT).as_posix(),
                     'interrupted_before_receipt':False,'infrastructure_error':'missing metrics after Isaac shutdown; cp949 console encoding'})
    events.append({'attempt_id':f'{case}:initial_controller_interrupted','stage':'control','status':'failed',
                   'costs':live.zero_cost(cpu_s=None,execution_wall_s=None),
                   'note':'Initial coordinator CPU/timeline and worker overhead interval were not durably stored; not reconstructed.'})
    started, wall0 = live.now(), time.perf_counter()
    new_gpu_wall = 0
    while (state := loop._load(case)[1])['pending'] is not None:
        rid = state['pending']
        _,_,ledger = loop._load(case)
        probe = ledger.requests()[rid]['probe']
        live.measured_action(events,case,'control',lambda:ledger.approve(rid,live.ACTOR))
        begin = live.now()
        cmd = [sys.executable,str(ROOT/'evals/live_loop_evidence.py'),'execute','--case',case,'--request-id',rid,'--tag',tag]
        proc = subprocess.run(cmd,cwd=ROOT)
        if proc.returncode:
            raise RuntimeError(f'preserved partial output; worker exit {proc.returncode}')
        meta = live.read(loop._execution_path(case,probe,tag))
        worker = live.read(ROOT/f'evals/results/{tag}/workers/{case}__{probe}.json')
        cpu = None if meta['cpu_s'] is None else meta['cpu_s']+worker['cpu_s']-meta['parent_cpu_s']
        new_gpu_wall += meta['gpu_wall_s']
        events.append({'attempt_id':f'{case}:{rid}','status':'completed' if meta['exit_code']==0 else 'failed',
                       'costs':{'cpu_s':cpu,'human_review_s':0,'simulator_steps':meta['simulator_steps']}})
        report = live.read(loop._probe_path(case,probe,tag))
        if report['checkpoint']['sha256'] != spec['checkpoint_sha256'] or report['seed'] != spec['seed']:
            raise ValueError('checkpoint/seed mismatch')
        agreement = live.measurement_agreement(probe,report['probes'][probe]['measurement'] or {},
                                               prior['probes'][probe]['measurement'],ref.get(probe),prereg)
        receipt = ledger.requests()[rid]['receipt']
        agreement.update(receipt_outcome=receipt['outcome'],receipt_agreement=receipt['outcome']==agreement['prior_outcome'])
        agreements.append(agreement)
        runs.append({'request_id':rid,'probe':probe,'worker_exit':proc.returncode,'process_started_at':begin,
                     'receipt_observed_at':live.now(),'process_to_receipt_s':(live.A._timestamp(live.now())-live.A._timestamp(begin)).total_seconds(),
                     'execution':loop._execution_path(case,probe,tag).relative_to(ROOT).as_posix(),'interrupted_before_receipt':False})
    action_cpu = sum(e['costs']['cpu_s'] for e in events if e.get('stage')=='control' and e['costs']['cpu_s'] is not None)
    action_wall = sum(e['costs']['execution_wall_s'] for e in events if e.get('stage')=='control' and e['costs']['execution_wall_s'] is not None)
    events.append({'attempt_id':f'{case}:resume_controller_overhead','stage':'control','status':'completed',
                   'started_at':started,'finished_at':live.now(),
                   'costs':live.zero_cost(cpu_s=max(0,time.process_time()-action_cpu),execution_wall_s=max(0,time.perf_counter()-wall0-new_gpu_wall-action_wall))})
    A._write_new(folder/'events.json',events)
    args=SimpleNamespace(loop_dir=loop._dir(case),contract=folder/'contract.json',reference=sources['reference'],
                         before=sources['before'],after=None,abstain=True,events=folder/'events.json',timeline=None)
    audit=A.finalize(args)
    A._write_new(folder/'audit.json',audit)
    result={'schema':'live_loop_case_v1','case':case,'prereg_sha256_lf':live.sha(prereg_path),
            'first_prereg_sha256_lf':live.sha(ROOT/'bench/protocols/live_loop_v1.json'),'actor':live.ACTOR,'scope':prereg['diagnosis_scope'],
            'runs':runs,'recovery':None,'agreements':agreements,'raw_classification_agreement':all(a['live_outcome']==a['prior_outcome'] for a in agreements),
            'classification_agreement':all(a['receipt_agreement'] for a in agreements),'scalar_agreement':all(a['scalar_agreement'] for a in agreements),
            'audit_finalize_error':None,'loop_status':audit['loop']['diagnosis_status'],'observed':audit['loop']['observed'],
            'costs':audit['costs']['totals'],'end_to_end_wall':audit['end_to_end_wall'],
            'checkpoint_sha256_verified_against_prereg':True,'repair_executed':False,'none_wrong_confirmation':False,
            'infrastructure_errors':['initial metric sidecar missing after Isaac shutdown','initial cp949 console encoding failure'],
            'missing_initial_costs_preserved':True,'resume_controller_sha256_lf':live.sha(Path(__file__)), 'initial_failure_prefix_verified':True}
    A._write_new(folder/'result.json',result)
    print(json.dumps({'case':case,'status':result['loop_status'],'costs':result['costs'],'classification_agreement':result['classification_agreement']}))

    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prereg',type=Path,required=True)
    parser.add_argument('--tag',required=True)
    parser.add_argument('--case',required=True)
    args=parser.parse_args(argv)
    resume_case(args.prereg,args.tag,args.case)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
