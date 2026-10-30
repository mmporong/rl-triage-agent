"""Offline recorded-evidence checks. No credentials, network, approval or GPU calls."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def require(condition, reason):
    if not condition:
        raise ValueError(reason)


def unique(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, 'duplicate JSON key')
        obj[key] = value
    return obj


def load(name):
    path = (ROOT / name).resolve()
    require(path.is_relative_to(ROOT) and path.is_file(), 'evidence path outside bundle')
    return json.loads(path.read_text(encoding='utf-8-sig'), object_pairs_hook=unique)


def main():
    index = load('PUBLIC_INDEX.json')
    for name, expected in index['files'].items():
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), 'manifest path outside bundle')
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == expected['sha256']
                and len(raw) == expected['bytes'], 'changed evidence bytes: ' + name)
    training = load('evidence/training_source.json')
    valid, holds, proposals = 0, 0, 0
    for record in index['core_records']:
        request, response, metadata = (load(record[key]) for key in ['request', 'response', 'metadata'])
        require(request['model'] == response['model'] == index['core_model'], 'served model mismatch')
        require(metadata['http_status'] == 200 and metadata['attempted_http'] is True
                and metadata['automatic_retries'] == 0 and not metadata['error'], 'HTTP evidence mismatch')
        require(response['choices'][0]['finish_reason'] == 'stop', 'unfinished model response')
        decision = json.loads(response['choices'][0]['message']['content'], object_pairs_hook=unique)
        require(set(decision) == {'action', 'target', 'fact_ids'}, 'decision fields')
        text = request['messages'][1]['content']
        if isinstance(text, list):
            text = text[0]['text']
        view = json.loads(text, object_pairs_hook=unique)
        require(decision['action'] in view['available_actions']
                and decision['target'] in view['available_actions'][decision['action']], 'unavailable action/target')
        facts = decision['fact_ids']
        require(isinstance(facts, list) and facts and all(isinstance(x, str) for x in facts)
                and len(facts) == len(set(facts)) and set(facts) <= set(view['evidence_facts']), 'unknown or duplicate fact citation')
        for key, value in view['evidence_facts'].items():
            if key.startswith('telemetry:'):
                _, role, metric = key.split(':', 2)
                arm = record['training_arms'][role]
                require(value == training['arms'][arm]['telemetry_final_20pct'][metric], 'model fact differs from training source')
        for role, arm in record['training_arms'].items():
            require(record['checkpoint_sha256'][role] == training['arms'][arm]['checkpoint']['sha256'], 'checkpoint source mismatch')
        require(decision == record['decision'], 'recorded decision mismatch')
        require(record['approval_granted'] is False and record['execution_allowed'] is False,
                'unexpected approval or execution claim')
        if decision['action'] == 'run_probe':
            require(record['probe_proposal']['probe'] == decision['target']
                    and record['probe_proposal']['requires_new_authorization'] is True, 'probe proposal binding')
            proposals += 1
        elif decision['action'] == 'hold':
            require(decision['target'] is None, 'hold target must be null')
            holds += 1
        valid += 1
    for record in index['earlier_provider_samples']:
        request, response, metadata = (load(record[key]) for key in ['request', 'response', 'metadata'])
        require(request['model'] == response['model'] == record['model'], 'earlier provider identity mismatch')
        require(metadata['http_status'] == 200 and response['usage'] == metadata['usage'], 'earlier provider receipt mismatch')
        require(response['choices'][0]['finish_reason'] == 'stop', 'earlier completion unfinished')
        decision = json.loads(response['choices'][0]['message']['content'], object_pairs_hook=unique)
        packet = json.loads(request['messages'][-1]['content'], object_pairs_hook=unique)
        require(set(decision) == {'probe', 'fact_ids'} and decision['probe'] in packet['available_probes'], 'earlier probe contract')
        require(decision['fact_ids'] and len(decision['fact_ids']) == len(set(decision['fact_ids']))
                and set(decision['fact_ids']) <= set(packet['allowed_fact_ids']), 'earlier citation contract')
    require(valid == 8 and holds == 4 and proposals == 4, 'recorded aggregate differs')
    print(json.dumps({'status': 'PASS', 'files_verified': len(index['files']), 'core_decisions': valid,
                      'holds': holds, 'approval_required_proposals': proposals,
                      'earlier_provider_samples': len(index['earlier_provider_samples']),
                      'new_http_requests': 0, 'gpu_requests': 0,
                      'scope': 'Recorded evidence bytes, visible actions/citations and published training joins; no live approval/resource or provider authenticity reauthentication'}))


if __name__ == '__main__':
    main()
