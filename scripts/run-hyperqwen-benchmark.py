#!/usr/bin/env python3
"""Direct and proof v0.5 benchmark on the pinned HyperQwen A10 serving profile."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mathagent import benchmark


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def prepare(args):
    lock, lock_hash = read(Path(__file__).resolve().parents[1] / 'deployment/hyperqwen-a10.lock.json')
    launch, launch_hash = read(args.launch_record)
    if launch.get('image') != lock['image'] or any(launch.get('env', {}).get(k) != v
                                                  for k, v in lock['server_env'].items()):
        raise ValueError('Launch does not match the pinned HyperQwen profile')
    os.environ.update(lock['thinking_environment'])
    cli = ['--manifest', str(args.manifest), '--output', str(args.output),
           '--backend', 'openai', '--host', lock['host'], '--model', lock['model'],
           '--model-revision', lock['model_revision'], '--server-image', lock['image'],
           '--dtype', lock['dtype'], '--arms', 'raw-single', 'proof',
           '--ctx', str(lock['context']),
           '--tokens', str(lock['harness_tokens']), '--predict', str(lock['solver_predict']),
           '--verify-tokens', str(lock['verify_tokens']), '--rounds', str(lock['rounds']),
           '--raw-seconds', str(lock['raw_seconds']), '--seconds', str(lock['harness_seconds']),
           '--request-timeout', str(lock['request_timeout']), '--seed', str(lock['seed']),
           '--temperature', '0.6', '--top-p', '0.95', '--workers', '1', '--max-in-flight', '1']
    parsed = benchmark.parser().parse_args(cli)
    data, plan = benchmark.preflight(parsed)
    if len(data['problems']) != 10:
        raise ValueError('This frozen profile requires exactly ten problems')
    acceptance, acceptance_hash = (None, None) if args.acceptance is None else read(args.acceptance)
    if acceptance is not None:
        if (acceptance.get('ok') is not True or acceptance.get('launch_sha256') != launch_hash
                or acceptance.get('profile_lock_sha256') != lock_hash
                or acceptance.get('code_sha256') != plan['code_sha256']
                or acceptance.get('thinking_budget_policy') != plan['openai_thinking_budget_policy']):
            raise ValueError('Acceptance must pass for this launch, source and thinking policy')
    plan['serving'] = {'profile': lock['profile'], 'lock_sha256': lock_hash,
                       'launch_record_sha256': launch_hash, 'launch_record': launch,
                       'acceptance_sha256': acceptance_hash, 'acceptance': acceptance}
    plan['resource_comparison'] = ('Both initial solves allow 32768 output tokens; proof allows 120000 total '
        'including verification and repairs. Total budgets are not matched to direct. This v0.5 configuration has not been GPU-validated by the software tests.')
    plan['nominal_job_time_ceiling_seconds'] = 10 * (lock['raw_seconds'] + lock['harness_seconds'])
    return lock, launch, parsed, data, plan


def verify_live(lock, launch):
    current = json.loads(subprocess.check_output(
        ['sudo', 'docker', 'inspect', launch['container']], text=True))[0]
    actual_env = dict(item.split('=', 1) for item in current['Config']['Env'])
    if (not current['State']['Running'] or current['Id'] != launch['container_id']
            or current['Config']['Image'] != lock['image']
            or any(actual_env.get(k) != v for k, v in lock['server_env'].items())):
        raise ValueError('Serving container changed or stopped after acceptance')
    with urllib.request.urlopen(lock['host'] + '/v1/models', timeout=15) as response:
        models = json.load(response)['data']
    if not any(m['id'] == lock['model'] and m.get('max_model_len') == lock['context'] for m in models):
        raise ValueError('Model/context differs from accepted profile')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--launch-record', type=Path, required=True)
    p.add_argument('--acceptance', type=Path, help='Optional prior readiness record; if provided, its source/launch/policy hashes must match')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args(argv)
    lock, launch, parsed, data, plan = prepare(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    verify_live(lock, launch)
    records = benchmark.execute(parsed, data, plan)
    return 1 if any(r['status'] == 'error' for r in records) else 0


if __name__ == '__main__':
    raise SystemExit(main())
