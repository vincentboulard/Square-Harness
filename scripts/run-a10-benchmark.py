#!/usr/bin/env python3
"""Run the matched-budget A10 Q4 pilot with one active proof branch using verified serving provenance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from mathagent import benchmark
from mathagent.agent import AgentError
from mathagent.backends import create_client


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def prepare(args):
    launch, launch_hash = read_json(args.launch_record)
    required = {'schema_version', 'weights_sha256', 'model_alias', 'model_path', 'llama_commit',
                'image_id', 'context_per_slot', 'parallel', 'host', 'quantization',
                'cache_type_k', 'cache_type_v', 'context_shift', 'template_sha256'}
    if not isinstance(launch, dict) or not required <= launch.keys() or launch['schema_version'] != 1:
        raise ValueError('Expected the launch record created by scripts/serve-llama-a10.sh')
    if (launch['context_per_slot'] != 32768 or launch['parallel'] != 1
            or str(launch['quantization']).lower() != 'q4_k_m'
            or launch['cache_type_k'] != 'f16' or launch['cache_type_v'] != 'f16'
            or launch['context_shift'] is not False):
        raise ValueError('This pilot requires Q4_K_M, one 32768-token slot, F16 KV, and disabled context shift')
    lock_path = Path(__file__).resolve().parents[1] / 'deployment/llama-a10.lock.json'
    lock, _ = read_json(lock_path)
    for key in ('weights_sha256', 'template_sha256', 'llama_commit', 'model_alias'):
        if launch[key] != lock[key]:
            raise ValueError(f'A10 launch differs from the pinned experimental profile: {key}')
    for key, length in (('weights_sha256', 64), ('template_sha256', 64), ('llama_commit', 40)):
        value = launch[key]
        if not isinstance(value, str) or len(value) != length or any(c not in '0123456789abcdef' for c in value):
            raise ValueError(f'Invalid immutable serving identifier: {key}')
    if not isinstance(launch['image_id'], str) or not launch['image_id'].startswith('sha256:'):
        raise ValueError('The launch record must identify the actual Docker image')
    acceptance, acceptance_hash = None, None
    if args.acceptance is not None:
        acceptance, acceptance_hash = read_json(args.acceptance)
        if (not isinstance(acceptance, dict) or acceptance.get('ok') is not True
                or acceptance.get('accepted_for_benchmark') is not True
                or acceptance.get('backend') != 'llamacpp'
                or acceptance.get('host') != launch['host']
                or acceptance.get('model') != launch['model_alias']
                or acceptance.get('launch_record', {}).get('sha256') != launch_hash):
            raise ValueError('Acceptance must pass and match this exact launch record, endpoint and model')
        settings = acceptance.get('settings', {})
        if settings.get('context_per_slot') != 32768 or settings.get('parallel') != 1:
            raise ValueError('Acceptance does not verify the one-slot 32768-token configuration')
    elif not args.dry_run:
        raise ValueError('Run scripts/smoke-serving.py first and provide its passing --acceptance report')
    cli = ['--manifest', str(args.manifest), '--output', str(args.output),
           '--backend', 'llamacpp', '--host', launch['host'], '--model', launch['model_alias'],
           '--model-revision', 'sha256:' + launch['weights_sha256'], '--server-image', launch['image_id'],
           '--dtype', 'q4_k_m', '--ctx', '32768', '--tokens', '60000',
           '--predict', '8192', '--max-predict', '8192', '--branches', '3',
           '--selection-tokens', '6144', '--replicates', str(args.replicates),
           '--branch-concurrency', '1', '--workers', '1', '--max-in-flight', '1', '--seed', str(args.seed),
           '--temperature', '0.6', '--top-p', '0.95',
           '--seconds', str(args.seconds), '--request-timeout', str(args.request_timeout),
           '--selection-seconds', str(args.selection_seconds)]
    parsed = benchmark.parser().parse_args(cli)
    data, plan = benchmark.preflight(parsed)
    plan['serving'] = {'launch_record_sha256': launch_hash, 'launch_record': launch,
                       'acceptance_sha256': acceptance_hash, 'acceptance': acceptance,
                       'profile': 'a10-q4-serial-branches',
                       'experimental_condition': 'Qwen3.8-27B Q4_K_M on one A10; three logical attempts executed one at a time. Distinct from Q8/BF16 and not a concurrent-inference speed test.'}
    return parsed, data, plan


def verify_current_server(launch):
    """Reject a replaced/reconfigured server before sending scored statements."""
    client = create_client('llamacpp', launch['host'], timeout=30)
    # Public native endpoint, no model input or generation occurs here.
    with client.opener.open(launch['host'].rstrip('/') + '/props', timeout=30) as response:
        props = json.load(response)
    settings = props.get('default_generation_settings', {})
    template = props.get('chat_template')
    build = str(props.get('build_info', ''))
    if (props.get('model_path') != launch['model_path']
            or props.get('total_slots') != launch['parallel']
            or settings.get('n_ctx') != launch['context_per_slot']
            or not isinstance(template, str)
            or hashlib.sha256(template.encode()).hexdigest() != launch['template_sha256']
            or launch['llama_commit'][:7] not in build):
        raise ValueError('Current server differs from the accepted launch; rerun serving validation')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch-record', type=Path, default=Path.home() / 'square-runs/a10-server/launch.json')
    parser.add_argument('--acceptance', type=Path)
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--seed', type=int, default=20260926)
    parser.add_argument('--seconds', type=float, default=43200, help='Wall-time ceiling per job; calibrate on synthetic problems')
    parser.add_argument('--request-timeout', type=float, default=7200)
    parser.add_argument('--selection-seconds', type=float, default=3600)
    parser.add_argument('--dry-run', action='store_true', help='Validate files and print the plan without network or writes')
    args = parser.parse_args(argv)
    try:
        parsed, data, plan = prepare(args)
        if args.dry_run:
            print(json.dumps(plan, ensure_ascii=False, indent=2))
            return 0
        verify_current_server(plan['serving']['launch_record'])
        records = benchmark.execute(parsed, data, plan)
        return 1 if any(record['status'] == 'error' for record in records) else 0
    except (ValueError, OSError, AgentError) as exc:
        print(f'A10 benchmark: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
