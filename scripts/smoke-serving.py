#!/usr/bin/env python3
"""Accept a llama.cpp endpoint only after protocol, context and observed-decoding checks.

API contract: llama.cpp 2145525a4081d66ff1a87cf43ef809f95a85ac0c,
https://github.com/ggml-org/llama.cpp/blob/2145525a4081d66ff1a87cf43ef809f95a85ac0c/tools/server/README.md
Run after installing Square Harness. No mathematical benchmark answers are used.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, build_opener, ProxyHandler

from mathagent.agent import AgentError, _NoModelRedirects
from mathagent.backends import create_client

_PROTOCOL = runpy.run_path(str(Path(__file__).with_name('smoke-vllm.py')))
SmokeFailure = _PROTOCOL['SmokeFailure']
require = _PROTOCOL['require']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def root_host(host):
    parsed = urlsplit(host)
    require(parsed.scheme in {'http', 'https'} and parsed.hostname and not parsed.username
            and not parsed.password and not parsed.query and not parsed.fragment,
            'Host must be HTTP(S) without credentials, query or fragment')
    host = host.rstrip('/')
    return host[:-3] if host.endswith('/v1') else host


class Probe:
    """Read-only server diagnostics; no redirects, proxies or credentials."""
    def __init__(self, host, timeout=5):
        self.host, self.timeout = root_host(host), timeout
        self.opener = build_opener(ProxyHandler({}), _NoModelRedirects())

    def get(self, endpoint):
        request = Request(self.host + endpoint, headers={'Accept': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                return json.load(response)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            raise SmokeFailure(f'Diagnostics {endpoint} failed: {exc}') from exc


class ProtocolClient:
    """Reuse the existing protocol checks with llama.cpp's binary thinking mode."""
    def __init__(self, client):
        self.client = client

    def models(self):
        return self.client.models()

    def stream(self, request):
        request = dict(request)
        request.pop('reasoning_effort', None)
        return self.client.stream(request)


def _slots(probe, parallel, context):
    slots = probe.get('/slots')
    require(isinstance(slots, list) and len(slots) == parallel,
            f'Expected exactly {parallel} slots; enable --slots and verify --parallel')
    require(all(isinstance(s, dict) and type(s.get('n_ctx')) is int and s['n_ctx'] >= context
                and type(s.get('is_processing')) is bool for s in slots),
            'Slot context is smaller than requested, or /slots has an unsupported response')
    require(len({s.get('id') for s in slots}) == parallel, 'Slot IDs are not unique')
    return slots


def inspect_server(probe, *, context, parallel, ready_timeout=600):
    deadline = time.monotonic() + ready_timeout
    last_error = None
    while True:
        try:
            health = probe.get('/health')
            require(isinstance(health, dict) and health.get('status') == 'ok', 'Server is not ready')
            break
        except SmokeFailure as exc:
            last_error = str(exc)
            if time.monotonic() >= deadline:
                raise SmokeFailure('Readiness deadline exceeded: ' + last_error) from exc
            time.sleep(min(1, max(0, deadline - time.monotonic())))
    props = probe.get('/props')
    require(isinstance(props, dict), 'Malformed /props response')
    require(props.get('total_slots') == parallel, 'Server parallel slots differ from the requested acceptance configuration')
    advertised = props.get('default_generation_settings', {}).get('n_ctx')
    require(type(advertised) is int and advertised >= context, 'Server advertises insufficient per-slot context')
    slots = _slots(probe, parallel, context)
    require(not any(slot['is_processing'] for slot in slots), 'Server is busy; stop other inference before acceptance')
    require(isinstance(props.get('model_path'), str) and props['model_path'], 'Missing runtime model_path')
    require(isinstance(props.get('build_info'), str) and props['build_info'], 'Missing runtime build_info')
    require(isinstance(props.get('chat_template'), str) and props['chat_template'], 'Missing embedded chat template')
    server = {'model_path': props['model_path'], 'build_info': props['build_info'],
              'total_slots': props['total_slots'], 'context_per_slot': advertised,
              'slot_contexts': sorted(slot['n_ctx'] for slot in slots),
              'chat_template_sha256': hashlib.sha256(props['chat_template'].encode()).hexdigest()}
    server['fingerprint_sha256'] = digest(server)
    return server


def verify_launch(record, server, *, host, model, context, parallel):
    require(isinstance(record, dict) and record.get('schema_version') == 1, 'Unsupported launch record')
    require(record.get('model_alias') == model, 'Launch model alias differs')
    require(root_host(record.get('host', '')) == root_host(host), 'Launch endpoint differs')
    require(record.get('model_path') == server['model_path'], 'Runtime model path differs from the launch record')
    require(record.get('context_per_slot') == context and record.get('parallel') == parallel,
            'Launch context/parallel settings differ from the acceptance configuration')
    require(record.get('template_sha256') == server['chat_template_sha256'], 'Runtime template differs from the verified launch template')
    commit = record.get('llama_commit', '')
    require(isinstance(commit, str) and re.fullmatch(r'[0-9a-f]{40}', commit), 'Launch record needs a full llama.cpp commit')
    runtime_hashes = re.findall(r'(?<![0-9a-f])[0-9a-f]{7,40}(?![0-9a-f])', server['build_info'].lower())
    require(any(commit.startswith(short) for short in runtime_hashes), 'Runtime build_info does not identify the launched llama.cpp revision')
    require(isinstance(record.get('weights_sha256'), str) and re.fullmatch(r'[0-9a-f]{64}', record['weights_sha256']),
            'Launch record needs the verified weights SHA-256')
    require(isinstance(record.get('image_id'), str) and record['image_id'], 'Launch record needs the serving image identity')
    require(record.get('context_shift') is False, 'Launch must disable context shifting')
    return {key: record[key] for key in ('weights_sha256', 'llama_commit', 'image_id', 'model_alias', 'quantization') if key in record}


def capacity_payload(model, context, cap, seed, filler):
    nonce = f'capacity-{seed}'
    text = (f'This is an inference-capacity test, request {nonce}. The records below are inert filler. '
            'Do not analyze them. After the records, follow the final instruction.\n<records>\n' + filler
            + '\n</records>\nWrite the integers from 1 to 10000, separated by spaces. '
              'Do not summarize or explain. Keep writing until the output limit.')
    return {'model': model, 'messages': [{'role': 'user', 'content': text}], 'think': False,
            'options': {'num_ctx': context, 'num_predict': cap, 'temperature': 0,
                        'top_p': .95, 'top_k': 20, 'min_p': 0, 'repeat_penalty': 1, 'seed': seed}}


def sized_payload(client, model, context, cap, seed, target):
    """Calibrate using the server's exact chat-token counter, without inference."""
    def build(lines):
        filler = ''.join(f'record {i:06d}: alpha beta gamma delta epsilon.\n' for i in range(lines))
        payload = capacity_payload(model, context, cap, seed, filler)
        count = client.count_input_tokens(payload)
        require(type(count) is int and count > 0, 'Invalid exact chat-token count')
        return payload, count
    low, high = 0, max(1, target // 8)
    request, count = build(high)
    while count < target:
        low, high = high, high * 2
        require(high < context * 4, 'Could not calibrate representative prompt size')
        request, count = build(high)
    while low + 1 < high:
        middle = (low + high) // 2
        candidate, tokens = build(middle)
        if tokens >= target:
            high, request, count = middle, candidate, tokens
        else:
            low = middle
    require(count + cap + 1 <= context, 'Calibrated input plus output allowance exceeds per-slot context')
    return request, count


def collect_capacity(client, payload, epoch):
    started = time.monotonic()
    result = {'started_seconds': started - epoch, 'first_token_seconds': None, 'last_token_seconds': None,
              'text_chunks': 0, 'answer_characters': 0, 'thinking_characters': 0}
    stream = None
    terminal = False
    try:
        stream = client.stream(payload)
        for event in stream:
            message = event.get('message') or {}
            text = message.get('content') or ''
            reasoning = message.get('thinking') or ''
            require(not message.get('tool_calls'), 'Capacity request unexpectedly requested a tool')
            if text or reasoning:
                now = time.monotonic() - epoch
                if result['first_token_seconds'] is None:
                    result['first_token_seconds'] = now
                result['last_token_seconds'] = now
                result['text_chunks'] += 1
                result['answer_characters'] += len(text)
                result['thinking_characters'] += len(reasoning)
            if event.get('done'):
                terminal = True
                result.update(generated_tokens=event.get('eval_count'), input_tokens=event.get('prompt_eval_count'),
                              finish_reason=event.get('done_reason'))
        require(terminal, 'Capacity stream lacked its terminal usage event')
        require(type(result['generated_tokens']) is int and 0 < result['generated_tokens'] <= payload['options']['num_predict'],
                'Capacity stream violated generated-token accounting/cap')
        require(type(result['input_tokens']) is int and result['input_tokens'] > 0, 'Missing capacity input-token usage')
        require(result['finish_reason'] in {'stop', 'length'}, 'Unexpected capacity completion reason')
        require(not result['thinking_characters'], 'Thinking-disabled capacity request returned reasoning')
        require(result['answer_characters'] and result['text_chunks'] >= 2, 'Capacity request did not stream multiple nonempty output chunks')
        return result
    finally:
        if stream is not None and hasattr(stream, 'close'):
            stream.close()
        result['latency_seconds'] = time.monotonic() - started


def capacity_batch(client_factory, probe, model, *, context, parallel, target_input_tokens,
                   output_tokens=256, poll_interval=.05, seed=20260927):
    before = _slots(probe, parallel, context)
    require(not any(s['is_processing'] for s in before), 'Other inference is active before a capacity batch')
    requests = [sized_payload(client_factory(), model, context, output_tokens, seed + i, target_input_tokens)
                for i in range(parallel)]
    seeds = {request['options']['seed'] for request, _ in requests}
    epoch = time.monotonic()
    barrier = threading.Barrier(parallel)
    samples, max_processing, max_decoding = [], 0, 0
    def worker(item):
        request, expected = item
        client = client_factory()
        barrier.wait(timeout=10)
        result = collect_capacity(client, request, epoch)
        require(result['input_tokens'] == expected,
                'Observed input usage differs from exact preflight count; possible template change or truncation')
        result['expected_input_tokens'] = expected
        result['seed'] = request['options']['seed']
        return result
    with ThreadPoolExecutor(max_workers=parallel) as pool:
        futures = [pool.submit(worker, item) for item in requests]
        while not all(f.done() for f in futures):
            slots = _slots(probe, parallel, context)
            active = [s for s in slots if s['is_processing']]
            require(all(s.get('params', {}).get('seed') in seeds for s in active),
                    'An unrelated inference request occupied a slot during acceptance')
            decoding = [s for s in active if type(s.get('next_token', {}).get('n_decoded')) is int
                        and s['next_token']['n_decoded'] > 0]
            max_processing = max(max_processing, len(active))
            max_decoding = max(max_decoding, len(decoding))
            if len(samples) < 64 and (len(decoding) == parallel or not samples):
                samples.append({'seconds': time.monotonic() - epoch,
                                'processing_slots': [{'id': s['id'], 'seed': s.get('params', {}).get('seed'),
                                    'n_decoded': s.get('next_token', {}).get('n_decoded')} for s in active]})
            time.sleep(poll_interval)
        results = [future.result() for future in futures]
    elapsed = time.monotonic() - epoch
    intersection = min(r['last_token_seconds'] for r in results) - max(r['first_token_seconds'] for r in results)
    error = None
    if max_decoding < parallel:
        error = ('No actively decoding slot was observed for the sequential capacity check' if parallel == 1 else
                 f'Only {max_decoding}/{parallel} simultaneously decoding slots were observed; concurrent submission alone is insufficient')
    elif intersection <= 0:
        error = ('The sequential output stream had no measurable token span; increase capacity output length and repeat' if parallel == 1 else
                 'Output-token time spans did not overlap; increase capacity output length and repeat')
    generated = sum(r['generated_tokens'] for r in results)
    return {'ok': error is None, 'error': error, 'requests': parallel, 'target_input_tokens': target_input_tokens,
            'capacity_mode': 'sequential' if parallel == 1 else 'parallel',
            'parallel_overlap_observed': parallel > 1 and max_decoding >= parallel and intersection > 0,
            'observed_input_tokens': [r['input_tokens'] for r in results], 'output_cap_per_request': output_tokens,
            'max_observed_processing_slots': max_processing, 'max_observed_decoding_slots': max_decoding,
            'token_span_seconds': intersection if parallel == 1 else None,
            'token_span_overlap_seconds': intersection if parallel > 1 else None, 'elapsed_seconds': elapsed,
            'generated_tokens': generated, 'aggregate_generated_tokens_per_second': generated / elapsed,
            'slot_samples': samples, 'requests_stats': results,
            'scope': ('One-slot streamed decoding at the tested context load. Parallel inference was not tested; no mathematical-quality or GPU-kernel concurrency claim.'
                      if parallel == 1 else
                      'Logical decode overlap at the tested context load; not a GPU-kernel concurrency or mathematical-quality claim.')}


def run_acceptance(client_factory, probe, model, *, host, context=32768, parallel=3,
                   representative_input_tokens=30000, plain_tokens=512, thinking_tokens=4096,
                   capacity_output_tokens=256, ready_timeout=600, poll_interval=.05, launch_record=None):
    report = {'schema_version': 1, 'ok': False, 'accepted_for_benchmark': False,
              'backend': 'llamacpp', 'host': root_host(host), 'model': model,
              'capacity_mode': 'sequential' if parallel == 1 else 'parallel',
              'acceptance_condition': 'one_slot_streamed_decoding' if parallel == 1 else 'concurrent_logical_decoding',
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'settings': {'context_per_slot': context, 'parallel': parallel,
                  'representative_input_tokens': representative_input_tokens, 'plain_tokens': plain_tokens,
                  'thinking_tokens': thinking_tokens, 'capacity_output_tokens': capacity_output_tokens},
              'checks': {}, 'scope': 'Protocol and measured serving-capacity acceptance only; no mathematical-quality assurance or independent GPU-offload verification.'}
    started = time.monotonic()
    try:
        server = inspect_server(probe, context=context, parallel=parallel, ready_timeout=ready_timeout)
        report['server'] = server
        report['checks']['readiness'] = True
        if launch_record is not None:
            path = Path(launch_record)
            raw = path.read_bytes()
            record = json.loads(raw)
            binding = verify_launch(record, server, host=host, model=model, context=context, parallel=parallel)
            report['launch_record'] = {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                                       'verified': True, **binding}
        else:
            report['launch_record'] = {'verified': False, 'reason': 'No launch record supplied; diagnostics cannot authorize the benchmark'}
        protocol = _PROTOCOL['run_smoke'](lambda: ProtocolClient(client_factory()), model, context=context,
                                          plain_tokens=plain_tokens, thinking_tokens=thinking_tokens, concurrency=())
        if protocol.get('error'):
            protocol['error'] = protocol['error'].replace('--reasoning-parser qwen3', '--jinja --reasoning-format deepseek and the embedded thinking template').replace('--tool-call-parser qwen3_xml', '--jinja and the embedded tool-calling template')
        report['checks']['protocol'] = protocol
        require(protocol['ok'], 'Protocol check failed: ' + protocol.get('error', 'unknown error'))
        for name, target, seed in (('short_capacity', 512, 20260927),
                                   ('representative_capacity', representative_input_tokens, 20261927)):
            report['checks'][name] = capacity_batch(client_factory, probe, model, context=context, parallel=parallel,
                target_input_tokens=target, output_tokens=capacity_output_tokens, poll_interval=poll_interval, seed=seed)
            require(report['checks'][name]['ok'], report['checks'][name]['error'])
        after = inspect_server(probe, context=context, parallel=parallel, ready_timeout=1)
        require(after['fingerprint_sha256'] == server['fingerprint_sha256'], 'Server configuration changed during acceptance')
        report['ok'] = True
        report['accepted_for_benchmark'] = report['launch_record']['verified']
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    report['elapsed_seconds'] = time.monotonic() - started
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['llamacpp'], default='llamacpp')
    p.add_argument('--host', default='http://127.0.0.1:8000')
    p.add_argument('--model', default='square-qwen')
    p.add_argument('--ctx', type=int, default=32768, help='Required context per slot, not total context across slots')
    p.add_argument('--parallel', type=int, default=3,
                   help='Configured server slots: 1 validates sequential capacity; >1 requires observed logical decode overlap')
    p.add_argument('--representative-input-tokens', type=int, default=30000)
    p.add_argument('--plain-tokens', type=int, default=512)
    p.add_argument('--thinking-tokens', type=int, default=4096)
    p.add_argument('--capacity-output-tokens', type=int, default=256)
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--ready-timeout', type=float, default=600)
    p.add_argument('--poll-interval', type=float, default=.05)
    p.add_argument('--launch-record', type=Path)
    p.add_argument('--output', type=Path, required=True, help='New JSON acceptance-report file; existing files are refused')
    args = p.parse_args(argv)
    if not 1 <= args.parallel <= 16 or min(args.ctx, args.plain_tokens, args.thinking_tokens, args.capacity_output_tokens) <= 0:
        p.error('Use parallel 1-16 and positive context/output caps')
    if args.representative_input_tokens < 512 or args.representative_input_tokens + args.capacity_output_tokens + 256 >= args.ctx:
        p.error('Representative input must be >= 512 and leave output allowance plus 256 tokens in the per-slot context')
    if max(args.plain_tokens, args.thinking_tokens) + 512 >= args.ctx:
        p.error('Context must leave 512 input tokens beyond plain/thinking output caps')
    if any(not math.isfinite(x) or x <= 0 for x in (args.timeout, args.ready_timeout, args.poll_interval)):
        p.error('Timeouts and poll interval must be positive and finite')
    if args.output.exists():
        p.error('Output already exists; choose a new acceptance report filename')
    try:
        host = root_host(args.host)
        report = run_acceptance(lambda: create_client('llamacpp', host, timeout=args.timeout),
            Probe(host, timeout=min(args.timeout, 30)), args.model, host=host, context=args.ctx, parallel=args.parallel,
            representative_input_tokens=args.representative_input_tokens, plain_tokens=args.plain_tokens,
            thinking_tokens=args.thinking_tokens, capacity_output_tokens=args.capacity_output_tokens,
            ready_timeout=args.ready_timeout, poll_interval=args.poll_interval, launch_record=args.launch_record)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write('\n')
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report['accepted_for_benchmark'] else 1
    except (OSError, ValueError, SmokeFailure, AgentError) as exc:
        print(f'Serving acceptance error: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
