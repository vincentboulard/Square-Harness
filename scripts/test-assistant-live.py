#!/usr/bin/env python3
"""Small opt-in Assistant smoke tests using the configured local model endpoint.

Every case uses its own temporary offline workspace. This script does not
restart or write to an existing GUI, model server, or user workspace. Pass
--parallel to compare two explicitly requested short critic workers with one
and two available inference slots. These timings are smoke evidence, not a
quality benchmark or a general performance estimate.
"""
import argparse
import json
from pathlib import Path
import sys
import tempfile
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mathagent import cli
from mathagent.gui import store
from mathagent.gui.hub import Hub, InterruptibleOpenAI


def run_case(host, model, query, concurrency=2, seconds=60):
    observations = []
    active = 0
    peak = 0
    lock = threading.Lock()
    original_stream = InterruptibleOpenAI.stream

    def tracked_stream(client, payload):
        nonlocal active, peak
        started = time.monotonic()
        record = {'thinking_enabled': payload.get('think'),
                  'output_cap': payload['options']['num_predict'],
                  'kind': 'main' if any(tool['function']['name'] == 'delegate'
                                        for tool in payload.get('tools', [])) else 'worker'}
        with lock:
            active += 1
            peak = max(peak, active)
            observations.append(record)
        try:
            for event in original_stream(client, payload):
                if event.get('done'):
                    record['output_tokens'] = event.get('eval_count')
                    record['finish_reason'] = event.get('done_reason')
                if event.get('message', {}).get('thinking'):
                    record['thinking_characters'] = record.get('thinking_characters', 0) + len(event['message']['thinking'])
                if event.get('message', {}).get('content'):
                    record['text'] = record.get('text', '') + event['message']['content']
                if event.get('message', {}).get('tool_calls'):
                    record['tools_called'] = [call.get('function', {}).get('name')
                                              for call in event['message']['tool_calls']]
                yield event
        finally:
            record['seconds'] = round(time.monotonic() - started, 3)
            with lock:
                active -= 1

    with tempfile.TemporaryDirectory(prefix='square-assistant-live-') as directory:
        args = cli.parser().parse_args([
            '--workspace', directory, '--backend', 'openai', '--host', host,
            '--model', model, '--ctx', '40960', '--predict', '512', '--seed', '42',
            '--request-timeout', str(seconds), '--offline', '--proof-rounds', '1',
            '--proof-solve-tokens', '2048', '--proof-verify-tokens', '1024',
            '--proof-tokens', '6000', '--proof-seconds', str(seconds),
            '--research-rounds', '1', '--research-tokens', '6000',
            '--research-input-tokens', '60000', '--research-seconds', str(seconds),
        ])
        args.workspace = Path(directory)
        args.assistant_concurrency = concurrency
        args.assistant_tokens = 6000
        args.assistant_seconds = seconds
        hub = Hub(args)
        chat = store.create_chat(Path(directory), 'free', model=model, online=False)
        started = time.monotonic()
        InterruptibleOpenAI.stream = tracked_stream
        try:
            hub.route(chat['id'], query)
            while time.monotonic() - started < seconds + 15:
                task = hub.current()
                if task and task.state in ('done', 'paused', 'error'):
                    break
                time.sleep(0.05)
            else:
                hub.shutdown(5)
                raise RuntimeError('Assistant smoke test did not finish within its time guard')
            saved = store.load_chat(Path(directory), chat['id'])
            state = saved.get('assistant_state') or {}
            answer = next((item.get('content', '') for item in reversed(saved['transcript'])
                           if item.get('role') == 'assistant'), state.get('answer', ''))
            return {'query': query, 'concurrency': concurrency,
                    'seconds': round(time.monotonic() - started, 3),
                    'task_state': task.state, 'task_error': task.error,
                    'status': state.get('status'), 'answer': answer,
                    'children': [{'mode': child['action']['mode'], 'status': child['status'],
                                  'worker_status': (child.get('result') or {}).get('status'),
                                  'answer': (child.get('result') or {}).get('answer')}
                                 for child in state.get('children', [])],
                    'budget': state.get('budget'), 'peak_inference_calls': peak,
                    'calls': observations}
        finally:
            hub.shutdown(5)
            InterruptibleOpenAI.stream = original_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='http://127.0.0.1:18020')
    parser.add_argument('--model', default='qwen3.8-27b')
    parser.add_argument('--seconds', type=int, default=60)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    queries = [
        'Give a short proof that a^2 >= 0 for every real number a.',
        'Is a^2 > 0 for every real number a? Answer briefly.',
        'If a=i, is a^2 >= 0 true? Answer briefly.',
    ]
    cases = [run_case(args.host, args.model, query, seconds=args.seconds) for query in queries]
    if args.parallel:
        query = ('Use exactly two separate critic workers to independently check the claim '
                 'a^2 >= 0 for all real a. Request both delegate calls in the same model turn, '
                 'with low effort. Each worker must answer in at most three sentences. '
                 'After both return, summarize their checks in at most three sentences.')
        cases.extend(run_case(args.host, args.model, query, concurrency=concurrency,
                              seconds=args.seconds) for concurrency in (1, 2))
    report = {'model': args.model, 'cases': cases}
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.write_text(encoded)
    print(encoded)
    complete = all(case['status'] == 'complete' and
                   all(child['worker_status'] == 'complete' for child in case['children'])
                   for case in cases)
    direct = all(not case['children'] and case['peak_inference_calls'] == 1 for case in cases[:3])
    parallel = (not args.parallel or all(
        len(case['children']) == 2 and all(child['mode'] == 'critic' for child in case['children'])
        and case['peak_inference_calls'] == case['concurrency'] for case in cases[3:]))
    return 0 if complete and direct and parallel else 1


if __name__ == '__main__':
    raise SystemExit(main())
