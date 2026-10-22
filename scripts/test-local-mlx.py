#!/usr/bin/env python3
"""Live checks against an explicitly loaded local model, without server changes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import secrets
import time

from mathagent.api_profiles import MLX


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Exact ID of an already loaded model')
    p.add_argument('--host', default='http://localhost:8000')
    p.add_argument('--report', type=Path, required=True, help='Where to save the local test report')
    args = p.parse_args()
    report = {'model': args.model, 'host': args.host, 'profile': 'mlx', 'checks': {},
              'schema_enforcement': False, 'scope': 'Short protocol checks, not mathematical quality or full-context performance.'}

    def client():
        return MLX(args.host, timeout=120)

    def payload(text, think=False, cap=1024, **fields):
        return {'model': args.model, 'messages': [{'role': 'user', 'content': text}],
                'think': think, 'options': {'num_ctx': 32768, 'num_predict': cap, 'temperature': 0.6}, **fields}

    def collect(body):
        out = {'text': '', 'thinking': '', 'calls': []}
        start = time.monotonic()
        for event in client().stream(body):
            message = event.get('message', {})
            out['text'] += message.get('content', '')
            out['thinking'] += message.get('thinking', '')
            out['calls'].extend(message.get('tool_calls') or [])
            if event.get('done'):
                out.update(output_tokens=event['eval_count'], input_tokens=event['prompt_eval_count'],
                           finish=event['done_reason'])
        assert 'finish' in out, 'Missing finish/usage'
        out['seconds'] = time.monotonic() - start
        return out

    def check(name, action):
        try:
            report['checks'][name] = {'passed': True, 'result': action()}
        except Exception as error:
            report['checks'][name] = {'passed': False, 'error': str(error)}
        print(name, 'PASS' if report['checks'][name]['passed'] else 'FAIL', flush=True)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + '\n')

    def discovery():
        c = client()
        assert args.model in c.models()
        count = c.count_input_tokens(payload('What is 19 plus 23?'))
        assert count > 0
        return {'exact_prompt_tokens': count, 'advertised_context': c.observed_context_tokens}

    def plain():
        out = collect(payload('What is 19 plus 23? Reply with only the integer.'))
        assert out['text'].strip() == '42', out
        # Some checkpoints' templates always reason even with thinking disabled.
        out['thinking_off_obeyed'] = not bool(out['thinking'].strip())
        return out

    def thinking():
        out = collect(payload('Compute 37 times 41. Include the integer in the final answer.', think=True, cap=4096))
        assert out['thinking'].strip() and '1517' in out['text'], out
        return out

    def tools():
        schema = {'type': 'function', 'function': {'name': 'read_smoke_nonce',
                  'description': 'Read the fresh verification code needed for this answer.',
                  'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
        body = payload('Call read_smoke_nonce now. You need the fresh code to answer. Then reply with that code.', tools=[schema])
        out = collect(body)
        assert len(out['calls']) == 1, out
        assert out['calls'][0]['function'] == {'name': 'read_smoke_nonce', 'arguments': {}}, out
        nonce = secrets.token_hex(12)
        body['messages'].extend([
            {'role': 'assistant', 'content': out['text'], 'thinking': out['thinking'], 'tool_calls': out['calls']},
            {'role': 'tool', 'tool_name': 'read_smoke_nonce', 'tool_call_id': out['calls'][0]['id'], 'content': nonce}])
        body.pop('tools')
        followup = collect(body)
        assert nonce in followup['text'], followup
        return {'call': out, 'followup': followup}

    def structured():
        schema = {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['ok']},
                  'count': {'type': 'integer', 'enum': [2]}}, 'required': ['status', 'count'], 'additionalProperties': False}
        out = collect(payload('Return status ok and count 2 as JSON.', format=schema))
        assert json.loads(out['text']) == {'status': 'ok', 'count': 2}, out
        return out

    def cap():
        out = collect(payload('Write integers from 1 to 100, separated by commas.', cap=1))
        assert out['finish'] == 'length' and out['output_tokens'] == 1, out
        return out

    def parallel():
        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            answers = list(pool.map(lambda _: collect(payload('Write a short paragraph explaining why reproducibility matters.')), range(2)))
        assert all(a['text'].strip() for a in answers)
        return {'requests': 2, 'seconds': time.monotonic() - start, 'results': answers}

    for name, action in [('model_and_tokenizer', discovery), ('plain_stream', plain), ('reasoning_stream', thinking),
                         ('tool_roundtrip', tools), ('prompted_json', structured), ('output_cap', cap), ('parallel_requests', parallel)]:
        check(name, action)
    report['passed'] = all(c['passed'] for c in report['checks'].values())
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
