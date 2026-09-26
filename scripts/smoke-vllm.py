#!/usr/bin/env python3
"""Live protocol smoke checks; run after installing Square Harness with pip -e ."""
import argparse
import json
import secrets
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from mathagent.backends import OpenAICompatible


class SmokeFailure(Exception):
    pass


def require(condition, message):
    if not condition:
        raise SmokeFailure(message)


def collect(client, payload):
    started = time.monotonic()
    result = {'text': '', 'thinking': '', 'calls': []}
    complete = False
    for event in client.stream(payload):
        message = event.get('message', {})
        result['text'] += message.get('content', '')
        result['thinking'] += message.get('thinking', '')
        result['calls'].extend(message.get('tool_calls') or [])
        if event.get('done'):
            complete = True
            result.update(generated_tokens=event.get('eval_count'), input_tokens=event.get('prompt_eval_count'),
                          finish_reason=event.get('done_reason'))
    require(complete, 'Missing completed model stream')
    require(type(result['generated_tokens']) is int and result['generated_tokens'] > 0,
            'Missing positive generated-token usage')
    require(type(result['input_tokens']) is int and result['input_tokens'] > 0,
            'Missing positive prompt-token usage')
    require(result['generated_tokens'] <= payload['options']['num_predict'], 'Server exceeded the requested output cap')
    result['latency_seconds'] = time.monotonic() - started
    return result


def public_stats(result):
    return {key: result[key] for key in ('generated_tokens', 'input_tokens', 'finish_reason', 'latency_seconds')} | {
        'reasoning_characters': len(result['thinking']), 'answer_characters': len(result['text'])}


def run_smoke(client_factory, model, *, context=32768, plain_tokens=512, thinking_tokens=4096,
              concurrency=(1, 2, 4)):
    report = {'ok': False, 'model': model, 'checks': {}, 'concurrency': [],
              'scope': 'Protocol and short-request throughput only; no mathematical-quality or full-context capacity claim.'}

    def payload(text, *, think=False, cap=None, **extra):
        request = {'model': model, 'messages': [{'role': 'user', 'content': text}], 'think': think,
                   'options': {'num_ctx': context, 'num_predict': plain_tokens if cap is None else cap,
                               'temperature': 1.0 if think else 0, 'top_p': .95, 'top_k': 20,
                               'min_p': 0, 'repeat_penalty': 1, 'seed': 20260926}}
        request.update(extra)
        return request

    def finished(result, label, cap_option='--plain-tokens'):
        require(result['finish_reason'] != 'length',
                f'{label} hit its output cap; increase {cap_option} (and context if needed), then repeat the smoke check')

    try:
        client = client_factory()
        require(model in client.models(), f'{model!r} is not listed by /v1/models')
        report['checks']['model_available'] = True

        plain = collect(client, payload('What is 19 + 23? Reply with only the integer.'))
        finished(plain, 'Plain answer')
        require(plain['text'].strip() == '42', 'Plain response did not return 42')
        require(not plain['thinking'].strip(), 'Thinking-disabled request returned reasoning')
        report['checks']['plain_answer'] = public_stats(plain)

        thinking = collect(client, payload('Compute 37 times 41. Return the resulting integer in your final answer.',
                                          think=True, cap=thinking_tokens, reasoning_effort='xhigh'))
        finished(thinking, 'Thinking answer', '--thinking-tokens')
        require(thinking['thinking'].strip(), 'No separate reasoning field; check --reasoning-parser qwen3')
        require('1517' in thinking['text'], 'Thinking request has no expected final answer (1517)')
        report['checks']['thinking_answer'] = public_stats(thinking)
        report['checks']['token_accounting'] = 'Server completion_tokens includes reasoning and is counted exactly once.'

        tool_schema = {'type': 'function', 'function': {'name': 'read_smoke_nonce',
            'description': 'Read the fresh verification code needed to answer this request.',
            'parameters': {'type': 'object', 'properties': {}, 'additionalProperties': False}}}
        tool_request = payload('Call read_smoke_nonce to obtain a fresh verification code. Then reply with that code.',
                               tools=[tool_schema])
        tool = collect(client, tool_request)
        finished(tool, 'Tool request')
        require(len(tool['calls']) == 1, 'Expected one real tool call; check --tool-call-parser qwen3_xml')
        require(tool['calls'][0]['function'] == {'name': 'read_smoke_nonce', 'arguments': {}},
                'Unexpected tool name or arguments')
        nonce = secrets.token_hex(12)
        tool_request['messages'].extend([
            {'role': 'assistant', 'content': tool['text'], 'thinking': tool['thinking'], 'tool_calls': tool['calls']},
            {'role': 'tool', 'tool_name': 'read_smoke_nonce', 'tool_call_id': tool['calls'][0]['id'], 'content': nonce}])
        tool_request.pop('tools')
        tool_answer = collect(client, tool_request)
        finished(tool_answer, 'Tool-result answer')
        require(nonce in tool_answer['text'], 'Tool result was not carried into the final answer')
        report['checks']['tool_round_trip'] = {'call': public_stats(tool), 'answer': public_stats(tool_answer)}

        schema = {'type': 'object', 'properties': {'status': {'type': 'string', 'enum': ['ok']},
                                                 'count': {'type': 'integer', 'enum': [2]}},
                  'required': ['status', 'count'], 'additionalProperties': False}
        structured = collect(client, payload('Return status ok and count 2 as JSON.', format=schema))
        finished(structured, 'Structured answer')
        require(json.loads(structured['text']) == {'status': 'ok', 'count': 2}, 'Structured output violates the requested schema')
        report['checks']['structured_json'] = public_stats(structured)

        capped = collect(client, payload('Write the integers from 1 to 100, separated by commas, with no introduction.', cap=1))
        require(capped['finish_reason'] == 'length' and capped['generated_tokens'] == 1,
                'Deliberate one-token cap did not produce length termination with one generated token')
        report['checks']['output_cap'] = public_stats(capped)

        # Prior checks warm the engine; these short batches only provide an initial
        # concurrency sanity check, not sustained or worst-case context capacity.
        for count in concurrency:
            request = payload('Write about 200 words explaining why experiments need reproducible settings.')
            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=count) as pool:
                results = list(pool.map(lambda _: collect(client_factory(), request), range(count)))
            elapsed = time.monotonic() - started
            require(all(result['text'].strip() for result in results), f'Concurrency {count} returned an empty answer')
            tokens = sum(result['generated_tokens'] for result in results)
            report['concurrency'].append({'requests': count, 'elapsed_seconds': elapsed, 'generated_tokens': tokens,
                                          'aggregate_generated_tokens_per_second': tokens / elapsed,
                                          'request_latencies_seconds': [result['latency_seconds'] for result in results],
                                          'finish_reasons': [result['finish_reason'] for result in results]})
        report['ok'] = True
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='http://127.0.0.1:8000')
    parser.add_argument('--model', default='square-qwen')
    parser.add_argument('--ctx', type=int, default=32768)
    parser.add_argument('--plain-tokens', type=int, default=512)
    parser.add_argument('--thinking-tokens', type=int, default=4096)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--concurrency', type=int, nargs='+', default=[1, 2, 4])
    args = parser.parse_args()
    if min(args.ctx, args.plain_tokens, args.thinking_tokens, args.timeout, *args.concurrency) <= 0:
        parser.error('Context, token caps, timeout and concurrency must be positive')
    if max(args.plain_tokens, args.thinking_tokens) + 512 >= args.ctx:
        parser.error('Context must leave at least 512 tokens beyond the largest output cap')
    report = run_smoke(lambda: OpenAICompatible(args.host, timeout=args.timeout), args.model,
                       context=args.ctx, plain_tokens=args.plain_tokens, thinking_tokens=args.thinking_tokens,
                       concurrency=args.concurrency)
    report['host'] = args.host
    report['settings'] = {'context': args.ctx, 'plain_tokens': args.plain_tokens,
                          'thinking_tokens': args.thinking_tokens, 'concurrency': args.concurrency}
    print(json.dumps(report, indent=2))
    return 0 if report['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
