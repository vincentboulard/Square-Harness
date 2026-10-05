"""Effort levels: the budgets behind Low … Poincaré.

The interface's forms use the same table (frontend/src/effort.ts); a test keeps
the two identical. The Default notebook starts jobs on the server, so it needs
the budgets here too.
"""

# tries are proof attempts or investigation rounds; tokens are generated tokens.
EFFORTS = {
    'low': dict(tries=1, minutes=1, tokens=30_000, input=120_000, requests=20, chars=15_000),
    'medium': dict(tries=3, minutes=15, tokens=60_000, input=240_000, requests=50, chars=30_000),
    'high': dict(tries=5, minutes=30, tokens=100_000, input=400_000, requests=80, chars=60_000),
    'xhigh': dict(tries=7, minutes=60, tokens=150_000, input=600_000, requests=120, chars=100_000),
    'poincare': dict(tries=10, minutes=120, tokens=200_000, input=800_000, requests=160, chars=150_000),
}
LABELS = {'low': 'Low', 'medium': 'Medium', 'high': 'High', 'xhigh': 'Extra high', 'poincare': 'Poincaré'}


def _proof_ceilings(args, tokens):
    """Fit one full solve and its review in the budget, keeping the launch repair floors."""
    solve, verify = args.proof_solve_tokens, args.proof_verify_tokens
    if tokens >= solve + verify:
        return tokens, solve, verify
    floor = max(128, args.proof_repair_tokens or 0, args.proof_min_solve_tokens or 0)
    fitted = min(solve, max(floor, tokens * solve // (solve + verify)))
    review = min(verify, max(128, tokens - fitted))
    return max(tokens, fitted + review), fitted, review


def limits(mode, level, args):
    """Budget keyword arguments for Hub.start_route; answers in the conversation have none."""
    effort = EFFORTS.get(level, EFFORTS['medium'])
    seconds = effort['minutes'] * 60
    if mode == 'prove':
        tokens, solve, verify = _proof_ceilings(args, effort['tokens'])
        return dict(rounds=effort['tries'], tokens=tokens, seconds=seconds, solve_tokens=solve, verify_tokens=verify)
    if mode == 'writeup':
        return dict(tokens=effort['tokens'], input_tokens=effort['input'], seconds=seconds)
    if mode == 'check':
        # A literature check has its own small budgets (refcheck.EFFORTS), chosen by level.
        return dict(effort=level if level in EFFORTS else 'medium')
    if mode == 'detailed_review':
        # A detailed review also checks the proofs: twice the tokens, context and time of a review.
        return dict(rounds=effort['tries'], tokens=2 * effort['tokens'], input_tokens=2 * effort['input'], seconds=2 * seconds,
                    requests=effort['requests'], chars=effort['chars'])
    if mode in ('literature', 'referee', 'quick_review', 'explain'):
        return dict(rounds=effort['tries'], tokens=effort['tokens'], input_tokens=effort['input'], seconds=seconds,
                    requests=effort['requests'], chars=effort['chars'])
    return {}
