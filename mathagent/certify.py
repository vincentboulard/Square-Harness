"""Independent checker for explicit counterexample certificates.

This file is deliberately standalone (it needs only SymPy and mpmath), so it can be
copied next to a certificate and rerun by anyone:

    python verify_certificate.py certificate.json

A certificate claims that an explicit witness violates a universally quantified
statement: every hypothesis holds at the witness, and the asserted relation fails.
Each relation ``lhs REL rhs`` is decided from the exact difference ``lhs - rhs``:

* exactly, when SymPy reduces the difference to a rational number (or to the
  literal zero for an equality);
* otherwise by outward-rounded interval arithmetic (mpmath.iv) at increasing
  precision, which certifies a strict sign when the enclosure excludes zero.

Floating-point literals are rejected; integrals and sums must evaluate in closed
form. What this checks is the arithmetic of the certificate. Whether the
certificate encodes the intended statement is a separate, human question.

Certificate format (JSON)::

    {
      "version": 1,
      "statement": "For every u in H^1_0(0,1): int u'^2 >= 10 int u^2",
      "variables": ["t"],                       # bound variables (integration etc.)
      "symbols": {"c": "10"},                   # witness values, exact expressions
      "functions": {"u": {"args": ["t"], "expr": "t*(1-t)"}},
      "assumptions": [{"lhs": "u(0)", "relation": "==", "rhs": "0"},
                      {"lhs": "u(1)", "relation": "==", "rhs": "0"}],
      "claim": {"lhs": "integrate(diff(u(t),t)**2,(t,0,1))", "relation": ">=",
                "rhs": "c*integrate(u(t)**2,(t,0,1))"}
    }
"""
import json
import re
import signal
import sys

RELATIONS = ('<', '<=', '>', '>=', '==', '!=')
NAME = re.compile(r'[A-Za-z][A-Za-z0-9_]{0,30}')
SAFE = re.compile(r'[A-Za-z0-9_+\-*/^(), ]*')
ALLOWED = ('sin', 'cos', 'tan', 'exp', 'log', 'sqrt', 'atan', 'asin', 'acos', 'sinh', 'cosh', 'tanh',
           'pi', 'E', 'Abs', 'factorial', 'binomial', 'Rational', 'integrate', 'diff', 'Sum', 'Integral',
           'Derivative', 'oo')
RESERVED = set(ALLOWED) | {'I', 'S', 'N', 'O', 'Q', 'beta', 'gamma', 'zeta', 'lambda'}
PRECISIONS = (64, 128, 256, 1024, 4096)
MAX_TEXT = 4000
TIME_LIMIT = 300


class CertificateError(ValueError):
    pass


def _namespace():
    import sympy
    return {name: getattr(sympy, name) for name in ALLOWED}


def _parse(text, names, where):
    import sympy
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_TEXT:
        raise CertificateError(f'{where}: expected a nonempty expression of at most {MAX_TEXT} characters')
    if not SAFE.fullmatch(text) or '__' in text:
        raise CertificateError(f'{where}: only names, integers, + - * / ^ ( ) and commas are allowed '
                               '(write rationals as 1/3; no decimal points)')
    for word in re.findall(r'[A-Za-z_][A-Za-z0-9_]*', text):
        if word not in names and word not in ALLOWED:
            raise CertificateError(f'{where}: unknown name {word!r}; declare it in variables, symbols or functions')
    local = dict(_namespace(), **names)
    try:
        value = parse_expr(text.replace('^', '**'), local_dict=local, global_dict={'Integer': sympy.Integer,
                           'Symbol': sympy.Symbol, 'Function': sympy.Function, 'Rational': sympy.Rational},
                           transformations=standard_transformations, evaluate=True)
    except Exception as exc:  # parse errors are certificate errors, never crashes
        raise CertificateError(f'{where}: cannot parse ({type(exc).__name__}: {exc})') from None
    if not isinstance(value, sympy.Basic):
        raise CertificateError(f'{where}: not a mathematical expression')
    if value.atoms(sympy.Float):
        raise CertificateError(f'{where}: floating-point numbers are not exact; use rationals')
    return value


def _interval(expr):
    """Outward-rounded enclosure of a closed-form real number (mpmath.iv)."""
    import sympy
    from mpmath import iv

    def walk(e):
        if e.is_Integer:
            return iv.mpf(str(int(e)))
        if e.is_Rational:
            return iv.mpf(str(e.p)) / iv.mpf(str(e.q))
        if e is sympy.pi:
            return iv.pi
        if e is sympy.E:
            return iv.e
        if e.is_Add:
            total = iv.mpf(0)
            for arg in e.args:
                total += walk(arg)
            return total
        if e.is_Mul:
            total = iv.mpf(1)
            for arg in e.args:
                total *= walk(arg)
            return total
        if e.is_Pow:
            base, power = e.args
            if power.is_Integer:
                b = walk(base)
                n = int(power)
                if n < 0 and b.a <= 0 <= b.b:
                    raise CertificateError('division by an interval containing zero')
                return b ** n
            if power == sympy.Rational(1, 2):
                return iv.sqrt(walk(base))
            b = walk(base)
            if not b.a > 0:
                raise CertificateError('non-integer power of a quantity not certified positive')
            return iv.exp(walk(power) * iv.log(b))
        functions = {getattr(sympy, name): getattr(iv, name) for name in
                     ('sin', 'cos', 'tan', 'exp', 'log', 'atan', 'sinh', 'cosh', 'tanh') if hasattr(iv, name)}
        if e.func in functions and len(e.args) == 1:
            value = walk(e.args[0])
            if e.func is sympy.log and not value.a > 0:
                raise CertificateError('logarithm of a quantity not certified positive')
            return functions[e.func](value)
        if e.func is sympy.Abs:
            return abs(walk(e.args[0]))
        raise CertificateError(f'no interval rule for {e.func.__name__}')
    return walk(expr)


def decide(lhs, relation, rhs):
    """Return (truth, method, detail) for a closed-form relation, or raise."""
    import sympy
    from mpmath import iv
    difference = sympy.simplify(sympy.sympify(lhs - rhs).doit())
    if difference.free_symbols:
        raise CertificateError('the relation still depends on free symbols ' + str(sorted(map(str, difference.free_symbols))))
    if difference.has(sympy.Integral, sympy.Sum, sympy.Derivative):
        raise CertificateError('an integral, sum or derivative did not evaluate in closed form')
    if difference.has(sympy.I) or difference.has(sympy.zoo, sympy.nan, sympy.oo, -sympy.oo):
        raise CertificateError('the difference is not a finite real number')
    sign, method, detail = None, None, str(difference)
    if difference.is_Rational:
        sign = int(bool(difference > 0)) - int(bool(difference < 0))
        method = 'exact rational arithmetic'
    elif relation in ('==', '!=') and difference == 0:
        sign, method = 0, 'exact symbolic simplification'
    else:
        for bits in PRECISIONS:
            old = iv.prec
            try:
                iv.prec = bits
                box = _interval(difference)
            finally:
                iv.prec = old
            if box.a > 0:
                sign = 1
            elif box.b < 0:
                sign = -1
            if sign is not None:
                method, detail = f'interval arithmetic at {bits} bits', f'{difference} in {box}'
                break
        if sign is None:
            if relation in ('==', '!='):
                raise CertificateError('equality could not be decided exactly (difference ' + str(difference) + ')')
            raise CertificateError('the sign of ' + str(difference) + ' could not be certified (zero not excluded)')
    truth = {'<': sign < 0, '<=': sign <= 0, '>': sign > 0, '>=': sign >= 0, '==': sign == 0, '!=': sign != 0}[relation]
    return truth, method, detail


def _relation(item, names, where):
    if not isinstance(item, dict) or set(item) - {'lhs', 'relation', 'rhs', 'note'} or item.get('relation') not in RELATIONS:
        raise CertificateError(f'{where}: expected {{"lhs", "relation" in {RELATIONS}, "rhs"}}')
    return _parse(item['lhs'], names, where + '.lhs'), item['relation'], _parse(item['rhs'], names, where + '.rhs')


def check(certificate):
    """Check a certificate dictionary; return a JSON-serialisable verdict."""
    import mpmath
    import sympy
    report = {'certified': False, 'claim_violated': False, 'assumptions_hold': False, 'checks': [],
              'reason': '', 'versions': {'sympy': sympy.__version__, 'mpmath': mpmath.__version__}}
    try:
        if not isinstance(certificate, dict) or certificate.get('version') != 1:
            raise CertificateError('certificate must be a JSON object with "version": 1')
        unknown = set(certificate) - {'version', 'statement', 'variables', 'symbols', 'functions', 'assumptions', 'claim', 'note'}
        if unknown:
            raise CertificateError('unknown certificate fields: ' + ', '.join(sorted(unknown)))
        names = {}
        for name in certificate.get('variables', []):
            if not isinstance(name, str) or not NAME.fullmatch(name) or name in RESERVED or name in names:
                raise CertificateError(f'invalid bound variable name {name!r}')
            names[name] = sympy.Symbol(name, real=True)
        symbols = certificate.get('symbols', {})
        functions = certificate.get('functions', {})
        if not isinstance(symbols, dict) or not isinstance(functions, dict):
            raise CertificateError('symbols and functions must be objects')
        for name, text in symbols.items():
            if not NAME.fullmatch(name) or name in RESERVED or name in names:
                raise CertificateError(f'invalid witness symbol name {name!r}')
            names[name] = _parse(text, names, f'symbols.{name}')
        for name, spec in functions.items():
            if not NAME.fullmatch(name) or name in RESERVED or name in names:
                raise CertificateError(f'invalid function name {name!r}')
            if not isinstance(spec, dict) or not isinstance(spec.get('args'), list) or not spec['args']:
                raise CertificateError(f'functions.{name}: expected {{"args": [...], "expr": "..."}}')
            args = []
            for arg in spec['args']:
                if not isinstance(arg, str) or not NAME.fullmatch(arg) or arg in RESERVED:
                    raise CertificateError(f'functions.{name}: invalid argument name {arg!r}')
                args.append(sympy.Symbol(arg, real=True))
            inner = dict(names, **{str(a): a for a in args})
            names[name] = sympy.Lambda(tuple(args), _parse(spec.get('expr'), inner, f'functions.{name}.expr'))
        assumptions = certificate.get('assumptions', [])
        if not isinstance(assumptions, list) or len(assumptions) > 50:
            raise CertificateError('assumptions must be a list of at most 50 relations')
        all_hold = True
        for index, item in enumerate(assumptions, 1):
            lhs, rel, rhs = _relation(item, names, f'assumptions[{index}]')
            truth, method, detail = decide(lhs, rel, rhs)
            report['checks'].append({'what': f'hypothesis {index}', 'relation': f'{item["lhs"]} {rel} {item["rhs"]}',
                                     'holds': truth, 'method': method, 'detail': detail})
            all_hold = all_hold and truth
        report['assumptions_hold'] = all_hold
        lhs, rel, rhs = _relation(certificate.get('claim'), names, 'claim')
        truth, method, detail = decide(lhs, rel, rhs)
        claim = certificate['claim']
        report['checks'].append({'what': 'claim at the witness', 'relation': f'{claim["lhs"]} {rel} {claim["rhs"]}',
                                 'holds': truth, 'method': method, 'detail': detail})
        report['claim_violated'] = not truth
        report['certified'] = all_hold and not truth
        if report['certified']:
            report['reason'] = 'Every hypothesis holds at the witness and the claimed relation fails there.'
        elif not all_hold:
            report['reason'] = 'Not a counterexample: a hypothesis fails at the witness.'
        else:
            report['reason'] = 'Not a counterexample: the claimed relation holds at the witness.'
    except CertificateError as exc:
        report['reason'] = 'Not certified: ' + str(exc)
    except Exception as exc:  # any SymPy failure means "not certified", never "certified"
        report['reason'] = f'Not certified: checker error ({type(exc).__name__}: {exc})'
    return report


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print('usage: python verify_certificate.py certificate.json', file=sys.stderr)
        return 2
    with open(argv[0], encoding='utf-8') as handle:
        try:
            certificate = json.load(handle)
        except ValueError as exc:
            certificate = None
            print(json.dumps({'certified': False, 'reason': f'Not certified: invalid JSON ({exc})'}))
            return 1
    if hasattr(signal, 'SIGALRM'):
        # SymPy can spend unbounded time on an integral without a closed form.
        def stop(*_):
            raise CertificateError(f'gave up after {TIME_LIMIT} s')
        signal.signal(signal.SIGALRM, stop)
        signal.alarm(TIME_LIMIT)
    report = check(certificate)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['certified'] else 1


if __name__ == '__main__':
    sys.exit(main())
