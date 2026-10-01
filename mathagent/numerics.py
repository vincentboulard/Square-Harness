"""Tested numerical helpers for experiments (import as ``from mathagent import numerics as sq``).

Model-written experiments should call these instead of re-deriving schemes, and
must report through :func:`record`, :func:`validate` and :func:`convergence`: the
controller reads ``results.json``, not the model's description of its output.

Conventions: ``laplacian_*`` return the matrix of ``-Δ`` (positive), so its
smallest eigenvalues approximate the Dirichlet/Neumann spectrum.
"""
import json
import math
import os

RESULTS = 'results.json'


# -- reporting ------------------------------------------------------------------------

def _plain(value):
    """Convert NumPy and other numeric types into JSON values."""
    try:
        import numpy as np
    except ImportError:  # pragma: no cover - numpy is a declared extra
        np = None
    if np is not None:
        if isinstance(value, np.ndarray):
            return _plain(value.tolist())
        if isinstance(value, np.generic):
            return _plain(value.item())
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, complex):
        return {'re': _plain(value.real), 'im': _plain(value.imag)}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return str(value)


def _load():
    if os.path.exists(RESULTS):
        with open(RESULTS, encoding='utf-8') as handle:
            return json.load(handle)
    return {'values': {}, 'validations': [], 'convergence': [], 'searches': []}


def _store(data):
    tmp = RESULTS + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as handle:
        json.dump(data, handle, ensure_ascii=False, indent=1)
    os.replace(tmp, RESULTS)


def record(name, value):
    """Save a named result the controller and the interpretation will see."""
    data = _load()
    data['values'][str(name)] = _plain(value)
    _store(data)
    return value


def validate(name, computed, exact, rtol=1e-3, atol=0.0):
    """Compare the method with a case whose answer is known. Returns True if it passed.

    An experiment whose validations fail is reported as unreliable by the controller.
    """
    import numpy as np
    computed_a, exact_a = np.asarray(computed, dtype=float), np.asarray(exact, dtype=float)
    if computed_a.shape != exact_a.shape:
        passed, error = False, float('inf')
    else:
        scale = np.maximum(np.abs(exact_a), 1e-300)
        error = float(np.max(np.abs(computed_a - exact_a) / scale)) if computed_a.size else 0.0
        passed = bool(np.all(np.abs(computed_a - exact_a) <= atol + rtol * np.abs(exact_a)))
    data = _load()
    data['validations'].append({'name': str(name), 'computed': _plain(computed_a), 'exact': _plain(exact_a),
                                'max_relative_error': _plain(error), 'rtol': rtol, 'atol': atol, 'passed': passed})
    _store(data)
    print(f'[validation] {name}: {"passed" if passed else "FAILED"} (max relative error {error:.3e}, rtol {rtol})')
    return passed


def convergence(name, sizes, values, exact=None, rtol=1e-3):
    """Study a quantity under refinement (``sizes`` increasing, e.g. grid points).

    Records the observed order, a Richardson-type extrapolation and whether the last
    refinement changed the value by less than ``rtol`` (relative). Returns that dict.
    """
    import numpy as np
    n = np.asarray(sizes, dtype=float)
    v = np.asarray(values, dtype=float)
    if n.ndim != 1 or n.shape != v.shape or len(n) < 3 or np.any(np.diff(n) <= 0):
        raise ValueError('convergence needs at least three increasing sizes and matching scalar values')
    if exact is not None:
        errors = np.abs(v - exact)
        positive = errors > 0
        order = (float(-np.polyfit(np.log(n[positive]), np.log(errors[positive]), 1)[0])
                 if positive.sum() >= 2 else float('inf'))
    else:
        steps = np.abs(np.diff(v))
        positive = steps > 0
        order = (float(-np.polyfit(np.log(n[1:][positive]), np.log(steps[positive]), 1)[0])
                 if positive.sum() >= 2 else float('inf'))
        errors = None
    last_change = abs(v[-1] - v[-2]) / max(abs(v[-1]), 1e-300)
    extrapolated = None
    if math.isfinite(order) and order > 0:
        ratio = (n[-1] / n[-2]) ** order
        extrapolated = float(v[-1] + (v[-1] - v[-2]) / (ratio - 1))
    entry = {'name': str(name), 'sizes': _plain(n), 'values': _plain(v), 'exact': _plain(exact),
             'errors': _plain(errors), 'observed_order': _plain(order), 'extrapolated': _plain(extrapolated),
             'last_relative_change': _plain(last_change), 'rtol': rtol, 'converged': bool(last_change <= rtol)}
    data = _load()
    data['convergence'].append(entry)
    _store(data)
    print(f'[convergence] {name}: order ≈ {order:.2f}, last relative change {last_change:.2e}, '
          f'{"converged" if entry["converged"] else "NOT converged"} at rtol {rtol}')
    return entry


# -- discretisations --------------------------------------------------------------------

def grid(n, length=1.0, bc='dirichlet'):
    """Nodes used by :func:`laplacian_1d` (interior nodes for Dirichlet)."""
    import numpy as np
    if bc == 'dirichlet':
        return np.linspace(0, length, n + 2)[1:-1]
    if bc == 'neumann':
        return (np.arange(n) + 0.5) * length / n
    if bc == 'periodic':
        return np.linspace(0, length, n, endpoint=False)
    raise ValueError('bc must be dirichlet, neumann or periodic')


def laplacian_1d(n, length=1.0, bc='dirichlet'):
    """Second-order finite-difference matrix of -d²/dx² on (0, length), CSR sparse.

    Dirichlet: n interior nodes, h = L/(n+1). Neumann: n cell centres, h = L/n
    (symmetric finite-volume closure, eigenvalue 0 exact). Periodic: n nodes, h = L/n.
    """
    import numpy as np
    import scipy.sparse as sp
    if type(n) is not int or n < 2:
        raise ValueError('n must be an integer >= 2')
    if bc == 'dirichlet':
        h = length / (n + 1)
        main = 2 * np.ones(n)
        A = sp.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1])
    elif bc == 'neumann':
        h = length / n
        main = 2 * np.ones(n)
        main[0] = main[-1] = 1
        A = sp.diags([-np.ones(n - 1), main, -np.ones(n - 1)], [-1, 0, 1])
    elif bc == 'periodic':
        h = length / n
        A = sp.diags([-np.ones(n - 1), 2 * np.ones(n), -np.ones(n - 1)], [-1, 0, 1]).tolil()
        A[0, n - 1] = A[n - 1, 0] = -1
    else:
        raise ValueError('bc must be dirichlet, neumann or periodic')
    return (sp.csr_matrix(A) / h ** 2)


def laplacian_2d(nx, ny=None, lx=1.0, ly=1.0):
    """Five-point -Δ on the rectangle (0,lx)×(0,ly), Dirichlet, interior nodes, CSR."""
    import scipy.sparse as sp
    ny = nx if ny is None else ny
    ax, ay = laplacian_1d(nx, lx), laplacian_1d(ny, ly)
    return sp.csr_matrix(sp.kron(sp.identity(ny), ax) + sp.kron(ay, sp.identity(nx)))


def smallest_eigenvalues(A, k=6, M=None):
    """The k smallest eigenvalues of a symmetric (generalized) eigenproblem, ascending."""
    import numpy as np
    import scipy.sparse as sp
    import scipy.linalg as la
    size = A.shape[0]
    if sp.issparse(A) and size > 400 and k < size - 1:
        from scipy.sparse.linalg import eigsh
        values = eigsh(A, k=k, M=M, sigma=0, which='LM', return_eigenvectors=False)
    else:
        dense = A.toarray() if sp.issparse(A) else np.asarray(A)
        mass = None if M is None else (M.toarray() if sp.issparse(M) else np.asarray(M))
        values = la.eigh(dense, mass, eigvals_only=True)[:k]
    return np.sort(np.real(values))


def dirichlet_eigenvalues_interval(k, length=1.0):
    """Exact Dirichlet eigenvalues (jπ/L)², j = 1..k, of -d²/dx² on (0, L)."""
    import numpy as np
    return (np.arange(1, k + 1) * np.pi / length) ** 2


def dirichlet_eigenvalues_rectangle(k, lx=1.0, ly=1.0):
    """The k smallest exact Dirichlet eigenvalues of -Δ on (0,lx)×(0,ly)."""
    import numpy as np
    m = int(np.ceil(np.sqrt(k))) + 3
    values = sorted((i * np.pi / lx) ** 2 + (j * np.pi / ly) ** 2 for i in range(1, m + 1) for j in range(1, m + 1))
    return np.array(values[:k])


def heat_1d(u0, T, nt, length=1.0, kappa=1.0, source=None):
    """Crank–Nicolson for u_t = κ u_xx + f(t, x) on (0, L), homogeneous Dirichlet.

    ``u0`` is an array on :func:`grid` (Dirichlet) or a callable of x. ``source`` is
    an optional callable f(t, x) (for a distributed control, e.g. f = χ_ω v).
    Returns (times, x, U) with U[i] the solution at times[i].
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu
    if callable(u0):
        raise ValueError('pass u0 as an array on sq.grid(n, length); call it first')
    u0 = np.asarray(u0, dtype=float)
    n = len(u0)
    x = grid(n, length)
    A = kappa * laplacian_1d(n, length)
    dt = T / nt
    I = sp.identity(n, format='csc')
    left = splu(sp.csc_matrix(I + 0.5 * dt * A))
    right = sp.csr_matrix(I - 0.5 * dt * A)
    times = np.linspace(0, T, nt + 1)
    U = np.empty((nt + 1, n))
    U[0] = u0
    for i in range(nt):
        rhs = right @ U[i]
        if source is not None:
            rhs = rhs + 0.5 * dt * (np.asarray(source(times[i], x)) + np.asarray(source(times[i + 1], x)))
        U[i + 1] = left.solve(rhs)
    return times, x, U


def wave_1d(u0, u1, T, nt, length=1.0, c=1.0, source=None):
    """Leapfrog for u_tt = c² u_xx + f(t, x) on (0, L), homogeneous Dirichlet.

    Stable when c·dt/h ≤ 1 (checked). Returns (times, x, U, energy) with the
    discrete energy ½(|u_t|² + c²|u_x|²) at each step (conserved without source).
    """
    import numpy as np
    u0, u1 = np.asarray(u0, dtype=float), np.asarray(u1, dtype=float)
    n = len(u0)
    x = grid(n, length)
    h = length / (n + 1)
    dt = T / nt
    if c * dt / h > 1 + 1e-12:
        raise ValueError(f'CFL condition violated: c·dt/h = {c * dt / h:.3f} > 1; increase nt')
    A = c ** 2 * laplacian_1d(n, length)
    times = np.linspace(0, T, nt + 1)
    U = np.empty((nt + 1, n))
    U[0] = u0
    f0 = np.zeros(n) if source is None else np.asarray(source(0.0, x))
    U[1] = u0 + dt * u1 + 0.5 * dt ** 2 * (-(A @ u0) + f0)
    for i in range(1, nt):
        f = 0 if source is None else np.asarray(source(times[i], x))
        U[i + 1] = 2 * U[i] - U[i - 1] + dt ** 2 * (-(A @ U[i]) + f)
    velocity = np.vstack([(U[1] - U[0]) / dt, (U[2:] - U[:-2]) / (2 * dt), (U[-1] - U[-2]) / dt])
    energy = 0.5 * h * (np.sum(velocity ** 2, axis=1) + np.einsum('ij,ij->i', U, (A @ U.T).T))
    return times, x, U, energy


def observability_gramian(A, C, T, nodes=64):
    """W(T) = ∫₀ᵀ e^{Aᵀt} CᵀC e^{At} dt for x' = Ax, y = Cx (Gauss–Legendre on 8 panels).

    Returns (W, lambda_min). The best constant in |x(0)|² ≤ K ∫₀ᵀ |y|² dt is
    K = 1/lambda_min (infinite when unobservable). Dense matrices: keep n ≲ 400.
    """
    import numpy as np
    from scipy.linalg import expm, eigh
    A = A.toarray() if hasattr(A, 'toarray') else np.asarray(A, dtype=float)
    C = C.toarray() if hasattr(C, 'toarray') else np.atleast_2d(np.asarray(C, dtype=float))
    points, weights = np.polynomial.legendre.leggauss(nodes)
    W = np.zeros_like(A)
    panels = 8
    for p in range(panels):
        a, b = T * p / panels, T * (p + 1) / panels
        for t, w in zip(0.5 * (b - a) * points + 0.5 * (a + b), 0.5 * (b - a) * weights):
            E = expm(A * t)
            CE = C @ E
            W += w * CE.T @ CE
    W = 0.5 * (W + W.T)
    return W, float(eigh(W, eigvals_only=True)[0])


# -- search and sweeps -----------------------------------------------------------------

def maximize(f, bounds, name='search', seed=0, maxiter=200, popsize=20):
    """Globally maximize f over a box (differential evolution, then local polish).

    Use it adversarially: maximize LHS/RHS of an inequality to estimate the best
    constant or find a violating input. Records the best point; returns (x, f(x)).
    """
    import numpy as np
    from scipy.optimize import differential_evolution
    history = []

    def negative(x):
        value = float(f(x))
        history.append(value)
        return -value if math.isfinite(value) else float('inf')
    result = differential_evolution(negative, bounds, seed=seed, maxiter=maxiter, popsize=popsize,
                                    polish=True, tol=1e-10)
    x, value = np.asarray(result.x), -float(result.fun)
    data = _load()
    data['searches'].append({'name': str(name), 'bounds': _plain(bounds), 'argmax': _plain(x), 'max': _plain(value),
                             'evaluations': len(history), 'seed': seed})
    _store(data)
    print(f'[search] {name}: max ≈ {value:.10g} at {np.array2string(x, precision=8)} ({len(history)} evaluations)')
    return x, value


def sweep(f, **grids):
    """Evaluate f on the Cartesian product of named 1-D grids; returns a list of dicts."""
    import itertools
    names = list(grids)
    rows = []
    for combo in itertools.product(*(list(grids[n]) for n in names)):
        point = dict(zip(names, combo))
        rows.append(dict(point, value=_plain(f(**point))))
    return rows


# -- figures ---------------------------------------------------------------------------

def _figure(name, title=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure = plt.figure(name, figsize=(6.4, 4.2))
    figure.clf()
    axes = figure.add_subplot(111)
    if title:
        axes.set_title(title)
    return plt, figure, axes


def plot_convergence(name, sizes, errors, title=None, reference_orders=(1, 2)):
    """Log-log error plot with reference slopes; saved as figures/<name>.png/.svg."""
    import numpy as np
    plt, figure, axes = _figure(name, title or 'Convergence')
    n, e = np.asarray(sizes, float), np.asarray(errors, float)
    axes.loglog(n, e, 'o-', label='error')
    for p in reference_orders:
        axes.loglog(n, e[0] * (n / n[0]) ** (-p), '--', alpha=.5, label=f'order {p}')
    axes.set_xlabel('size')
    axes.set_ylabel('error')
    axes.grid(True, which='both', alpha=.3)
    axes.legend()
    return figure


def plot_curves(name, x, curves, title=None, xlabel='', ylabel='', logx=False, logy=False):
    """Several curves {label: y-values} against x."""
    plt, figure, axes = _figure(name, title)
    for label, y in curves.items():
        axes.plot(x, y, label=label)
    axes.set_xscale('log' if logx else 'linear')
    axes.set_yscale('log' if logy else 'linear')
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.grid(True, alpha=.3)
    if len(curves) > 1:
        axes.legend()
    return figure


def plot_spectrum(name, computed, exact=None, title=None):
    """Computed eigenvalues (and exact ones when known) against their index."""
    import numpy as np
    plt, figure, axes = _figure(name, title or 'Spectrum')
    index = np.arange(1, len(computed) + 1)
    axes.plot(index, computed, 'o', label='computed')
    if exact is not None:
        axes.plot(np.arange(1, len(exact) + 1), exact, 'x', label='exact')
        axes.legend()
    axes.set_xlabel('index')
    axes.set_ylabel('eigenvalue')
    axes.grid(True, alpha=.3)
    return figure


def plot_heatmap(name, x, y, Z, title=None, xlabel='', ylabel='', level=None):
    """Colour map of Z[y, x]; ``level`` draws the contour Z = level (e.g. where an inequality fails)."""
    plt, figure, axes = _figure(name, title)
    mesh = axes.pcolormesh(x, y, Z, shading='auto')
    figure.colorbar(mesh, ax=axes)
    if level is not None:
        axes.contour(x, y, Z, levels=[level], colors='white', linewidths=1.5)
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    return figure


def plot_solution(name, x, times, U, title=None, snapshots=6):
    """Snapshots of a time-dependent 1-D solution U[t, x]."""
    import numpy as np
    plt, figure, axes = _figure(name, title or 'Solution snapshots')
    for i in np.linspace(0, len(times) - 1, snapshots).astype(int):
        axes.plot(x, U[i], label=f't = {times[i]:.3g}')
    axes.set_xlabel('x')
    axes.legend(fontsize='small')
    axes.grid(True, alpha=.3)
    return figure
