#!/usr/bin/env python3
"""Fail-fast and feasibility gates of the analyses (CHECK.md B06, B07, B11).
Run: python3 -m pytest -q tests/test_analysis_gates.py"""
import ast, math, pathlib, sys
sys.path.insert(0, 'bench')
import numpy as np
import pytest
from common import spearman, avg_ranks
from analyze_review import run_costs

def stream(method, rounds, measured, recall=1.0, qps=100.0, **extra):
    rs = []
    for r in rounds:
        row = {'round': r, 'cum_maint_s': float(r)}
        if r in measured: row['sweep'] = [{'recall': recall, 'qps': {'mean': qps}}]
        rs.append(row)
    return {'method': method, 'rounds': rs, **extra}

def test_run_costs_rejects_a_partial_run():
    # the audit's counterexample: an IVF run holding round 0 only produced a complete frontier
    with pytest.raises(ValueError): run_costs(stream('ivf', [0], [0]), 10, 0.95)
    with pytest.raises(ValueError): run_costs(stream('ivf', range(11), [0, 5]), 10, 0.95)       # would extrapolate 6..10
    c = run_costs(stream('ivf', range(11), [0, 5, 10]), 10, 0.95)
    assert c['maint'] == 10.0 and len(c['t_query']) == 10 and c['maintenance_source'] == 'measured'

def test_run_costs_marks_estimated_rebuild_maintenance():
    d = stream('fdiskann_rebuild', [0, 5, 10], [0, 5, 10])
    for r in d['rounds']: r['rebuild_s'] = 2.0
    c = run_costs(d, 10, 0.95); assert c['maintenance_source'] == 'estimated' and c['maint'] == 20.0
    d = stream('fdiskann_rebuild', range(11), [0, 5, 10], maintenance_source='measured')
    for r in d['rounds']: r['rebuild_s'] = 1.0 + r['round']
    c = run_costs(d, 10, 0.95); assert c['maintenance_source'] == 'measured' and c['maint'] == sum(1.0 + r for r in range(1, 11))

def test_run_costs_infeasible_round():
    assert run_costs(stream('ivf', range(11), [0, 5, 10], recall=0.9), 10, 0.95) is None

def test_spearman_ties_and_constant():
    assert spearman([1] * 10, list(range(10)), perms=9) == (None, None)       # the first version returned rho = 1
    assert list(avg_ranks([3, 1, 3, 2])) == [2.5, 0.0, 2.5, 1.0]
    r, _ = spearman([1, 2, 2, 3], [1, 2, 3, 4], perms=0); assert abs(r - 0.9486832980505138) < 1e-12   # scipy's value
    r, _ = spearman([1, 2, 3, 4], [4, 3, 2, 1], perms=0); assert r == pytest.approx(-1.0)

def follow3_functions():
    src = pathlib.Path('bench/analyze_followup3.py').read_text(); tree = ast.parse(src)
    body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('score_rule', 'transfers')]
    env = {'math': math, 'np': np, 'INF': 'inf'}; exec(compile(ast.Module(body=body, type_ignores=[]), 'f3', 'exec'), env)
    return env['score_rule'], env['transfers']

def test_follow3_infinite_regret_fails_the_transfer_test():
    score, transfers = follow3_functions()
    # one wrong choice of an IVF that misses the target (infinite cost) and one finite 2% regret: the first version
    # dropped the infinite value and passed (wrong 1 <= 1, max finite regret 0.02 <= 0.10)
    ws = [{'wl': 'a', 'm': 10, 'c_scan': 1.0, 'c_ivf': math.inf}, {'wl': 'b', 'm': 10, 'c_scan': 1.02, 'c_ivf': 1.0}]
    r = score(ws, lambda w: 5)     # T = 5 < m: always IVF
    assert r['infeasible_choices'] == 1 and r['regret_max'] == 'inf' and not transfers(r)
    assert not transfers(score([], lambda w: 5))                          # an empty evaluation is not a success
    ok = score([{'wl': 'b', 'm': 10, 'c_scan': 1.02, 'c_ivf': 1.0}], lambda w: 5); assert transfers(ok)
