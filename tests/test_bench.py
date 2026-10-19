"""Unit tests for the benchmark harness (bench/): the functions that turn raw measurements into the
paper's numbers. Small, hand-computable cases; no index builds."""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'bench'))
import numpy as np
import pytest
import common
from analyze import frontier, qps_at, dispatcher, merge_stream_passes, break_even


# ---- recall -------------------------------------------------------------------------------------
def test_recall_basic_and_padding():
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert common.recall(np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]), gt) == 1.0
    assert common.recall(np.array([[0, 1, 2, 3, 4, -1, -1, -1, -1, -1]]), gt) == 0.5   # -1 never matches
    # fewer than k admissible points: denominator is |A|, and -1 padding of the ground truth is ignored
    gt3 = np.array([[7, 8, 9, -1, -1, -1, -1, -1, -1, -1]])
    assert common.recall(np.array([[9, 7, -1, -1, -1, -1, -1, -1, -1, -1]]), gt3) == pytest.approx(2 / 3)
    # no admissible point at all counts as a perfect answer
    assert common.recall(np.full((1, 10), -1), np.full((1, 10), -1)) == 1.0

def test_recall_duplicates_do_not_inflate():
    gt = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]])
    assert common.recall(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), gt) == pytest.approx(0.1)

def test_recall_vec_matches_mean():
    rng = np.random.default_rng(0); gt = rng.permutation(100)[:30].reshape(3, 10); out = gt.copy(); out[1, :4] = -1
    v = common.recall_vec(out, gt)
    assert list(np.round(v, 3)) == [1.0, 0.6, 1.0] and common.recall(out, gt) == pytest.approx(v.mean())

def test_ci_and_timed_search_median():
    calls = iter([(np.array([1]), 1.0), (np.array([2]), 3.0), (np.array([3]), 2.0), (np.array([4]), 4.0)])
    out, q = common.timed_search(lambda: next(calls), nq=10, reps=3)   # warm-up (1.0) then 3 timed runs
    assert out[0] == 2                                                   # output of the median-time run (3.0 s of 2.0, 3.0, 4.0)
    assert q['reps'] == 3 and q['mean'] == pytest.approx(np.mean([10 / 3, 10 / 2, 10 / 4]))

def test_groups_and_admissible():
    pred = {'kind': np.array('label'), 'attr': np.array([0, 1, 1, 2]), 'qlab': np.array([1, 2, 1])}
    g = dict(common.groups(pred, 3))
    assert list(g[1]) == [0, 2] and list(g[2]) == [1]
    assert list(common.admissible(pred, 1, 4)) == [False, True, True, False]
    csr = {'kind': np.array('csr'), 'qoff': np.array([0, 2, 3]), 'qids': np.array([1, 3, 0])}
    assert [k for k, _ in common.groups(csr, 2)] == [0, 1]
    assert list(common.admissible(csr, 0, 4)) == [False, True, False, True] and list(common.admissible_ids(csr, 1)) == [0]


# ---- frontier and interpolation -------------------------------------------------------------------
def test_frontier_drops_dominated_points():
    fr = frontier([(0.9, 100), (0.95, 50), (0.93, 40), (0.99, 10), (0.85, 90)])
    assert fr == [(0.9, 100), (0.95, 50), (0.99, 10)]

def test_qps_at_interpolation():
    fr = [(0.9, 100.0), (0.99, 10.0)]
    assert qps_at(fr, 0.995) is None                    # never reached
    assert qps_at(fr, 0.99) == pytest.approx(10.0)
    assert qps_at(fr, 0.85) == pytest.approx(100.0)     # below the frontier: the fastest point
    w = (0.95 - 0.9) / (0.99 - 0.9)                     # log-linear between the bracketing points
    assert qps_at(fr, 0.95) == pytest.approx(np.exp(np.log(100) * (1 - w) + np.log(10) * w))


# ---- dispatcher (oracle and held-out) -----------------------------------------------------------------
def test_dispatcher_hybrid_recall_and_qps():
    # 4 queries: two small admissible sets (size 10), two large (size 1e6). The scan is exact and fast on small
    # sets, slow on large ones; the index fails on small sets and is fast on large ones.
    sizes = np.array([10, 10, 10**6, 10**6])
    scan = {'param': 'per_query', 'rq': [1, 1, 1, 1], 'tq': [0.001, 0.001, 1.0, 1.0], 'recall': 1.0, 'qps': {'mean': 1}}
    idx = {'param': 'ef=1', 'rq': [0, 0, 1, 1], 'tq': [0.01, 0.01, 0.01, 0.01], 'recall': 0.5, 'qps': {'mean': 1}}
    H = dispatcher('wl', {'scan_part': [scan], 'ivf': [idx]}, sizes)
    b = H['ivf']['best']['0.99']                        # best measured configuration reaching recall 0.99
    assert b['recall'] == 1.0 and 10 < b['T'] <= 10**6  # scan the small sets, index the large ones
    assert b['qps'] == pytest.approx(4 / (0.001 + 0.001 + 0.01 + 0.01))
    cv = H['ivf']['heldout']['0.99']                    # chosen on even queries, measured on odd ones
    assert cv['test_recall'] == 1.0 and cv['test_qps'] == pytest.approx(2 / (0.001 + 0.01))
    assert cv['p99_ms'] == pytest.approx(1e3 * np.quantile([0.001, 0.01], 0.99))   # latency of the held-out half only


# ---- streams ------------------------------------------------------------------------------------------
def test_merge_stream_passes_best_qps_min_maintenance(tmp_path):
    def run(qps, maint, rounds=(0, 4)):
        return {'method': 'ivf', 'rounds': [{'round': r, 'cum_maint_s': maint * r,
                'sweep': [{'param': 'nprobe=8', 'recall': 0.9, 'qps': {'mean': qps}}]} for r in rounds]}
    a, b = tmp_path / 'a.json', tmp_path / 'b.json'
    a.write_text(json.dumps(run(100, 2.0))); b.write_text(json.dumps(run(150, 1.0, rounds=(4,))))   # pass b lacks round 0
    m = merge_stream_passes([str(a), str(b)])
    r0, r4 = m['rounds']
    assert r0['round'] == 0 and r0['sweep'][0]['qps']['mean'] == 100       # round 0 kept from pass a
    assert r4['sweep'][0]['qps']['mean'] == 150 and r4['cum_maint_s'] == 4.0   # best QPS, minimum maintenance

def test_break_even():
    # rebuild 100 s vs IVF 1 s per round; per-query 1/50 s (IVF) vs 1/100 s (F-DiskANN): Q = 99 / 0.01
    assert break_even(100.0, 1.0, 50.0, 100.0) == pytest.approx(9900.0)
    assert break_even(100.0, 1.0, 100.0, 50.0) is None and break_even(100.0, 1.0, None, 50.0) is None


# ---- workload generation ---------------------------------------------------------------------------------
def test_labels_balanced_and_correlated():
    from prep import labels
    X = np.random.default_rng(1).standard_normal((20000, 16)).astype(np.float32)
    for beta in (0.0, 0.9):
        a = labels(X, beta, 10)
        assert a.min() == 0 and a.max() == 9 and np.bincount(a).min() == np.bincount(a).max() == 2000
    w = np.random.default_rng(11).standard_normal(16).astype(np.float32); w /= np.linalg.norm(w)
    proj = X @ w
    c0, c9 = (abs(np.corrcoef(proj, labels(X, b, 10))[0, 1]) for b in (0.0, 0.9))
    assert c0 < 0.05 and c9 > 0.8                       # beta controls the label--geometry correlation

def test_crossset_dispatcher_uses_train_choice_on_test():
    from analyze import crossset
    sizes = np.array([10, 10, 10**6, 10**6])
    scan = {'param': 'per_query', 'rq': [1, 1, 1, 1], 'tq': [0.001, 0.001, 1.0, 1.0]}
    idx = {'param': 'ef=1', 'rq': [0, 0, 1, 1], 'tq': [0.01] * 4}
    idx_te = {'param': 'ef=1', 'rq': [0, 0, 1, 0.5], 'tq': [0.01] * 4}       # the index is worse on the test set
    X = crossset({'scan_part': [scan], 'ivf': [idx]}, sizes, {'scan_part': [scan], 'ivf': [idx_te]}, sizes)
    c = X['ivf']['0.99']
    assert c['train_recall'] == 1.0 and c['test_recall'] == pytest.approx(3.5 / 4)   # chosen on train, measured on test

def test_pair_dispatch_routes_by_admissible_size():
    from analyze import pair_dispatch
    sizes = np.array([10, 10, 10**6, 10**6])
    small = {'param': 'a', 'rq': [1, 1, 0, 0], 'tq': [0.001] * 4}      # exact on selective queries only
    large = {'param': 'b', 'rq': [0, 0, 1, 1], 'tq': [0.002] * 4}      # exact on common queries only
    P = pair_dispatch({'s': [small], 'l': [large]}, sizes, {'s': [small], 'l': [large]}, sizes, 's', 'l')
    c = P['0.99']
    assert c['recall'] == 1.0 and 10 < c['T'] <= 10**6 and c['qps'] == pytest.approx(4 / 0.006) and c['test_recall'] == 1.0
