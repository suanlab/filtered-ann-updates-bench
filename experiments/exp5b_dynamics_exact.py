#!/usr/bin/env python3
"""exp5b: EXACT stable-carving maintenance — canonical tie-break + exact affected sets via dmax.
Assignment is then a pure function of (alive set); CSD must hold bit-exactly."""
import sys, json, time
sys.path.insert(0, 'experiments')
import numpy as np
from exp0_regen_gap import make_gmm, log

def priority(ids, seed=7):
    x = (np.asarray(ids, dtype=np.uint64) + np.uint64(seed)) * np.uint64(0x9E3779B97F4A7C15)
    x ^= x >> np.uint64(31); x *= np.uint64(0xBF58476D1CE4E5B9); x ^= x >> np.uint64(27)
    return x

class ExactStableCarve:
    """Leaders = L0 smallest-priority alive ids. Point q assigned to f0 nearest leaders
    (ties broken by leader id). dmax[q] = distance to its f0-th (last) assigned leader."""
    def __init__(self, X, ids, L0, f0):
        self.X, self.L0, self.f0 = X, L0, f0
        self.alive = set(int(i) for i in ids)
        self.batch_build()
    def _leaders_now(self):
        ids = np.fromiter(self.alive, dtype=np.int64)
        pr = priority(ids)
        take = ids[np.lexsort((ids, pr))[:self.L0]]
        return np.sort(take)
    def _assign_block(self, qids):
        Lx = self.X[self.leaders]
        qn = (self.X[qids] ** 2).sum(1)[:, None]; ln = (Lx ** 2).sum(1)[None, :]
        D = qn + ln - 2.0 * (self.X[qids] @ Lx.T)
        order = np.lexsort((np.broadcast_to(self.leaders, D.shape), D), axis=1)[:, :self.f0]
        for i, q in enumerate(qids.tolist()):
            ls = tuple(sorted(int(self.leaders[j]) for j in order[i]))
            dmx = float(D[i, order[i]].max())
            old = self.assign.get(q)
            if old != ls:
                if old:
                    for l in old: self.buckets.get(l, set()).discard(q)
                for l in ls: self.buckets.setdefault(l, set()).add(q)
            self.assign[q] = ls; self.dmax[q] = dmx
    def batch_build(self):
        self.leaders = self._leaders_now()
        self.assign, self.dmax = {}, {}
        self.buckets = {int(l): set() for l in self.leaders}
        ids = np.fromiter(self.alive, dtype=np.int64); ids.sort()
        for s in range(0, len(ids), 8192):
            self._assign_block(ids[s:s+8192])
    def _leader_change(self, new_leaders):
        old_set = set(self.leaders.tolist()); new_set = set(new_leaders.tolist())
        demoted, promoted = old_set - new_set, new_set - old_set
        affected = set()
        for l in demoted:
            affected |= {q for q, ls in self.assign.items() if l in ls}
            self.buckets.pop(l, None)
        self.leaders = new_leaders
        for l in promoted:
            self.buckets.setdefault(int(l), set())
            lx = self.X[l]
            ids = np.fromiter(self.alive, dtype=np.int64)
            d = ((self.X[ids] - lx) ** 2).sum(1)
            dm = np.array([self.dmax.get(int(q), np.inf) for q in ids])
            hit = ids[d <= dm + 1e-6]          # exact affected superset (ties margin)
            affected |= set(int(h) for h in hit)
        affected &= self.alive
        if affected:
            self._assign_block(np.fromiter(affected, dtype=np.int64))
        return len(affected)
    def insert(self, pid):
        self.alive.add(pid)
        new_leaders = self._leaders_now()
        if not np.array_equal(new_leaders, self.leaders):
            t = self._leader_change(new_leaders)
            if pid not in self.assign: self._assign_block(np.array([pid], dtype=np.int64))
            return t + 1, True
        self._assign_block(np.array([pid], dtype=np.int64))
        return 1, False
    def delete(self, pid):
        self.alive.discard(pid)
        ls = self.assign.pop(pid, ()); self.dmax.pop(pid, None)
        for l in ls: self.buckets.get(l, set()).discard(pid)
        new_leaders = self._leaders_now()
        if not np.array_equal(new_leaders, self.leaders):
            t = self._leader_change(new_leaders)
            return t + 1, True
        return 1, False
    def state_hash(self):
        import hashlib
        h = hashlib.sha256()
        for pid in sorted(self.assign):
            h.update(f"{pid}:{self.assign[pid]};".encode())
        return h.hexdigest()

def main():
    d, f0, seed = 128, 8, 42
    out = {'locality': {}, 'csd': {}}
    # A) CSD
    n = 20000
    rng = np.random.default_rng(seed)
    X, C = make_gmm(n + 2000, d, k=64, rng=rng, spread=1.5)
    kill = np.random.default_rng(3).choice(n, 1000, replace=False)
    final_alive = np.array(sorted(set(range(n + 2000)) - set(kill.tolist())), dtype=np.int64)
    scB = ExactStableCarve(X, final_alive, max(64, n // 100), f0)
    hashes = set()
    for perm_seed in (0, 1, 2):
        sc = ExactStableCarve(X, np.arange(n, dtype=np.int64), max(64, n // 100), f0)
        ops = [('i', int(n + i)) for i in range(2000)] + [('d', int(p)) for p in kill]
        np.random.default_rng(perm_seed).shuffle(ops)
        for op, pid in ops:
            (sc.insert if op == 'i' else sc.delete)(pid)
        hashes.add(sc.state_hash())
    ident = scB.state_hash() in hashes and len(hashes) == 1
    out['csd'] = {'streamed_unique': len(hashes), 'identical': bool(ident)}
    log(f"CSD: {len(hashes)} unique streamed state(s); identical to batch = {ident}")
    # B) locality
    for n2 in (20000, 40000, 80000, 160000):
        rng = np.random.default_rng(seed)
        X2, _ = make_gmm(int(n2 * 1.1), d, k=64, rng=rng, spread=1.5)
        sc = ExactStableCarve(X2, np.arange(n2, dtype=np.int64), max(64, n2 // 100), f0)
        ins_t, del_t, lead = [], [], 0
        for i in range(300):
            t, L = sc.insert(n2 + i); ins_t.append(t); lead += L
        for pid in np.random.default_rng(1).choice(n2, 300, replace=False).tolist():
            t, L = sc.delete(int(pid)); del_t.append(t); lead += L
        out['locality'][str(n2)] = {'ins_med': float(np.median(ins_t)), 'ins_p99': float(np.percentile(ins_t, 99)),
                                    'del_med': float(np.median(del_t)), 'del_p99': float(np.percentile(del_t, 99)),
                                    'leader_events': lead}
        log(f"n={n2}: ins med={np.median(ins_t):.0f} p99={np.percentile(ins_t,99):.0f} | "
            f"del med={np.median(del_t):.0f} p99={np.percentile(del_t,99):.0f} | leader_ev={lead}/600")
    json.dump(out, open('results/exp5b.json', 'w'), indent=2)
    log("DONE")

if __name__ == '__main__':
    main()
