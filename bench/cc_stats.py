"""Visibility checks of the concurrency runs (bench/stream_concurrent.py; follow-up 4, paper_audit/followup4_prereg.md).
A Tracker keeps, for the whole stream, when each id's deletion completed and when its insertion was requested, and
checks every returned id of a query execution against that history (tests/test_cc_stats.py)."""
import threading
import numpy as np
from common import recall_vec

class Tracker:
    def __init__(self, n, N0, attr, qlab, gt, lock='rw'):
        self.n, self.attr, self.qlab, self.gt, self.lock = n, attr, qlab, gt, lock
        self.deleted_at = np.full(n, np.inf); self.born_at = np.full(n, np.inf); self.born_at[:N0] = -np.inf

    def record_updates(self, dtime, itime):
        """dtime: (id, t) records, t = time the deleting batch's write lock was released; itime: (id, t) records,
        t = time the inserting batch's write lock was requested. Kept for the whole stream: a later round's query is
        checked against every earlier deletion. Ids are never reused (the schedules insert unseen ids), so each id is
        deleted at most once."""
        if len(dtime):
            assert np.all(~np.isfinite(self.deleted_at[dtime['id']])), 'an id deleted twice'
            self.deleted_at[dtime['id']] = dtime['t']
        if len(itime): self.born_at[itime['id']] = itime['t']

    def stats(self, recs, r, writer_s):
        """recs: query-execution records (q, phase, t0 request, ta lock granted, t1 end, ids) of round r, after
        record_updates() for this round."""
        n, attr, qlab, gt = self.n, self.attr, self.qlab, self.gt
        out = {'round': r, 'writer_s': writer_s, 'lock': self.lock}
        for ph, lab in ((0, 'concurrent'), (1, 'quiet')):
            x = recs[recs['phase'] == ph]
            if not len(x): out[lab] = {'queries': 0}; continue
            lat = (x['t1'] - x['t0']) * 1e3; wait = (x['ta'] - x['t0']) * 1e3; ids = x['ids']; valid = ids >= 0
            oor = valid & (ids >= n); valid &= ~oor; safe = np.where(valid, ids, 0)
            tdel = np.where(valid, self.deleted_at[safe], np.inf); tins = np.where(valid, self.born_at[safe], -np.inf)
            stale = (tdel < x['t0'][:, None]) & valid; during = (tdel >= x['t0'][:, None]) & (tdel < x['t1'][:, None]) & valid
            unborn = (tins > x['t1'][:, None]) & valid
            inadm = valid & (attr[safe] != qlab[x['q']][:, None]); nv = max(1, int(valid.sum())); ids_ = np.where(oor, -1, ids)
            o = {'queries': int(len(x)), 'duration_s': float(x['t1'].max() - x['t0'].min()),
                 'p50_ms': float(np.percentile(lat, 50)), 'p95_ms': float(np.percentile(lat, 95)), 'p99_ms': float(np.percentile(lat, 99)),
                 'lock_wait_p50_ms': float(np.percentile(wait, 50)), 'lock_wait_p99_ms': float(np.percentile(wait, 99)),
                 'stale_share': float(stale.sum() / nv), 'stale_ids': int(stale.sum()), 'deleted_during_share': float(during.sum() / nv),
                 'unborn_ids': int(unborn.sum()), 'out_of_range_ids': int(oor.sum()), 'returned_ids': int(valid.sum()),
                 'inadmissible_share': float(inadm.sum() / nv), 'recall_end': float(recall_vec(ids_, gt[r][x['q']]).mean())}
            if ph == 0:
                rs_, re_ = recall_vec(ids_, gt[r - 1][x['q']]), recall_vec(ids_, gt[r][x['q']])
                o['recall_start'] = float(rs_.mean()); o['recall_best'] = float(np.maximum(rs_, re_).mean())   # mid-round, neither endpoint is exact
                o['qps'] = float(len(x) / writer_s) if writer_s > 0 else None
            out[lab] = o
        return out

class FairRW:
    """Phase-fair reader-writer lock, the same policy as FairRW in src/cc_ivf.cpp. A reader that arrives while a writer
    holds or waits for the lock enters only after that writer's release, and a writer enters only when no reader is
    active and every reader that was waiting at the previous write release has entered. Readers therefore cannot
    starve a writer, and a writer cannot starve readers by re-acquiring the lock batch after batch."""
    def __init__(self):
        self.cv = threading.Condition(); self.readers = 0; self.writer = False; self.wwait = 0; self.gen = 0; self.waiting = {}
    def acquire_read(self):
        with self.cv:
            g = self.gen
            if self.writer or self.wwait:
                self.waiting[g] = self.waiting.get(g, 0) + 1
                while self.writer or (self.wwait and self.gen == g): self.cv.wait()
                self.waiting[g] -= 1
                if not self.waiting[g]: del self.waiting[g]
                self.cv.notify_all()
            self.readers += 1
    def release_read(self):
        with self.cv:
            self.readers -= 1
            if not self.readers: self.cv.notify_all()
    def acquire_write(self):
        with self.cv:
            self.wwait += 1
            while self.writer or self.readers or any(g < self.gen for g in self.waiting): self.cv.wait()
            self.wwait -= 1; self.writer = True
    def release_write(self):
        with self.cv: self.writer = False; self.gen += 1; self.cv.notify_all()
class NoLock:
    def acquire_read(self): pass
    def release_read(self): pass
    def acquire_write(self): pass
    def release_write(self): pass
