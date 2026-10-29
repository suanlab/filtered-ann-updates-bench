#!/usr/bin/env python3
"""Visibility checks and the reader-writer lock of the concurrency runs (CHECK.md B01/B02).
Run: python3 -m pytest -q tests/test_cc_stats.py"""
import sys, threading, time
sys.path.insert(0, 'bench')
import numpy as np
from cc_stats import Tracker, FairRW

K = 1
REC = np.dtype([('q', '<i4'), ('phase', '<i4'), ('t0', '<f8'), ('ta', '<f8'), ('t1', '<f8'), ('ids', '<i4', (K,))])
DT = np.dtype([('id', '<i4'), ('t', '<f8')])

def tracker(n=6, N0=4):
    gt = np.full((4, 1, K), 2)
    return Tracker(n, N0, np.zeros(n, int), np.zeros(1, int), gt)

def rec(t0, t1, i):
    r = np.zeros(1, REC); r['t0'] = t0; r['ta'] = t0; r['t1'] = t1; r['ids'] = i; return r

def test_deletion_of_an_earlier_round_is_stale():
    # the audit's counterexample: id 0 deleted in round 1 at t=1, id 1 in round 2 at t=2; a round-2 query at t=3
    # returns id 0. The first detector forgot round 1 and reported 0.
    T = tracker()
    T.record_updates(np.array([(0, 1.0)], DT), np.zeros(0, DT)); T.stats(rec(1.5, 1.6, 2), 1, 1.0)
    T.record_updates(np.array([(1, 2.0)], DT), np.zeros(0, DT))
    o = T.stats(rec(3.0, 4.0, 0), 2, 1.0)['concurrent']
    assert o['stale_share'] == 1.0 and o['stale_ids'] == 1

def test_deleted_while_running_is_not_stale():
    T = tracker(); T.record_updates(np.array([(0, 3.5)], DT), np.zeros(0, DT))
    o = T.stats(rec(3.0, 4.0, 0), 1, 1.0)['concurrent']
    assert o['stale_share'] == 0.0 and o['deleted_during_share'] == 1.0

def test_unborn_and_out_of_range_ids():
    T = tracker(); T.record_updates(np.zeros(0, DT), np.array([(4, 5.0)], DT))
    assert T.stats(rec(3.0, 4.0, 4), 1, 1.0)['concurrent']['unborn_ids'] == 1        # inserted after the query ended
    assert T.stats(rec(6.0, 7.0, 4), 1, 1.0)['concurrent']['unborn_ids'] == 0
    assert T.stats(rec(6.0, 7.0, 99), 1, 1.0)['concurrent']['out_of_range_ids'] == 1

def test_an_id_deleted_twice_fails():
    T = tracker(); T.record_updates(np.array([(0, 1.0)], DT), np.zeros(0, DT))
    try: T.record_updates(np.array([(0, 2.0)], DT), np.zeros(0, DT))
    except AssertionError: return
    raise AssertionError('a second deletion of the same id was accepted')

def test_rw_lock_excludes_readers_during_a_write():
    rw, state, bad = FairRW(), {'w': 0, 'r': 0}, []
    def reader():
        for _ in range(200):
            rw.acquire_read(); state['r'] += 1
            if state['w']: bad.append('read during write')
            state['r'] -= 1; rw.release_read()
    def writer():
        for _ in range(50):
            rw.acquire_write(); state['w'] = 1
            if state['r']: bad.append('write during read')
            time.sleep(0.0005); state['w'] = 0; rw.release_write()
    th = [threading.Thread(target=reader) for _ in range(4)] + [threading.Thread(target=writer)]
    for t in th: t.start()
    for t in th: t.join(timeout=30)
    assert not any(t.is_alive() for t in th) and not bad

def test_rw_lock_is_phase_fair():
    # a writer that re-acquires the lock batch after batch must not starve readers (the first Python lock did), and a
    # reader waits for at most the write in progress plus one more
    rw, writes, worst, stop = FairRW(), [0], [0], threading.Event()
    def writer():
        while not stop.is_set():
            rw.acquire_write(); writes[0] += 1; time.sleep(0.0002); rw.release_write()
    def reader():
        for _ in range(100):
            w0 = writes[0]; rw.acquire_read(); worst[0] = max(worst[0], writes[0] - w0); rw.release_read()
    tw = threading.Thread(target=writer); tw.start()
    th = [threading.Thread(target=reader) for _ in range(4)]
    for t in th: t.start()
    for t in th: t.join(timeout=30)
    stop.set(); tw.join(timeout=30)
    assert not any(t.is_alive() for t in th + [tw]) and worst[0] <= 2
