#!/usr/bin/env python3
"""Run lineage (CHECK.md B14): output directories of the helper scripts and the workload-statistics cache check.
Run: python3 -m pytest -q tests/test_lineage.py"""
import sys, json
sys.path.insert(0, 'bench')
import pytest
import common

def test_output_dir_never_overwrites_the_original_under_a_tag(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('OUT_DIR', raising=False); monkeypatch.delenv('BENCH_TAG', raising=False)
    assert common.output_dir('results/bench_h5') == 'results/bench_h5'                  # the historical default
    monkeypatch.setenv('BENCH_TAG', 'rep_main'); monkeypatch.setattr(common, 'RES', 'results/bench_rep_main')
    assert common.output_dir('results/bench_h5') == 'results/bench_rep_main'           # a re-execution keeps the original
    monkeypatch.setenv('OUT_DIR', 'elsewhere')
    assert common.output_dir('results/bench_h5') == 'elsewhere'

def test_stale_cache_entry_fails(tmp_path, monkeypatch):
    f = tmp_path / 'stats.json'; f.write_text(json.dumps({'admissible_sizes': {}, 'streams': {'s': {'R': 10, 'updates_per_round': 5.0}}}))
    monkeypatch.setattr(common, 'STATS', str(f)); monkeypatch.delenv('STATS_REGEN', raising=False)
    common._check_cache('streams', 's', {'R': 10, 'updates_per_round': 5.0})            # matches: no error
    common._check_cache('streams', 'other', {'R': 1})                                  # not cached: no error
    with pytest.raises(SystemExit): common._check_cache('streams', 's', {'R': 10, 'updates_per_round': 6.0})
    monkeypatch.setenv('STATS_REGEN', '1')
    common._check_cache('streams', 's', {'R': 10, 'updates_per_round': 6.0})            # regenerating: skipped
