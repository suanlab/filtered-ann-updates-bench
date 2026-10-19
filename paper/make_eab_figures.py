#!/usr/bin/env python3
"""Figures for paper_eab.tex from results/bench_summary.json (run bench/analyze.py first)."""
import json, os
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'pdf.fonttype': 42, 'ps.fonttype': 42})
S = json.load(open('results/bench_summary.json')); ST = S['static']; os.makedirs('paper/figures', exist_ok=True)
M = [('scan_part', 'exact scan (per-label)', 'k', 's', '-'), ('scan_mask', 'exact scan (bitmap)', '#888', 'x', ':'),
     ('ivf', 'IVF-Flat + bitmap', '#1f5fa8', 'o', '-'), ('postfilter', 'HNSW post-filter', '#999', '^', '--'),
     ('acorn', 'ACORN-$\\gamma$', '#d07a00', 'D', '-'), ('acorn1', 'ACORN-1', '#f0b050', 'd', '--'),
     ('fdiskann', 'Filtered-DiskANN', '#2a8a4a', 'v', '-'), ('curator', 'Curator', '#6a3d9a', 'X', '-'), ('rig', 'RIG (ours)', '#8E2B3B', 'P', '-')]
get = lambda wl, m, k: (ST.get(wl, {}).get(m) or {}).get(k)

# Fig A: scaling at fixed s=0.1, beta=0.9 -- QPS at recall 0.95 and 0.99 vs n
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
ns = [('deep1m', 1e6), ('deep3m', 3.16e6), ('deep10m', 1e7)]
for ax, t in zip(axes, ('0.95', '0.99')):
    for m, lab, c, mk, ls in M:
        xs = [n for ds, n in ns if get(f'{ds}/b0.9_s0.1', m, f'qps@{t}')]
        ys = [get(f'{ds}/b0.9_s0.1', m, f'qps@{t}') for ds, n in ns if get(f'{ds}/b0.9_s0.1', m, f'qps@{t}')]
        if xs: ax.plot(xs, ys, color=c, marker=mk, ls=ls, lw=1.1, ms=4, label=lab)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('$n$ (DEEP subsets; $s{=}0.1$, $\\beta{=}0.9$)')
    ax.set_ylabel(f'QPS at recall {t} (1 thread)'); ax.set_title(f'recall $\\geq${t}', fontsize=8)
axes[0].legend(fontsize=5.8, frameon=False, ncol=2, loc='lower left')
plt.tight_layout(); plt.savefig('paper/figures/eab_scaling.pdf'); plt.close()

# Fig B: speedup over the per-label scan vs n at FIXED admissible-set size s*n (DEEP subsets, beta=0.9)
FIX = {'1e4': [('deep1m', 'b0.9_s0.01', 1e6), ('deep3m', 'b0.9_s0.003125', 3.16e6), ('deep10m', 'b0.9_s0.001', 1e7)],
       '1e5': [('deep1m', 'b0.9_s0.1', 1e6), ('deep3m', 'b0.9_s0.03125', 3.16e6), ('deep10m', 'b0.9_s0.01', 1e7)]}
fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.0), sharex=True)
for col, (A, wls) in enumerate(FIX.items()):
    for row, t in enumerate(('0.95', '0.99')):
        ax = axes[row][col]
        for m, lab, c, mk, ls in M:
            if m.startswith('scan'): continue
            xs, ys, xs0 = [], [], []
            for ds, sc, n in wls:
                wl = f'{ds}/{sc}'
                if ST.get(wl, {}).get(m) is None or not get(wl, 'scan_part', f'qps@{t}'): continue
                q = get(wl, m, f'qps@{t}')
                if q: xs.append(n); ys.append(q / get(wl, 'scan_part', f'qps@{t}'))
                else: xs0.append(n)
            if xs: ax.plot(xs, ys, color=c, marker=mk, ls=ls, lw=1.1, ms=4, label=lab)
            if xs0: ax.scatter(xs0, [0.03] * len(xs0), color=c, marker=mk, s=14, facecolors='none', lw=0.8)
        ax.axhline(1, color='k', lw=0.6, ls=':'); ax.set_xscale('log'); ax.set_yscale('log'); ax.set_ylim(0.02, 30)
        ax.set_title(f'$s\\cdot n={A[0]}0^{{{A[-1]}}}$, recall $\\geq${t}', fontsize=8)
        if col == 0: ax.set_ylabel('QPS / exact-scan QPS')
        if row == 1: ax.set_xlabel('$n$ (DEEP subsets, $\\beta{=}0.9$)')
from matplotlib.lines import Line2D
hs = [Line2D([], [], color=c, marker=mk, ls=ls, lw=1.1, ms=4, label=lab) for m, lab, c, mk, ls in M if not m.startswith('scan')]
fig.legend(handles=hs, fontsize=6, frameon=False, loc='upper center', ncol=6)
plt.tight_layout(rect=(0, 0, 1, 0.95)); plt.savefig('paper/figures/eab_crossover.pdf'); plt.close()

# Fig C: correlation on SIFT1M (s=0.1): max recall and QPS@0.95 vs beta
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.2))
betas = ['0.0', '0.5', '0.9', '1.0']
for m, lab, c, mk, ls in M:
    if m == 'scan_mask': continue
    r = [get(f'sift1m/b{b}_s0.1', m, 'max_recall') for b in betas]; q = [get(f'sift1m/b{b}_s0.1', m, 'qps@0.95') for b in betas]
    if any(v is not None for v in r): axes[0].plot([float(b) for b in betas], [v if v is not None else np.nan for v in r], color=c, marker=mk, ls=ls, lw=1.1, ms=4, label=lab)
    if any(v is not None for v in q): axes[1].plot([float(b) for b in betas], [v if v is not None else np.nan for v in q], color=c, marker=mk, ls=ls, lw=1.1, ms=4)
axes[0].set_ylabel('max recall@10 in sweep'); axes[1].set_ylabel('QPS at recall 0.95'); axes[1].set_yscale('log')
for ax in axes: ax.set_xlabel('label--geometry correlation $\\beta$ (SIFT1M, $s{=}0.1$)')
axes[0].legend(fontsize=5.8, frameon=False, loc='lower left')
plt.tight_layout(); plt.savefig('paper/figures/eab_correlation.pdf'); plt.close()
print('static figures written')

# Fig D: update streams -- per stream: max recall, QPS at recall 0.95, cumulative maintenance vs round
STR = S.get('streams', {})
names = [n for n in ('sift500k', 'deep5m', 'sift500k_adv') if n in STR]
if names:
    SM = [('scan_part', 'exact scan (per-label)', 'k', 's', '-'), ('ivf', 'IVF-Flat + bitmap', '#1f5fa8', 'o', '-'),
          ('postfilter', 'HNSW post-filter', '#999', '^', '--'), ('acorn', 'ACORN-$\\gamma$', '#d07a00', 'D', '-'),
          ('fdiskann_rebuild', 'Filtered-DiskANN (rebuilt)', '#2a8a4a', 'v', '-'), ('curator', 'Curator', '#6a3d9a', 'X', '-'), ('rig', 'RIG (ours)', '#8E2B3B', 'P', '-')]
    titles = {'sift500k': 'SIFT-500K (uniform deletes)', 'deep5m': 'DEEP-5M', 'sift500k_adv': 'SIFT-500K-adv (correlated deletes)'}
    fig, axes = plt.subplots(len(names), 3, figsize=(7.0, 1.75 * len(names) + 0.3), squeeze=False)
    for i, nm in enumerate(names):
        for m, lab, c, mk, ls in SM:
            d = STR[nm].get(m)
            if not d: continue
            rr = d['rounds']; x = [r['round'] for r in rr]
            axes[i][0].plot(x, [r['max_recall'] for r in rr], color=c, marker=mk, ls=ls, lw=1.0, ms=3, label=lab)
            q = [r['qps@0.95'] if r['qps@0.95'] else np.nan for r in rr]
            axes[i][1].plot(x, q, color=c, marker=mk, ls=ls, lw=1.0, ms=3)
            R = max(x)
            if m == 'fdiskann_rebuild':   # cost of staying fresh every round = one rebuild per round
                per = float(np.mean([r.get('rebuild_s') or 0 for r in rr]))
                axes[i][2].plot([1, R], [per, per * R], color=c, ls='--', lw=1.0)
            else:
                pts = [(r['round'], r['cum_maint_s']) for r in rr if r['round'] > 0 and r['cum_maint_s']]
                if pts: axes[i][2].plot(*zip(*pts), color=c, marker=mk, ls=ls, lw=1.0, ms=3)
        axes[i][0].set_ylabel(titles[nm].split(' (')[0] + '\nmax recall', fontsize=7); axes[i][1].set_ylabel('QPS @ 0.95', fontsize=7)
        axes[i][1].set_yscale('log'); axes[i][2].set_ylabel('cum. maintenance (s)', fontsize=7); axes[i][2].set_yscale('log')
        from matplotlib.ticker import LogLocator
        for ax in axes[i][1:]: ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    for ax in axes[-1]: ax.set_xlabel('round')
    h, l = axes[0][0].get_legend_handles_labels(); fig.legend(h, l, fontsize=6, frameon=False, loc='upper center', ncol=6)
    plt.tight_layout(rect=(0, 0, 1, 0.96)); plt.savefig('paper/figures/eab_streams.pdf'); plt.close(); print('stream figure written')
