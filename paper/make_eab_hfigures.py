#!/usr/bin/env python3
"""Figures for the follow-up experiments (main env): paper/figures/eab_law.pdf (H1: crossover admissible-set size
vs n per method, with the sqrt(n) bound's slope for reference) and paper/figures/eab_percolation.pdf (H3: per-label
maximum recall against the effective admissible degree; maximum recall by round with and without repair).
Reads results/bench_h_summary.json and the raw H3 stream results."""
import json, glob, os, sys
sys.path.insert(0, 'bench')
import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 8, 'axes.spines.top': False, 'axes.spines.right': False, 'figure.dpi': 150, 'pdf.fonttype': 42, 'ps.fonttype': 42})
S = json.load(open('results/bench_h_summary.json'))
LAB = {'acorn': 'tombstone', 'acorn_repair': 'level-0 repair', 'acorn_repair_all': 'all-level repair'}
STY = {'ivf': ('IVF-Flat + bitmap', 'C0', 'o'), 'acorn': ('ACORN-$\\gamma$', 'C1', 's'), 'postfilter': ('HNSW post-filter', 'C2', '^'),
       'curator': ('Curator', 'C4', 'D'), 'ivf_part': ('label-partitioned IVF', 'C3', 'v'), 'fdiskann': ('Filtered-DiskANN', 'C5', 'P')}
NS = {'deep300k': 316_228, 'deep1m': 1_000_000, 'deep3m': 3_162_278, 'deep10m': 10_000_000}

def law():
    H = S.get('H1')
    if not H: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True)
    for ax, t in zip(axs, ('0.95', '0.99')):
        rows = H['per_n'].get(t, {})
        for m, (lab, c, mk) in STY.items():
            xs, ys, cen = [], [], []
            for ds, r in rows.items():
                if m in r: xs.append(NS[ds]); ys.append(r[m]['m_star']); cen.append(r[m]['censor'])
            if not xs: continue
            o = np.argsort(xs); xs, ys, cen = np.array(xs)[o], np.array(ys)[o], np.array(cen)[o]
            ok = cen == 'interp'
            ax.plot(xs[ok], ys[ok], marker=mk, color=c, lw=1.2, ms=4, label=lab)
            ax.plot(xs[~ok], ys[~ok], marker=mk, color=c, lw=0, ms=4, mfc='none')
            ex = H['fit'].get(t, {}).get(m)
            if ex: ax.annotate(f"$n^{{{ex['exponent']:.2f}}}$", (xs[ok][-1], ys[ok][-1]), xytext=(4, 0), textcoords='offset points', color=c, fontsize=7, va='center')
        x = np.array([3e5, 1e7]); ax.plot(x, 3e4 * np.sqrt(x / 1e6), color='0.6', ls=':', lw=1); ax.text(1.2e7, 3e4 * np.sqrt(10), '$\\sqrt{n}$', color='0.4', fontsize=7)
        ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xlabel('n (DEEP subsets, $\\beta{=}0$)'); ax.set_title(f'recall $\\geq${t}', fontsize=8)
    axs[0].set_ylabel('crossover $|A|$ (index = scan)')
    h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='upper center', ncol=6, frameon=False, fontsize=7, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout(); fig.savefig('paper/figures/eab_law.pdf', bbox_inches='tight'); plt.close(fig)

def percolation():
    from analyze import passes
    rows = []
    for root in passes('h3') + passes('h3b'):
        for f in glob.glob(f'{root}/stream_*/acorn*.json'):
            d = json.load(open(f)); nm = f.split('/')[-2][len('stream_'):]
            for r in d['rounds']:
                if 'perc' not in r or 'sweep' not in r: continue
                for lab in r['perc']:
                    b = str(lab['label']); rec = max(s['recall_by_label'].get(b, np.nan) for s in r['sweep'])
                    rows.append((nm, d['method'], r['round'], lab['deg_mean'], lab['lcc_frac'], rec))
    if not rows: return
    fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.4))
    for nm, meth, c, mk in (('sift500k', 'acorn', 'C0', 'o'), ('sift500k_adv', 'acorn', 'C3', 's'), ('sift500k_adv', 'acorn_repair', 'C2', '^'), ('sift500k_adv', 'acorn_repair_all', 'C4', 'D')):
        R = [x for x in rows if x[0] == nm and x[1] == meth]
        if R: axs[0].scatter([x[3] for x in R], [x[5] for x in R], s=10, color=c, marker=mk, alpha=0.7, label=f"{nm.replace('sift500k_adv', 'SIFT-500K-adv').replace('sift500k', 'SIFT-500K')} {LAB[meth]}")
    axs[0].set_xscale('symlog', linthresh=1); axs[0].set_xlabel('mean effective admissible degree (label, round)'); axs[0].set_ylabel('max recall of the label')
    axs[0].axvline(1, color='0.6', ls=':', lw=1); axs[0].legend(frameon=False, fontsize=6.5)
    S3 = S.get('H3', {}).get('streams', {})
    for nm, ls in (('sift500k_adv', '-'), ('sift500k', '--')):
        for meth, c in (('acorn', 'C3'), ('acorn_repair', 'C2'), ('acorn_repair_all', 'C4')):
            d = S3.get(nm, {}).get(meth)
            if d: axs[1].plot([r[0] for r in d['rounds']], [r[1] for r in d['rounds']], ls=ls, color=c, marker='o', ms=3,
                              label=f"{nm.replace('sift500k_adv', 'SIFT-500K-adv').replace('sift500k', 'SIFT-500K')} {LAB[meth]}")
    axs[1].set_xlabel('round'); axs[1].set_ylabel('max recall'); axs[1].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(); fig.savefig('paper/figures/eab_percolation.pdf', bbox_inches='tight'); plt.close(fig)

law(); percolation(); print('follow-up figures written')
