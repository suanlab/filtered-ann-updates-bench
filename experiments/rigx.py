"""rigx — thin ctypes wrapper around the C++ RIG kernels (output arrays are allocated per call).

Holds a prebuilt partition (CSR of leaves and of point->leaf memberships, int64 offsets) and
calls rig_search64 directly, so a call costs only the search (no per-call list->CSR conversion,
which dominated single-query timings in the earlier harness). Also wraps the C++ carver and the
pivot-retaining dynamic path used by the TCO stream experiment."""
import ctypes, time
import numpy as np

_here = __file__.rsplit('/', 2)[0]
lib = ctypes.CDLL(f'{_here}/src/librig.so')
clib = ctypes.CDLL(f'{_here}/src/libcarve.so')
f32p = ctypes.POINTER(ctypes.c_float); i32p = ctypes.POINTER(ctypes.c_int32)
i64p = ctypes.POINTER(ctypes.c_int64); u8p = ctypes.POINTER(ctypes.c_uint8); i8p = ctypes.POINTER(ctypes.c_int8)
P = lambda a, t: a.ctypes.data_as(t)

lib.rig_search64.argtypes = [f32p, ctypes.c_int, ctypes.c_int, i64p, i32p, i64p, i32p, i64p, i32p, i32p, ctypes.c_int,
                             u8p, ctypes.c_int, f32p, ctypes.c_int, ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p, i64p]
lib.rig_set_quant.argtypes = [i8p, f32p, f32p]
lib.rig_set_leafmajor.argtypes = [i8p, f32p, f32p, i32p, i64p]
lib.brute_masked.argtypes = [f32p, ctypes.c_int, ctypes.c_int, u8p, f32p, ctypes.c_int, ctypes.c_int, ctypes.c_int, i32p]

clib.carve_run.restype = ctypes.c_void_p
clib.carve_run.argtypes = [f32p, ctypes.c_int64, ctypes.c_int, i32p, ctypes.c_int64, ctypes.c_uint64, ctypes.c_int,
                           ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clib.carve_sizes.argtypes = [ctypes.c_void_p, i64p, i64p, i64p, i64p]
clib.carve_copy.argtypes = [ctypes.c_void_p, i64p, i32p, i64p, i32p, i32p]
clib.carve_assign.restype = ctypes.c_int64
clib.carve_assign.argtypes = [ctypes.c_void_p, f32p, i32p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, i64p]
clib.carve_free.argtypes = [ctypes.c_void_p]
clib.carve_dyn_open.restype = ctypes.c_void_p
clib.carve_dyn_open.argtypes = [ctypes.c_void_p, i32p, ctypes.c_int64]
clib.carve_dyn_insert.restype = ctypes.c_int64
clib.carve_dyn_insert.argtypes = [ctypes.c_void_p, f32p, i32p, ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_int]
clib.carve_dyn_delete.argtypes = [ctypes.c_void_p, i32p, ctypes.c_int64]
clib.carve_dyn_ghost_pivots.restype = ctypes.c_int64
clib.carve_dyn_ghost_pivots.argtypes = [ctypes.c_void_p]
clib.carve_dyn_export.argtypes = [ctypes.c_void_p, i64p, i64p, i32p, i64p, i32p, ctypes.c_int]
clib.carve_dyn_nleaves.restype = ctypes.c_int64
clib.carve_dyn_nleaves.argtypes = [ctypes.c_void_p]
clib.carve_dyn_close.argtypes = [ctypes.c_void_p]


def _copy_partition(h, nrows):
    nl, ns, nn, nl0 = (ctypes.c_int64() for _ in range(4))
    clib.carve_sizes(h, ctypes.byref(nl), ctypes.byref(ns), ctypes.byref(nn), ctypes.byref(nl0))
    out = dict(leaf_off=np.empty(nl.value + 1, np.int64), leaf_ids=np.empty(ns.value, np.int32),
               mem_off=np.empty(nrows + 1, np.int64), mem_ids=np.empty(ns.value, np.int32),
               leaders=np.empty(nl0.value, np.int32))
    clib.carve_copy(h, P(out['leaf_off'], i64p), P(out['leaf_ids'], i32p), P(out['mem_off'], i64p),
                    P(out['mem_ids'], i32p), P(out['leaders'], i32p))
    out['nodes'] = nn.value
    return out


def carve(X, seed=44, f0=10, f1=3, Cmax=1024, Lcap=1000, T=60, min_leaf=8, threads=64, keep_handle=False, subset=None):
    """Run the C++ carver (optionally over a subset of rows). Returns CSR dict + build time."""
    X = np.ascontiguousarray(X, dtype=np.float32); n, d = X.shape
    sub = None if subset is None else np.ascontiguousarray(subset, np.int32)
    t0 = time.time()
    h = clib.carve_run(P(X, f32p), n, d, None if sub is None else P(sub, i32p), 0 if sub is None else len(sub),
                       seed, f0, f1, Cmax, Lcap, T, min_leaf, threads)
    t = time.time() - t0
    out = _copy_partition(h, n); out['carve_s'] = t
    if keep_handle: out['handle'] = h
    else: clib.carve_free(h)
    return out


def from_leaves(leaves, leaders, n):
    """Build the same CSR structure from a Python list-of-arrays partition (python carver)."""
    leaf_off = np.zeros(len(leaves) + 1, np.int64); leaf_off[1:] = np.cumsum([len(l) for l in leaves])
    leaf_ids = np.concatenate(leaves).astype(np.int32)
    cnt = np.bincount(leaf_ids, minlength=n)
    mem_off = np.zeros(n + 1, np.int64); mem_off[1:] = np.cumsum(cnt)
    order = np.argsort(leaf_ids, kind='stable')
    leaf_of_slot = np.repeat(np.arange(len(leaves), dtype=np.int32), np.diff(leaf_off))
    mem_ids = leaf_of_slot[order].astype(np.int32)
    return dict(leaf_off=leaf_off, leaf_ids=leaf_ids, mem_off=mem_off, mem_ids=mem_ids,
                leaders=np.asarray(leaders, np.int32))


class RigIndex:
    """A RIG over X with int8 codes and (optionally) the leaf-major layout installed."""
    def __init__(self, X, part, leafmajor=True):
        self.X = np.ascontiguousarray(X, dtype=np.float32); self.n, self.d = self.X.shape
        self.p = part
        self._goff = np.zeros(self.n + 1, np.int64); self._gids = np.zeros(1, np.int32)   # no explicit graph
        sx = (np.abs(self.X).max(1) / 127.0).astype(np.float32); sx[sx == 0] = 1.0
        self.sx = sx; self.X8 = np.ascontiguousarray(np.rint(self.X / sx[:, None]).astype(np.int8))
        self.xn = np.ascontiguousarray((self.X * self.X).sum(1).astype(np.float32))
        self.leafmajor = leafmajor
        if leafmajor: self._layout()
        self.install()

    def _layout(self):
        order = self.p['leaf_ids'].astype(np.int64)
        self.LX8 = np.ascontiguousarray(self.X8[order]); self.Lsx = np.ascontiguousarray(self.sx[order])
        self.Lxn = np.ascontiguousarray(self.xn[order]); self.Lid = np.ascontiguousarray(self.p['leaf_ids'])

    def nbytes(self, live=None):
        """Resident bytes of the search-time state. Per-point arrays are counted for `live` rows
        (default: all rows) -- the dynamic path keeps a capacity-sized Xall, which a production
        layout would compact."""
        n = self.n if live is None else live; slots = len(self.p['leaf_ids'])
        per_pt = self.d * 4 + self.d + 8 + 8                 # fp32 rerank copy, int8 code, scale+norm, mem_off
        b = n * per_pt + slots * 8 + len(self.p['leaf_off']) * 8   # leaf_ids + mem_ids, leaf_off
        if self.leafmajor: b += slots * (self.d + 12)        # per-slot int8 copy, scale, norm, id
        return int(b)

    def install(self):
        """(Re)install this index's arrays as the kernel's global state (see rig.cpp contract)."""
        lib.rig_set_quant(P(self.X8, i8p), P(self.sx, f32p), P(self.xn, f32p))
        if self.leafmajor:
            lib.rig_set_leafmajor(P(self.LX8, i8p), P(self.Lsx, f32p), P(self.Lxn, f32p), P(self.Lid, i32p),
                                  P(self.p['leaf_off'], i64p))

    def entry_set(self, frac):
        """Entry candidates: the `frac` share of points with the smallest hash priority (a fixed,
        data-independent sample). Top-level leaders are the frac~0.05 special case at small n."""
        n = self.n; k = max(len(self.p['leaders']), int(frac * n))
        pr = (np.arange(n, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(44)
        return np.ascontiguousarray(np.sort(np.argpartition(pr, k)[:k]).astype(np.int32))

    def search(self, Q, mask=None, ef=32, k=10, threads=64, t_entry=4, filtered_nav=False, entries=None):
        Q = np.ascontiguousarray(Q, dtype=np.float32); nq = len(Q)
        use_mask = mask is not None
        if use_mask and mask.dtype == np.uint8 and mask.flags.c_contiguous: M = mask   # no per-call copy (thread pools)
        else: M = np.ascontiguousarray((mask if use_mask else np.ones(1, bool)).astype(np.uint8))
        out = np.empty((nq, k), np.int32); nc = np.empty(nq, np.int64)
        mode = (8 if filtered_nav else 7) if self.leafmajor else 5
        p = self.p
        ent = p['leaders'] if entries is None else entries
        t0 = time.perf_counter()
        lib.rig_search64(P(self.X, f32p), self.n, self.d, P(p['leaf_off'], i64p), P(p['leaf_ids'], i32p),
                         P(p['mem_off'], i64p), P(p['mem_ids'], i32p), P(self._goff, i64p), P(self._gids, i32p),
                         P(ent, i32p), len(ent), P(M, u8p), int(use_mask), P(Q, f32p), nq,
                         1.2, 32, 256, ef, k, t_entry, mode, threads, P(out, i32p), P(nc, i64p))
        return out, nc, time.perf_counter() - t0


class DynamicRig:
    """Pivot-retaining dynamic RIG over a fixed row space Xall (capacity = len(Xall)).
    The initial carve covers rows `init_ids`; insert/delete take row ids; refresh() compacts the
    partition and returns (RigIndex, compaction seconds). Leaders never change on this path, so
    state identity with a fresh carve is NOT maintained here (the exact protocol is exp5b)."""
    def __init__(self, Xall, init_ids, threads=16, f0=10, f1=3):
        self.Xall = np.ascontiguousarray(Xall, np.float32); self.cap, self.d = self.Xall.shape
        self.f0, self.f1, self.threads = f0, f1, threads
        ids = np.ascontiguousarray(np.sort(np.asarray(init_ids)).astype(np.int32))
        t0 = time.time()
        self.h = clib.carve_run(P(self.Xall, f32p), self.cap, self.d, P(ids, i32p), len(ids), 44, f0, f1,
                                1024, 1000, 60, 8, threads)
        self.carve_s = time.time() - t0
        self.leaders0 = _copy_partition(self.h, self.cap)['leaders']
        self.dh = clib.carve_dyn_open(self.h, P(ids, i32p), len(ids))
        self.alive = np.zeros(self.cap, bool); self.alive[ids] = True
        sx = (np.abs(self.Xall).max(1) / 127.0).astype(np.float32); sx[sx == 0] = 1.0
        self.sx = sx; self.X8 = np.ascontiguousarray(np.rint(self.Xall / sx[:, None]).astype(np.int8))
        self.xn = np.ascontiguousarray((self.Xall * self.Xall).sum(1).astype(np.float32))

    def insert(self, ids):
        ids = np.ascontiguousarray(ids, np.int32); self.alive[ids] = True
        return clib.carve_dyn_insert(self.dh, P(self.Xall, f32p), P(ids, i32p), len(ids), self.f0, self.f1, self.threads)

    def delete(self, ids):
        ids = np.ascontiguousarray(ids, np.int32); self.alive[ids] = False
        clib.carve_dyn_delete(self.dh, P(ids, i32p), len(ids))

    def ghost_pivots(self):
        return int(clib.carve_dyn_ghost_pivots(self.dh))

    def live_entries(self, frac=0.01):
        """Entry candidates restricted to live points: the hash-priority sample of `frac` of all
        rows, intersected with the alive set. Deleted leaders stay as routing pivots but must never
        be search entry points (their memberships are empty after compaction)."""
        k = max(1000, int(frac * self.cap))
        pr = (np.arange(self.cap, dtype=np.uint64) * np.uint64(0x9E3779B97F4A7C15)) ^ np.uint64(44)
        cand = np.sort(np.argpartition(pr, k)[:k])
        return np.ascontiguousarray(cand[self.alive[cand]].astype(np.int32))

    def refresh(self, leafmajor=True):
        """Compaction: live-only CSR (+ leaf-major layout unless leafmajor=False, which keeps the
        gathered int8 layout without per-leaf code copies). Returns (RigIndex, seconds)."""
        t0 = time.time()
        ns = ctypes.c_int64(); nl = int(clib.carve_dyn_nleaves(self.dh))
        clib.carve_dyn_export(self.dh, ctypes.byref(ns), None, None, None, None, self.threads)
        part = dict(leaf_off=np.empty(nl + 1, np.int64), leaf_ids=np.empty(ns.value, np.int32),
                    mem_off=np.empty(self.cap + 1, np.int64), mem_ids=np.empty(ns.value, np.int32),
                    leaders=self.leaders0)
        clib.carve_dyn_export(self.dh, ctypes.byref(ns), P(part['leaf_off'], i64p), P(part['leaf_ids'], i32p),
                              P(part['mem_off'], i64p), P(part['mem_ids'], i32p), self.threads)
        idx = RigIndex.__new__(RigIndex)
        idx.X, idx.n, idx.d, idx.p = self.Xall, self.cap, self.d, part
        idx._goff = np.zeros(self.cap + 1, np.int64); idx._gids = np.zeros(1, np.int32)
        idx.sx, idx.X8, idx.xn, idx.leafmajor = self.sx, self.X8, self.xn, leafmajor
        if leafmajor: idx._layout()
        idx.install()
        return idx, time.time() - t0


def brute(X, Q, mask=None, k=10, threads=64):
    X = np.ascontiguousarray(X, dtype=np.float32); Q = np.ascontiguousarray(Q, dtype=np.float32)
    n, d = X.shape; M = np.ascontiguousarray((np.ones(n, bool) if mask is None else mask).astype(np.uint8))
    out = np.empty((len(Q), k), np.int32)
    t0 = time.perf_counter(); lib.brute_masked(P(X, f32p), n, d, P(M, u8p), P(Q, f32p), len(Q), k, threads, P(out, i32p))
    return out, time.perf_counter() - t0


def recall(out, gt, k=10):
    return float(np.mean([len(set(out[i, :k].tolist()) & set(gt[i, :k].tolist())) / k for i in range(len(gt))]))
