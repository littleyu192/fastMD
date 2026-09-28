"""Fixed-capacity, capture-safe neighbour lists for GPU-resident MACE MD.

MACE adaptation of matris ``FixedCapacityGraphBuilder`` (atom-graph half only),
extended so that the Verlet-candidate rebuild itself is capture-safe.

Convention (MACE / matscipy "ijS"): edge (sender s, receiver r, integer image U)
has ``vec = x[r] - x[s] + U @ cell``; ``edge_index[0] = s``, ``edge_index[1] = r``.
Positions are the *unwrapped* NHC positions (never re-wrapped); U is expressed
for the unwrapped positions, so it stays valid between candidate rebuilds.

Per MD step, :meth:`FixedCapacityNeighbors.step` enqueues 2 Triton launches with one
program per receiver atom (no host sync, fixed shapes -> CUDA-graph capturable):

1. ``_k1_kernel``: rebuild decision ``need = force_flag | policy(max displacement since
   the last build)`` (every program computes the same value from the same data;
   program 0 publishes it in a device control word), model-dtype copy of the float64
   positions, and per-receiver counts:
   * ``need``: brute-force count over all senders and the needed periodic images
     (positions wrapped in-kernel) of candidates (``d < r_cand = r_max + skin + 2 eps``)
     and of model edges (``d < r_max + eps``);
   * otherwise (filter mode): number of the receiver's candidates within
     ``r_max + eps`` at the current positions.
2. ``_k2_kernel``: offsets = exclusive prefix sums of the K1 counts (computed per
   program), then
   * ``need``: the brute force is repeated with identical arithmetic and writes the new
     candidate list (receiver-major, deterministic order) and the model edges, and
     updates the reference positions;
   * otherwise ``mode="filter"`` (option 2): re-filter the receiver's candidates and
     write the compacted model edges;
   * ``mode="candidates"`` (option 1): the candidates are written directly into the
     model edge buffers at a rebuild and nothing happens otherwise. Valid because MACE
     edges with d >= r_max contribute exactly 0.
   Remaining slots up to ``E_cap`` get padding edges; stats are updated.

Rebuild policies (``rebuild=``):
* ``"device"`` (default): rebuild inside the step when the max displacement since
  the last build exceeds ``skin/2 - eps`` (the Verlet criterion is then never
  violated; no host involvement).
* ``"host"``: rebuild only when the host sets the force flag
  (:meth:`request_rebuild`); the runner checks the displacement stat after each
  chunk (rollback on violation, proactive refresh) -- matris protocol.
* ``"always"``: rebuild every step (exact brute-force list each step; use with
  ``skin=0``).

Distance tests run in float32 with a safety margin ``eps`` (default 1e-3 A): the
edge set given to the model is a superset of the exact ``d < r_max`` set (the
extra edges have r_max <= d < r_max + eps and contribute exactly 0).  Candidates
use ``r_max + skin + 2 eps`` so that every pair with ``d < r_max + eps`` is a
candidate whenever the displacement since the build is <= skin/2.

Padding edges: self-loops ``i -> i`` spread over all real atoms (slot % N) with
image ``PAD_REP * a_k`` (a_k = longest lattice vector, length >= 2 r_max): finite,
non-zero length > r_max -> exactly zero contribution (docs/mace_internals.md s6).

The model edge buffers exist per capacity *tier* (``E_cap``); every tier has its
own static model-input dict (shapes differ), all tiers share the candidate list,
the reference positions and the model positions.

Device stats buffer (float64 [N_STATS], zeroed by the runner at chunk start):
    STAT_EDGES_MAX   max model-edge count needed (overflow iff > E_cap of the tier)
    STAT_CAND_MAX    max candidate total at a rebuild (overflow iff > C_cap)
    STAT_DISP2_EFF   max squared displacement of the list actually used
    STAT_REBUILDS    number of candidate rebuilds
    STAT_ENERGY      energy after the last step (set by the runner)
    STAT_DISP2_RAW   max squared displacement before rebuild decisions
    STAT_EDGES_SUM   sum over steps of the model-edge count
    STAT_STEPS       number of neighbour steps
    STAT_NL_ERR      internal consistency errors (count/fill mismatch); must be 0

Limitations: fully periodic cells only (pbc all True); brute force is
O(N^2 * images) per rebuild (fine for N up to a few thousand).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence

import numpy as np
import torch

import triton
import triton.language as tl

STAT_EDGES_MAX = 0
STAT_CAND_MAX = 1
STAT_DISP2_EFF = 2
STAT_REBUILDS = 3
STAT_ENERGY = 4
STAT_DISP2_RAW = 5
STAT_EDGES_SUM = 6
STAT_STEPS = 7
STAT_NL_ERR = 8
N_STATS = 9

CTRL_NEED, CTRL_FORCE = 0, 1
_REBUILD_CODES = {"host": 0, "device": 1, "always": 2}

# float32 parameter buffer layout
_P_CELL, _P_INV, _P_RC2, _P_RF2, _P_THR2 = 0, 9, 18, 19, 20
# model-dtype parameter buffer layout
_M_CELL, _M_PAD_US, _M_PAD_SH = 0, 9, 12


class CapacityOverflow(RuntimeError):
    def __init__(self, kind: str, required: int, capacity: int):
        super().__init__(f"{kind} capacity overflow: need {required} > {capacity}")
        self.kind = kind
        self.required = int(required)
        self.capacity = int(capacity)


def round_up(x: int, m: int) -> int:
    return int(-(-int(x) // m) * m)


def bucket_capacity(required: int, headroom: float = 0.0, step: int = 64,
                    minimum: int = 64) -> int:
    """required * (1 + headroom) rounded up to a multiple of ``step``."""
    return max(minimum, round_up(math.ceil(required * (1.0 + headroom)), step))


# --------------------------------------------------------------------------- #
# Triton kernels
# --------------------------------------------------------------------------- #
@triton.jit
def _load_cell(prm_ptr):
    c00 = tl.load(prm_ptr + 0)
    c01 = tl.load(prm_ptr + 1)
    c02 = tl.load(prm_ptr + 2)
    c10 = tl.load(prm_ptr + 3)
    c11 = tl.load(prm_ptr + 4)
    c12 = tl.load(prm_ptr + 5)
    c20 = tl.load(prm_ptr + 6)
    c21 = tl.load(prm_ptr + 7)
    c22 = tl.load(prm_ptr + 8)
    return c00, c01, c02, c10, c11, c12, c20, c21, c22


@triton.jit
def _wrap(x, y, z, prm_ptr):
    """float32 wrap into the cell: returns wrapped position and floor(frac)."""
    c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell(prm_ptr)
    i00, i01, i02, i10, i11, i12, i20, i21, i22 = _load_cell(prm_ptr + 9)
    ox = tl.floor(x * i00 + y * i10 + z * i20)
    oy = tl.floor(x * i01 + y * i11 + z * i21)
    oz = tl.floor(x * i02 + y * i12 + z * i22)
    wx = x - (ox * c00 + oy * c10 + oz * c20)
    wy = y - (ox * c01 + oy * c11 + oz * c21)
    wz = z - (ox * c02 + oy * c12 + oz * c22)
    return wx, wy, wz, ox, oy, oz


@triton.jit
def _k1_kernel(pos_ptr, ref_ptr, mpos_ptr, ctrl_ptr, stat_ptr, prm_ptr, cnt_ptr, vcnt_ptr,
               cs_ptr, csh_ptr, N, C_CAP,
               BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
               NX: tl.constexpr, NY: tl.constexpr, NZ: tl.constexpr,
               POLICY: tl.constexpr, FILTER: tl.constexpr):
    """Grid N (program r = receiver). Rebuild decision (every program computes the same
    value from the same data), model positions, and per-receiver counts:
    need -> brute-force candidate count cnt[r] and edge count vcnt[r];
    else (filter mode) -> number of candidates of r within the filter cutoff."""
    r = tl.program_id(0)
    a = tl.arange(0, BLOCK_N)
    am = a < N
    px = tl.load(pos_ptr + a * 3 + 0, mask=am, other=0.0)
    py = tl.load(pos_ptr + a * 3 + 1, mask=am, other=0.0)
    pz = tl.load(pos_ptr + a * 3 + 2, mask=am, other=0.0)
    dx = (px - tl.load(ref_ptr + a * 3 + 0, mask=am, other=0.0)).to(tl.float32)
    dy = (py - tl.load(ref_ptr + a * 3 + 1, mask=am, other=0.0)).to(tl.float32)
    dz = (pz - tl.load(ref_ptr + a * 3 + 2, mask=am, other=0.0)).to(tl.float32)
    disp2 = tl.max(dx * dx + dy * dy + dz * dz, axis=0)
    force = tl.load(ctrl_ptr + 1)
    if POLICY == 2:
        need = force * 0 + 1
    elif POLICY == 1:
        need = tl.where((force != 0) | (disp2 > tl.load(prm_ptr + 20)), 1, 0)
    else:
        need = tl.where(force != 0, 1, 0)
    rx64 = tl.load(pos_ptr + r * 3 + 0)
    ry64 = tl.load(pos_ptr + r * 3 + 1)
    rz64 = tl.load(pos_ptr + r * 3 + 2)
    odt = mpos_ptr.dtype.element_ty
    tl.store(mpos_ptr + r * 3 + 0, rx64.to(odt))
    tl.store(mpos_ptr + r * 3 + 1, ry64.to(odt))
    tl.store(mpos_ptr + r * 3 + 2, rz64.to(odt))
    if r == 0:
        tl.store(ctrl_ptr + 0, need)
        d64 = disp2.to(tl.float64)
        tl.store(stat_ptr + 2, tl.maximum(tl.load(stat_ptr + 2), tl.where(need != 0, 0.0, d64)))
        tl.store(stat_ptr + 5, tl.maximum(tl.load(stat_ptr + 5), d64))
    rx = rx64.to(tl.float32)
    ry = ry64.to(tl.float32)
    rz = rz64.to(tl.float32)
    if need != 0:
        c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell(prm_ptr)
        rc2 = tl.load(prm_ptr + 18)
        rf2 = tl.load(prm_ptr + 19)
        wrx, wry, wrz, orx, ory, orz = _wrap(rx, ry, rz, prm_ptr)
        tot_c = tl.sum(tl.zeros([BLOCK_S], dtype=tl.int32), axis=0)
        tot_e = tot_c
        for sb in range(0, N, BLOCK_S):
            s = sb + tl.arange(0, BLOCK_S)
            ms = s < N
            sx = tl.load(pos_ptr + s * 3 + 0, mask=ms, other=0.0).to(tl.float32)
            sy = tl.load(pos_ptr + s * 3 + 1, mask=ms, other=0.0).to(tl.float32)
            sz = tl.load(pos_ptr + s * 3 + 2, mask=ms, other=0.0).to(tl.float32)
            wsx, wsy, wsz, osx, osy, osz = _wrap(sx, sy, sz, prm_ptr)
            bx = wrx - wsx
            by = wry - wsy
            bz = wrz - wsz
            for img in range(0, (2 * NX + 1) * (2 * NY + 1) * (2 * NZ + 1)):
                SX = (img // ((2 * NY + 1) * (2 * NZ + 1)) - NX).to(tl.float32)
                SY = ((img // (2 * NZ + 1)) % (2 * NY + 1) - NY).to(tl.float32)
                SZ = (img % (2 * NZ + 1) - NZ).to(tl.float32)
                vx = bx + (SX * c00 + SY * c10 + SZ * c20)
                vy = by + (SX * c01 + SY * c11 + SZ * c21)
                vz = bz + (SX * c02 + SY * c12 + SZ * c22)
                d2 = vx * vx + vy * vy + vz * vz
                self_img = (SX == 0.0) & (SY == 0.0) & (SZ == 0.0)
                ok = ms & ~((s == r) & self_img)
                tot_c += tl.sum((ok & (d2 < rc2)).to(tl.int32), axis=0)
                tot_e += tl.sum((ok & (d2 < rf2)).to(tl.int32), axis=0)
        tl.store(cnt_ptr + r, tot_c)
        tl.store(vcnt_ptr + r, tot_e)
    else:
        if FILTER:
            rf2 = tl.load(prm_ptr + 19)
            cnts = tl.load(cnt_ptr + a, mask=am, other=0)
            coff = tl.sum(tl.where(a < r, cnts, 0), axis=0)
            mine = tl.sum(tl.where(a == r, cnts, 0), axis=0)
            v = mine * 0
            for kb in range(0, mine, BLOCK_K):
                kk = kb + tl.arange(0, BLOCK_K)
                k = coff + kk
                m = (kk < mine) & (k < C_CAP)
                s = tl.load(cs_ptr + k, mask=m, other=0)
                vx = rx - tl.load(pos_ptr + s * 3 + 0, mask=m, other=0.0).to(tl.float32) \
                    + tl.load(csh_ptr + k * 3 + 0, mask=m, other=0.0).to(tl.float32)
                vy = ry - tl.load(pos_ptr + s * 3 + 1, mask=m, other=0.0).to(tl.float32) \
                    + tl.load(csh_ptr + k * 3 + 1, mask=m, other=0.0).to(tl.float32)
                vz = rz - tl.load(pos_ptr + s * 3 + 2, mask=m, other=0.0).to(tl.float32) \
                    + tl.load(csh_ptr + k * 3 + 2, mask=m, other=0.0).to(tl.float32)
                v += tl.sum((m & ((vx * vx + vy * vy + vz * vz) < rf2)).to(tl.int32), axis=0)
            tl.store(vcnt_ptr + r, v)


@triton.jit
def _k2_kernel(pos_ptr, ref_ptr, ctrl_ptr, stat_ptr, prm_ptr, prmm_ptr, cnt_ptr, vcnt_ptr,
               cs_ptr, cr_ptr, cus_ptr, csh_ptr, ctot_ptr, ei_ptr, us_ptr, sh_ptr,
               N, C_CAP, E_CAP,
               BLOCK_N: tl.constexpr, BLOCK_S: tl.constexpr, BLOCK_K: tl.constexpr,
               BLOCK_P: tl.constexpr, NX: tl.constexpr, NY: tl.constexpr,
               NZ: tl.constexpr, FILTER: tl.constexpr):
    """Grid N. Writes (need) the new candidate list + model edges straight from the
    brute force, or (filter mode) the re-filtered model edges from the candidate list;
    padding; stats. Offsets = exclusive prefix sums of the K1 counts."""
    r = tl.program_id(0)
    need = tl.load(ctrl_ptr)
    a = tl.arange(0, BLOCK_N)
    am = a < N
    cnts = tl.load(cnt_ptr + a, mask=am, other=0)
    coff = tl.sum(tl.where(a < r, cnts, 0), axis=0)
    cmine = tl.sum(tl.where(a == r, cnts, 0), axis=0)
    ctot = tl.sum(cnts, axis=0)
    if FILTER:
        vcnts = tl.load(vcnt_ptr + a, mask=am, other=0)
        eoff = tl.sum(tl.where(a < r, vcnts, 0), axis=0)
        emine = tl.sum(tl.where(a == r, vcnts, 0), axis=0)
        etot = tl.sum(vcnts, axis=0)
    else:
        eoff = coff
        emine = cmine
        etot = ctot
    if r == 0:
        tl.store(ctrl_ptr + 1, need * 0)  # clear the host force flag (K1 has read it)
    rx64 = tl.load(pos_ptr + r * 3 + 0)
    ry64 = tl.load(pos_ptr + r * 3 + 1)
    rz64 = tl.load(pos_ptr + r * 3 + 2)
    rx = rx64.to(tl.float32)
    ry = ry64.to(tl.float32)
    rz = rz64.to(tl.float32)
    mdt = us_ptr.dtype.element_ty
    cidt = cs_ptr.dtype.element_ty
    cbase = coff
    ebase = eoff
    if need != 0:
        c00, c01, c02, c10, c11, c12, c20, c21, c22 = _load_cell(prm_ptr)
        m00, m01, m02, m10, m11, m12, m20, m21, m22 = _load_cell(prmm_ptr)
        rc2 = tl.load(prm_ptr + 18)
        rf2 = tl.load(prm_ptr + 19)
        wrx, wry, wrz, orx, ory, orz = _wrap(rx, ry, rz, prm_ptr)
        for sb in range(0, N, BLOCK_S):
            s = sb + tl.arange(0, BLOCK_S)
            ms = s < N
            sx = tl.load(pos_ptr + s * 3 + 0, mask=ms, other=0.0).to(tl.float32)
            sy = tl.load(pos_ptr + s * 3 + 1, mask=ms, other=0.0).to(tl.float32)
            sz = tl.load(pos_ptr + s * 3 + 2, mask=ms, other=0.0).to(tl.float32)
            wsx, wsy, wsz, osx, osy, osz = _wrap(sx, sy, sz, prm_ptr)
            bx = wrx - wsx
            by = wry - wsy
            bz = wrz - wsz
            for img in range(0, (2 * NX + 1) * (2 * NY + 1) * (2 * NZ + 1)):
                SX = (img // ((2 * NY + 1) * (2 * NZ + 1)) - NX).to(tl.float32)
                SY = ((img // (2 * NZ + 1)) % (2 * NY + 1) - NY).to(tl.float32)
                SZ = (img % (2 * NZ + 1) - NZ).to(tl.float32)
                vx = bx + (SX * c00 + SY * c10 + SZ * c20)
                vy = by + (SX * c01 + SY * c11 + SZ * c21)
                vz = bz + (SX * c02 + SY * c12 + SZ * c22)
                d2 = vx * vx + vy * vy + vz * vz
                self_img = (SX == 0.0) & (SY == 0.0) & (SZ == 0.0)
                ok = ms & ~((s == r) & self_img)
                # unwrapped image: vec = x_r - x_s + U@cell,  U = S - o_r + o_s
                ux = ((SX - orx) + osx).to(mdt)
                uy = ((SY - ory) + osy).to(mdt)
                uz = ((SZ - orz) + osz).to(mdt)
                hx = ux * m00 + uy * m10 + uz * m20
                hy = ux * m01 + uy * m11 + uz * m21
                hz = ux * m02 + uy * m12 + uz * m22
                vc = ok & (d2 < rc2)
                vci = vc.to(tl.int32)
                rk = cbase + tl.cumsum(vci, axis=0) - 1
                wm = vc & (rk < C_CAP)
                tl.store(cs_ptr + rk, s.to(cidt), mask=wm)
                tl.store(cr_ptr + rk, (s * 0 + r).to(cidt), mask=wm)
                tl.store(cus_ptr + rk * 3 + 0, ux, mask=wm)
                tl.store(cus_ptr + rk * 3 + 1, uy, mask=wm)
                tl.store(cus_ptr + rk * 3 + 2, uz, mask=wm)
                tl.store(csh_ptr + rk * 3 + 0, hx, mask=wm)
                tl.store(csh_ptr + rk * 3 + 1, hy, mask=wm)
                tl.store(csh_ptr + rk * 3 + 2, hz, mask=wm)
                cbase += tl.sum(vci, axis=0)
                if FILTER:
                    ve = ok & (d2 < rf2)
                    vei = ve.to(tl.int32)
                    rk = ebase + tl.cumsum(vei, axis=0) - 1
                    wm = ve & (rk < E_CAP)
                    tl.store(ei_ptr + rk, s.to(tl.int64), mask=wm)
                    tl.store(ei_ptr + E_CAP + rk, (s * 0 + r).to(tl.int64), mask=wm)
                    tl.store(us_ptr + rk * 3 + 0, ux, mask=wm)
                    tl.store(us_ptr + rk * 3 + 1, uy, mask=wm)
                    tl.store(us_ptr + rk * 3 + 2, uz, mask=wm)
                    tl.store(sh_ptr + rk * 3 + 0, hx, mask=wm)
                    tl.store(sh_ptr + rk * 3 + 1, hy, mask=wm)
                    tl.store(sh_ptr + rk * 3 + 2, hz, mask=wm)
                    ebase += tl.sum(vei, axis=0)
        tl.store(ref_ptr + r * 3 + 0, rx64)
        tl.store(ref_ptr + r * 3 + 1, ry64)
        tl.store(ref_ptr + r * 3 + 2, rz64)
        bad = (cbase - coff) != cmine
        if FILTER:
            bad = bad | ((ebase - eoff) != emine)
        tl.atomic_add(stat_ptr + 8, tl.where(bad, 1.0, 0.0).to(tl.float64))
        if r == 0:
            tl.store(ctot_ptr, ctot)
            tl.store(stat_ptr + 1, tl.maximum(tl.load(stat_ptr + 1), ctot.to(tl.float64)))
            tl.store(stat_ptr + 3, tl.load(stat_ptr + 3) + 1.0)
    else:
        if FILTER:
            rf2 = tl.load(prm_ptr + 19)
            for kb in range(0, cmine, BLOCK_K):
                kk = kb + tl.arange(0, BLOCK_K)
                k = coff + kk
                m = (kk < cmine) & (k < C_CAP)
                s = tl.load(cs_ptr + k, mask=m, other=0)
                h0 = tl.load(csh_ptr + k * 3 + 0, mask=m, other=0.0)
                h1 = tl.load(csh_ptr + k * 3 + 1, mask=m, other=0.0)
                h2 = tl.load(csh_ptr + k * 3 + 2, mask=m, other=0.0)
                vx = rx - tl.load(pos_ptr + s * 3 + 0, mask=m, other=0.0).to(tl.float32) \
                    + h0.to(tl.float32)
                vy = ry - tl.load(pos_ptr + s * 3 + 1, mask=m, other=0.0).to(tl.float32) \
                    + h1.to(tl.float32)
                vz = rz - tl.load(pos_ptr + s * 3 + 2, mask=m, other=0.0).to(tl.float32) \
                    + h2.to(tl.float32)
                ve = m & ((vx * vx + vy * vy + vz * vz) < rf2)
                vei = ve.to(tl.int32)
                rk = ebase + tl.cumsum(vei, axis=0) - 1
                wm = ve & (rk < E_CAP)
                tl.store(ei_ptr + rk, s.to(tl.int64), mask=wm)
                tl.store(ei_ptr + E_CAP + rk, (s * 0 + r).to(tl.int64), mask=wm)
                tl.store(us_ptr + rk * 3 + 0, tl.load(cus_ptr + k * 3 + 0, mask=m, other=0.0), mask=wm)
                tl.store(us_ptr + rk * 3 + 1, tl.load(cus_ptr + k * 3 + 1, mask=m, other=0.0), mask=wm)
                tl.store(us_ptr + rk * 3 + 2, tl.load(cus_ptr + k * 3 + 2, mask=m, other=0.0), mask=wm)
                tl.store(sh_ptr + rk * 3 + 0, h0, mask=wm)
                tl.store(sh_ptr + rk * 3 + 1, h1, mask=wm)
                tl.store(sh_ptr + rk * 3 + 2, h2, mask=wm)
                ebase += tl.sum(vei, axis=0)
            bad = (ebase - eoff) != emine
            tl.atomic_add(stat_ptr + 8, tl.where(bad, 1.0, 0.0).to(tl.float64))
    # padding: slots [etot, E_CAP) spread over the programs (slot % N = pad atom)
    if FILTER or need != 0:
        pu0 = tl.load(prmm_ptr + 9)
        pu1 = tl.load(prmm_ptr + 10)
        pu2 = tl.load(prmm_ptr + 11)
        ps0 = tl.load(prmm_ptr + 12)
        ps1 = tl.load(prmm_ptr + 13)
        ps2 = tl.load(prmm_ptr + 14)
        start = tl.minimum(etot, E_CAP) + r
        for pb in range(start, E_CAP, N * BLOCK_P):
            k = pb + N * tl.arange(0, BLOCK_P)
            pm = k < E_CAP
            atom = (k % N).to(tl.int64)
            z = tl.zeros([BLOCK_P], dtype=mdt)
            tl.store(ei_ptr + k, atom, mask=pm)
            tl.store(ei_ptr + E_CAP + k, atom, mask=pm)
            tl.store(us_ptr + k * 3 + 0, z + pu0, mask=pm)
            tl.store(us_ptr + k * 3 + 1, z + pu1, mask=pm)
            tl.store(us_ptr + k * 3 + 2, z + pu2, mask=pm)
            tl.store(sh_ptr + k * 3 + 0, z + ps0, mask=pm)
            tl.store(sh_ptr + k * 3 + 1, z + ps1, mask=pm)
            tl.store(sh_ptr + k * 3 + 2, z + ps2, mask=pm)
    if r == 0:
        c64 = etot.to(tl.float64)
        tl.store(stat_ptr + 0, tl.maximum(tl.load(stat_ptr + 0), c64))
        tl.store(stat_ptr + 6, tl.load(stat_ptr + 6) + c64)
        tl.store(stat_ptr + 7, tl.load(stat_ptr + 7) + 1.0)


# --------------------------------------------------------------------------- #
# Host side
# --------------------------------------------------------------------------- #
@dataclass
class EdgeTier:
    e_cap: int
    edge_index: torch.Tensor  # [2, E_cap] int64
    unit_shifts: torch.Tensor  # [E_cap, 3] model dtype
    shifts: torch.Tensor  # [E_cap, 3] model dtype
    inputs: Dict[str, torch.Tensor]  # static model input dict


class FixedCapacityNeighbors:
    """Verlet candidate list + fixed-capacity MACE edge buffers (see module doc).

    Parameters
    ----------
    n_atoms, cell (3x3, rows = lattice vectors), r_max, skin
    mode : "filter" (option 2) or "candidates" (option 1)
    rebuild : "device" | "host" | "always"
    model_dtype : dtype of model-side buffers (positions, unit_shifts, shifts)
    c_cap : candidate capacity (filter mode; only Triton kernels see it -> be generous)
    e_caps : initial edge-capacity tiers (filter mode) / single capacity (candidates
             mode, where the model edges are the candidates, so c_cap = e_cap)
    eps : float32 distance-test margin (A)
    """

    def __init__(self, n_atoms: int, cell, r_max: float, skin: float, *,
                 node_attrs: torch.Tensor, head: int = 0, mode: str = "filter",
                 rebuild: str = "device", model_dtype: torch.dtype = torch.float64,
                 c_cap: int = 0, e_caps: Sequence[int] = (), pbc=(True, True, True),
                 device="cuda", eps: float = 1e-3, pad_margin: float = 2.0):
        if mode not in ("filter", "candidates"):
            raise ValueError(f"unknown mode {mode!r}")
        if rebuild not in _REBUILD_CODES:
            raise ValueError(f"unknown rebuild policy {rebuild!r}")
        if not all(bool(p) for p in pbc):
            raise NotImplementedError("only fully periodic cells are supported")
        self.n_atoms = n = int(n_atoms)
        self.device = torch.device(device)
        self.mode = mode
        self.rebuild = rebuild
        self.model_dtype = model_dtype
        self.r_max = float(r_max)
        self.skin = float(skin)
        self.eps = float(eps)
        self.r_cand = self.r_max + self.skin + 2.0 * self.eps
        self.r_filter = self.r_max + self.eps
        # rebuild threshold on max displacement (device policy)
        self.disp_thr = max(0.0, 0.5 * self.skin - self.eps)
        self.head = int(head)
        cell_np = np.asarray(cell, dtype=np.float64).reshape(3, 3)
        self.cell_np = cell_np
        vol = abs(np.linalg.det(cell_np))
        heights = [vol / np.linalg.norm(np.cross(cell_np[(k + 1) % 3], cell_np[(k + 2) % 3]))
                   for k in range(3)]
        self.n_img = [int(math.floor(self.r_cand / h + 1e-6)) + 1 for h in heights]
        norms = np.linalg.norm(cell_np, axis=1)
        self.pad_axis = int(np.argmax(norms))
        self.pad_rep = int(math.ceil(self.r_max * pad_margin / float(norms[self.pad_axis])))
        dev = self.device
        prm = np.zeros(24, dtype=np.float64)
        prm[_P_CELL:_P_CELL + 9] = cell_np.reshape(-1)
        prm[_P_INV:_P_INV + 9] = np.linalg.inv(cell_np).reshape(-1)
        prm[_P_RC2] = self.r_cand ** 2
        prm[_P_RF2] = self.r_filter ** 2
        prm[_P_THR2] = self.disp_thr ** 2
        self.prm32 = torch.tensor(prm, dtype=torch.float32, device=dev)
        pad_us = np.zeros(3)
        pad_us[self.pad_axis] = float(self.pad_rep)
        prmm = np.zeros(16, dtype=np.float64)
        prmm[_M_CELL:_M_CELL + 9] = cell_np.reshape(-1)
        prmm[_M_PAD_US:_M_PAD_US + 3] = pad_us
        # pad shift computed in the model dtype like MACE (unit_shifts @ cell)
        cm = torch.tensor(cell_np, dtype=model_dtype)
        prmm_t = torch.tensor(prmm, dtype=model_dtype)
        prmm_t[_M_PAD_SH:_M_PAD_SH + 3] = torch.tensor(pad_us, dtype=model_dtype) @ cm
        self.prm_m = prmm_t.to(dev)
        # persistent state (address-stable)
        self.ref_pos = torch.zeros(n, 3, dtype=torch.float64, device=dev)
        self.model_pos = torch.zeros(n, 3, dtype=model_dtype, device=dev)
        self.ctrl = torch.zeros(4, dtype=torch.int32, device=dev)
        self.stats = torch.zeros(N_STATS, dtype=torch.float64, device=dev)
        self.cnt = torch.zeros(n, dtype=torch.int32, device=dev)
        self.vcnt = torch.zeros(n, dtype=torch.int32, device=dev)
        self.cand_total = torch.zeros(1, dtype=torch.int32, device=dev)
        self.node_attrs = node_attrs
        self.cell_m = torch.tensor(cell_np, dtype=model_dtype, device=dev)
        self.batch = torch.zeros(n, dtype=torch.int64, device=dev)
        self.ptr = torch.tensor([0, n], dtype=torch.int64, device=dev)
        self.head_t = torch.full((1,), self.head, dtype=torch.int64, device=dev)
        self._block_n = max(16, triton.next_power_of_2(n))
        self._block_s = min(256, self._block_n)
        self._block_k = 64
        self._block_p = 16
        self.tiers: Dict[int, EdgeTier] = {}
        self.c_cap = 0
        if mode == "candidates":
            if len(e_caps) != 1:
                raise ValueError("candidates mode needs exactly one capacity")
            self.set_candidate_capacity(int(e_caps[0]))
        else:
            if c_cap <= 0 or not e_caps:
                raise ValueError("filter mode needs c_cap > 0 and e_caps")
            self.set_candidate_capacity(int(c_cap))
            for e in e_caps:
                self.add_tier(int(e))
        self.request_rebuild()

    # ------------------------------------------------------------ buffers
    def _make_tier(self, e_cap: int) -> EdgeTier:
        dev, dt = self.device, self.model_dtype
        ei = torch.zeros(2, e_cap, dtype=torch.int64, device=dev)
        us = torch.zeros(e_cap, 3, dtype=dt, device=dev)
        sh = torch.zeros(e_cap, 3, dtype=dt, device=dev)
        inputs = {
            "positions": self.model_pos,
            "node_attrs": self.node_attrs,
            "edge_index": ei,
            "unit_shifts": us,
            "shifts": sh,
            "cell": self.cell_m,
            "batch": self.batch,
            "ptr": self.ptr,
            "head": self.head_t,
        }
        t = EdgeTier(e_cap, ei, us, sh, inputs)
        self._pad_eager(t, 0)
        return t

    @torch.no_grad()
    def _pad_eager(self, t: EdgeTier, start: int) -> None:
        k = torch.arange(start, t.e_cap, device=self.device)
        atom = k % self.n_atoms
        t.edge_index[:, start:] = atom
        t.unit_shifts[start:] = self.prm_m[_M_PAD_US:_M_PAD_US + 3]
        t.shifts[start:] = self.prm_m[_M_PAD_SH:_M_PAD_SH + 3]

    def add_tier(self, e_cap: int) -> EdgeTier:
        if self.mode != "filter":
            raise RuntimeError("tiers only exist in filter mode")
        e_cap = int(e_cap)
        if e_cap not in self.tiers:
            self.tiers[e_cap] = self._make_tier(e_cap)
        return self.tiers[e_cap]

    def set_candidate_capacity(self, c_cap: int) -> None:
        """(Re)allocate candidate buffers (invalidates captured graphs) and force a
        rebuild at the next step."""
        dev, dt = self.device, self.model_dtype
        c_cap = int(c_cap)
        self.c_cap = c_cap
        if self.mode == "candidates":
            # the model edges ARE the candidates
            self.tiers = {c_cap: self._make_tier(c_cap)}
            t = self.tiers[c_cap]
            self.cand_send = t.edge_index[0]
            self.cand_recv = t.edge_index[1]
            self.cand_us = t.unit_shifts
            self.cand_sh = t.shifts
        else:
            self.cand_send = torch.zeros(c_cap, dtype=torch.int32, device=dev)
            self.cand_recv = torch.zeros(c_cap, dtype=torch.int32, device=dev)
            self.cand_us = torch.zeros(c_cap, 3, dtype=dt, device=dev)
            self.cand_sh = torch.zeros(c_cap, 3, dtype=dt, device=dev)
        self.request_rebuild()

    @property
    def tier_caps(self):
        return sorted(self.tiers)

    # ------------------------------------------------------------ control
    def request_rebuild(self) -> None:
        """Force a candidate rebuild at the next :meth:`step` (stream ordered)."""
        self.ctrl[CTRL_FORCE:CTRL_FORCE + 1].fill_(1)

    def reset_stats(self) -> None:
        self.stats.zero_()

    # ------------------------------------------------------------ in-graph step
    def step(self, positions: torch.Tensor, e_cap: Optional[int] = None) -> EdgeTier:
        """Capture-safe per-step update from the float64 (N,3) contiguous positions.
        Writes model positions and the model edges of tier ``e_cap`` (filter mode;
        default: the smallest tier); returns that tier. Two grid-N Triton launches."""
        n = self.n_atoms
        if self.mode == "candidates":
            t = self.tiers[self.c_cap]
        else:
            t = self.tiers[e_cap if e_cap is not None else min(self.tiers)]
        nx, ny, nz = self.n_img
        filt = self.mode == "filter"
        _k1_kernel[(n,)](positions, self.ref_pos, self.model_pos, self.ctrl, self.stats,
                         self.prm32, self.cnt, self.vcnt, self.cand_send, self.cand_sh,
                         n, self.c_cap, BLOCK_N=self._block_n, BLOCK_S=self._block_s,
                         BLOCK_K=self._block_k, NX=nx, NY=ny, NZ=nz,
                         POLICY=_REBUILD_CODES[self.rebuild], FILTER=filt,
                         num_warps=4, enable_fp_fusion=False)
        _k2_kernel[(n,)](positions, self.ref_pos, self.ctrl, self.stats, self.prm32,
                         self.prm_m, self.cnt, self.vcnt, self.cand_send, self.cand_recv,
                         self.cand_us, self.cand_sh, self.cand_total, t.edge_index,
                         t.unit_shifts, t.shifts, n, self.c_cap, t.e_cap,
                         BLOCK_N=self._block_n, BLOCK_S=self._block_s,
                         BLOCK_K=self._block_k, BLOCK_P=self._block_p, NX=nx, NY=ny, NZ=nz,
                         FILTER=filt, num_warps=4, enable_fp_fusion=False)
        return t

    # ------------------------------------------------------------ test helpers
    def is_pad(self, i: int, j: int, s) -> bool:
        return i == j and int(s[self.pad_axis]) == self.pad_rep and \
            sum(abs(int(x)) for x in s) == self.pad_rep

    def edge_list(self, e_cap: Optional[int] = None, include_pad: bool = False):
        """Host helper (syncs): list of (sender, receiver, U0, U1, U2) of tier e_cap."""
        t = self.tiers[self.c_cap] if self.mode == "candidates" else \
            self.tiers[e_cap if e_cap is not None else min(self.tiers)]
        ei = t.edge_index.cpu().numpy()
        us = np.rint(t.unit_shifts.double().cpu().numpy()).astype(np.int64)
        out = []
        for k in range(ei.shape[1]):
            i, j = int(ei[0, k]), int(ei[1, k])
            s = tuple(int(x) for x in us[k])
            if not include_pad and self.is_pad(i, j, s):
                continue
            out.append((i, j) + s)
        return out

    def candidate_set(self) -> set:
        n = int(min(int(self.cand_total.item()), self.c_cap))
        s = self.cand_send[:n].cpu().numpy()
        r = self.cand_recv[:n].cpu().numpy()
        u = np.rint(self.cand_us[:n].double().cpu().numpy()).astype(np.int64)
        return {(int(s[k]), int(r[k]), int(u[k, 0]), int(u[k, 1]), int(u[k, 2]))
                for k in range(n)}

    def host_stats(self) -> dict:
        s = self.stats.cpu().numpy()
        return {"edges_max": int(s[STAT_EDGES_MAX]), "cand_max": int(s[STAT_CAND_MAX]),
                "disp_eff": float(np.sqrt(s[STAT_DISP2_EFF])),
                "rebuilds": int(s[STAT_REBUILDS]), "disp_raw": float(np.sqrt(s[STAT_DISP2_RAW])),
                "edges_sum": float(s[STAT_EDGES_SUM]), "steps": int(s[STAT_STEPS]),
                "nl_err": int(s[STAT_NL_ERR])}


# --------------------------------------------------------------------------- #
# Reference helpers (host; tests)
# --------------------------------------------------------------------------- #
def edge_set_from(edge_index, unit_shifts) -> set:
    ei = np.asarray(edge_index.cpu().numpy() if torch.is_tensor(edge_index) else edge_index)
    us = unit_shifts.cpu().numpy() if torch.is_tensor(unit_shifts) else np.asarray(unit_shifts)
    us = np.rint(us).astype(np.int64)
    return {(int(ei[0, k]), int(ei[1, k]), int(us[k, 0]), int(us[k, 1]), int(us[k, 2]))
            for k in range(ei.shape[1])}


def edge_lengths(edges, positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """|x_r - x_s + U @ cell| for an iterable of (s, r, U0, U1, U2)."""
    e = np.asarray(list(edges), dtype=np.int64).reshape(-1, 5)
    v = positions[e[:, 1]] - positions[e[:, 0]] + e[:, 2:5].astype(np.float64) @ cell
    return np.linalg.norm(v, axis=1)


def unwrap_trajectory(pos: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Unwrap a wrapped trajectory [T,N,3] (cartesian) with the minimum-image
    displacement between consecutive frames (fixed cell)."""
    inv = np.linalg.inv(cell)
    frac = pos @ inv
    d = np.diff(frac, axis=0)
    d -= np.round(d)
    out = np.concatenate([frac[:1], frac[:1] + np.cumsum(d, axis=0)], axis=0)
    return out @ cell
