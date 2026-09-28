"""Triton kernels: fused MACE edge embedding (forward + analytic backward).

One forward kernel computes, per edge e = (sender s -> receiver r, unit shift u):

    vec0 = pos[r] - pos[s] + u @ cell               (MACE get_edge_vectors_and_lengths)
    vec  = vec0 + vec0 @ S,  S = sym(displacement)   (only if HAS_DISP; S == 0 numerically)
    len  = |vec|
    edge_attrs[e, 0:16] = e3nn SphericalHarmonics(lmax=3, normalize=True,
                          normalization="component")(vec)       (e3nn 0.4.4 polynomials)
    cut  = PolynomialCutoff(p=P)(len) * (len < r_max)
    x'   = Agnesi(len; r0 = (cov[Zs]+cov[Zr])/2)          (only if AGNESI; else x' = len)
    edge_feats[e, k] = prefactor * sin(w_k x') / x' * cut        (BesselBasis x cutoff)
    density[r, l] += tanh((edge_feats[e] . wd_l)^2)   (l < ND; MACE density_fn = bias-free linear)
    zbl[r]        += ZBLBasis edge energy              (only if ZBL; on the untransformed len)
    lengths[e]     = len                               (only if STORE_LEN)

The backward kernel recomputes everything from the inputs (only the inputs are saved),
chains the incoming cotangents (edge_attrs, edge_feats, lengths, density, zbl) down to
dE/dvec and accumulates

    grad_pos[r] += (I+S) dE/dvec ,  grad_pos[s] -= (I+S) dE/dvec    (fp atomics, any edge order)
    grad_disp   += sym( sum_e vec0_e (x) dE/dvec_e )                  (block-reduced, 9 atomics/program)

i.e. no sort-based ``index_put_`` backward of ``positions[edge_index]`` and no 3x3 bmm/einsum.

Cost-driven formulation (GB300 FP64 throughput is low, the kernels are latency bound with
one edge per thread):
  * Bessel: sin(k*theta), k = 1..NB, by the angle-addition (rotation) recurrence from one
    sin/cos pair (BESSEL_REC, used when the model's weights are exactly k*w_1, checked on the
    host); error grows ~k ulp.  Otherwise NB direct libdevice sin/cos.
  * PolynomialCutoff / ZBL envelope: integer powers by repeated multiplication (P constexpr).
  * one reciprocal per edge for F.normalize and for sin(x)/x.
  * ZBL: the whole ZBL chain is skipped for a program when none of its edges is inside the
    ZBL cutoff (cov(Zs)+cov(Zr)); the result is then exactly 0 like MACE's envelope mask.
These give differences to the torch reference of a few ulp (fp64 ~1e-15 relative), see
docs/fusion_edge.md.  Constants MACE keeps as tensor buffers (Bessel weights, r_max, Agnesi
a/q/p, ZBL c/a, covalent radii, density weights) are read from device tensors in the model
dtype; e3nn's Python-float constants (sqrt(3), ...) are compile-time constants.
"""

from __future__ import annotations

import math

import triton
import triton.language as tl
from triton.language.extra import libdevice

# e3nn _spherical_harmonics constants (Python doubles, exactly as e3nn uses them)
_S3 = tl.constexpr(math.sqrt(3.0))
_S3_2 = tl.constexpr(math.sqrt(3.0) / 2.0)
_S5 = tl.constexpr(math.sqrt(5.0))
_S5_6 = tl.constexpr(math.sqrt(5.0 / 6.0))
_S3_8 = tl.constexpr(math.sqrt(3.0 / 8.0))
# 'component' normalisation sqrt(2l+1)
_N1 = tl.constexpr(math.sqrt(3.0))
_N2 = tl.constexpr(math.sqrt(5.0))
_N3 = tl.constexpr(math.sqrt(7.0))

# parameter-vector layout (model dtype); see edge_fusions.EdgeEmbedding.__init__
# (plain ints for host code in LAYOUT, tl.constexpr copies for the kernels)
LAYOUT = dict(
    RMAX=0,
    BPREF=1,
    AG_A=2,
    AG_Q=3,
    AG_QMP=4,
    AG_P=5,
    ZBL_APREF=6,  # a_prefactor * 0.529
    ZBL_C0=7,  # c0..c3
    BW=16,  # Bessel weights, NB_PAD entries (zero padded)
    WD=32,  # density weights, ND x NB_PAD entries (zero padded)
    SIZE=32 + 4 * 16,
    NODE_COLS=4,  # node table [N,4]: cov (Agnesi buffer), cov (ZBL buffer), Z**a_exp, float(Z)
)
PRM_RMAX = tl.constexpr(LAYOUT["RMAX"])
PRM_BPREF = tl.constexpr(LAYOUT["BPREF"])
PRM_AG_A = tl.constexpr(LAYOUT["AG_A"])
PRM_AG_Q = tl.constexpr(LAYOUT["AG_Q"])
PRM_AG_QMP = tl.constexpr(LAYOUT["AG_QMP"])
PRM_AG_P = tl.constexpr(LAYOUT["AG_P"])
PRM_ZBL_APREF = tl.constexpr(LAYOUT["ZBL_APREF"])
PRM_ZBL_C0 = tl.constexpr(LAYOUT["ZBL_C0"])
PRM_BW = tl.constexpr(LAYOUT["BW"])
PRM_WD = tl.constexpr(LAYOUT["WD"])
NODE_COLS = tl.constexpr(LAYOUT["NODE_COLS"])


@triton.jit
def _poly_env(x, rmax, P: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr):
    """MACE PolynomialCutoff.calculate_envelope(x, r_max, p) (without the mask) and d/dx.

    Integer powers by multiplication (P is a constexpr int >= 2)."""
    u = libdevice.div_rn(x, rmax)
    um1 = u
    for _ in tl.static_range(P - 2):
        um1 = um1 * u
    up = um1 * u
    up1 = up * u
    up2 = up1 * u
    env = 1.0 - C1 * up + C2 * up1 - C3 * up2
    denv = libdevice.div_rn(-C1 * P * um1 + C2 * (P + 1.0) * up - C3 * (P + 2.0) * up1, rmax)
    return env, denv


@triton.jit
def _edge_geometry(pos_ptr, cell_ptr, disp_ptr, ei_ptr, us_ptr, E, offs, m, HAS_DISP: tl.constexpr):
    snd = tl.load(ei_ptr + offs, mask=m, other=0)
    rcv = tl.load(ei_ptr + E + offs, mask=m, other=0)
    u0 = tl.load(us_ptr + offs * 3 + 0, mask=m, other=0.0)
    u1 = tl.load(us_ptr + offs * 3 + 1, mask=m, other=0.0)
    u2 = tl.load(us_ptr + offs * 3 + 2, mask=m, other=0.0)
    c00 = tl.load(cell_ptr + 0)
    c01 = tl.load(cell_ptr + 1)
    c02 = tl.load(cell_ptr + 2)
    c10 = tl.load(cell_ptr + 3)
    c11 = tl.load(cell_ptr + 4)
    c12 = tl.load(cell_ptr + 5)
    c20 = tl.load(cell_ptr + 6)
    c21 = tl.load(cell_ptr + 7)
    c22 = tl.load(cell_ptr + 8)
    # shift = u @ cell (rows of cell are lattice vectors)
    sx = u0 * c00 + u1 * c10 + u2 * c20
    sy = u0 * c01 + u1 * c11 + u2 * c21
    sz = u0 * c02 + u1 * c12 + u2 * c22
    prx = tl.load(pos_ptr + rcv * 3 + 0, mask=m, other=0.0)
    pry = tl.load(pos_ptr + rcv * 3 + 1, mask=m, other=0.0)
    prz = tl.load(pos_ptr + rcv * 3 + 2, mask=m, other=0.0)
    psx = tl.load(pos_ptr + snd * 3 + 0, mask=m, other=0.0)
    psy = tl.load(pos_ptr + snd * 3 + 1, mask=m, other=0.0)
    psz = tl.load(pos_ptr + snd * 3 + 2, mask=m, other=0.0)
    v0x = prx - psx + sx
    v0y = pry - psy + sy
    v0z = prz - psz + sz
    # dead lanes: make the vector finite and non-zero (results are masked anyway)
    v0x = tl.where(m, v0x, 1.0)
    if HAS_DISP:
        # S = 0.5 (D + D^T); vec = vec0 + vec0 @ S
        d00 = tl.load(disp_ptr + 0)
        d01 = tl.load(disp_ptr + 1)
        d02 = tl.load(disp_ptr + 2)
        d10 = tl.load(disp_ptr + 3)
        d11 = tl.load(disp_ptr + 4)
        d12 = tl.load(disp_ptr + 5)
        d20 = tl.load(disp_ptr + 6)
        d21 = tl.load(disp_ptr + 7)
        d22 = tl.load(disp_ptr + 8)
        s01 = 0.5 * (d01 + d10)
        s02 = 0.5 * (d02 + d20)
        s12 = 0.5 * (d12 + d21)
        vx = v0x + (v0x * d00 + v0y * s01 + v0z * s02)
        vy = v0y + (v0x * s01 + v0y * d11 + v0z * s12)
        vz = v0z + (v0x * s02 + v0y * s12 + v0z * d22)
    else:
        vx = v0x
        vy = v0y
        vz = v0z
    return snd, rcv, v0x, v0y, v0z, vx, vy, vz


@triton.jit
def _agnesi(ln, snd, rcv, m, node_ptr, prm_ptr):
    """x' = 1 / (1 + a t^q / (1 + t^(q-p))), t = len / r0, and dx'/dlen."""
    cov_s = tl.load(node_ptr + snd * NODE_COLS + 0, mask=m, other=1.0)
    cov_r = tl.load(node_ptr + rcv * NODE_COLS + 0, mask=m, other=1.0)
    ag_a = tl.load(prm_ptr + PRM_AG_A)
    ag_q = tl.load(prm_ptr + PRM_AG_Q)
    ag_qmp = tl.load(prm_ptr + PRM_AG_QMP)
    ag_p = tl.load(prm_ptr + PRM_AG_P)
    r0 = 0.5 * (cov_s + cov_r)
    t = libdevice.div_rn(ln, r0)
    tA = libdevice.pow(t, ag_q)
    tB = libdevice.pow(t, ag_qmp)
    opb = 1.0 + tB
    xp = libdevice.rcp_rn(1.0 + libdevice.div_rn(ag_a * tA, opb))
    # d g / d t with g = a t^q / (1 + t^(q-p)):  a t^q (q + p t^(q-p)) / (t (1+B)^2)
    dg_dt = libdevice.div_rn(ag_a * tA * (ag_q + ag_p * tB), t * opb * opb)
    dxp_dlen = -(xp * xp) * libdevice.div_rn(dg_dt, r0)
    return xp, dxp_dlen


@triton.jit
def _zbl_terms(ln, snd, rcv, m, node_ptr, prm_ptr,
               ZP: tl.constexpr, ZC1: tl.constexpr, ZC2: tl.constexpr, ZC3: tl.constexpr):
    """MACE ZBLBasis edge energy v(len) (0.5 * ... * envelope, masked) and dv/dlen."""
    zc_s = tl.load(node_ptr + snd * NODE_COLS + 1, mask=m, other=1.0)
    zc_r = tl.load(node_ptr + rcv * NODE_COLS + 1, mask=m, other=1.0)
    zp_s = tl.load(node_ptr + snd * NODE_COLS + 2, mask=m, other=1.0)
    zp_r = tl.load(node_ptr + rcv * NODE_COLS + 2, mask=m, other=1.0)
    zf_s = tl.load(node_ptr + snd * NODE_COLS + 3, mask=m, other=1.0)
    zf_r = tl.load(node_ptr + rcv * NODE_COLS + 3, mask=m, other=1.0)
    apref = tl.load(prm_ptr + PRM_ZBL_APREF)
    c0 = tl.load(prm_ptr + PRM_ZBL_C0 + 0)
    c1 = tl.load(prm_ptr + PRM_ZBL_C0 + 1)
    c2 = tl.load(prm_ptr + PRM_ZBL_C0 + 2)
    c3 = tl.load(prm_ptr + PRM_ZBL_C0 + 3)
    za = libdevice.div_rn(apref, zp_s + zp_r)
    ra = libdevice.div_rn(ln, za)
    e0 = libdevice.exp(-3.2 * ra)
    e1 = libdevice.exp(-0.9423 * ra)
    e2 = libdevice.exp(-0.4028 * ra)
    e3 = libdevice.exp(-0.2016 * ra)
    phi = c0 * e0 + c1 * e1 + c2 * e2 + c3 * e3
    dphi = libdevice.div_rn(-3.2 * c0 * e0 - 0.9423 * c1 * e1 - 0.4028 * c2 * e2 - 0.2016 * c3 * e3, za)
    kz = 14.3996 * zf_s * zf_r
    f = libdevice.div_rn(kz, ln) * phi
    df = kz * libdevice.div_rn(dphi - libdevice.div_rn(phi, ln), ln)
    rmz = zc_s + zc_r
    zenv, zdenv = _poly_env(ln, rmz, ZP, ZC1, ZC2, ZC3)
    inz = (ln < rmz) & m
    v = tl.where(inz, 0.5 * f * zenv, 0.0)
    dv = tl.where(inz, 0.5 * (df * zenv + f * zdenv), 0.0)
    return v, dv


@triton.jit
def _zbl_any(ln, snd, rcv, m, node_ptr):
    zc_s = tl.load(node_ptr + snd * NODE_COLS + 1, mask=m, other=0.0)
    zc_r = tl.load(node_ptr + rcv * NODE_COLS + 1, mask=m, other=0.0)
    inz = (ln < zc_s + zc_r) & m
    return tl.max(inz.to(tl.int32), axis=0)


@triton.jit
def _edge_embed_fwd_kernel(
    pos_ptr, cell_ptr, disp_ptr, ei_ptr, us_ptr, node_ptr, prm_ptr,
    attrs_ptr, feats_ptr, len_ptr, nout_ptr,
    E,
    HAS_DISP: tl.constexpr, AGNESI: tl.constexpr, ND: tl.constexpr, ZBL: tl.constexpr,
    STORE_LEN: tl.constexpr, NO: tl.constexpr, BESSEL_REC: tl.constexpr,
    NB: tl.constexpr, NB_PAD: tl.constexpr,
    P: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    ZP: tl.constexpr, ZC1: tl.constexpr, ZC2: tl.constexpr, ZC3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    snd, rcv, v0x, v0y, v0z, vx, vy, vz = _edge_geometry(
        pos_ptr, cell_ptr, disp_ptr, ei_ptr, us_ptr, E, offs, m, HAS_DISP
    )
    ln = libdevice.sqrt_rn(vx * vx + vy * vy + vz * vz)
    if STORE_LEN:
        tl.store(len_ptr + offs, ln, mask=m)

    # ---- spherical harmonics (F.normalize: v / max(|v|, 1e-12)) ----
    inv = libdevice.rcp_rn(tl.maximum(ln, 1e-12))
    x = vx * inv
    y = vy * inv
    z = vz * inv
    sh20 = _S3 * x * z
    sh21 = _S3 * x * y
    y2 = y * y
    x2z2 = x * x + z * z
    sh22 = y2 - 0.5 * x2z2
    sh23 = _S3 * y * z
    sh24 = _S3_2 * (z * z - x * x)
    sh30 = _S5_6 * (sh20 * z + sh24 * x)
    sh31 = _S5 * sh20 * y
    sh32 = _S3_8 * (4.0 * y2 - x2z2) * x
    sh33 = 0.5 * y * (2.0 * y2 - 3.0 * x2z2)
    sh34 = _S3_8 * z * (4.0 * y2 - x2z2)
    sh35 = _S5 * sh24 * y
    sh36 = _S5_6 * (sh24 * z - sh20 * x)
    ab = attrs_ptr + offs * 16
    tl.store(ab + 0, tl.full([BLOCK], 1.0, x.dtype), mask=m)
    tl.store(ab + 1, x * _N1, mask=m)
    tl.store(ab + 2, y * _N1, mask=m)
    tl.store(ab + 3, z * _N1, mask=m)
    tl.store(ab + 4, sh20 * _N2, mask=m)
    tl.store(ab + 5, sh21 * _N2, mask=m)
    tl.store(ab + 6, sh22 * _N2, mask=m)
    tl.store(ab + 7, sh23 * _N2, mask=m)
    tl.store(ab + 8, sh24 * _N2, mask=m)
    tl.store(ab + 9, sh30 * _N3, mask=m)
    tl.store(ab + 10, sh31 * _N3, mask=m)
    tl.store(ab + 11, sh32 * _N3, mask=m)
    tl.store(ab + 12, sh33 * _N3, mask=m)
    tl.store(ab + 13, sh34 * _N3, mask=m)
    tl.store(ab + 14, sh35 * _N3, mask=m)
    tl.store(ab + 15, sh36 * _N3, mask=m)

    # ---- radial: cutoff(len) * Bessel(Agnesi(len)) ----
    rmax = tl.load(prm_ptr + PRM_RMAX)
    env, _denv = _poly_env(ln, rmax, P, C1, C2, C3)
    cut = tl.where(ln < rmax, env, 0.0)
    if AGNESI:
        xp, _dxp = _agnesi(ln, snd, rcv, m, node_ptr, prm_ptr)
    else:
        xp = ln
    bpref = tl.load(prm_ptr + PRM_BPREF)
    rx = libdevice.rcp_rn(xp)
    fb = feats_ptr + offs * NB
    if BESSEL_REC:
        w1 = tl.load(prm_ptr + PRM_BW)
        th = w1 * xp
        s1 = libdevice.sin(th)
        c1 = libdevice.cos(th)
        s = s1
        c = c1
    d0 = tl.zeros([BLOCK], dtype=x.dtype)
    d1 = tl.zeros([BLOCK], dtype=x.dtype)
    d2 = tl.zeros([BLOCK], dtype=x.dtype)
    d3 = tl.zeros([BLOCK], dtype=x.dtype)
    for k in tl.static_range(NB):
        if BESSEL_REC:
            if k > 0:
                s_new = s * c1 + c * s1
                c = c * c1 - s * s1
                s = s_new
            sk = s
        else:
            sk = libdevice.sin(tl.load(prm_ptr + PRM_BW + k) * xp)
        fk = bpref * (sk * rx) * cut
        tl.store(fb + k, fk, mask=m)
        if ND > 0:
            d0 += fk * tl.load(prm_ptr + PRM_WD + 0 * NB_PAD + k)
        if ND > 1:
            d1 += fk * tl.load(prm_ptr + PRM_WD + 1 * NB_PAD + k)
        if ND > 2:
            d2 += fk * tl.load(prm_ptr + PRM_WD + 2 * NB_PAD + k)
        if ND > 3:
            d3 += fk * tl.load(prm_ptr + PRM_WD + 3 * NB_PAD + k)

    # ---- density (MACE density_fn: Linear(NB->1, no bias), tanh(d^2)) ----
    if ND > 0:
        tl.atomic_add(nout_ptr + rcv * NO + 0, libdevice.tanh(d0 * d0), mask=m, sem="relaxed")
    if ND > 1:
        tl.atomic_add(nout_ptr + rcv * NO + 1, libdevice.tanh(d1 * d1), mask=m, sem="relaxed")
    if ND > 2:
        tl.atomic_add(nout_ptr + rcv * NO + 2, libdevice.tanh(d2 * d2), mask=m, sem="relaxed")
    if ND > 3:
        tl.atomic_add(nout_ptr + rcv * NO + 3, libdevice.tanh(d3 * d3), mask=m, sem="relaxed")

    # ---- ZBL pair repulsion (on the untransformed length) ----
    if ZBL:
        if _zbl_any(ln, snd, rcv, m, node_ptr) > 0:
            v, _dv = _zbl_terms(ln, snd, rcv, m, node_ptr, prm_ptr, ZP, ZC1, ZC2, ZC3)
            tl.atomic_add(nout_ptr + rcv * NO + ND, v, mask=m, sem="relaxed")


@triton.jit
def _edge_embed_bwd_kernel(
    pos_ptr, cell_ptr, disp_ptr, ei_ptr, us_ptr, node_ptr, prm_ptr,
    g_attrs_ptr, g_feats_ptr, g_len_ptr, g_nout_ptr,
    gpos_ptr, gdisp_ptr,
    E,
    HAS_DISP: tl.constexpr, AGNESI: tl.constexpr, ND: tl.constexpr, ZBL: tl.constexpr,
    NO: tl.constexpr, BESSEL_REC: tl.constexpr,
    HAS_GA: tl.constexpr, HAS_GF: tl.constexpr, HAS_GL: tl.constexpr, HAS_GN: tl.constexpr,
    NEED_DISP_GRAD: tl.constexpr,
    NB: tl.constexpr, NB_PAD: tl.constexpr,
    P: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr, C3: tl.constexpr,
    ZP: tl.constexpr, ZC1: tl.constexpr, ZC2: tl.constexpr, ZC3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    snd, rcv, v0x, v0y, v0z, vx, vy, vz = _edge_geometry(
        pos_ptr, cell_ptr, disp_ptr, ei_ptr, us_ptr, E, offs, m, HAS_DISP
    )
    ln = libdevice.sqrt_rn(vx * vx + vy * vy + vz * vz)
    inv = libdevice.rcp_rn(tl.maximum(ln, 1e-12))
    x = vx * inv
    y = vy * inv
    z = vz * inv
    zero = tl.zeros([BLOCK], dtype=vx.dtype)
    g_len = zero
    if HAS_GL:
        g_len += tl.load(g_len_ptr + offs, mask=m, other=0.0)

    # ---------------- radial part ----------------
    DO_DENS: tl.constexpr = ND > 0 and HAS_GN
    if HAS_GF or DO_DENS:
        rmax = tl.load(prm_ptr + PRM_RMAX)
        env, denv = _poly_env(ln, rmax, P, C1, C2, C3)
        inr = ln < rmax
        cut = tl.where(inr, env, 0.0)
        dcut = tl.where(inr, denv, 0.0)
        if AGNESI:
            xp, dxp_dlen = _agnesi(ln, snd, rcv, m, node_ptr, prm_ptr)
        else:
            xp = ln
        bpref = tl.load(prm_ptr + PRM_BPREF)
        rx = libdevice.rcp_rn(xp)
        gfb = g_feats_ptr + offs * NB
        if BESSEL_REC:
            w1 = tl.load(prm_ptr + PRM_BW)
            th = w1 * xp
            s1 = libdevice.sin(th)
            c1 = libdevice.cos(th)
        # density cotangents: dd_l = g_dens[r,l] * (1 - tanh(d_l^2)^2) * 2 d_l  (needs d_l first)
        dd0 = zero
        dd1 = zero
        dd2 = zero
        dd3 = zero
        if DO_DENS:
            d0 = zero
            d1 = zero
            d2 = zero
            d3 = zero
            if BESSEL_REC:
                s = s1
                c = c1
            for k in tl.static_range(NB):
                if BESSEL_REC:
                    if k > 0:
                        s_new = s * c1 + c * s1
                        c = c * c1 - s * s1
                        s = s_new
                    sk = s
                else:
                    sk = libdevice.sin(tl.load(prm_ptr + PRM_BW + k) * xp)
                fk = bpref * (sk * rx) * cut
                if ND > 0:
                    d0 += fk * tl.load(prm_ptr + PRM_WD + 0 * NB_PAD + k)
                if ND > 1:
                    d1 += fk * tl.load(prm_ptr + PRM_WD + 1 * NB_PAD + k)
                if ND > 2:
                    d2 += fk * tl.load(prm_ptr + PRM_WD + 2 * NB_PAD + k)
                if ND > 3:
                    d3 += fk * tl.load(prm_ptr + PRM_WD + 3 * NB_PAD + k)
            if ND > 0:
                th0 = libdevice.tanh(d0 * d0)
                dd0 = tl.load(g_nout_ptr + rcv * NO + 0, mask=m, other=0.0) * (1.0 - th0 * th0) * (2.0 * d0)
            if ND > 1:
                th1 = libdevice.tanh(d1 * d1)
                dd1 = tl.load(g_nout_ptr + rcv * NO + 1, mask=m, other=0.0) * (1.0 - th1 * th1) * (2.0 * d1)
            if ND > 2:
                th2 = libdevice.tanh(d2 * d2)
                dd2 = tl.load(g_nout_ptr + rcv * NO + 2, mask=m, other=0.0) * (1.0 - th2 * th2) * (2.0 * d2)
            if ND > 3:
                th3 = libdevice.tanh(d3 * d3)
                dd3 = tl.load(g_nout_ptr + rcv * NO + 3, mask=m, other=0.0) * (1.0 - th3 * th3) * (2.0 * d3)
        # feats_k = rad_k * cut, rad_k = pref sin(w_k x)/x, d rad_k/dx = pref (w_k cos(w_k x) - sin/x)/x
        g_cut = zero
        g_xp = zero
        if BESSEL_REC:
            s = s1
            c = c1
        for k in tl.static_range(NB):
            wk = tl.load(prm_ptr + PRM_BW + k)
            if BESSEL_REC:
                if k > 0:
                    s_new = s * c1 + c * s1
                    c = c * c1 - s * s1
                    s = s_new
                sk = s
                ck = c
            else:
                sk = libdevice.sin(wk * xp)
                ck = libdevice.cos(wk * xp)
            gk = zero
            if HAS_GF:
                gk += tl.load(gfb + k, mask=m, other=0.0)
            if DO_DENS:
                if ND > 0:
                    gk += dd0 * tl.load(prm_ptr + PRM_WD + 0 * NB_PAD + k)
                if ND > 1:
                    gk += dd1 * tl.load(prm_ptr + PRM_WD + 1 * NB_PAD + k)
                if ND > 2:
                    gk += dd2 * tl.load(prm_ptr + PRM_WD + 2 * NB_PAD + k)
                if ND > 3:
                    gk += dd3 * tl.load(prm_ptr + PRM_WD + 3 * NB_PAD + k)
            sx_ = sk * rx
            g_cut += gk * (bpref * sx_)
            g_xp += gk * (bpref * ((wk * ck - sx_) * rx))
        g_xp = g_xp * cut
        if AGNESI:
            g_len += g_xp * dxp_dlen
        else:
            g_len += g_xp
        g_len += g_cut * dcut

    # ---------------- ZBL ----------------
    if ZBL:
        if HAS_GN:
            if _zbl_any(ln, snd, rcv, m, node_ptr) > 0:
                gz = tl.load(g_nout_ptr + rcv * NO + ND, mask=m, other=0.0)
                _v, dv = _zbl_terms(ln, snd, rcv, m, node_ptr, prm_ptr, ZP, ZC1, ZC2, ZC3)
                g_len += gz * dv

    # ---------------- d len / d vec = vec / len ----------------
    lpos = ln > 0.0
    gl = tl.where(lpos, g_len, 0.0)
    gvx = gl * x
    gvy = gl * y
    gvz = gl * z

    # ---------------- spherical harmonics adjoint ----------------
    if HAS_GA:
        gab = g_attrs_ptr + offs * 16
        G1 = tl.load(gab + 1, mask=m, other=0.0) * _N1
        G2 = tl.load(gab + 2, mask=m, other=0.0) * _N1
        G3 = tl.load(gab + 3, mask=m, other=0.0) * _N1
        a20 = tl.load(gab + 4, mask=m, other=0.0) * _N2
        a21 = tl.load(gab + 5, mask=m, other=0.0) * _N2
        a22 = tl.load(gab + 6, mask=m, other=0.0) * _N2
        a23 = tl.load(gab + 7, mask=m, other=0.0) * _N2
        a24 = tl.load(gab + 8, mask=m, other=0.0) * _N2
        b30 = tl.load(gab + 9, mask=m, other=0.0) * _N3
        b31 = tl.load(gab + 10, mask=m, other=0.0) * _N3
        b32 = tl.load(gab + 11, mask=m, other=0.0) * _N3
        b33 = tl.load(gab + 12, mask=m, other=0.0) * _N3
        b34 = tl.load(gab + 13, mask=m, other=0.0) * _N3
        b35 = tl.load(gab + 14, mask=m, other=0.0) * _N3
        b36 = tl.load(gab + 15, mask=m, other=0.0) * _N3
        # recompute intermediates
        sh20 = _S3 * x * z
        y2 = y * y
        x2z2 = x * x + z * z
        sh24 = _S3_2 * (z * z - x * x)
        q = 4.0 * y2 - x2z2
        gx = G1
        gy = G2
        gz_ = G3
        ay2 = zero
        ax2z2 = zero
        aq = zero
        # sh30 = S5_6 (sh20 z + sh24 x)
        a20 += _S5_6 * z * b30
        a24 += _S5_6 * x * b30
        gz_ += _S5_6 * sh20 * b30
        gx += _S5_6 * sh24 * b30
        # sh31 = S5 sh20 y
        a20 += _S5 * y * b31
        gy += _S5 * sh20 * b31
        # sh32 = S3_8 q x
        gx += _S3_8 * q * b32
        aq += _S3_8 * x * b32
        # sh33 = 0.5 y (2 y2 - 3 x2z2)
        gy += 0.5 * (2.0 * y2 - 3.0 * x2z2) * b33
        ay2 += y * b33
        ax2z2 += -1.5 * y * b33
        # sh34 = S3_8 z q
        gz_ += _S3_8 * q * b34
        aq += _S3_8 * z * b34
        # sh35 = S5 sh24 y
        a24 += _S5 * y * b35
        gy += _S5 * sh24 * b35
        # sh36 = S5_6 (sh24 z - sh20 x)
        a24 += _S5_6 * z * b36
        gz_ += _S5_6 * sh24 * b36
        a20 -= _S5_6 * x * b36
        gx -= _S5_6 * sh20 * b36
        # q = 4 y2 - x2z2
        ay2 += 4.0 * aq
        ax2z2 -= aq
        # l = 2
        gx += _S3 * z * a20
        gz_ += _S3 * x * a20
        gx += _S3 * y * a21
        gy += _S3 * x * a21
        ay2 += a22
        ax2z2 -= 0.5 * a22
        gy += _S3 * z * a23
        gz_ += _S3 * y * a23
        # sh24 = S3_2 (z^2 - x^2): d/dz = S3 z, d/dx = -S3 x
        gz_ += _S3 * z * a24
        gx -= _S3 * x * a24
        gy += 2.0 * y * ay2
        gx += 2.0 * x * ax2z2
        gz_ += 2.0 * z * ax2z2
        # through the normalisation n = v / max(|v|, eps)
        big = ln > 1e-12
        ndot = tl.where(big, x * gx + y * gy + z * gz_, 0.0)
        gvx += (gx - x * ndot) * inv
        gvy += (gy - y * ndot) * inv
        gvz += (gz_ - z * ndot) * inv

    gvx = tl.where(m, gvx, 0.0)
    gvy = tl.where(m, gvy, 0.0)
    gvz = tl.where(m, gvz, 0.0)

    # ---------------- displacement: vec = vec0 (I + S) ----------------
    if HAS_DISP:
        d00 = tl.load(disp_ptr + 0)
        d01 = tl.load(disp_ptr + 1)
        d02 = tl.load(disp_ptr + 2)
        d10 = tl.load(disp_ptr + 3)
        d11 = tl.load(disp_ptr + 4)
        d12 = tl.load(disp_ptr + 5)
        d20 = tl.load(disp_ptr + 6)
        d21 = tl.load(disp_ptr + 7)
        d22 = tl.load(disp_ptr + 8)
        s01 = 0.5 * (d01 + d10)
        s02 = 0.5 * (d02 + d20)
        s12 = 0.5 * (d12 + d21)
        g0x = gvx + (d00 * gvx + s01 * gvy + s02 * gvz)
        g0y = gvy + (s01 * gvx + d11 * gvy + s12 * gvz)
        g0z = gvz + (s02 * gvx + s12 * gvy + d22 * gvz)
    else:
        g0x = gvx
        g0y = gvy
        g0z = gvz

    tl.atomic_add(gpos_ptr + rcv * 3 + 0, g0x, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + rcv * 3 + 1, g0y, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + rcv * 3 + 2, g0z, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + snd * 3 + 0, -g0x, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + snd * 3 + 1, -g0y, mask=m, sem="relaxed")
    tl.atomic_add(gpos_ptr + snd * 3 + 2, -g0z, mask=m, sem="relaxed")

    if NEED_DISP_GRAD:
        # dE/dS_ab = sum_e vec0_a g_b ; dE/dD = sym(dE/dS)
        v0x = tl.where(m, v0x, 0.0)
        wxx = tl.sum(v0x * gvx, axis=0)
        wyy = tl.sum(v0y * gvy, axis=0)
        wzz = tl.sum(v0z * gvz, axis=0)
        wxy = 0.5 * (tl.sum(v0x * gvy, axis=0) + tl.sum(v0y * gvx, axis=0))
        wxz = 0.5 * (tl.sum(v0x * gvz, axis=0) + tl.sum(v0z * gvx, axis=0))
        wyz = 0.5 * (tl.sum(v0y * gvz, axis=0) + tl.sum(v0z * gvy, axis=0))
        tl.atomic_add(gdisp_ptr + 0, wxx, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 4, wyy, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 8, wzz, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 1, wxy, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 3, wxy, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 2, wxz, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 6, wxz, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 5, wyz, sem="relaxed")
        tl.atomic_add(gdisp_ptr + 7, wyz, sem="relaxed")


@triton.jit
def _zbl_fwd_kernel(
    len_ptr, ei_ptr, node_ptr, prm_ptr, out_ptr, E,
    ZP: tl.constexpr, ZC1: tl.constexpr, ZC2: tl.constexpr, ZC3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Standalone ZBL on precomputed lengths: out[rcv] += v_e (the zbl="triton" option)."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    snd = tl.load(ei_ptr + offs, mask=m, other=0)
    rcv = tl.load(ei_ptr + E + offs, mask=m, other=0)
    ln = tl.load(len_ptr + offs, mask=m, other=1.0)
    if _zbl_any(ln, snd, rcv, m, node_ptr) > 0:
        v, _dv = _zbl_terms(ln, snd, rcv, m, node_ptr, prm_ptr, ZP, ZC1, ZC2, ZC3)
        tl.atomic_add(out_ptr + rcv, v, mask=m, sem="relaxed")


@triton.jit
def _zbl_bwd_kernel(
    len_ptr, ei_ptr, node_ptr, prm_ptr, gout_ptr, glen_ptr, E,
    ZP: tl.constexpr, ZC1: tl.constexpr, ZC2: tl.constexpr, ZC3: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < E
    snd = tl.load(ei_ptr + offs, mask=m, other=0)
    rcv = tl.load(ei_ptr + E + offs, mask=m, other=0)
    ln = tl.load(len_ptr + offs, mask=m, other=1.0)
    glen = tl.zeros([BLOCK], dtype=ln.dtype)
    if _zbl_any(ln, snd, rcv, m, node_ptr) > 0:
        gz = tl.load(gout_ptr + rcv, mask=m, other=0.0)
        _v, dv = _zbl_terms(ln, snd, rcv, m, node_ptr, prm_ptr, ZP, ZC1, ZC2, ZC3)
        glen = gz * dv
    tl.store(glen_ptr + offs, glen, mask=m)
