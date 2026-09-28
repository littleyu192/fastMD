"""GPU port of ASE 3.29 ``NoseHooverChainNVT`` and ``MaxwellBoltzmannDistribution``.

Used by :class:`~matris.applications.gpu_md.GPUResidentMolecularDynamics`
with ``ensemble="nvt_nhc"`` in the dynamic, model-graph and whole-step CUDA
graph paths. The matbench-discovery MD protocol, for example, is::

    MaxwellBoltzmannDistribution(atoms, temperature_K=T, rng=np.random.default_rng(0))
    NoseHooverChainNVT(atoms, timestep=0.25 * units.fs, temperature_K=T,
                       tdamp=25 * units.fs)             # tchain=3, tloop=1

Design:

* All integrator state (positions, momenta, cached forces, thermostat
  ``eta``/``p_eta``, potential energy) lives in ONE preallocated float64
  device buffer (views into it), so a transactional backup/restore is a
  single ``copy_``.
* A step is split into :meth:`GPUNoseHooverChainNVT.step_pre_force` (NHC half
  step, kick with the cached F(q_n), drift) and
  :meth:`GPUNoseHooverChainNVT.step_post_force` (kick with F(q_{n+1}), NHC half
  step, cache the forces). ASE evaluates forces exactly once per step (its
  first ``_integrate_p`` hits the calculator cache), so
  pre -> model -> post reproduces ``NoseHooverChainNVT.step()``.
* No host syncs, no data-dependent Python control flow, no allocation of
  state: both halves are CUDA-graph capturable.

Two numerically equivalent backends:

* ``"triton"`` (default on CUDA, 3N <= 16384): one single-CTA fused kernel per
  half (2 launches per step). FP contraction is disabled
  (``enable_fp_fusion=False``) so every elementwise op is rounded exactly like
  NumPy. Thermostat variables live in registers as fp64 scalars; the only
  reduction is sum(p**2/m). Optional ``fast_math=True`` uses reciprocal
  multiplies instead of the fp64 divisions and allows FMA contraction
  (<= 1 ulp per affected op vs ASE).
* ``"torch"``: op-by-op reference (several hundred tiny kernels per step),
  used for validation and as fallback for large systems or CPU tensors.

The order of floating-point operations follows ASE line by line. The only
intended differences to ASE are (a) the reduction order of sum(p**2/m)
(NumPy pairwise vs GPU tree), (b) CUDA ``exp`` vs glibc ``exp`` (last bit
for a few percent of arguments) and (c) NumPy scalar ``x**2`` vs ``x*x``,
i.e. ~1e-16 relative per operation. With ``force_dtype=torch.float32`` the
kick ``dt/2 * F`` is formed in float32 exactly like NumPy 2 (NEP 50) does
for the float32 forces a float32 calculator returns; the state stays
float64.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from ase import units
from torch import Tensor

from .gpu_integrator import StatefulIntegrator

try:  # Triton is optional; the torch backend works everywhere
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None
    tl = None
    libdevice = None
    _HAS_TRITON = False

# Identical Python expressions to ase/md/nose_hoover_chain.py -> identical floats
FOURTH_ORDER_COEFFS = [
    1 / (2 - 2 ** (1 / 3)),
    -(2 ** (1 / 3)) / (2 - 2 ** (1 / 3)),
    1 / (2 - 2 ** (1 / 3)),
]

_TRITON_MAX_ELEMS = 16384  # single-CTA register-resident limit (3N)


def maxwell_boltzmann_momenta(masses, temperature_K: float, seed=0) -> np.ndarray:
    """Bit-identical ``MaxwellBoltzmannDistribution(atoms, temperature_K=T,
    rng=np.random.default_rng(seed))`` momenta (ASE 3.29).

    ASE path: ``thermalize_momenta`` (``exact_temperature=False``, no
    Stationary/ZeroRotation): ``xi = rng.standard_normal((N, 3))``;
    ``p = xi * sqrt(masses * kB * T)[:, None]``. ``masses`` =
    ``atoms.get_masses()`` in amu; ``seed`` is an int or a
    ``np.random.Generator`` (consumed exactly like ASE consumes it).
    """
    masses = np.asarray(masses, dtype=np.float64).reshape(-1)
    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    temp = units.kB * temperature_K
    xi = rng.standard_normal((len(masses), 3))
    return xi * np.sqrt(masses * temp)[:, np.newaxis]


@dataclass
class NHCState:
    """Views into one flat float64 buffer ``flat``.

    positions/momenta/forces: (N, 3); eta/p_eta: (tchain,); potential_energy:
    (1,). ``forces`` always holds F(positions) (the ASE calculator cache).
    """

    flat: Tensor
    positions: Tensor
    momenta: Tensor
    forces: Tensor
    eta: Tensor
    p_eta: Tensor
    potential_energy: Tensor

    @staticmethod
    def allocate(n_atoms: int, tchain: int, device) -> "NHCState":
        n3 = 3 * n_atoms
        flat = torch.zeros(3 * n3 + 2 * tchain + 1, dtype=torch.float64, device=device)
        offset = 0
        positions = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        momenta = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        forces = flat[offset: offset + n3].view(n_atoms, 3)
        offset += n3
        eta = flat[offset: offset + tchain]
        offset += tchain
        p_eta = flat[offset: offset + tchain]
        offset += tchain
        energy = flat[offset: offset + 1]
        return NHCState(flat, positions, momenta, forces, eta, p_eta, energy)

    def numpy(self) -> dict:
        return {
            "positions": self.positions.detach().cpu().numpy().copy(),
            "momenta": self.momenta.detach().cpu().numpy().copy(),
            "forces": self.forces.detach().cpu().numpy().copy(),
            "eta": self.eta.detach().cpu().numpy().copy(),
            "p_eta": self.p_eta.detach().cpu().numpy().copy(),
            "potential_energy": float(self.potential_energy.item()),
        }


# --------------------------------------------------------------------------- #
# Triton fused kernels
# --------------------------------------------------------------------------- #
# constants buffer (float64): dt, dt/2, kT, 3N*kT, 3 x (delta, delta/2,
# -(delta/4), -delta), Q[tchain], 1/Q[tchain]
_C_Q = 16

if _HAS_TRITON:
    # Thermostat variables are carried as Triton tuples of fp64 scalars
    # (uniform, thread-redundant); only sum(p**2/m) is a block reduction.

    @triton.jit
    def _tset(t, J: tl.constexpr, v):
        return t[:J] + (v,) + t[J + 1:]

    @triton.jit
    def _tload(ptr, M: tl.constexpr):
        t = ()
        for j in tl.static_range(M):
            t = t + (tl.load(ptr + j),)
        return t

    @triton.jit
    def _tstore(ptr, t, M: tl.constexpr):
        for j in tl.static_range(M):
            tl.store(ptr + j, t[j])

    @triton.jit
    def _qdiv(x, Q, J: tl.constexpr, FAST: tl.constexpr):
        # exact: x / Q[J] (as ASE); fast: x * (1/Q[J]) (Q tuple holds inverses)
        if FAST:
            r = x * Q[J]
        else:
            r = x / Q[J]
        return r

    @triton.jit
    def _mdiv(x, m, FAST: tl.constexpr):
        # exact: x / m (as ASE); fast: x * (1/m) (m holds inverse masses)
        if FAST:
            r = x * m
        else:
            r = x / m
        return r

    @triton.jit
    def _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4,
                 J: tl.constexpr, M: tl.constexpr, FAST: tl.constexpr):
        # ASE _integrate_p_eta_j; ke2 = sum(p**2 / m) of the current p (J == 0)
        pj = pe[J]
        if J < M - 1:
            s = libdevice.exp(_qdiv(neg_delta4 * pe[J + 1], Q, J + 1, FAST))
            pj = pj * s
        if J == 0:
            g = ke2 - ndof_kT
        else:
            g = _qdiv(pe[J - 1] * pe[J - 1], Q, J - 1, FAST) - kT
        pj = pj + delta2 * g
        if J < M - 1:
            pj = pj * s  # p_eta[j+1] unchanged in between -> identical factor
        return _tset(pe, J, pj)

    @triton.jit
    def _nhc_half(p, m, pe, eta, Q, c_ptr, M: tl.constexpr, TLOOP: tl.constexpr,
                  FAST: tl.constexpr):
        # ASE integrate_nhc with _integrate_nhc_loop
        kT = tl.load(c_ptr + 2)
        ndof_kT = tl.load(c_ptr + 3)
        # sum(p**2 / masses) only changes when p changes: once at entry and
        # after every p scaling (the backward sweep reuses the same value).
        ke2 = tl.sum(_mdiv(p * p, m, FAST), axis=0)
        for _t in tl.static_range(TLOOP):
            for k in tl.static_range(3):
                delta = tl.load(c_ptr + 4 + 4 * k)
                delta2 = tl.load(c_ptr + 5 + 4 * k)
                neg_delta4 = tl.load(c_ptr + 6 + 4 * k)
                neg_delta = tl.load(c_ptr + 7 + 4 * k)
                for jj in tl.static_range(M):
                    pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4,
                                  M - 1 - jj, M, FAST)
                for j in tl.static_range(M):
                    eta = _tset(eta, j, eta[j] + _qdiv(delta * pe[j], Q, j, FAST))
                p = p * libdevice.exp(_qdiv(neg_delta * pe[0], Q, 0, FAST))
                ke2 = tl.sum(_mdiv(p * p, m, FAST), axis=0)
                for j in tl.static_range(M):
                    pe = _p_eta_j(ke2, pe, Q, kT, ndof_kT, delta2, neg_delta4, j, M,
                                  FAST)
        return p, pe, eta

    @triton.jit
    def _nhc_pre_force_kernel(q_ptr, p_ptr, f_ptr, m_ptr, eta_ptr, pe_ptr, c_ptr,
                              n_elem, BLOCK: tl.constexpr, M: tl.constexpr,
                              TLOOP: tl.constexpr, F32_KICK: tl.constexpr,
                              FAST: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
        pe = _tload(pe_ptr, M)
        eta = _tload(eta_ptr, M)
        if FAST:
            Q = _tload(c_ptr + 16 + M, M)  # 1/Q
        else:
            Q = _tload(c_ptr + 16, M)
        # (1) p = integrate_nhc(p, dt/2)
        p, pe, eta = _nhc_half(p, m, pe, eta, Q, c_ptr, M, TLOOP, FAST)
        # (2) p += dt/2 * F(q_n) (cached forces)
        dt = tl.load(c_ptr + 0)
        dt2 = tl.load(c_ptr + 1)
        f = tl.load(f_ptr + offs, mask=mask, other=0.0)
        if F32_KICK:  # NumPy NEP50: python float * float32 array -> float32
            kick = (dt2.to(tl.float32) * f.to(tl.float32)).to(tl.float64)
        else:
            kick = dt2 * f
        p = p + kick
        # (3) q += dt * p / m
        q = tl.load(q_ptr + offs, mask=mask, other=0.0)
        q = q + _mdiv(dt * p, m, FAST)
        tl.store(q_ptr + offs, q, mask=mask)
        tl.store(p_ptr + offs, p, mask=mask)
        _tstore(pe_ptr, pe, M)
        _tstore(eta_ptr, eta, M)

    @triton.jit
    def _nhc_post_force_kernel(p_ptr, fin_ptr, fcache_ptr, m_ptr, eta_ptr, pe_ptr,
                               c_ptr, ein_ptr, eout_ptr, n_elem,
                               BLOCK: tl.constexpr, M: tl.constexpr,
                               TLOOP: tl.constexpr, F32_KICK: tl.constexpr,
                               HAS_E: tl.constexpr, FAST: tl.constexpr):
        offs = tl.arange(0, BLOCK)
        mask = offs < n_elem
        p = tl.load(p_ptr + offs, mask=mask, other=0.0)
        m = tl.load(m_ptr + offs // 3, mask=mask, other=1.0)
        pe = _tload(pe_ptr, M)
        eta = _tload(eta_ptr, M)
        if FAST:
            Q = _tload(c_ptr + 16 + M, M)  # 1/Q
        else:
            Q = _tload(c_ptr + 16, M)
        dt2 = tl.load(c_ptr + 1)
        f_in = tl.load(fin_ptr + offs, mask=mask, other=0.0)
        f64 = f_in.to(tl.float64)
        # (4) p += dt/2 * F(q_{n+1})
        if F32_KICK:
            kick = (dt2.to(tl.float32) * f_in.to(tl.float32)).to(tl.float64)
        else:
            kick = dt2 * f64
        p = p + kick
        # (5) p = integrate_nhc(p, dt/2)
        p, pe, eta = _nhc_half(p, m, pe, eta, Q, c_ptr, M, TLOOP, FAST)
        tl.store(p_ptr + offs, p, mask=mask)
        tl.store(fcache_ptr + offs, f64, mask=mask)
        _tstore(pe_ptr, pe, M)
        _tstore(eta_ptr, eta, M)
        if HAS_E:
            tl.store(eout_ptr, tl.load(ein_ptr).to(tl.float64))


class GPUNoseHooverChainNVT(StatefulIntegrator):
    """GPU/CUDA-graph friendly port of ``ase.md.nose_hoover_chain.NoseHooverChainNVT``.

    Parameters mirror ASE (ASE units): ``timestep`` and ``tdamp`` in ASE time
    units (e.g. ``0.25 * units.fs``), ``temperature_K`` in K, ``tchain``,
    ``tloop``. ``masses``: (N,) amu (``atoms.get_masses()``);
    ``positions``/``momenta``: (N, 3) (``atoms.get_positions()/get_momenta()``,
    use the float64 host arrays for ASE-exact trajectories). ``forces``:
    F(positions), may be given later via :meth:`set_forces`.
    ``force_dtype``: dtype of the model forces (MatRIS: float32, see the module
    docstring for the NumPy kick semantics). ``fast_math`` (triton only):
    reciprocal multiplies and FMA contraction (<= 1 ulp per op vs ASE).

    Per step (eager or inside one CUDA graph)::

        q = nhc.step_pre_force()           # q_{n+1}, float64 (N, 3) view
        forces, energy = model(q)          # any float dtype
        nhc.step_post_force(forces, energy=energy)

    or ``positions, momenta, forces = nhc.step(force_fn)`` with
    ``force_fn(q) -> forces`` or ``(forces, energy)``.
    """

    ensemble = "nvt_nhc"

    def __init__(
        self,
        masses,
        positions,
        momenta,
        *,
        temperature_K: float,
        timestep: float,
        tdamp: float,
        tchain: int = 3,
        tloop: int = 1,
        forces=None,
        potential_energy=None,
        device="cuda",
        backend: str = "auto",
        force_dtype: torch.dtype = torch.float64,
        numpy_kick_semantics: bool = True,
        fast_math: bool = False,
    ) -> None:
        if tchain < 1 or tloop < 1:
            raise ValueError("tchain and tloop must be >= 1")
        if tdamp <= 0 or timestep < 0:
            raise ValueError("tdamp must be positive and timestep non-negative")
        self.device = torch.device(device)
        masses_np = np.asarray(
            masses.detach().cpu().numpy() if torch.is_tensor(masses) else masses,
            dtype=np.float64,
        ).reshape(-1)
        self.n_atoms = n = int(masses_np.shape[0])
        self.tchain = int(tchain)
        self.tloop = int(tloop)
        self.temperature_K = float(temperature_K)
        self.dt = float(timestep)
        self.tdamp = float(tdamp)
        self.force_dtype = force_dtype
        self.f32_kick = bool(numpy_kick_semantics and force_dtype == torch.float32)

        # scalars with the exact Python/NumPy expressions of ASE
        self.kT = units.kB * self.temperature_K
        Q = np.zeros(self.tchain)
        Q[0] = 3 * n * self.kT * self.tdamp**2
        Q[1:] = self.kT * self.tdamp**2
        self.Q_np = Q
        self.ndof_kT = 3 * n * self.kT
        self.dt2 = self.dt / 2
        self.sub = []  # per Suzuki-Yoshida coefficient
        for coeff in FOURTH_ORDER_COEFFS:
            delta = coeff * self.dt2 / self.tloop
            self.sub.append((delta, delta / 2, -(delta / 4), -delta))

        consts = [self.dt, self.dt2, self.kT, self.ndof_kT]
        for sub in self.sub:
            consts.extend(sub)
        consts.extend(Q.tolist())
        consts.extend((1.0 / Q).tolist())  # fast_math only
        self.consts = torch.tensor(consts, dtype=torch.float64, device=self.device)
        self.Q = self.consts[_C_Q: _C_Q + self.tchain]
        self.masses = torch.tensor(
            masses_np, dtype=torch.float64, device=self.device
        ).reshape(n, 1).contiguous()
        self.inv_masses = (1.0 / self.masses).contiguous()  # fast_math only

        # state + transactional backup (single flat buffers)
        self.state = NHCState.allocate(n, self.tchain, self.device)
        self._backup = NHCState.allocate(n, self.tchain, self.device)
        self.set_state(positions=positions, momenta=momenta, forces=forces,
                       potential_energy=potential_energy)

        if backend == "auto":
            backend = (
                "triton"
                if _HAS_TRITON and self.device.type == "cuda" and 3 * n <= _TRITON_MAX_ELEMS
                else "torch"
            )
        if backend == "triton":
            if not _HAS_TRITON or self.device.type != "cuda":
                raise RuntimeError("triton backend needs triton + CUDA")
            if 3 * n > _TRITON_MAX_ELEMS:
                raise ValueError(f"triton backend supports 3N <= {_TRITON_MAX_ELEMS}")
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")
        self.backend = backend
        self.fast_math = bool(fast_math) and backend == "triton"
        self._block = max(16, triton.next_power_of_2(3 * n)) if backend == "triton" else 0
        # GB300 (fp64 latency bound): 4 warps best for 3N in [405, 1500]
        self._num_warps = (1 if self._block <= 64 else 2 if self._block <= 128
                           else 4 if self._block <= 2048 else 8 if self._block <= 8192
                           else 16)

    @classmethod
    def from_atoms(cls, atoms, *, temperature_K, timestep, tdamp, tchain=3, tloop=1,
                   **kwargs) -> "GPUNoseHooverChainNVT":
        """Build from ASE Atoms whose momenta are already set. Forces are not
        evaluated here."""
        return cls(atoms.get_masses(), atoms.get_positions(), atoms.get_momenta(),
                   temperature_K=temperature_K, timestep=timestep, tdamp=tdamp,
                   tchain=tchain, tloop=tloop, **kwargs)

    # ------------------------------------------------------------------ state
    @property
    def positions(self) -> Tensor:
        return self.state.positions

    @property
    def momenta(self) -> Tensor:
        return self.state.momenta

    @property
    def forces(self) -> Tensor:
        return self.state.forces

    @property
    def potential_energy(self) -> Tensor:
        return self.state.potential_energy

    @property
    def eta(self) -> Tensor:
        return self.state.eta

    @property
    def p_eta(self) -> Tensor:
        return self.state.p_eta

    def _as_f64(self, x) -> Tensor:
        if torch.is_tensor(x):
            return x.detach().to(device=self.device, dtype=torch.float64)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), device=self.device)

    @torch.no_grad()
    def set_state(self, *, positions=None, momenta=None, forces=None, eta=None,
                  p_eta=None, potential_energy=None, state: NHCState | None = None):
        """(Re)load state from host/device arrays (not for use inside capture)."""
        st = self.state if state is None else state
        if positions is not None:
            st.positions.copy_(self._as_f64(positions).reshape(self.n_atoms, 3))
        if momenta is not None:
            st.momenta.copy_(self._as_f64(momenta).reshape(self.n_atoms, 3))
        if forces is not None:
            st.forces.copy_(self._as_f64(forces).reshape(self.n_atoms, 3))
        if eta is not None:
            st.eta.copy_(self._as_f64(eta).reshape(-1))
        if p_eta is not None:
            st.p_eta.copy_(self._as_f64(p_eta).reshape(-1))
        if potential_energy is not None:
            st.potential_energy.copy_(self._as_f64(potential_energy).reshape(1))

    @torch.no_grad()
    def set_forces(self, forces, energy=None, state: NHCState | None = None) -> None:
        """Set F(q) (and E(q)) for the current positions (capture-safe for
        device tensors)."""
        st = self.state if state is None else state
        st.forces.copy_(
            forces.reshape(self.n_atoms, 3) if torch.is_tensor(forces)
            else self._as_f64(forces).reshape(self.n_atoms, 3)
        )
        if energy is not None:
            st.potential_energy.copy_(
                energy.reshape(1) if torch.is_tensor(energy)
                else self._as_f64(energy).reshape(1)
            )

    # -------------------------------------------------------------- step halves
    def step_pre_force(self, state: NHCState | None = None) -> Tensor:
        """ASE ``step()`` first half: NHC(dt/2); p += dt/2*F(q_n); q += dt*p/m.

        Uses the cached ``state.forces`` = F(q_n). Returns ``state.positions``
        (float64 (N, 3) view, updated in place) for the force evaluation.
        """
        st = self.state if state is None else state
        if self.backend == "triton":
            _nhc_pre_force_kernel[(1,)](
                st.positions, st.momenta, st.forces,
                self.inv_masses if self.fast_math else self.masses,
                st.eta, st.p_eta, self.consts, 3 * self.n_atoms,
                BLOCK=self._block, M=self.tchain, TLOOP=self.tloop,
                F32_KICK=self.f32_kick, FAST=self.fast_math,
                num_warps=self._num_warps, enable_fp_fusion=self.fast_math,
            )
        else:
            with torch.no_grad():
                self._torch_integrate_nhc(st)
                self._torch_kick(st, st.forces)
                st.positions.add_(st.momenta * self.dt / self.masses)
        return st.positions

    def step_post_force(self, forces: Tensor, energy: Tensor | None = None,
                        state: NHCState | None = None) -> None:
        """ASE ``step()`` second half: p += dt/2*F(q_{n+1}); NHC(dt/2).

        ``forces``: (N, 3) device tensor (float32/float64) = F(q_{n+1}); it is
        cached (as float64) for the next pre-force half. ``energy``: optional
        1-element device tensor stored as the potential energy.
        """
        st = self.state if state is None else state
        forces = forces.reshape(self.n_atoms, 3)
        if self.backend == "triton":
            if not forces.is_contiguous():
                forces = forces.contiguous()
            has_e = energy is not None
            e = energy.reshape(-1) if has_e else st.potential_energy
            _nhc_post_force_kernel[(1,)](
                st.momenta, forces, st.forces,
                self.inv_masses if self.fast_math else self.masses,
                st.eta, st.p_eta, self.consts, e, st.potential_energy,
                3 * self.n_atoms,
                BLOCK=self._block, M=self.tchain, TLOOP=self.tloop,
                F32_KICK=self.f32_kick, HAS_E=has_e, FAST=self.fast_math,
                num_warps=self._num_warps, enable_fp_fusion=self.fast_math,
            )
        else:
            with torch.no_grad():
                self._torch_kick(st, forces)
                self._torch_integrate_nhc(st)
                st.forces.copy_(forces)
                if energy is not None:
                    st.potential_energy.copy_(energy.reshape(1))

    def step(
        self, force_fn, state: NHCState | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Full step; same conventions as :meth:`GPUIntegrator.step` and
        ``GPUResidentMolecularDynamics.evaluate``.

        ``force_fn(positions_f64)`` returns the forces, or ``(forces, energy)``
        with an optional 1-element ``energy`` (``None`` keeps the stored
        potential energy). Returns ``(positions, momenta, forces)``, the
        float64 (N, 3) views of the updated state (forces = F(positions)).
        """
        st = self.state if state is None else state
        q = self.step_pre_force(st)
        result = force_fn(q)
        forces, energy = result if isinstance(result, tuple) else (result, None)
        self.step_post_force(forces, energy=energy, state=st)
        return st.positions, st.momenta, st.forces

    # ---------------------------------------------------------- torch reference
    def _torch_kick(self, st: NHCState, forces: Tensor) -> None:
        if self.f32_kick:  # NumPy NEP50 semantics for float32 forces
            st.momenta.add_((forces.to(torch.float32) * self.dt2).to(torch.float64))
        else:
            st.momenta.add_(forces.to(torch.float64) * self.dt2)

    def _torch_p_eta_j(self, st, j, delta2, neg_delta4) -> None:
        pe, Q, M = st.p_eta, self.Q, self.tchain
        if j < M - 1:
            s = torch.exp(pe[j + 1] * neg_delta4 / Q[j + 1])
            pe[j].mul_(s)
        if j == 0:
            g = (st.momenta * st.momenta / self.masses).sum() - self.ndof_kT
        else:
            g = pe[j - 1] * pe[j - 1] / Q[j - 1] - self.kT
        pe[j].add_(g * delta2)
        if j < M - 1:
            pe[j].mul_(s)

    def _torch_integrate_nhc(self, st: NHCState) -> None:
        M = self.tchain
        for _ in range(self.tloop):
            for delta, delta2, neg_delta4, neg_delta in self.sub:
                for j in range(M):
                    self._torch_p_eta_j(st, M - j - 1, delta2, neg_delta4)
                st.eta.add_(st.p_eta * delta / self.Q)
                st.momenta.mul_(torch.exp(st.p_eta[0] * neg_delta / self.Q[0]))
                for j in range(M):
                    self._torch_p_eta_j(st, j, delta2, neg_delta4)

    # -------------------------------------------------------------- observables
    def velocities(self, state: NHCState | None = None) -> Tensor:
        st = self.state if state is None else state
        return st.momenta / self.masses

    def kinetic_energy(self, state: NHCState | None = None) -> Tensor:
        """0.5 * vdot(p, p/m) (``Atoms.get_kinetic_energy``), 0-dim float64."""
        st = self.state if state is None else state
        return 0.5 * (st.momenta * (st.momenta / self.masses)).sum()

    def temperature(self, state: NHCState | None = None) -> Tensor:
        """``Atoms.get_temperature()``: 2*Ekin/(3N kB) (no constraints)."""
        return 2 * self.kinetic_energy(state) / (3 * self.n_atoms * units.kB)

    def thermostat_energy(self, state: NHCState | None = None) -> Tensor:
        """``NoseHooverChainThermostat.get_thermostat_energy``."""
        st = self.state if state is None else state
        return (self.ndof_kT * st.eta[0] + self.kT * st.eta[1:].sum()
                + (0.5 * (st.p_eta * st.p_eta) / self.Q).sum())

    def conserved_energy(self, potential_energy=None,
                         state: NHCState | None = None) -> Tensor:
        """``NoseHooverChainNVT.get_conserved_energy``: E_pot + E_kin +
        E_thermostat (uses the stored potential energy unless given)."""
        st = self.state if state is None else state
        epot = st.potential_energy[0] if potential_energy is None else potential_energy
        return epot + self.kinetic_energy(st) + self.thermostat_energy(st)

    # ------------------------------------------------------------ transactions
    def backup(self, state: NHCState | None = None) -> None:
        """Snapshot the full integrator state (one D2D copy, capture-safe)."""
        self._backup.flat.copy_((self.state if state is None else state).flat)

    def restore(self, state: NHCState | None = None) -> None:
        """Roll back to the last :meth:`backup` (one D2D copy, capture-safe)."""
        (self.state if state is None else state).flat.copy_(self._backup.flat)

    def snapshot(self) -> tuple[Tensor, Tensor]:
        """Clones of state and backup (eager; capture warmup, trials)."""
        return self.state.flat.clone(), self._backup.flat.clone()

    def load_snapshot(self, snapshot: tuple[Tensor, Tensor]) -> None:
        state, backup = snapshot
        self.state.flat.copy_(state)
        self._backup.flat.copy_(backup)

    def new_state(self) -> NHCState:
        """Allocate another state of identical shape (e.g. for replicas)."""
        return NHCState.allocate(self.n_atoms, self.tchain, self.device)

    def warmup_compile(self, force_dtypes=None, energy_dtype=torch.float32) -> None:
        """JIT-compile the Triton kernels outside of any graph capture for the
        given force dtypes (default ``self.force_dtype``), with and without
        energy. State is preserved."""
        if self.backend != "triton":
            return
        force_dtypes = [self.force_dtype] if force_dtypes is None else force_dtypes
        saved = self.snapshot()
        for dtype in force_dtypes:
            for with_energy in (True, False):
                self.step_pre_force()
                forces = self.state.forces.to(dtype).clone()
                energy = (
                    self.state.potential_energy.to(energy_dtype).clone()
                    if with_energy else None
                )
                self.step_post_force(forces, energy=energy)
                self.load_snapshot(saved)
        torch.cuda.synchronize(self.device)
