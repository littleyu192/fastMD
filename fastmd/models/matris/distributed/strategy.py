"""MatRIS's exchange strategy: the boundary-bond partial exchange as the object its adapter provides.

``MatRISAdapter.custom_strategy()`` returns :data:`BOND_PARTIAL`, a
:class:`fastmd.distributed.strategy.ExchangeStrategy`; the distributed runtime
reaches MatRIS's P>1 code only through it: the admission of a model and its
fingerprint (``admission``), the collective preparation and its padded layout
(``bond_partial``), the forward with the kernel scopes its rank-uniform
results need, and the memory admission with MatRIS's K6 tables
(``fastmd.engine.memory_admission``).
"""
from __future__ import annotations

from contextlib import nullcontext
from functools import partial

from .admission import check_converter_cutoffs, model_fingerprint, reject_enable_compile, validate_model


class BondPartialStrategy:
    """The symmetric boundary-bond partial exchange (``exchange_scheme="bond_partial"``); stateless."""

    name = "bond_partial"

    def admit_model(self, model, *, single_rank: bool) -> None:
        """A stock, unhooked FP32 MatRIS model (``validate_model``)."""
        validate_model(model, single_rank=single_rank)

    def admit_single_rank(self, model) -> None:
        """The single-GPU engine's requirements on the model beyond ``admit_model``."""
        # A runtime=None forward swaps interaction-block layers for torch.compile
        # wrappers (Interaction_Block.forward): after admission, so the
        # per-evaluation guard would reject every step and the caller's model
        # would stay rebound (validate_model refuses it for every engine).
        reject_enable_compile(model)
        # The builder takes model.config cutoffs like the distributed path,
        # single-GPU applications read the converter: both must be one graph.
        check_converter_cutoffs(model)

    def model_fingerprint(self, model) -> str:
        return model_fingerprint(model)

    def contract(self, model) -> dict:
        """The strategy's entries of the rank contract."""
        return {"name": self.name}

    def prepare(self, model, shard, context, **options):
        """``prepare_bond_partial`` (collective); the preparation's forward runs on this strategy."""
        from .bond_partial import prepare_bond_partial
        prepared = prepare_bond_partial(model, shard, context, **options)
        object.__setattr__(prepared, "strategy", self)
        return prepared

    def pad(self, prepared, capacity, **options):
        """``pad_bond_partial``: the private padded view in ``capacity``'s layout (local)."""
        from .bond_partial import pad_bond_partial
        return pad_bond_partial(prepared, capacity, **options)

    def evaluation_scope(self, recompute, admission):
        """The forwards use the mode (and the C3 and C2 decisions, see memory_admission.single_bwd_mode and
        gated_tail_fits) the evaluator's collective admission chose (its predictor's ``bind``)."""
        if admission is not None:
            return admission.bind()
        if recompute == "auto":
            from fastmd.engine import saved_recompute
            return saved_recompute.bound_admission(None)
        return nullcontext()

    def forward_scope(self, local_backend: str):
        """The P>1 forward's configuration scope: the resolved local backend profile (unified engine; §4.6,
        DSP-18)."""
        from fastmd.distributed.config import distributed_resolved
        return distributed_resolved(local_backend).scope()

    def backward_scope(self, prepared):
        """The wide GEMMs' row thresholds on the agreed padded layout (rank-uniform); unpadded: the scope's."""
        node_rows = prepared.stats.get("capacity", {}).get("node_rows")
        if node_rows is None:
            return nullcontext()
        from fastmd.engine.kernels import triton_wide_gemm
        from fastmd.distributed.config import wide_gemm_min_rows
        return triton_wide_gemm.rank_uniform(node_rows, wide_gemm_min_rows(prepared.local_backend))

    def forward(self, model, context, prepared, *, stress: bool = False):
        """The E/F forward of a validated preparation, leaving the VJP to the returned result.

        The forward (``run_bond_partial_forward``) runs inside the resolved
        configuration of the local backend's profile (nested inside the
        evaluator's identical scope) with the P>1 OpDispatcher of the prepared
        layout, at a private sink, inside the rank-uniform wide-GEMM band
        (``backward_scope``, which the result's VJP runs in too). ``stress``: with the strain leaf
        (``run_bond_partial_forward``).
        """
        from fastmd.engine.dispatch import current_policy, dispatcher_for
        from fastmd.engine.layout import ExecutionLayout
        from fastmd.engine.kernels.sink_atomics import private_sink_atomics
        from .bond_partial import MARKER, minimum_rows, run_bond_partial_forward
        assert MARKER in prepared.tensors, "expected a bond_partial preparation"
        if prepared.stats.get("min_row_padding") and "real_atom_mask" not in prepared.tensors:
            prepared = minimum_rows(prepared)
        with self.forward_scope(prepared.local_backend):
            layout = ExecutionLayout.from_bond_partial(has_line=prepared.has_line,
                                                       padded="real_atom_mask" in prepared.tensors)
            # INV-4: strict at P>1; the backend is part of the cache key (design §4.2).
            dispatcher = dispatcher_for(current_policy(), layout, strict_errors=True,
                                        backend=prepared.local_backend)
            # D3: at a private sink the atomic VJPs of this forward skip zero addends.
            with private_sink_atomics(dispatcher.private_sink), self.backward_scope(prepared):
                result = run_bond_partial_forward(model, context, prepared, dispatcher,
                                                  **({"stress": True} if stress else {}))
            result.backward_scope = partial(self.backward_scope, prepared)
            return result

    def memory_admission(self, model, *, resolved, local_backend: str, device, recompute: str, policy: str,
                         epoch: bool):
        """The evaluator's device-memory admission: the generic admission with MatRIS's memory predictor (the K6
        tables, ``K6MemoryPredictor``) of the single-GPU engine (``resolved``, one CUDA rank) or of the
        bond-partial engine (P>1, the local backend's profile)."""
        from fastmd.distributed.config import distributed_resolved
        from fastmd.engine import memory_admission
        from fastmd.engine.saved_recompute import internal_mode
        options = (resolved if resolved is not None else distributed_resolved(local_backend)).kernel_options
        predictor = memory_admission.K6MemoryPredictor.for_model(
            model, engine="single" if resolved is not None else "distributed_bond_partial",
            profile=(local_backend if resolved is None else resolved.optimization_profile),
            checkpoint=resolved is not None and resolved.checkpoint_enabled is not False,
            recompute=internal_mode(recompute), kernel_options=options, policy=policy, epoch=epoch)
        return memory_admission.MemoryAdmission(device=device, predictor=predictor)

    def close(self) -> None:
        """Nothing to release: the strategy holds no graph and installs no hook."""


# The one instance: MatRISAdapter.custom_strategy() and prepare_bond_partial's preparations.
BOND_PARTIAL = BondPartialStrategy()
