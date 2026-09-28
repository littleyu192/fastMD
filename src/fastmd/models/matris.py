"""MatRIS adapter with topology-directed inference lowerings."""
from ase import units

from .base import ModelBackend, ModelCapabilities


class MatRISModel(ModelBackend):
    """MatRIS inference with variable-cell graph reuse.

    compute_stress=True always returns energy, forces and stress together, for
    ASE NPT or cell relaxation without separate force/stress evaluations.
    """

    capabilities = ModelCapabilities(
        frozenset({"energy", "forces", "stress", "magmoms"}),
        frozenset({"energy", "forces", "stress", "magmoms"}),
        cuda_graph_variable_cell=True,
    )

    def __init__(self, *, checkpoint=None, model_name="matris_10m_oam",
                 compute_stress=False, compile_lowerings=False, expert_overrides=None, **kwargs):
        super().__init__(**kwargs)
        # NPT requests forces and stress separately. Return both from one model
        # evaluation so ASE can cache them and the captured task stays stable.
        self.compute_stress = bool(compute_stress)
        from fastmd._vendor.matris.applications.base import MatRISCalculator
        from fastmd._vendor.matris.config import CapacityConfig, InferenceConfig, resolve_config
        overrides = dict(expert_overrides or {})
        if compile_lowerings:
            overrides["MATRIS_COMPILED_LOWERINGS"] = True
        options = dict(
            optimization_profile="topology" if self.config.enable_fusions else "generic",
            checkpoint="off", report=False, expert_overrides=overrides,
            capacity=CapacityConfig(u_step=self.config.edge_capacity_step or 512,
                                    t_step=self.config.triplet_capacity_step or 8192,
                                    warmup=self.config.warmup_steps),
        )
        self._graph_config = resolve_config(InferenceConfig(execution="model_graph", **options))
        self.calculator = MatRISCalculator(model_path=str(checkpoint) if checkpoint is not None else None,
                                          model=model_name, device=str(self.device), task="ef",
                                          config=InferenceConfig(execution="eager", **options))
        self.model = self.calculator.model.eval()
        self.model.enable_checkpoint = False
        for layer in self.model.interaction_block:
            layer.enable_checkpoint = False
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        # Honor explicit CPU even on hosts with CUDA and GPU neighbor operators.
        if self.device.type == "cpu":
            self.model.graph_converter.algorithm = "legacy"
            self.calculator._gpu_graph = False
        self.runner = None
        self._task = None

    def _prediction_task(self, properties):
        return "efsm" if "magmoms" in properties else (
            "efs" if self.compute_stress or "stress" in properties else "ef")

    def graph_unavailable_reason(self, properties):
        reason = super().graph_unavailable_reason(properties)
        if reason:
            return reason
        if self.model.reference_energy is None:
            return "MatRIS CUDA Graph requires a reference-energy table for isolated-atom handling"
        from fastmd._vendor.matris.graph.gpu_graph_builder import op_available
        return None if op_available else "GPU neighbor operators missing; install fastMD[cuda]"

    def _predict_eager(self, atoms, properties):
        calc = self.calculator
        calc.task = self._prediction_task(properties)
        calc.key = set(calc.task) | {"atoms_per_graph", "ref_energy"}
        calc.calculate(atoms.copy(), list(properties), ["positions"])
        return {k: v for k, v in calc.results.items() if v is not None and k in self.capabilities.properties}

    def _predict_graph(self, atoms, properties):
        from fastmd._vendor.matris.applications.cuda_graph import BucketedGraphRunner
        from fastmd._vendor.matris.graph.gpu_graph_builder import atoms_to_graph_gpu
        task = self._prediction_task(properties)
        if self.runner is None or task != self._task:
            self.runner = BucketedGraphRunner(
                self.model, task=task,
                enable_model_fusions=self.config.enable_fusions,
                config=self._graph_config,
            )
            self._task = task
        converter = self.model.graph_converter
        with self._graph_config.scope():
            # Rebuild topology from the current cell, including periodic images.
            # The runner copies lattice/coordinates/topology into address-stable
            # buffers and selects a larger capacity bucket when necessary.
            graph = atoms_to_graph_gpu(atoms, atom_graph_cutoff=converter.atom_graph_cutoff,
                                       line_graph_cutoff=converter.line_graph_cutoff, device=self.device)
            output, n = self.runner.run(graph)
        scale = n if self.model.is_intensive else 1
        results = {"energy": float(output["e"][0].detach()) * scale,
                   "forces": output["f"][0][:n].detach().cpu().numpy().copy()}
        if "s" in task:
            results["stress"] = output["s"][0].detach().cpu().numpy().copy() * units.GPa
        if "magmoms" in properties:
            results["magmoms"] = output["m"][0][:n].detach().cpu().numpy().copy()
        if len(self.runner.cache) > self.config.max_cached_graphs:
            self.clear_graphs()
        return results

    def clear_graphs(self):
        self.runner = None
        self._task = None

    def stats(self):
        return {**super().stats(), "optimization_profile": self._graph_config.optimization_profile,
                "compute_stress": self.compute_stress,
                "cache": self.runner.stats() if self.runner else {},
                "kernel_options": dict(self._graph_config.kernel_options),
                "merged_projection_caches": sum(
                    getattr(module, "_merged_attn_proj_weight", None) is not None
                    for module in self.model.modules()),
                "checkpointing": self.model.enable_checkpoint}
