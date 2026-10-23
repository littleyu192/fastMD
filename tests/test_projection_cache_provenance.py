"""Single-device projection-cache seals, capture retention, and mutation rejection.

Tests the shared provenance API directly; distributed model-admission boundaries
are outside FastMD and are intentionally excluded.
"""
import gc
import unittest
from unittest.mock import patch
import weakref

import torch

from fastmd.models.matris import feature_embed
from fastmd.engine.cache_provenance import (
    PROJECTION_CACHE_GROUPS as REGISTERED_GROUPS, StaleCaptureError, capture_retention, publish_projection_cache,
    register_projection_cache_groups, validate_projection_caches,
)
from fastmd.models.matris.functions import IndexedProjectionConfig
from fastmd.models.matris.projection_caches import PROJECTION_CACHE_GROUPS
from single_gpu_fixtures import toy_model


def _build(model, group):
    layer = model.interaction_block[0].attn_block_atom_graph
    gate = layer.edge_nonlinear_update
    if group == "indexed_split":
        result = gate.aligned_first_linear_block(
            30000, 6, 3, torch.device("cpu"), torch.float32,
            projection_config=IndexedProjectionConfig())
        owner = gate
    elif group == "attention_merged":
        result = layer._merged_attention_projection(
            torch.empty(30000, 6), projection_config=IndexedProjectionConfig())
        owner = layer
    else:
        owner = model.three_body_embedding
        with patch.object(feature_embed, "_use_fused_threebody_linear", return_value=True):
            result = owner._combined_swish_linear_weight(torch.empty(2, 7))
    assert result is not None, group
    return owner


def _owner(model, group):
    layer = model.interaction_block[0].attn_block_atom_graph
    return {"indexed_split": layer.edge_nonlinear_update, "attention_merged": layer,
            "combined_swish": model.three_body_embedding}[group]


def _sources(owner, group):
    """The two frozen weights the group's official builder sealed as sources.

    (attention_merged also seals the aligned indexed block, itself a cache.)
    """
    if group == "combined_swish":
        return owner.angle_embedding.weight, owner.swish_layer.linear.weight
    if group == "attention_merged":
        return owner.source_weight_linear.weight, owner.target_weight_linear.weight
    return owner.mlp_gate.layers[0].weight, owner.mlp_core.layers[0].weight


def _source(owner, group, kind):
    """The sealed source a ``source_*`` (first) or ``source2_*`` (second) kind mutates."""
    return _sources(owner, group)[kind.startswith("source2_")]


def _cache_tensor(owner, group):
    value = getattr(owner, PROJECTION_CACHE_GROUPS[group][1])
    return value[0] if isinstance(value, tuple) else value


def _forbidden_payload(*args, **kwargs):
    raise AssertionError("model payload ran before model/cache mutation admission")


def _change_tensor_metadata(value, kind):
    """Use public tensor metadata operations; assert the version-blind fixture."""
    version, identity = value._version, id(value)
    shape, stride, pointer = tuple(value.shape), tuple(value.stride()), value.data_ptr()
    if kind == "pointer":
        value.data = value.detach().clone()
        assert value.data_ptr() != pointer
        assert tuple(value.stride()) == stride
    elif kind == "stride":
        assert value.ndim == 2 and value.is_contiguous()
        # Column-major interpretation of the same dense storage: same pointer,
        # dimensions and Tensor object, but a different address calculation.
        value.data = value.detach().as_strided(shape, (1, shape[0]))
        assert tuple(value.stride()) != stride
        assert value.data_ptr() == pointer
    elif kind == "requires_grad":
        assert not value.requires_grad
        value.requires_grad_(True)
    else:
        raise AssertionError(kind)
    assert value._version == version, "fixture must bypass only the version counter"
    assert id(value) == identity and tuple(value.shape) == shape


# Mutations of a sealed group. TRUSTED_AFTER leave a trusted cache (only the
# program changes); every other one also invalidates the cache seal.
SEALED_MUTATIONS = (
    "cache_inplace", "cache_rebind", "cache_resize", "cache_key", "cache_reset",
    "cache_pointer", "cache_stride", "cache_requires_grad", "cache_remove", "cache_field_remove",
    "source_inplace", "source_pointer", "source_stride", "source_requires_grad",
    "source2_inplace", "source2_pointer",
    "alias_pointer", "alias_stride", "alias_requires_grad",
    "alias_rebind", "source_inplace_rebuilt", "source2_inplace_rebuilt",
)
TRUSTED_AFTER = ("alias_rebind", "source_inplace_rebuilt", "source2_inplace_rebuilt")


def _mutate(model, owner, group, kind):
    """Apply one SEALED_MUTATIONS kind to a model whose ``group`` cache is sealed."""
    names = PROJECTION_CACHE_GROUPS[group]
    if kind == "cache_inplace":
        _cache_tensor(owner, group).add_(1)
    elif kind == "cache_rebind":
        value = getattr(owner, names[1])
        setattr(owner, names[1], tuple(part.clone() for part in value)
                if isinstance(value, tuple) else value.clone())
    elif kind == "cache_resize":
        value = _cache_tensor(owner, group)
        value.resize_(value.numel())
    elif kind == "cache_key":
        setattr(owner, names[0], getattr(owner, names[0]) + (123,))
    elif kind in ("cache_reset", "cache_remove"):
        for name in names:
            if kind == "cache_reset":
                setattr(owner, name, None)
            else:
                delattr(owner, name)
    elif kind == "cache_field_remove":
        delattr(owner, names[-1])
    elif kind in ("cache_pointer", "cache_stride", "cache_requires_grad"):
        _change_tensor_metadata(_cache_tensor(owner, group), kind.removeprefix("cache_"))
    elif kind.startswith("source") and kind.endswith(("_inplace", "_inplace_rebuilt")):
        with torch.no_grad():
            _source(owner, group, kind).add_(0.01)
        if kind.endswith("_rebuilt"):
            # The official builder refreshes the cache; the program still changed.
            _build(model, group)
    elif kind.startswith("source"):
        _change_tensor_metadata(_source(owner, group, kind), kind.split("_", 1)[1])
    elif kind == "alias_rebind":
        model._guard_cache_alias = model._guard_cache_alias.clone()
    elif kind.startswith("alias_"):
        _change_tensor_metadata(model._guard_cache_alias, kind.removeprefix("alias_"))
    else:
        raise AssertionError(kind)


class ProjectionCacheProvenanceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)


    def test_cache_tensor_versions_identity_shape_keys_and_splits_are_sealed(self):
        cases = [(group, kind) for group in PROJECTION_CACHE_GROUPS for kind in SEALED_MUTATIONS]
        cases += [("attention_merged", "split_reversal")]
        cases += [(group, "copied_cache") for group in PROJECTION_CACHE_GROUPS]
        for group, kind in cases:
            with self.subTest(group=group, mutation=kind):
                model = toy_model(norm="layer")
                if kind == "copied_cache":
                    # An unpublished copy of another model's official cache,
                    # injected into a cold model, is untrusted.
                    donor = _build(toy_model(norm="layer"), group)
                    owner = _owner(model, group)
                else:
                    owner = _build(model, group)
                if kind.startswith("alias_"):
                    # A registered alias must not mask a sealed cache change.
                    model.register_buffer("_guard_cache_alias", _cache_tensor(owner, group))
                if kind == "copied_cache":
                    for name in PROJECTION_CACHE_GROUPS[group]:
                        setattr(owner, name, getattr(donor, name))
                elif kind == "split_reversal":
                    owner._merged_attn_proj_splits = tuple(reversed(owner._merged_attn_proj_splits))
                else:
                    _mutate(model, owner, group, kind)
                if kind in TRUSTED_AFTER:
                    self.assertEqual(validate_projection_caches(owner), ())
                else:
                    self.assertTrue(validate_projection_caches(owner))

    def test_a_capture_keeps_what_it_read_and_a_replay_rejects_changed_sources(self):
        class Graph:  # a CUDA graph's replay interface
            def replay(self):
                self.replays = getattr(self, "replays", 0) + 1

        for group in PROJECTION_CACHE_GROUPS:
            for check_replay in (True,):
                with self.subTest(group=group, check_replay=check_replay):
                    model, graph = toy_model(norm="layer"), Graph()
                    with capture_retention(graph, check_replay=check_replay):
                        owner = _build(model, group)
                    cached = weakref.ref(_cache_tensor(owner, group))
                    self.assertTrue(graph._matris_derived)
                    graph.replay()
                    self.assertEqual(graph.replays, 1)
                    source = _sources(owner, group)[-1]
                    with torch.no_grad():
                        source.add_(0.0)  # a new version, as an in-place weight load
                    _build(model, group)  # rebuilt for the new key: the module drops its cache
                    self.assertIsNot(_cache_tensor(owner, group), cached())
                    self.assertIsNotNone(cached())  # the capture still holds what it read
                    if check_replay:
                        with self.assertRaises(StaleCaptureError):
                            graph.replay()
                    self.assertEqual(graph.replays, 1)
                    if group != "combined_swish" and check_replay:
                        # Current MD/FastMD deliberately do not retain Python
                        # cache objects while Dynamo is tracing. The upstream
                        # expectation of automatic retention during _build's
                        # trace contradicts that guard on torch 2.8.
                        tracing = Graph()
                        with capture_retention(tracing, check_replay=True), \
                                patch("torch.compiler.is_compiling", return_value=True):
                            _build(model, group)
                        self.assertEqual(tracing._matris_derived, ())
                        compiled_builder = torch.compile(_build, backend="eager", dynamic=True)
                        compiled_owner = compiled_builder(model, group)
                        self.assertEqual(validate_projection_caches(compiled_owner), ())
                        # A later eager capture must still retain the cache,
                        # and its replay must reject a changed source weight.
                        compiled = Graph()
                        with capture_retention(compiled, check_replay=True):
                            compiled_owner = _build(model, group)
                        self.assertTrue(compiled._matris_derived)
                        compiled.replay()
                        with torch.no_grad():
                            _sources(compiled_owner, group)[-1].add_(0.0)
                        with self.assertRaises(StaleCaptureError):
                            compiled.replay()
                        self.assertEqual(compiled.replays, 1)
                    # No reference cycle: the graph and its caches go without a collection.
                    graph_ref = weakref.ref(graph)
                    collecting = gc.isenabled()
                    gc.disable()
                    try:
                        del graph
                        self.assertIsNone(graph_ref())
                        self.assertIsNone(cached())
                    finally:
                        if collecting:
                            gc.enable()

    def test_a_registered_cache_group_keeps_its_fields(self):
        register_projection_cache_groups(PROJECTION_CACHE_GROUPS)  # MatRIS's, registered again: nothing changes
        split, swish = PROJECTION_CACHE_GROUPS["indexed_split"], PROJECTION_CACHE_GROUPS["combined_swish"]
        for groups, error in (({"indexed_split": split[:2]}, "registered with other fields"),
                              ({"unregistered": ("_unregistered_field",), "other": swish[:1]}, "belongs to the group")):
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                register_projection_cache_groups(groups)
        self.assertNotIn("unregistered", REGISTERED_GROUPS)  # a call that raises registers nothing

    def test_sources_and_cache_tensors_are_not_retained_by_provenance(self):
        for group in ("indexed_split", "combined_swish"):
            with self.subTest(group=group):
                module = torch.nn.Module()
                source = torch.ones(2, 2)
                cached = source.clone()
                key, weight, *rest = PROJECTION_CACHE_GROUPS[group]
                setattr(module, key, (source.data_ptr(), source._version))
                setattr(module, weight, cached)
                for name in rest:
                    setattr(module, name, None)
                publish_projection_cache(module, group, (source,))
                # A boundary's model stamp keeps no source metadata alive.
                self.assertEqual(validate_projection_caches(module), ())
                source_ref, cached_ref, module_ref = weakref.ref(source), weakref.ref(cached), weakref.ref(module)
                del source
                gc.collect()
                self.assertIsNone(source_ref())
                self.assertTrue(validate_projection_caches(module))
                del module, cached
                gc.collect()
                self.assertIsNone(cached_ref())
                self.assertIsNone(module_ref())


if __name__ == "__main__":
    unittest.main()
