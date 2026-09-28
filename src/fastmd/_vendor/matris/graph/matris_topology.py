"""MatRIS graph bindings for the model-independent topology contract IR."""

from __future__ import annotations

from fastmd._vendor.matris.config import env_value

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from .radiusgraph import RadiusGraph
from .topology_contract import (
    BuilderEvidence,
    ContractProperty,
    DescriptorLifetime,
    DomainExtent,
    DomainSpec,
    ExtentPolicy,
    ExtentScope,
    PropertyKind,
    RelationSpec,
    TopologyDomain,
    TopologyCertificate,
    TopologyInstance,
    TopologySchema,
    VerificationMode,
    VerificationReport,
    certify_topology,
)


ATOM = "atom"
DIRECTED_EDGE = "directed_edge"
UNDIRECTED_PAIR = "undirected_pair"
TRIPLET = "triplet"
GRAPH = "graph"
DUMMY = "dummy"

EDGE_TARGET = "edge_target"
EDGE_SOURCE = "edge_source"
DIRECTED_TO_PAIR = "directed_to_pair"
PAIR_TO_DIRECTED = "pair_to_directed"
ATOM_GRAPH_ID = "atom_graph_id"
TRIPLET_OWNER = "triplet_owner"
TRIPLET_PAIR_A = "triplet_pair_a"
TRIPLET_PAIR_B = "triplet_pair_b"
TRIPLET_EDGE_A = "triplet_edge_a"
TRIPLET_EDGE_B = "triplet_edge_b"

_VERIFIED_BINDING_CACHE: dict[
    tuple, tuple[TopologyCertificate, VerificationReport]
] = {}


_BASE_RELATIONS = (
    RelationSpec(EDGE_TARGET, DIRECTED_EDGE, ATOM),
    RelationSpec(EDGE_SOURCE, DIRECTED_EDGE, ATOM),
    RelationSpec(DIRECTED_TO_PAIR, DIRECTED_EDGE, UNDIRECTED_PAIR),
    RelationSpec(PAIR_TO_DIRECTED, UNDIRECTED_PAIR, DIRECTED_EDGE),
    RelationSpec(
        ATOM_GRAPH_ID,
        ATOM,
        GRAPH,
        lifetime=DescriptorLifetime.CAPTURE,
        invalidated_by=("batch_changed",),
    ),
    RelationSpec(TRIPLET_OWNER, TRIPLET, ATOM),
    RelationSpec(TRIPLET_PAIR_A, TRIPLET, UNDIRECTED_PAIR),
    RelationSpec(TRIPLET_PAIR_B, TRIPLET, UNDIRECTED_PAIR),
    RelationSpec(TRIPLET_EDGE_A, TRIPLET, DIRECTED_EDGE),
    RelationSpec(TRIPLET_EDGE_B, TRIPLET, DIRECTED_EDGE),
)


def _matris_properties(
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> tuple[ContractProperty, ...]:
    properties: list[ContractProperty] = [
        ContractProperty(PropertyKind.BOUNDED, relation.name)
        for relation in _BASE_RELATIONS
    ]
    properties.extend(
        [
            ContractProperty(PropertyKind.GRAPH_CONTIGUOUS, ATOM_GRAPH_ID),
            ContractProperty(
                PropertyKind.PAIR_CARDINALITY,
                DIRECTED_TO_PAIR,
                (2,),
                ExtentScope.ACTIVE,
            ),
            ContractProperty(
                PropertyKind.REVERSE_PAIR_INVOLUTION,
                DIRECTED_TO_PAIR,
                (EDGE_SOURCE, EDGE_TARGET, PAIR_TO_DIRECTED),
                ExtentScope.ACTIVE,
            ),
        ]
    )
    if incidence_consistent:
        properties.append(
            ContractProperty(
                PropertyKind.INCIDENCE,
                TRIPLET_OWNER,
                (TRIPLET_EDGE_A, TRIPLET_EDGE_B, EDGE_TARGET),
                ExtentScope.ACTIVE,
            )
        )
    if atom_target_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                EDGE_TARGET,
                scope=ExtentScope.ACTIVE,
            )
        )
    if line_owner_sorted:
        properties.append(
            ContractProperty(
                PropertyKind.SORTED_BY,
                TRIPLET_OWNER,
                scope=ExtentScope.ACTIVE,
            )
        )
    if fixed_capacity:
        properties.extend(
            ContractProperty(PropertyKind.FIXED_CAPACITY, domain)
            for domain in (ATOM, DIRECTED_EDGE, UNDIRECTED_PAIR, TRIPLET)
        )
    if real_dummy_closed:
        properties.extend(
            ContractProperty(PropertyKind.REAL_DUMMY_CLOSED, relation.name)
            for relation in _BASE_RELATIONS
        )
    return tuple(properties)


def matris_builder_evidence(
    issuer: str,
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> BuilderEvidence:
    """Create static evidence emitted by a MatRIS graph builder."""

    return BuilderEvidence(
        issuer=issuer,
        properties=_matris_properties(
            atom_target_sorted=atom_target_sorted,
            line_owner_sorted=line_owner_sorted,
            fixed_capacity=fixed_capacity,
            real_dummy_closed=real_dummy_closed,
            incidence_consistent=incidence_consistent,
        ),
    )


def matris_schema(
    *,
    atom_target_sorted: bool,
    line_owner_sorted: bool,
    fixed_capacity: bool = False,
    real_dummy_closed: bool = False,
    incidence_consistent: bool = True,
) -> TopologySchema:
    policy = ExtentPolicy.FIXED_CAPACITY if fixed_capacity else ExtentPolicy.DYNAMIC
    domains = (
        DomainSpec(ATOM, TopologyDomain.ATOM, policy, has_real_prefix=fixed_capacity),
        DomainSpec(
            DIRECTED_EDGE,
            TopologyDomain.DIRECTED_EDGE,
            policy,
            has_real_prefix=fixed_capacity,
        ),
        DomainSpec(
            UNDIRECTED_PAIR,
            TopologyDomain.UNDIRECTED_PAIR,
            policy,
            has_real_prefix=fixed_capacity,
        ),
        DomainSpec(TRIPLET, TopologyDomain.TRIPLET, policy, has_real_prefix=fixed_capacity),
        DomainSpec(GRAPH, TopologyDomain.GRAPH, ExtentPolicy.DYNAMIC),
        DomainSpec(DUMMY, TopologyDomain.DUMMY, policy, has_real_prefix=fixed_capacity),
    )
    return TopologySchema(
        domains=domains,
        relations=_BASE_RELATIONS,
        properties=_matris_properties(
            atom_target_sorted=atom_target_sorted,
            line_owner_sorted=line_owner_sorted,
            fixed_capacity=fixed_capacity,
            real_dummy_closed=real_dummy_closed,
            incidence_consistent=incidence_consistent,
        ),
    )


def _extent(capacity: int, real: int | None = None, active: int | None = None) -> DomainExtent:
    return DomainExtent(
        real=capacity if real is None else int(real),
        active=capacity if active is None else int(active),
        capacity=int(capacity),
    )


@dataclass(frozen=True)
class MatrisTopologyBinding:
    instance: TopologyInstance
    report: VerificationReport


def bind_matris_topology(
    batched_graph: dict,
    graphs: Sequence[RadiusGraph],
    *,
    mode: VerificationMode | str | None = None,
) -> MatrisTopologyBinding:
    """Bind and certify the descriptor produced by :func:`process_graphs`."""

    graph_evidence = []
    for graph in graphs:
        evidence = getattr(graph, "topology_evidence", None)
        if evidence is None:
            evidence = matris_builder_evidence(
                "matris.legacy_radius_graph",
                atom_target_sorted=bool(getattr(graph, "atom_target_sorted", False)),
                line_owner_sorted=bool(getattr(graph, "line_atom_sorted", False)),
            )
        graph_evidence.append(evidence)
    atom_sorted_claim = ContractProperty(
        PropertyKind.SORTED_BY, EDGE_TARGET, scope=ExtentScope.ACTIVE
    )
    line_sorted_claim = ContractProperty(
        PropertyKind.SORTED_BY, TRIPLET_OWNER, scope=ExtentScope.ACTIVE
    )
    atom_sorted = all(
        atom_sorted_claim in evidence.properties for evidence in graph_evidence
    )
    line_sorted = all(
        line_sorted_claim in evidence.properties for evidence in graph_evidence
    )
    incidence_claim = ContractProperty(
        PropertyKind.INCIDENCE,
        TRIPLET_OWNER,
        (TRIPLET_EDGE_A, TRIPLET_EDGE_B, EDGE_TARGET),
        ExtentScope.ACTIVE,
    )
    fixed_capacity_claim = ContractProperty(PropertyKind.FIXED_CAPACITY, ATOM)
    fixed_capacity = all(
        fixed_capacity_claim in evidence.properties for evidence in graph_evidence
    )
    incidence_consistent = all(
        incidence_claim in evidence.properties for evidence in graph_evidence
    )
    schema = matris_schema(
        atom_target_sorted=atom_sorted,
        line_owner_sorted=line_sorted,
        fixed_capacity=fixed_capacity,
        incidence_consistent=incidence_consistent,
    )
    atom_graph = batched_graph["atom_graph_dict"]
    line_graph = batched_graph["line_graph_dict"]
    atom_count = int(batched_graph["atomic_numbers"].shape[0])
    edge_count = int(atom_graph["target_index"].shape[0])
    pair_count = int(batched_graph["undirected2directed"].shape[0])
    triplet_count = int(line_graph["line_graph"].shape[0])
    graph_count = int(batched_graph["num_graphs"])
    device = batched_graph["atomic_numbers"].device
    empty = torch.empty(0, dtype=torch.long, device=device)
    relations = {
        EDGE_TARGET: atom_graph["target_index"],
        EDGE_SOURCE: atom_graph["source_index"],
        DIRECTED_TO_PAIR: batched_graph["directed2undirected"],
        PAIR_TO_DIRECTED: batched_graph["undirected2directed"],
        ATOM_GRAPH_ID: batched_graph["atom_segment"],
        TRIPLET_OWNER: line_graph.get("atom_list", empty),
        TRIPLET_PAIR_A: line_graph.get("target_index", empty),
        TRIPLET_PAIR_B: line_graph.get("source_index", empty),
        TRIPLET_EDGE_A: line_graph.get("target_DE_index", empty),
        TRIPLET_EDGE_B: line_graph.get("source_DE_index", empty),
    }
    extents = {
        ATOM: _extent(atom_count),
        DIRECTED_EDGE: _extent(edge_count),
        UNDIRECTED_PAIR: _extent(pair_count),
        TRIPLET: _extent(triplet_count),
        GRAPH: _extent(graph_count),
        DUMMY: _extent(0),
    }
    issuers = sorted(
        {
            getattr(evidence, "issuer", "legacy")
            for evidence in graph_evidence
        }
    )
    common_properties = set(graph_evidence[0].properties)
    for graph_proof in graph_evidence[1:]:
        common_properties.intersection_update(graph_proof.properties)
    missing_claims = set(schema.properties) - common_properties
    if missing_claims:
        missing = ", ".join(
            sorted(f"{prop.kind.value}({prop.subject})" for prop in missing_claims)
        )
        raise ValueError(f"MatRIS builders do not jointly prove schema claims: {missing}")
    evidence = BuilderEvidence("+".join(issuers), schema.properties)
    instance = TopologyInstance(
        schema=schema,
        extents=extents,
        relations=relations,
        producer=evidence.issuer,
    )
    verify_mode = VerificationMode(
        mode or env_value("MATRIS_TOPOLOGY_VERIFY", VerificationMode.BUILDER.value)
    )
    first_relation = next(iter(relations.values()))
    cache_key = (
        schema.fingerprint,
        instance.extent_signature,
        evidence.issuer,
        first_relation.device.type,
        first_relation.device.index,
        verify_mode.value,
        tuple(id(graph) for graph in graphs),
    )
    cached = _VERIFIED_BINDING_CACHE.get(cache_key)
    if cached is not None:
        certificate, report = cached
        certified = instance.with_certificate(certificate)
    else:
        if (
            verify_mode != VerificationMode.BUILDER
            and first_relation.is_cuda
            and torch.cuda.is_current_stream_capturing()
        ):
            raise RuntimeError(
                "topology verification must run once before CUDA Graph capture"
            )
        certified, report = certify_topology(
            instance, evidence=evidence, mode=verify_mode
        )
        if verify_mode != VerificationMode.BUILDER:
            _VERIFIED_BINDING_CACHE[cache_key] = (certified.certificate, report)
    return MatrisTopologyBinding(certified, report)
