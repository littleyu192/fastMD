"""Execution layout: the capability fields the operator dispatcher reads (design §4.1).

Step 2.3 carries only the fields today's P=1 sites select on. Each is derived
from the topology lowering plan that ``process_graphs`` compiles from the graph
certificate (``fastmd.engine.topology_lowering``), so the layout says what the
certificate proves, never what a kernel would like:

* ``pair_layout``: ``"pair_index"`` when the plan lowers ``pair_algebra`` to
  paired rows (every undirected pair has exactly two directed rows), else
  ``"generic"``. ``"interleaved"`` is the P>1 layout (not used at P=1 yet).
* ``incidence``: line-graph incidence of the envelope product, ``"strong"``
  (``line_incidence.direct``), ``"bounded"`` (``line_incidence.direct_bounded``)
  or ``"none"``; ``compiled_envelope`` when the plan selects the compiled
  (Inductor) lowering for the envelope site.
* ``sorted_by``: relations whose indexed-affine VJP may assume sorted rows
  (``"atom_indexed_affine"`` also needs the contiguous sorted segment VJP).
  A pure performance hint (DSP-07).
* ``identity_csr``: graph kinds (``"atom"``, ``"line"``) whose target segment
  attention CSR is the identity permutation.

Step 2.4 adds the P>1 fields a distributed preparer declares
(``from_bond_partial``); at P=1 they keep their defaults and stay out of the
fingerprint, so P=1 fingerprints (and dispatcher cache keys) are unchanged:

* ``route``: ``"exchange"`` (P>1 transport between the model stages).
* ``pair_layout`` ``"interleaved"``: directed rows 2u and 2u+1 are the two
  orientations of undirected row u (E == 2U, padding included).
* ``sink``: ``"private_single"`` when padding rows of every domain are routed
  to one private sink row/segment (the padded model_graph layout); gathers and
  sums then use the zero-skipping kernels (their atomics omit the zero
  cotangents/addends of padding rows), and the weighted atomic segment sum is
  not selected. ``"none"`` otherwise.
* ``owner_csr``: graph kinds whose owner segments come as prepared CSR
  (``graph["_owner_{side}_csr"] = (perm, offsets, skip_segment)``, int64);
  segment attention reads them instead of sorting in the model body.
* ``shared_value_rows``: graph kinds whose source and target attention sides
  read the same value rows (both sides passed as one tensor object) with the
  prepared CSR over those rows; a necessary condition of the paired segment
  attention besides tensor identity.
* ``required_keys``: graph-dict keys the generic lowerings of this layout read
  (e.g. ``source_bincount``); ``check_required_keys`` rejects a graph dict
  missing one at construction instead of a KeyError inside the model.

Row counts, dtypes and strides are not layout fields: the dispatcher checks
them per call from host metadata. Precision is not a layout field (INV-5).
The fingerprint covers every field; it is stable across steps with the same
certified topology schema, so the dispatcher cache does not grow with ragged
shapes.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import asdict, dataclass
from functools import cached_property
import hashlib
import json

# Layouts per lowering plan object (plans are cached by topology_lowering).
_LAYOUT_CACHE_SIZE = 64
_LAYOUTS: OrderedDict[int, tuple[object, ExecutionLayout]] = OrderedDict()


# P>1 fields (step 2.4) and their P=1 defaults; a field at its default is left
# out of the fingerprint payload, so P=1 layouts keep their step-2.3 fingerprints.
_P1_DEFAULTS = {"sink": "none", "owner_csr": frozenset(), "shared_value_rows": frozenset(),
                "required_keys": frozenset()}


@dataclass(frozen=True)
class ExecutionLayout:
    route: str = "identity"
    pair_layout: str = "pair_index"
    incidence: str = "strong"
    compiled_envelope: bool = False
    sorted_by: frozenset[str] = frozenset()
    identity_csr: frozenset[str] = frozenset()
    sink: str = "none"
    owner_csr: frozenset[str] = frozenset()
    shared_value_rows: frozenset[str] = frozenset()
    required_keys: frozenset[str] = frozenset()

    @cached_property
    def fingerprint(self) -> str:
        fields = {name: sorted(value) if isinstance(value, frozenset) else value
                  for name, value in asdict(self).items()
                  if name not in _P1_DEFAULTS or value != _P1_DEFAULTS[name]}
        payload = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @classmethod
    def from_bond_partial(cls, *, has_line: bool, padded: bool = False) -> ExecutionLayout:
        """The P>1 layout of the A2 bond-partial scheme (``fastmd.models.matris.distributed.bond_partial``).

        Every rank runs the single-GPU layers on its local graph: interleaved
        pairs of every local bond, owner CSR of both sides of each kind over
        the local rows, and both attention sides of each kind read the same
        message rows (the CSR is over those rows, no folding). Triplet rows
        index the local bond rows directly (``"bounded"``: base[source] *
        base[target]). The padded (model_graph) layout routes padding to one
        private sink per domain (the last owned atom, the last bond). Every
        generic segment lowering has its bincount keys prepared. The only P>1
        layout since step 4.2 (the rows exchange's layout was removed with it).
        """
        kinds = frozenset(("atom", "line") if has_line else ("atom",))
        return cls(
            route="exchange",
            pair_layout="interleaved",
            incidence="bounded" if has_line else "none",
            sink="private_single" if padded else "none",
            owner_csr=kinds,
            shared_value_rows=kinds,
            required_keys=frozenset({"source_bincount", "target_bincount"}),
        )

    def check_required_keys(self, kind: str, graph: dict) -> None:
        """Reject a graph dict that lacks a key a generic lowering will read."""
        missing = sorted(key for key in self.required_keys if key not in graph)
        if missing:
            raise KeyError(f"{kind} graph lacks the keys {missing} required by layout "
                           f"{self.fingerprint} (ExecutionLayout.required_keys)")

    @classmethod
    def from_plan(cls, plan) -> ExecutionLayout:
        """The P=1 layout certified by a ``LoweringPlan`` (cached per plan)."""
        key = id(plan)
        entry = _LAYOUTS.get(key)
        if entry is not None and entry[0] is plan:
            _LAYOUTS.move_to_end(key)
            return entry[1]
        uses = plan.uses
        sorted_by = set()
        if (uses("atom_indexed_affine", "indexed_affine.decomposed_sorted_vjp")
                and uses("sorted_segment_vjp", "segment_vjp.contiguous")):
            sorted_by.add("atom_indexed_affine")
        if uses("line_indexed_affine", "indexed_affine.decomposed_sorted_vjp"):
            sorted_by.add("line_indexed_affine")
        identity_csr = {kind for kind, request in (("atom", "target_segment_attention"),
                                                   ("line", "line_target_segment_attention"))
                        if uses(request, "segment_attention.identity_csr")}
        layout = cls(
            pair_layout="pair_index" if uses("pair_algebra", "pair_algebra.paired_rows") else "generic",
            incidence=("strong" if uses("line_incidence", "line_incidence.direct")
                       else "bounded" if uses("line_incidence", "line_incidence.direct_bounded")
                       else "none"),
            compiled_envelope=uses("line_envelope", "line_incidence.compiled"),
            sorted_by=frozenset(sorted_by),
            identity_csr=frozenset(identity_csr),
        )
        # Keep the plan alive with its entry so that its id cannot be reused.
        _LAYOUTS[key] = (plan, layout)
        if len(_LAYOUTS) > _LAYOUT_CACHE_SIZE:
            _LAYOUTS.popitem(last=False)
        return layout
