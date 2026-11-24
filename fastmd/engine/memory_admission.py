"""Device-memory admission: reject a mode that cannot fit before it OOMs (K6).

Before the first evaluation and before any later capture, the predicted peak
device memory of the chosen mode is compared with what this process can still
obtain: ``torch.cuda.mem_get_info()`` free bytes plus the caching allocator's
reserved bytes, minus the bytes that were allocated before the engine's first
evaluation (model and caller tensors) and a safety margin. The predictions are
process reserved peaks, which include the engine's own graph and state; the
P=1 entry points' auto decisions (``saved_recompute.AutoRecompute.decide_local``)
take the same baseline, the bytes allocated when the entry point was created. An explicitly
requested mode that does not fit fails fast with predicted vs available and
what would fit; ``execution="auto"`` only chooses among fitting modes. At P>1
every rank gathers (predictions, available) once per admission event, so all
ranks take the same decision; no admission work runs on a replay or an
ordinary eager step. ``MATRIS_MEMORY_ADMISSION=off`` disables it (the
behavior before K6); it is part of the evaluator's rank contract.

The admission itself (``MemoryAdmission``: the available memory and margin,
the gather over ranks, the fitting modes, the decision record and the error)
does not depend on the model; the peak predictions and the memory decisions a
model takes itself are its ``MemoryPredictor``. MatRIS's is
``K6MemoryPredictor``: the tables below, and its recompute level, C3 and C2
decisions.

Model: peak = base + a * rows (+ b * line_rows), where rows are the engine's
triplet rows (single engine: its own; distributed bond_partial: the installed
capacity's triplet rows, see "Recalibration" below) and line_rows the rows
exchange's local line rows (owned + halo received; history: that exchange and
its "distributed" entries were removed in step 4.2, bond_partial has no line
term). The peak is the allocator's reserved
peak over the first three evaluations; for model_graph it holds the side-stream
warmup and the graph pool together (the warmup cache is released only after
capture). Edges and atoms scale with the triplet rows on the calibration
material and are not separate terms. Calibrated on GB300 (RDX, FP32
highest, native, checkpoint off, PYTORCH_CUDA_ALLOC_CONF unset; the K6
admission measurements, peak reserved GiB). The single-engine topology and the distributed bond_partial
native entries, all three saved_tensor_recompute levels, were refitted on
GB200 (2026-09-29) with the mixed-width indexed projections (step 2.5b D8,
``MATRIS_MIXED_WIDTH_CAT``, default on): D8 changes those peaks (off lower,
on/deep higher; the earlier bond_partial eager on/deep and single on/deep
coefficients under-predicted D8 by up to 18% / 16%). Where D8 is inactive
(``MATRIS_MIXED_WIDTH_CAT=0`` or the indexed projection off; see
``mixed_width_active``) those twelve entries come from
``CALIBRATION_MIXED_WIDTH_OFF``, fitted the same way on the same node on the
parent tree (without D8 the off peaks are up to 9% higher and the
bond_partial model_graph on/deep peaks up to 20% higher than the D8 fits).
The legacy profile, the torch_reference backend and the (removed) rows scheme
do not run D8 (unchanged). Where the merged-projection single-GEMM backward runs (C3,
``single_bwd_mode``), six of these entries add a slope increment
(``SINGLE_BWD_GIB_PER_MILLION_ROWS``; its gradient buffer holds columns longer).

  single engine, triplet rows  951,366 / 3,805,464 / 7,610,928 / 15,221,856
    topology eager (D8)   11.29 /  44.59 /  87.76 / 174.69
    topology model_graph  23.21 /  44.87 /  91.89 / 178.59  (D8; the P=1
                          graph pool fix and D8 took it from 244.34 at 7.6M)
    legacy eager          13.53 /  53.49 / 106.68 / 213.42
    legacy model_graph    25.33 / 103.93 / 219.17 / OOM in capture
  distributed rows (history: the rows exchange and these entries were
  removed in step 4.2; max rank), 10,752 / 43,008 atoms; recalibrated on the
  merged m1 code (2026-09-29; the coefficients fitted on the points of
  2026-09-28, e.g. 2 GPUs eager 71.51 / 272.20, which predate steps
  2.4-2.5a, over-predicted m1 by 19-31%)
    2 GPUs eager          59.72 / 237.79
    2 GPUs model_graph    64.37 / 262.61
    4 GPUs eager          30.74 / 122.10
    4 GPUs model_graph    34.21 / 132.41

  distributed bond_partial (engine "distributed_bond_partial", max rank;
  owned triplet rows 1,902,732 / 3,805,464 / 7,610,928 / 15,221,856 =
  10,752 P=4 / 10,752 P=2 / 43,008 P=4 / 43,008 P=2; native with D8 on
  GB200, 2026-09-29, the last column OOM on GB200 (183.6 GiB in use);
  torch_reference 2026-09-28/29)
    eager (D8)            24.10 /  47.91 /  95.51 / OOM
    model_graph (D8)      26.03 /  49.63 /  99.13 / OOM
    torch_reference eager 44.26 /  86.92 / 173.87 / OOM (predicted 354)
    torch_reference mg    47.18 /  90.88 / 181.95 / OOM (predicted 370)
  (native predicted within [+2.0%, +2.1%] eager and [+2.0%, +3.7%] model_graph;
  torch_reference within [+2.0%, +3.5%])

Coefficients are least squares plus the base shift that puts every measured
point at or above the line; with ``_SLACK`` each is predicted within
[+2%, +13%] (model_graph is slightly convex in the rows); the recalibrated
distributed rows native points (history) within [+2.0%, +2.4%] (eager) and
[+2.0%, +5.5%] (model_graph). Owned and line rows are nearly collinear on the
symmetric calibration supercells, so only the combination is meaningful.
The fit pairs each run's largest peak with the largest rows over ranks;
eager admission predicts every rank from its own rows, within -7.3% to
+3.8% of that rank's own peak (the admission margin covers the deficit).

The distributed rows coefficients above (history, removed in step 4.2) are
the native local backend's. The torch_reference backend (the CLI default)
peaked 1.06-1.45x above the native prediction and had its own rows
coefficients (from the two-node validation runs, same protocol and fit, every
rank of P=2/4/8 at 10,752 and 43,008 atoms except P=2 at 43,008, GB300,
2026-09-29; peak reserved GiB, max rank):

  torch_reference, 10,752 / 43,008 atoms
    2 GPUs eager          102.2 / -          model_graph 100.1 / -
    4 GPUs eager           52.8 / 210.3      model_graph  53.0 / 205.8
    8 GPUs eager           28.1 / 107.4      model_graph  27.5 / 103.9

predicted within [+2%, +14%] (eager) and [+2%, +20%] (model_graph). On the
same runs the native coefficients predict every native point at +17..+47%.

``InferenceConfig.saved_tensor_recompute="on"`` and ``"deep"`` have their own
entries, keyed ``(engine, profile, execution, "narrow" | "deep")``: same
protocol and fit, peak reserved GiB. "on" (legacy on GB200, 2026-09-29, after
the legacy recipes were added; topology and bond_partial native with D8 on
GB200, 2026-09-29):

  single engine, triplet rows  951,366 / 3,805,464 / 7,610,928 / 15,221,856
    topology eager (D8)    5.72 /  21.55 /  42.93 /  85.17
    topology model_graph   12.56 /  23.12 /  46.00 /  91.80  (D8)
    legacy eager           7.76 /  23.48 /  47.55 /  92.55
    legacy model_graph    16.67 /  24.31 /  51.13 / 102.03
    (D8, 43,008 atoms = 30,443,712 rows: topology eager 167.82, model_graph 182.97)
  distributed bond_partial, owned rows 1,902,732 / 3,805,464 / 7,610,928 / 15,221,856
    eager (D8)            14.37 /  27.60 /  54.62 / 108.06
    model_graph (D8)      15.28 /  29.02 /  59.81 / 121.44

"deep" (GB200, 2026-09-29; topology and bond_partial native with D8):

  single engine, triplet rows  951,366 / 3,805,464 / 7,610,928 / 15,221,856
    topology eager (D8)    4.23 /  15.64 /  31.12 /  62.24
    topology model_graph   9.59 /  17.68 /  35.08 /  64.56  (D8)
    legacy eager           5.43 /  17.45 /  34.22 /  68.34
    legacy model_graph    13.70 /  20.07 /  37.46 /  71.09
    (D8, 43,008 atoms = 30,443,712 rows: topology eager 127.89, model_graph 139.82)
  distributed bond_partial, owned rows 1,902,732 / 3,805,464 / 7,610,928 / 15,221,856
    eager (D8)            11.81 /  22.81 /  45.53 /  90.24
    model_graph (D8)      13.21 /  25.85 /  51.60 / 100.97

predicted within [+2%, +15%] (eager) and [+2%, +47%] (model_graph: convex at
small N, where memory does not bind); the D8 refits (the single engine up to
43,008 atoms, where on/deep still fit a GB200) within [+2.0%, +20%] (eager)
and [+2.0%, +29%] (model_graph), CALIBRATION_MIXED_WIDTH_OFF within
[+2.0%, +51%], the largest at 1,344 atoms.

GB300 check of the D8 table (2026-09-29, same probe, with the gated-LN bias
fold D13a): every point above measures the same peak as on GB200 (the
same where both devices ran the same tree; the D13a tree differs from the D8
tree by -6% to +2% at some small points, from the allocator's layout).
Points only GB300 holds, up to where the admission stops admitting (its
budget is about 268 GiB there), stay within the predictions, except for
three model_graph entries. Their slopes are raised by the minimum that keeps every GB200 and
GB300 point at or below the prediction:
  single topology model_graph (off)   11.118 -> 11.122  32,256 atoms,
                                                        22,832,784 rows: 271.90
  single topology model_graph deep     4.455 ->  4.527  64,512 / 80,640 atoms,
                                     45,665,568 / 57,081,960 rows: 215.32 / 269.03
  bond_partial model_graph (off)      12.835 -> 12.865  53,760 atoms on 2 GPUs,
                                                        19,176,304 rows: 253.27
Near the device limit, model_graph peaks do not grow monotonically in the rows
(single deep: 1.011 / 0.989 / 1.016 of the earlier prediction at 64,512 /
72,576 / 80,640 atoms), so a size between the measured points can still
exceed the prediction by about 1%; the admission margin covers that.

Recalibration on the engine of 2026-09-30 (steps 2.5-C1/C2/C3/C7 and the
bond_partial topology epoch; GB200, same probe with C2 on and C3 off, max
rank): single topology 1,344 to 59,136 atoms and bond_partial at P=2 and P=4
(automatic grid; P=4 also 4x1x1) from 1,344 atoms, every level up to its
largest admitted size. The topology epoch's capacity pads every rank to the
tier of the MAX demand over ranks plus a growth band (1/128 eager, 1/32
model_graph), in eager as in model_graph, while the admission predicted each
rank from its own demand: with main's coefficients bond_partial eager peaked
up to 16.8% above the prediction off (P=4 at 1,344 atoms, 336 per rank;
4x1x1 at 10,752 atoms 9.0%), 5.3% on and 6.3% deep (4x1x1 at 10,752), 3.0%
on at P=4 43,008; bond_partial model_graph on 9.1% (P=4 at 1,344); single
topology eager on 1.0% (43,008 atoms, C2 on). Since then the bond_partial
admission reads the topology epoch's installed capacity rows (the tier a
model_graph capacity installs for its demand), and every bond_partial
coefficient is main's mapped to them (slope / (1 + band), base less the
slope times min_pad_t: at least main's prediction for every demand; the
torch_reference and D8-inactive entries only so, not re-measured).
Entries where a peak still exceeded its prediction keep their slope and
raise their base by the least amount that covers every peak:
  single topology eager on     1.499 -> 3.132   43,008 atoms: 173.89
  bond_partial eager off       0.254 -> 0.976   P=4 43,008: 98.19
  bond_partial eager on        1.016 -> 4.065   P=4 129,024: 168.13
  bond_partial eager deep      0.646 -> 0.750   P=4 10,752: 12.24
  bond_partial model_graph on  0.003 -> 0.180   P=4 1,344: 2.25
(mapped base -> new base; the binding point, peak reserved GiB). Every other
entry keeps main's line; C3's increments cover every C3 peak of the sweep
unchanged, and C2 (on in the sweep) needs none. The largest admitted rows on
a 184 GiB GB200 shrink by 0.93% (single eager on), 0.43% / 1.75% / 0.07%
(bond_partial eager off / on / deep) and at most 0.14% (bond_partial
model_graph, the mapping's rounding); every other entry and level is
unchanged.

The P>1 entries were measured with the StageTape, which step 4.2 deleted; the
six bond_partial native entries are refit on autograd (GB200, 2026-09-30, the
same probe with C2 on and C3 off, P=2 and P=4 on the automatic grid from 1,344
atoms, P=4 up to each level's largest admitted size; capacity rows, max rank).
Peak reserved GiB, capacity rows 1.9M / 3.8M / 7.7M / 13.5M / 15.4M (and the
largest):
  eager off          22.96 / 44.51 / 89.19 / 155.65 / 177.85
  eager on           11.32 / 22.47 / 45.99 /   -    /  87.60  (23.0M: 136.23)
  eager deep          8.91 / 16.62 / 34.01 /   -    /  65.34  (29.0M: 123.49)
  model_graph off    23.50 / 45.27 / 90.52 / 158.16 / 178.63
  model_graph on     11.42 / 22.35 / 44.81 /   -    /  88.75  (22.2M: 125.38)
  model_graph deep    8.69 / 17.14 / 34.44 /   -    /  64.67  (26.7M: 112.46)
With the tape's coefficients these were predicted at 1.10 (off) to 1.66 (deep)
times the peak, so auto climbed a level early. Each entry takes the tightest
cover (least-squares slope of the points from 1M rows, the base that puts every
point at or below the prediction with the slack): eager off 0.976 + 12.415 ->
0.780 + 11.307, on 4.065 + 6.985 -> 1.105 + 5.754, deep 0.750 + 5.850 -> 1.481 +
4.151; model_graph off 1.562 + 12.476 -> 1.275 + 11.134, on 0.180 + 7.770 ->
0.432 + 5.537, deep 1.422 + 6.391 -> 1.610 + 4.093. C3 (forced on, same sweep)
adds nothing at recompute on on autograd (its two increments are removed) and
up to +3.0 GiB at deep (eager 7.7M rows), which the deep increments cover. The
torch_reference and D8-inactive entries keep the tape's (higher) lines.
The bond_partial model_graph "on" slope is raised to 5.655 (2026-10-01): at
P=8, 268,800 atoms (24.8M capacity rows per rank, ``main`` after #33) it
peaked at 143.42 GiB, 2.1% above the prediction with the slack (allocated
126.83 GiB: about 1% more per row than the P=4 sweep, and 1% more allocator
cache). The excess grows with the rows, so the slope covers it and the base
stays; at 16 GPUs (12.4M rows) the same configuration peaked at 68.47 GiB.

Refit of 2026-09-30 (same engine, GB200, the K6 probe and the single-GPU GPU MD entry point, off, near
the device limit: 20,160 / 21,504 / 22,848 atoms = 14,270,490 / 15,221,856 / 16,173,222 rows, K6 probe
162.20 / 170.01 / 180.80 GiB, GPU MD 158.37 / 170.79 / 179.46): the single topology eager off entry
over-predicted every point by 4.4% to 5.4% (21,504 atoms: 178.63 predicted) and is refit to the tightest
cover, the least-squares slope of the points from 3.8M rows and the base that puts every point at or below
the prediction with the slack (1.081 + 11.434 -> 3.289 + 10.913; binding: 20,160 atoms, K6 probe). The
single topology model_graph off entry (over by 7.6% at 21,504 atoms) stays: its line is bound by the
GB300 32,256-atom point of an earlier tree (271.90 GiB), which the GB200 cannot re-measure.

Model-aware calibration (2026-09-30). Every table above is the RDX 1M
fine-tune's (4 layers, node/edge/three-body widths 128/32/32). Peaks per
triplet row scale with the architecture: the 4M FeCOH model (4 layers,
128/128/128) peaked 3.1-3.4x and the library default 10M OAM model (10
layers, 128/128/128, three-body cutoff 4.5 A) 6.3-7.9x above these
predictions (10M, 1,344 RDX atoms, eager, recompute off: 170.3 GiB measured,
23.9 predicted; recompute auto kept "off" and OOMed from 2,688 atoms on, and
at P=4 from 10,752). ``memory_model(model)`` selects the calibration from the
model's ``Architecture``: the measured architecture with the smallest per-row
bytes whose envelope holds it (no deeper or wider, no more pair edges per
triplet row: pair cutoff at most, three-body cutoff at least, the same layer
kinds), else ``derived``. The 4M and 10M tables (``ARCHITECTURE_CALIBRATION``)
are fitted like the RDX ones (least squares plus the base shift that puts
every measured point at or below the line) on GB200 points of the single
topology engine, eager and model_graph at every recompute level (1,165 to
10,752 atoms: RDX, FeCOH, Si, SiO2, LiFePO4, Cu), and the bond_partial
entries are the RDX ones scaled by the architecture's measured single-engine
slope ratio at the same level, their base shifted to cover the P=4 points
(4M 10,752 atoms off / on; 10M 1,344 off, 2,688 on, 5,376 deep, 10,752 on /
deep). They include C3 (on in every point) and the C2 path, so they get no
increments. Every measured 4M and 10M point is within its prediction
(0.20-0.98 of it), and every configuration that OOMed there is predicted above
the device. Above ``GRAPH_GAIN_MAX_ROWS`` a P=1 model_graph runner of these
models holds its envelope bucket only (engine.capacity_policy), about
half the multi-bucket peak the model_graph tables were fitted on: their
model_graph entries there are ``ENVELOPE_CALIBRATION`` (2026-10-01, ``main``
after #39, the same probe, 44 points from 414,720 to 7,610,928 triplet rows,
each with one bucket; 0.68-0.98 of the new prediction, 0.23-0.97 of the old
one). Up to the limit the multi-bucket entries stay (4M model_graph off: base
2.317 -> 3.578, to cover Cu 1,000 and SiO2 1,125 atoms with four buckets,
29.00 and 30.40 GiB, 1.7% and 2.3% above the old prediction).
Entries a measured architecture was not measured with (the
legacy and generic profiles, torch_reference, the rows scheme) and every
architecture outside the measured envelopes are ``derived``: the RDX entry
times the analytic per-row ratio of ``derived_bytes_per_row`` (saved FP32
bytes per triplet row from widths and depth, with 0.25 pair edges per
triplet row, twice RDX's, or more where the cutoffs give more:
``derived_edges_per_row``) times ``DERIVED_SAFETY`` (1.25; the analytic
ratio alone under-predicts five of the 56 measured 4M and 10M runs, all
model_graph, by up to 9%: 10M deep at 1,344 atoms).
For a model outside every measured envelope (``MemoryModel.calibrated``
False), or a derived entry of a measured one (``entry_calibrated``), no
automatic choice relaxes memory: the recompute ladder climbs on the derived
bound, C3 stays off, execution="auto" runs eager, and C2 still runs (it
lowers the peak). For every model but RDX an unpredicted (None) peak does
not count as fitting an automatic choice: the ladder takes "deep" and auto
runs eager (an explicit mode and the layer checkpoint are admitted unchecked,
as before). The RDX model's predictions, and so its admission boundary, are
unchanged.
A model from which no MatRIS ``Architecture`` can be derived (``architecture``
None: no ``config`` dict whose MatRIS keys it can read, e.g. another
potential's module, or no model at all) has no bound: ``memory_model`` returns
``UNCALIBRATED_MEMORY_MODEL`` (calibration "uncalibrated", architecture
None), which predicts no peak, is not calibrated and is strict, so the
ladder takes "deep", C3 stays off, execution="auto" runs eager and a P=1
runner precaptures no neighbor bucket. (Before, such a model got the RDX
tables, the only calibration whose unpredicted peaks count as fitting.)

Uncalibrated configurations (checkpoint on/auto, the single engine's generic
profile, recompute with the torch_reference backend) are admitted unchecked
for the RDX model and recorded as such: checkpointing exists to fit larger systems, and an eager
upper bound would reject exactly those. The layer checkpoint's peaks are
linear (single topology eager: 5.00 / 36.97 / 73.89 / 147.63 / 166.07 GiB at
1,344 / 10,752 / 21,504 / 43,008 / 48,384 atoms; 59,136 OOMs on GB200), but
saved_tensor_recompute "deep" holds less at every size (60.51 / 120.63 GiB at
21,504 / 43,008 atoms), costs less time, and is admitted; the single engine's
GPU MD and calculator entry points run no execution-mode admission, so an
entry would cover only the distributed API at P=1.
"""
from __future__ import annotations

from dataclasses import astuple, dataclass, replace
import hashlib
import os
from typing import ContextManager, Protocol, runtime_checkable

import torch

ENVIRONMENT = "MATRIS_MEMORY_ADMISSION"
_GIB = 1 << 30
# Relative slack on every prediction; the margin kept free is the larger of
# 2 GiB and 2% of the device (43,008 atoms on 2 GPUs eager peaked 1.5% under
# the device and OOMed at its third step from fragmentation).
_SLACK = 0.02
_MARGIN_BYTES = 2 * _GIB
_MARGIN_FRACTION = 0.02


@dataclass(frozen=True)
class Coefficients:
    base_gib: float
    gib_per_million_rows: float
    gib_per_million_line_rows: float = 0.0

    def predict(self, rows: int, line_rows: int = 0) -> int:
        gib = (self.base_gib + self.gib_per_million_rows * rows / 1e6
               + self.gib_per_million_line_rows * line_rows / 1e6)
        return int(max(gib, 0.0) * (1 + _SLACK) * _GIB)


# (engine, profile, execution[, recompute]) -> peak reserved; see the module docstring.
# Distributed: profile None = the native local backend, else the backend.
CALIBRATION = {
    ("single", "topology", "eager"): Coefficients(3.289, 10.913),
    ("single", "topology", "model_graph"): Coefficients(12.638, 11.122),
    ("single", "legacy", "eager"): Coefficients(0.212, 14.007),
    ("single", "legacy", "model_graph"): Coefficients(-2.423, 29.170),
    # The bond_partial exchange (no routed rows; rows: the installed capacity's triplet rows); see the docstring.
    ("distributed_bond_partial", None, "eager"): Coefficients(0.780, 11.307),
    ("distributed_bond_partial", None, "model_graph"): Coefficients(1.275, 11.134),
    ("distributed_bond_partial", "torch_reference", "eager"): Coefficients(0.922, 22.550),
    ("distributed_bond_partial", "torch_reference", "model_graph"): Coefficients(2.075, 22.941),
    # saved_tensor_recompute "on" (internal "narrow"); see the docstring.
    ("single", "topology", "eager", "narrow"): Coefficients(3.132, 5.497),
    ("single", "topology", "model_graph", "narrow"): Coefficients(6.967, 5.876),
    ("single", "legacy", "eager", "narrow"): Coefficients(2.074, 5.976),
    ("single", "legacy", "model_graph", "narrow"): Coefficients(10.767, 6.203),
    ("distributed_bond_partial", None, "eager", "narrow"): Coefficients(1.105, 5.754),
    ("distributed_bond_partial", None, "model_graph", "narrow"): Coefficients(0.432, 5.655),
    # saved_tensor_recompute "deep"; see the docstring.
    ("single", "topology", "eager", "deep"): Coefficients(0.239, 4.196),
    ("single", "topology", "model_graph", "deep"): Coefficients(5.348, 4.527),
    ("single", "legacy", "eager", "deep"): Coefficients(1.226, 4.420),
    ("single", "legacy", "model_graph", "deep"): Coefficients(9.756, 4.140),
    ("distributed_bond_partial", None, "eager", "deep"): Coefficients(1.481, 4.151),
    ("distributed_bond_partial", None, "model_graph", "deep"): Coefficients(1.610, 4.093),
}


# The entries whose peaks depend on the mixed-width indexed projections (step 2.5b D8), with D8
# inactive (the parent tree, GB200, 2026-09-29; same probe and fit, peak reserved GiB, triplet
# rows as in the tables above):
#   single topology  eager off 12.40 / 49.12 / 97.56 / OOM, model_graph off 25.49 / 49.99 / 99.84 / OOM,
#                    eager on 7.42 / 21.09 / 41.29 / 82.17 / 165.47, model_graph on 16.09 / 21.97 / 45.58 /
#                    90.96 / 181.70, eager deep 5.38 / 14.50 / 29.79 / 59.54 / 115.23, model_graph deep 11.06 /
#                    14.92 / 30.54 / 60.92 / 121.66 (the fifth: 43,008 atoms, 30,443,712 rows)
#   bond_partial     eager off 26.63 / 51.51 / 104.60 / OOM, model_graph off 28.16 / 54.26 / 109.99 / OOM,
#                    eager on 13.97 / 25.03 / 51.56 / 99.08, model_graph on 18.75 / 36.51 / 72.93 / 143.91,
#                    eager deep 10.32 / 19.04 / 37.58 / 74.93, model_graph deep 14.97 / 29.01 / 57.94 / 117.68
CALIBRATION_MIXED_WIDTH_OFF = {
    ("single", "topology", "eager"): Coefficients(6.913, 11.910),
    ("single", "topology", "model_graph"): Coefficients(14.768, 11.269),
    ("single", "topology", "eager", "narrow"): Coefficients(2.294, 5.384),
    ("single", "topology", "model_graph", "narrow"): Coefficients(10.594, 5.779),
    ("single", "topology", "eager", "deep"): Coefficients(2.380, 3.755),
    ("single", "topology", "model_graph", "deep"): Coefficients(7.388, 3.861),
    ("distributed_bond_partial", None, "eager"): Coefficients(0.507, 13.595),
    ("distributed_bond_partial", None, "model_graph"): Coefficients(0.737, 13.946),
    ("distributed_bond_partial", None, "eager", "narrow"): Coefficients(2.595, 6.381),
    ("distributed_bond_partial", None, "model_graph", "narrow"): Coefficients(1.328, 9.119),
    ("distributed_bond_partial", None, "eager", "deep"): Coefficients(1.043, 4.828),
    ("distributed_bond_partial", None, "model_graph", "deep"): Coefficients(0.237, 7.493),
}


# C3 (``MATRIS_MERGED_PROJECTION_SINGLE_BWD``, default auto): the merged attention projection's
# consumers write their gradients into one [rows, sum(splits)] buffer, allocated by the first of
# them in the backward (the segment attention: the alphas), so the p0 columns are held until the
# cat3 epilogue backward. The last block (whose backward runs first, before the caching allocator
# holds a free block of that size) keeps the per-output backward. Measured on GB200 (2026-09-29,
# same probe and sweep as above, peak reserved GiB, C3 on minus off; recompute =
# saved_tensor_recompute; every other point of the sweep: +0.00):
#   single topology eager, recompute deep         7,610,928 rows: +2.32 (0 at 1.0M, 3.8M, 15.2M)
#   single topology model_graph, recompute deep     951,366 rows: +0.46;  15,221,856 rows: +7.27
#   bond_partial eager, recompute on and deep     1.9M-15.2M rows: +0.48 GiB per M rows (P=2 and 4)
#   bond_partial model_graph, recompute on        1,902,732 rows: +0.94 (P=4); P=2 up to +1.80
#   bond_partial model_graph, recompute deep    1.9M-15.2M rows: +0.63 .. +7.49
# With the fused narrow tail (C2, ``MATRIS_FUSED_GATED_TAIL``) on, which changes the GatedMLP
# backward's allocations inside the buffer's claim window (C3 on minus C2 alone, same sweep):
#   single topology eager, recompute deep     1.0M / 3.8M / 7.6M rows: +0.46 / +1.82 / +1.81
#   bond_partial model_graph, recompute deep       3.8M / 15.2M rows: +1.88 / +7.49 (P=2)
# The increments below cover each such C3 peak, or at least C3's own share of it, plus 0.02; on the
# recalibrated coefficients (see the module docstring) every C3 peak of that sweep, P=2 and P=4 on the
# automatic and 4x1x1 grids, is within the prediction with them. They are added to the slope where C3 runs: always
# under "on", and under "auto" (the default) for its decision, which keeps C3 only while the
# prediction with it fits (``saved_recompute.AUTO_THRESHOLD``); the recompute ladder and the
# execution modes read the predictions without it.
SINGLE_BWD_GIB_PER_MILLION_ROWS = {
    ('single', 'topology', 'eager', 'deep'): 0.20,
    ('single', 'topology', 'model_graph', 'deep'): 0.23,
    ('distributed_bond_partial', None, 'eager', 'deep'): 0.51,
    ('distributed_bond_partial', None, 'model_graph', 'deep'): 0.39,
}


def single_bwd_mode(kernel_options) -> str:
    """C3 of a resolved configuration's kernel options: "off" without the merged attention
    projections and the indexed projection they need, else its switch ("on", "off" or "auto":
    on only while the prediction with it fits, see ``saved_recompute.AUTO_THRESHOLD``)."""
    if not (kernel_options.get("MATRIS_MERGED_ATTENTION_PROJECTIONS", True)
            and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False)):
        return "off"
    return kernel_options.get("MATRIS_MERGED_PROJECTION_SINGLE_BWD", "auto")


# C2 (``MATRIS_FUSED_GATED_TAIL``, default on): the fused GatedMLP narrow tail writes its linear outputs
# and gradients over dead tensors and lowers the allocated peak at every point of the sweep (up to
# -5.4 GiB), and the reserved peak at almost every point (up to -7.3 GiB). With less memory cached
# after the first evaluations, the caching allocator adds segments later at two points, where the
# reserved peak ends above the entry's prediction (GB200 2026-09-30, same probe and sweep; C2 on /
# off, GiB; off = main's peak):
#   single topology eager, recompute on         15,221,856 rows: 87.01 / 85.17 (predicted 86.88)
#   bond_partial model_graph, recompute on (P=4) 1,902,732 rows: 15.64 / 15.55 (predicted 15.58)
# (bond_partial eager on the epoch topology exceeded its prediction without C2, by up to 5.5%.) C2
# runs only while the prediction with its increment fits AUTO_THRESHOLD of the budget at the
# recompute level chosen without it (the P=1 entry point's state, auto or explicit level, or the
# evaluator's admission: ``gated_tail_fits``); the admission, the recompute ladder and the execution
# modes read the predictions without it, so the largest system per level is unchanged.
# Entries without an increment always run C2. The increments were 0.03 and 0.10 (the second on C7 and
# C3); the recalibration fitted every entry with C2 on (its peaks, with C3 off and on, are within the
# predictions without an increment), so none is needed and C2 always runs.
GATED_TAIL_GIB_PER_MILLION_ROWS = {}


def gated_tail_active(kernel_options) -> bool:
    """Whether a resolved configuration's kernel options run C2 (its switch and the indexed
    projection whose Cat3/Cat4 outputs it fuses)."""
    return bool(kernel_options.get("MATRIS_FUSED_GATED_TAIL", True)
                and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False))


def gated_tail_fits(engine: str, profile: str | None, execution: str, rows: int, line_rows: int, *,
                    recompute: str, mixed_width: bool, budget: int, single_bwd: bool = True,
                    memory: "MemoryModel | None" = None, threshold: float | None = None) -> bool:
    """C2's memory decision for one rank at a recompute level: no increment for the entry, or the
    prediction with it (and with C3's where ``single_bwd``) fits ``threshold`` (default
    ``saved_recompute.AUTO_THRESHOLD``, see ``saved_recompute.policy_threshold``) of ``budget``
    (uncalibrated: no)."""
    if not _gated_tail_increment(engine, profile, execution, recompute, memory):
        return True
    if threshold is None:
        from fastmd.engine.saved_recompute import AUTO_THRESHOLD as threshold
    peak = predict(engine, profile, execution, rows, line_rows, recompute=recompute, mixed_width=mixed_width,
                   single_bwd=single_bwd, gated_tail=True, memory=memory)
    return peak is not None and peak <= threshold * budget


def _gated_tail_increment(engine, profile, execution, recompute, memory=None) -> float:
    if memory is not None and memory.calibration != "rdx_1m":
        return 0.0  # the other tables were fitted with C2's path (no increment)
    if engine.startswith("distributed") and profile != "torch_reference":
        profile = None
    key = (engine, profile, execution) if recompute == "off" else (engine, profile, execution, recompute)
    return GATED_TAIL_GIB_PER_MILLION_ROWS.get(key, 0.0)


def mixed_width_active(kernel_options) -> bool:
    """Whether a resolved configuration's kernel options run D8 (the mixed-width switch
    and the indexed projection it needs): selects the calibration of the twelve D8 entries."""
    return bool(kernel_options.get("MATRIS_MIXED_WIDTH_CAT", True)
                and kernel_options.get("MATRIS_FUSED_INDEXED_CAT_LINEAR", False))


@dataclass(frozen=True)
class Architecture:
    """The hyperparameters of a MatRIS model that set its bytes per triplet row (``architecture``)."""
    num_layers: int
    node_feat_dim: int
    edge_feat_dim: int
    three_body_feat_dim: int
    mlp_hidden: int  # the widest MLP hidden layer
    num_radial: int
    num_angular: int
    max_n: int
    max_l: int
    pairwise_cutoff: float
    three_body_cutoff: float
    kinds: tuple = ("layer", "fourier", "gatemlp", False)  # norm, three-body expansion, conv MLP, smoothed delta
    dtype_bytes: int = 4

    def fingerprint(self) -> str:
        return hashlib.sha256(repr(astuple(self)).encode()).hexdigest()[:12]

    def within(self, other: "Architecture") -> bool:
        """Whether ``other``'s calibration bounds this model: no deeper or wider, no more pair edges per
        triplet row (pair cutoff at most, three-body cutoff at least), the same layer kinds."""
        sizes = ("num_layers", "node_feat_dim", "edge_feat_dim", "three_body_feat_dim", "mlp_hidden",
                 "num_radial", "num_angular", "max_n", "max_l", "dtype_bytes")
        return (all(getattr(self, size) <= getattr(other, size) for size in sizes) and self.kinds == other.kinds
                and self.pairwise_cutoff <= other.pairwise_cutoff and self.three_body_cutoff >= other.three_body_cutoff)


def architecture(model) -> Architecture | None:
    """A model's ``Architecture`` from its ``config``; None without one (``memory_model``: uncalibrated)."""
    config = getattr(model, "config", None)
    if not isinstance(config, dict):
        return None
    try:
        hidden = config["mlp_hidden_dims"]
        # As the energy head reads it: one width, or any iterable of widths (a list, tuple, range or array; a
        # string is one value).
        hidden = max([hidden] if isinstance(hidden, (int, str)) else list(hidden), default=0)
        kinds = tuple(value.lower() if isinstance(value, str) else bool(value) for value in (
            config["norm_type"], config["three_body_expansion"], config["graph_conv_mlp"],
            config["use_smoothed_for_delta_edge"]))
        parameter = next(iter(model.parameters()), None) if hasattr(model, "parameters") else None
        return Architecture(
            int(config["num_layers"]), int(config["node_feat_dim"]), int(config["edge_feat_dim"]),
            int(config["three_body_feat_dim"]), int(hidden), int(config["num_radial"]), int(config["num_angular"]),
            int(config["max_n"]), int(config["max_l"]), float(config["pairwise_cutoff"]),
            float(config["three_body_cutoff"]), kinds, 4 if parameter is None else int(parameter.element_size()))
    except (KeyError, TypeError, ValueError):
        return None


# The measured architectures: the RDX 1M fine-tune (the tables above), the 4M FeCOH fine-tune and the
# library default 10M OAM model (``matris_10m_oam``).
RDX_1M = Architecture(4, 128, 32, 32, 256, 7, 7, 4, 4, 6.0, 4.0)
FECOH_4M = Architecture(4, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.0)
OAM_10M = Architecture(10, 128, 128, 128, 256, 7, 7, 4, 4, 6.0, 4.5)

# Per-architecture tables (GB200, 2026-09-30; see the module docstring). Peak reserved GiB, max rank:
#   4M single topology, triplet rows 152,262 / 951,366 / 1,218,096 / 3,805,464 / 7,610,928 (FeCOH 1,165,
#   RDX 1,344, FeCOH 9,320, RDX 5,376, RDX 10,752 atoms)
#     eager off 6.60 / 38.36 / 51.46 / 152.76 / OOM, on 3.15 / 17.35 / 23.42 / 67.09 / 133.98,
#     deep 2.41 / 13.71 / 18.77 / 61.63 / 104.93; model_graph off 14.13 / 78.71 / 104.35 / - / -,
#     on 6.96 / 35.34 / 49.03 / 69.24 / -, deep 5.68 / 31.62 / 43.19 / 65.60 / -
#   10M single topology (cutoff 4.5 A), eager off: Si 1,000 / 1,728 / 4,096 (240,000 / 414,720 / 983,040 rows)
#     24.74 / 42.34 / 99.71, SiO2 1,125 (935,250) 83.36, LiFePO4 1,008 (1,099,440) 97.93, RDX 1,344
#     (1,951,244) 170.27, RDX 2,688 OOM; on Si 1,000 / 4,096 10.07 / 40.59, RDX 1,344 / 2,688 65.01 / 127.83,
#     Cu 4,096 (7,053,312) OOM; deep Si 1,000 / 4,096 7.54 / 28.87, RDX 1,344 / 2,688 49.18 / 98.04,
#     Cu 4,096 159.67, RDX 5,376 OOM; model_graph off Si 1,000 / 1,728 51.83 / 86.76 (Si 4,096 and RDX 1,344
#     OOM), on Si 1,000 / 4,096 21.39 / 83.38, RDX 1,344 132.01 (2,688 OOM), deep 15.81 / 67.71 / 101.10
#   bond_partial native P=4 eager, capacity triplet rows: 4M RDX 10,752 (1,925,120) off 79.76, on 37.33;
#     10M RDX 1,344 (524,288) off 47.23 (model_graph 48.11), 2,688 (1,024,000) on 35.39, 5,376 (1,974,272)
#     deep 52.98, 10,752 (3,940,352) on 135.67, deep 99.91
_TABLE_4M = {
    ("single", "topology", "eager"): Coefficients(2.991, 39.791),
    ("single", "topology", "model_graph"): Coefficients(3.578, 83.764),
    ("single", "topology", "eager", "narrow"): Coefficients(2.149, 17.463),
    ("single", "topology", "model_graph", "narrow"): Coefficients(30.870, 14.909),
    ("single", "topology", "eager", "deep"): Coefficients(8.672, 13.916),
    ("single", "topology", "model_graph", "deep"): Coefficients(25.439, 14.573),
    ("distributed_bond_partial", None, "eager"): Coefficients(3.397, 43.205),
    ("distributed_bond_partial", None, "model_graph"): Coefficients(5.436, 43.417),
    ("distributed_bond_partial", None, "eager", "narrow"): Coefficients(12.914, 23.810),
    ("distributed_bond_partial", None, "model_graph", "narrow"): Coefficients(0.572, 26.241),
    ("distributed_bond_partial", None, "eager", "deep"): Coefficients(2.374, 20.133),
    ("distributed_bond_partial", None, "model_graph", "deep"): Coefficients(4.501, 21.466),
}
_TABLE_10M = {
    ("single", "topology", "eager"): Coefficients(17.173, 83.960),
    ("single", "topology", "model_graph"): Coefficients(3.849, 199.920),
    ("single", "topology", "eager", "narrow"): Coefficients(9.664, 31.460),
    ("single", "topology", "model_graph", "narrow"): Coefficients(20.533, 63.932),
    ("single", "topology", "eager", "deep"): Coefficients(11.210, 22.250),
    ("single", "topology", "model_graph", "deep"): Coefficients(19.459, 49.083),
    ("distributed_bond_partial", None, "eager"): Coefficients(7.167, 91.163),
    ("distributed_bond_partial", None, "model_graph"): Coefficients(11.470, 91.611),
    ("distributed_bond_partial", None, "eager", "narrow"): Coefficients(23.264, 42.895),
    ("distributed_bond_partial", None, "model_graph", "narrow"): Coefficients(1.030, 47.273),
    ("distributed_bond_partial", None, "eager", "deep"): Coefficients(3.796, 32.191),
    ("distributed_bond_partial", None, "model_graph", "deep"): Coefficients(7.197, 34.321),
}
# name -> (architecture, table); the RDX table is CALIBRATION (with its D8 and C3 variants).
CALIBRATED_ARCHITECTURES = {"rdx_1m": (RDX_1M, None), "fecoh_4m": (FECOH_4M, _TABLE_4M),
                            "oam_10m": (OAM_10M, _TABLE_10M)}
ARCHITECTURE_CALIBRATION = {name: table for name, (_, table) in CALIBRATED_ARCHITECTURES.items() if table}
# P=1 model_graph runners of these calibrations hold their envelope bucket only above these triplet rows
# (the graph-gain limit of engine.capacity_policy, which reads this table); there their single
# topology model_graph peaks are ENVELOPE_CALIBRATION's (main after #39, GB200, peak reserved GiB with C3 on,
# one bucket each):
#   4M off  Si 1,728 / LiFePO4 1,008 / RDX 1,344 / Si 4,096 / FeCOH 9,320 / RDX 2,688 / RDX 5,376 (414,720 /
#     617,472 / 951,366 / 983,040 / 1,218,096 / 1,902,732 / 3,805,464 rows) 18.50 / 25.83 / 38.99 / 43.47 /
#     51.91 / 76.24 / 155.33; on the same and RDX 10,752 (7,610,928) 9.07 / 11.99 / 17.85 / 20.92 / 24.49 /
#     34.15 / 69.44 / 142.39; deep 7.45 / 11.39 / 15.06 / 15.68 / 20.85 / 28.61 / 54.87 / 113.30
#   10M off Si 1,728 / SiO2 1,125 / Si 4,096 / LiFePO4 1,008 / Cu 1,000 / RDX 1,344 (414,720 / 943,388 /
#     983,040 / 1,099,440 / 1,722,000 / 1,951,244 rows) 43.76 / 85.96 / 101.95 / 99.89 / 149.92 / 171.79; on the
#     same and RDX 2,688 (3,902,488) 18.00 / 33.22 / 41.54 / 38.52 / 57.13 / 65.81 / 132.43; deep the same and
#     Cu 4,096 (7,053,312) 13.74 / 24.09 / 33.44 / 28.43 / 43.05 / 49.85 / 98.82 / 174.23
# About the eager entries (one working set); Si 4,096 binds every level.
GRAPH_GAIN_MAX_ROWS = {"fecoh_4m": 400_000, "oam_10m": 400_000}
ENVELOPE_CALIBRATION = {
    "fecoh_4m": {("single", "topology", "model_graph"): Coefficients(3.921, 40.232),
                 ("single", "topology", "model_graph", "narrow"): Coefficients(2.770, 18.464),
                 ("single", "topology", "model_graph", "deep"): Coefficients(3.117, 14.558)},
    "oam_10m": {("single", "topology", "model_graph"): Coefficients(22.006, 81.324),
                ("single", "topology", "model_graph", "narrow"): Coefficients(9.557, 32.535),
                ("single", "topology", "model_graph", "deep"): Coefficients(9.659, 24.192)},
}

# Derived entries: the RDX entry times DERIVED_SAFETY x the analytic per-row ratio (derived_bytes_per_row at
# derived_edges_per_row, at least _DERIVED_EDGES_PER_ROW, over RDX's at _RDX_EDGES_PER_ROW); the ratio alone under-predicts five measured 4M/10M
# model_graph points by up to 9% (10M deep, 1,344 atoms).
DERIVED_SAFETY = 1.25
_DERIVED_EDGES_PER_ROW = 0.25
_RDX_EDGES_PER_ROW = 0.128  # RDX: 121,778 directed edges per 951,366 triplet rows (6 / 4.0 A)
# (tau, fixed bytes) per level: the next layer's inputs held while one layer is rebuilt, and per-row constants
_DERIVED_OVERLAP = {"off": (0.60, 50), "narrow": (0.72, 100), "deep": (0.81, 160)}


def derived_bytes_per_row(arch: Architecture, recompute: str, edges_per_row: float) -> float:
    """Saved bytes per triplet row of one evaluation at a recompute level: per layer the line-graph row's
    tensors plus ``edges_per_row`` directed edges' (FP32 widths from the kept activations of each level),
    plus the overlap of the next layer and a fixed part (fitted on RDX and 4M; the analytic model of the
    task that measured the 4M and 10M peaks)."""
    n, e, t = arch.node_feat_dim, arch.edge_feat_dim, arch.three_body_feat_dim
    line = {"off": 6 * e + 10 * t + 8, "narrow": t + 4 * e + 8, "deep": t + 2 * e + 4}
    edge = {"off": 6 * n + 11 * e + 8, "narrow": 4 * n + 2 * e + 8, "deep": 2 * n + e + 4}
    tau, fixed = _DERIVED_OVERLAP[recompute]
    per_layer = arch.dtype_bytes * (line[recompute] + edges_per_row * edge[recompute])
    return arch.num_layers * per_layer + tau * arch.dtype_bytes * (line["off"] + edges_per_row * edge["off"]) + fixed


def derived_edges_per_row(arch: Architecture) -> float:
    """Directed pair edges per triplet row for the derived bound: RDX's at 6 / 4.0 A scaled by the pair
    edges per atom ((pair cutoff / 6)^3) over the triplets per atom ((three-body cutoff / 4)^6) at the same
    density, and at least ``_DERIVED_EDGES_PER_ROW``."""
    ratio = (arch.pairwise_cutoff / RDX_1M.pairwise_cutoff) ** 3 / max(
        arch.three_body_cutoff / RDX_1M.three_body_cutoff, 1e-3) ** 6
    return max(_DERIVED_EDGES_PER_ROW, _RDX_EDGES_PER_ROW * ratio)


def derived_scale(arch: Architecture, recompute: str) -> float:
    """The factor on an RDX entry that bounds ``arch`` (``DERIVED_SAFETY`` included)."""
    return DERIVED_SAFETY * (derived_bytes_per_row(arch, recompute, derived_edges_per_row(arch))
                             / derived_bytes_per_row(RDX_1M, recompute, _RDX_EDGES_PER_ROW))


@dataclass(frozen=True)
class MemoryModel:
    """Which calibration predicts a model's peaks (``memory_model``): ``calibration`` is a measured
    architecture's name, "derived" or "uncalibrated" (no architecture, no prediction); for the last two
    ``calibrated`` is False: no automatic choice may relax memory."""
    architecture: Architecture | None = None
    calibration: str = "rdx_1m"

    @property
    def calibrated(self) -> bool:
        return self.calibration in CALIBRATED_ARCHITECTURES

    @property
    def strict(self) -> bool:
        """An unpredicted (None) peak does not count as fitting: every model but the RDX calibration's."""
        return self.calibration != "rdx_1m"

    def summary(self) -> dict:
        return {"calibration": self.calibration,
                "architecture": None if self.architecture is None else self.architecture.fingerprint()}


RDX_MEMORY_MODEL = MemoryModel()
# A model without a MatRIS ``Architecture`` (``memory_model``): every prediction None, strict.
UNCALIBRATED_MEMORY_MODEL = MemoryModel(None, "uncalibrated")


def no_bound_reason(memory: MemoryModel, outside: str) -> str:
    """Why K6 bounds no capture of a model whose automatic choices may not rely on it: no MatRIS
    architecture (``UNCALIBRATED_MEMORY_MODEL``), else ``outside`` (the caller's wording for a derived model
    or an entry its architecture was not measured with)."""
    return "no MatRIS architecture (K6 uncalibrated)" if memory.calibration == "uncalibrated" else outside


def entry_calibrated(memory: MemoryModel | None, engine: str, profile: str | None, execution: str,
                     recompute: str = "off") -> bool:
    """Whether a prediction comes from a measured table: the RDX model's always (its uncalibrated entries
    predict None), another measured architecture's where its table has the entry, a derived model's or
    ``UNCALIBRATED_MEMORY_MODEL``'s never."""
    if memory is None or memory.calibration == "rdx_1m":
        return True
    return _key(engine, profile, execution, recompute) in ARCHITECTURE_CALIBRATION.get(memory.calibration, {})


def memory_model(model) -> MemoryModel:
    """The calibration for a model (or an ``Architecture``): the measured architecture with the smallest
    per-row bytes whose envelope holds it; else the RDX tables where the derived bound is at most RDX's at
    every level (``derived_scale`` <= 1: e.g. the small test models with other cutoffs or norms); else
    "derived". A model without a MatRIS ``Architecture`` (``architecture`` None, e.g. another potential's
    module, a config lacking a key, or None) is ``UNCALIBRATED_MEMORY_MODEL``."""
    arch = model if isinstance(model, Architecture) else architecture(model)
    if arch is None:
        return UNCALIBRATED_MEMORY_MODEL
    covering = [name for name, (calibrated, _) in CALIBRATED_ARCHITECTURES.items() if arch.within(calibrated)]
    if covering:
        return MemoryModel(arch, min(covering, key=lambda name: derived_bytes_per_row(
            CALIBRATED_ARCHITECTURES[name][0], "off", _RDX_EDGES_PER_ROW)))
    if all(derived_scale(arch, level) <= 1.0 for level in _DERIVED_OVERLAP):
        return MemoryModel(arch, "rdx_1m")
    return MemoryModel(arch, "derived")


class MemoryAdmissionError(RuntimeError):
    """A requested mode, or every mode under auto, exceeds the device budget."""


def enabled(value: str | None = None) -> bool:
    value = os.environ.get(ENVIRONMENT, "on") if value is None else value
    if value not in ("on", "off", "1", "0"):
        raise ValueError(f"{ENVIRONMENT} must be on/off/1/0, got {value!r}")
    return value in ("on", "1")


def predict(engine: str, profile: str | None, execution: str, rows: int, line_rows: int = 0, *,
            checkpoint: bool = False, recompute: str = "off", mixed_width: bool = True,
            single_bwd: bool = True, gated_tail: bool = False, memory: MemoryModel | None = None) -> int | None:
    """Predicted peak bytes, or None where the table has no calibration.

    ``recompute`` ("narrow" = saved_tensor_recompute "on", "deep") other than "off" reads the
    ``(engine, profile, execution, recompute)`` entries. ``mixed_width``: whether D8 runs
    (``mixed_width_active``); False reads ``CALIBRATION_MIXED_WIDTH_OFF`` where it has the key.
    ``single_bwd``: whether C3 runs (``single_bwd_mode``); True adds its slope increment.
    ``gated_tail``: with C2's slope increment (``GATED_TAIL_GIB_PER_MILLION_ROWS``).
    ``memory``: the model's calibration (``memory_model``; None: the RDX tables); ``UNCALIBRATED_MEMORY_MODEL``
    predicts None everywhere."""
    if checkpoint:
        return None
    if memory is not None and memory.calibration != "rdx_1m":
        table = ARCHITECTURE_CALIBRATION.get(memory.calibration, {})
        key = _key(engine, profile, execution, recompute)
        entry = table.get(key)
        if engine == "single" and execution == "model_graph" and rows > GRAPH_GAIN_MAX_ROWS.get(memory.calibration, rows):
            entry = ENVELOPE_CALIBRATION.get(memory.calibration, {}).get(key, entry)  # an envelope-only runner
        if entry is not None:
            return entry.predict(rows, line_rows)
        # Not measured for this architecture: the RDX entry scaled by the derived per-row ratio.
        arch = CALIBRATED_ARCHITECTURES[memory.calibration][0] if memory.calibrated else memory.architecture
        if arch is None:
            return None  # uncalibrated: no architecture to derive a bound from
        rdx = predict(engine, profile, execution, rows, line_rows, recompute=recompute, mixed_width=mixed_width,
                      single_bwd=single_bwd, gated_tail=gated_tail)
        return None if rdx is None else int(rdx * derived_scale(arch, recompute))
    key = _key(engine, profile, execution, recompute)
    coefficients = None if mixed_width else CALIBRATION_MIXED_WIDTH_OFF.get(key)
    coefficients = CALIBRATION.get(key) if coefficients is None else coefficients
    if coefficients is not None and single_bwd and key in SINGLE_BWD_GIB_PER_MILLION_ROWS:
        coefficients = replace(coefficients, gib_per_million_rows=coefficients.gib_per_million_rows
                               + SINGLE_BWD_GIB_PER_MILLION_ROWS[key])
    if coefficients is None:
        return None
    increment = GATED_TAIL_GIB_PER_MILLION_ROWS.get(key, 0.0) if gated_tail else 0.0
    return coefficients.predict(rows, line_rows) + int(increment * rows / 1e6 * (1 + _SLACK) * _GIB)


def _key(engine, profile, execution, recompute):
    if engine.startswith("distributed") and profile != "torch_reference":
        # Distributed ``profile`` is the local backend; native (or None) keeps the original key.
        profile = None
    return (engine, profile, execution) if recompute == "off" else (engine, profile, execution, recompute)


def margin_bytes(total: int) -> int:
    return max(_MARGIN_BYTES, int(_MARGIN_FRACTION * total))


def fitting_modes(predicted: list[dict], budgets: list[int], strict: bool = False) -> set:
    """Modes whose prediction fits every rank's budget; None predictions always fit (``strict``: only
    eager's).

    Pure decision on gathered data: ``budgets`` are available minus margin.
    """
    return {mode for mode in predicted[0]
            if all((p[mode] is None and (not strict or mode == "eager")) or (p[mode] is not None and p[mode] <= budget)
                   for p, budget in zip(predicted, budgets))}


def _gib(value) -> str:
    return "uncalibrated" if value is None else f"{value / _GIB:.1f} GiB"


@runtime_checkable
class MemoryPredictor(Protocol):
    """A model's part of the device-memory admission: its peak predictions and the memory decisions it takes itself.

    ``MemoryAdmission`` keeps the generic part: the available memory and margin, the gather over ranks,
    ``fitting_modes``, the decision record and the error. MatRIS's predictor is ``K6MemoryPredictor`` (the tables
    of this module). ``extents`` are the sizes a prediction reads: ``{"rows": ..., "line_rows": ...}`` of
    ``MemoryAdmission.check`` (triplet rows, and the rank's line rows at P>1), or the sizes the caller passes as
    ``check(extents=...)`` for another predictor (e.g. a declared shard's rows, ghost rows and edges; an optional
    ``describe(entry)`` then words them in the error). A check calls, on every rank in this order and on the same
    gathered data, so that every rank decides alike:

    * ``predictions(modes, extents)``: this rank's predicted peaks (bytes; None: no prediction) of the execution
      ``modes`` at each level of ``levels()``, and whatever else ``decide`` reads: a dict, gathered with the
      extents and the memory parts, and left out of the decision record;
    * ``decide(gathered, budgets, modes=, requested=, event=)``: the model's own decisions from every rank's
      gathered entry and budget (available minus margin); returns the level the forwards use;
    * ``peaks(entry, level)``: per mode, the peak a gathered entry predicts at that level under the decisions;
      ``predict(level, mode, extents)``: one peak under them (the ``shared`` modes at the largest extents over
      ranks);
    * ``strict``: an unpredicted peak does not fit an automatic choice; ``calibrated(modes)``: the automatic
      choices may rely on the predictions of ``modes`` (gathered over the ranks: where it is False on any rank, execution="auto" runs eager);
    * ``record(modes, event=, requested=, fits=, shared_rows_modes=, by_rank=, calibrated=)``: the check's entry
      of ``MemoryAdmission.decisions``, from the admission's fields (``calibrated``: every mode predicted on this
      rank) and the model's decisions.

    ``bind()`` is the context in which the forwards of one evaluation use the decisions. ``slack`` (the relative
    slack every prediction includes), ``label`` (what is admitted, in messages) and ``summary()`` (the model's
    entries of ``MemoryAdmission.summary``) describe the predictor.
    """

    slack: float
    label: str
    strict: bool

    def levels(self) -> tuple:
        ...

    def calibrated(self, modes=("eager", "model_graph")) -> bool:
        ...

    def predictions(self, modes, extents: dict) -> dict:
        ...

    def decide(self, gathered: list[dict], budgets: list[int], *, modes, requested: str | None, event: str) -> str:
        ...

    def peaks(self, entry: dict, level: str) -> dict:
        ...

    def predict(self, level: str, mode: str, extents: dict) -> int | None:
        ...

    def record(self, modes, *, event: str, requested: str, fits: list, shared_rows_modes: list, by_rank: list[dict],
               calibrated: bool) -> dict:
        ...

    def summary(self) -> dict:
        ...

    def bind(self) -> ContextManager:
        ...


class K6MemoryPredictor:
    """MatRIS's ``MemoryPredictor``: the K6 tables (``predict``) of one engine and profile for the model's
    calibration (``memory_model``), and the memory decisions MatRIS takes at each check: the recompute level
    (saved_tensor_recompute "auto": ``saved_recompute.AutoRecompute``), C3 (``single_bwd_mode`` "auto") and C2
    (``gated_tail_fits``). The forwards of an evaluation read ``recompute``, ``single_bwd_decision`` and
    ``gated_tail_decision`` (``bind``)."""

    def __init__(self, *, engine: str, profile: str | None, checkpoint: bool, recompute: str = "off",
                 mixed_width: bool = True, single_bwd: str = "auto", gated_tail: bool = False,
                 memory: MemoryModel | None = None, policy: str = "safe", epoch: bool = False):
        self.engine, self.profile, self.checkpoint = engine, profile, checkpoint
        # The model's calibration (memory_model). Not calibrated: C3 stays off and execution="auto" runs
        # eager; strict: an unpredicted peak does not fit (the ladder climbs to "deep").
        self.memory = RDX_MEMORY_MODEL if memory is None else memory
        # InferenceConfig.recompute_policy: the share of the budget the memory decisions (recompute
        # auto, C3 auto, C2) may use; prediction only (no allocator switch, no fallback: ranks cannot
        # retry). "fast" applies to bond_partial on the topology epoch (``epoch``), and only to a check
        # whose predictions all come from a measured table (``decide``); see saved_recompute.FAST_THRESHOLD.
        from fastmd.engine.saved_recompute import policy_threshold
        self.policy = policy
        self.threshold = policy_threshold(
            policy if self.memory.calibrated else "safe",
            "distributed_epoch" if engine == "distributed_bond_partial" and epoch else engine)
        self.mixed_width = mixed_width  # D8 runs: selects its calibration (see predict)
        # C3 ("on"/"off"/"auto", ``single_bwd_mode``): "on" adds its slope increments to every
        # prediction; "auto" decides at each check (``single_bwd_decision``; the rule of
        # ``saved_recompute.AutoRecompute.update``, never back on within a level).
        self.single_bwd = single_bwd
        self.single_bwd_decision = single_bwd == "on"
        self._single_bwd_possible = True
        # C2 runs (gated_tail_active): decided at each check (gated_tail_fits), after the level and C3.
        self.gated_tail = gated_tail
        self.gated_tail_decision = gated_tail
        # The mode the forwards use ("off"/"narrow"/"deep"); saved_tensor_recompute
        # "auto" decides it at each check from the gathered counts.
        self.auto = None
        if recompute == "auto":
            from fastmd.engine.saved_recompute import LADDER, AutoRecompute
            self.auto = AutoRecompute(f"{engine} admission", self.threshold, memory=self.memory)
            self._levels = LADDER
            recompute = "off"
        self.recompute = recompute

    @classmethod
    def for_model(cls, model, *, engine: str, profile: str | None, checkpoint: bool, recompute: str, kernel_options,
                  policy: str = "safe", epoch: bool = False) -> "K6MemoryPredictor":
        """The predictor of ``model`` (its calibration: ``memory_model``) on an engine and profile with resolved
        ``kernel_options`` (D8: ``mixed_width_active``, C3: ``single_bwd_mode``, C2: ``gated_tail_active``);
        ``recompute``: an internal level or "auto" (``saved_recompute.internal_mode``)."""
        return cls(engine=engine, profile=profile, checkpoint=checkpoint, recompute=recompute,
                   mixed_width=mixed_width_active(kernel_options), single_bwd=single_bwd_mode(kernel_options),
                   gated_tail=gated_tail_active(kernel_options), memory=memory_model(model), policy=policy,
                   epoch=epoch)

    @property
    def slack(self) -> float:
        return _SLACK

    @property
    def label(self) -> str:
        return f"{self.engine} engine{'' if self.profile is None else ', ' + self.profile}"

    @property
    def strict(self) -> bool:
        # The layer checkpoint is admitted unchecked (see the module docstring) for every model.
        return self.memory.strict and not self.checkpoint

    def levels(self) -> tuple:
        """The levels a check predicts: every level of the ladder under "auto", else the explicit one."""
        return (self.recompute,) if self.auto is None else self._levels

    def calibrated(self, modes=("eager", "model_graph")) -> bool:
        """Whether the automatic choices (execution="auto" preferring model_graph, C3) may rely on the
        predictions: the model is inside a measured envelope (``memory_model``) and every entry a check of
        ``modes`` reads comes from a measured table (``entry_calibrated``)."""
        return self.memory.calibrated and all(entry_calibrated(self.memory, self.engine, self.profile, mode, level)
                                              for mode in modes for level in self.levels())

    def _tables(self) -> tuple:
        # C3 auto: the ladder reads the predictions without it; C3 is decided after (see saved_recompute).
        return ("by_recompute", "by_recompute_single_bwd") if self.single_bwd == "auto" else ("by_recompute",)

    def predictions(self, modes, extents: dict) -> dict:
        """Per table ("by_recompute"; C3 auto: also "by_recompute_single_bwd", with its increment), level and mode."""
        return {table: {key: {mode: predict(self.engine, self.profile, mode, extents["rows"], extents["line_rows"],
                                            checkpoint=self.checkpoint, recompute=key, mixed_width=self.mixed_width,
                                            single_bwd=self.single_bwd == "on" or table != "by_recompute",
                                            memory=self.memory)
                              for mode in modes} for key in self.levels()} for table in self._tables()}

    def decide(self, gathered: list[dict], budgets: list[int], *, modes, requested: str | None, event: str) -> str:
        """The recompute level (the ladder under "auto"), C3 ("auto") and C2 from every rank's gathered predictions."""
        tables, c3_auto, strict = self._tables(), self.single_bwd == "auto", self.strict
        # Every prediction this check reads comes from a measured table (else no automatic choice relaxes memory).
        calibrated = self.calibrated(modes)
        # recompute_policy "fast" only on those predictions (else "safe"'s share).
        from fastmd.engine.saved_recompute import AUTO_THRESHOLD
        threshold = self.threshold if calibrated else AUTO_THRESHOLD
        # Every rank decides from the same gathered data (collective, deterministic):
        # the requested execution's peak, else (execution auto) the largest candidate's.
        considered = (requested,) if requested in modes else modes
        if requested is None and not calibrated and "eager" in modes:
            considered = ("eager",)  # auto runs eager outside every measured envelope (MemoryAdmission.check)
        worst = lambda g, key, table="by_recompute": max((g[table][key][m] for m in considered
                                                          if g[table][key][m] is not None), default=None)
        if self.auto is not None:
            self.recompute = self.auto.update(
                max(g["rows"] for g in gathered), {key: [worst(g, key) for g in gathered] for key in self._levels},
                budgets, event,
                peaks_single_bwd=({key: [worst(g, key, tables[1]) for g in gathered] for key in self._levels}
                                  if c3_auto else None), strict=strict, calibrated=calibrated,
                threshold=threshold)
        if c3_auto and self.auto is not None:
            self.single_bwd_decision = self.auto.single_bwd
        elif c3_auto:  # an explicit level: the same rule, C3 only
            fits_c3 = calibrated and all(
                peak is not None and peak <= threshold * budget
                for peak, budget in zip([worst(g, self.recompute, tables[1]) for g in gathered], budgets))
            self.single_bwd_decision = self._single_bwd_possible = self._single_bwd_possible and fits_c3
        if self.gated_tail:
            # C2 after the level and C3: every rank, every considered mode (collective data).
            self.gated_tail_decision = all(
                gated_tail_fits(self.engine, self.profile, mode, g["rows"], g["line_rows"], recompute=self.recompute,
                                mixed_width=self.mixed_width, budget=g["available"] - g["margin"],
                                single_bwd=self.single_bwd_decision, memory=self.memory, threshold=threshold)
                for g in gathered for mode in ((requested,) if requested in modes else modes))
        return self.recompute

    def peaks(self, entry: dict, level: str) -> dict:
        tables = self._tables()
        return entry[tables[1] if self.single_bwd == "auto" and self.single_bwd_decision else "by_recompute"][level]

    def predict(self, level: str, mode: str, extents: dict) -> int | None:
        return predict(self.engine, self.profile, mode, extents["rows"], extents["line_rows"],
                       checkpoint=self.checkpoint, recompute=level, mixed_width=self.mixed_width,
                       single_bwd=self.single_bwd_decision, memory=self.memory)

    def record(self, modes, *, event: str, requested: str, fits: list, shared_rows_modes: list, by_rank: list[dict],
               calibrated: bool) -> dict:
        return {"event": event, "requested": requested, "fits": fits, "single_bwd": self.single_bwd_decision,
                "shared_rows_modes": shared_rows_modes, "by_rank": by_rank, "calibrated": calibrated,
                "memory_model": {**self.memory.summary(), "entries_calibrated": self.calibrated(modes)},
                "gated_tail": self.gated_tail_decision}

    def summary(self) -> dict:
        return {"engine": self.engine, "profile": self.profile, "checkpoint": self.checkpoint,
                "recompute": self.recompute, "recompute_policy": self.policy, "threshold": self.threshold,
                "mixed_width": self.mixed_width, "single_bwd": self.single_bwd,
                "memory_model": self.memory.summary(),
                "single_bwd_decision": self.single_bwd_decision, "gated_tail": self.gated_tail_decision,
                "recompute_auto": None if self.auto is None else list(self.auto.decisions),
                "model": ("peak reserved = base + a * triplet rows (+ b * local line rows at P>1); "
                          "fastmd.engine.memory_admission.CALIBRATION (D8 inactive: "
                          "CALIBRATION_MIXED_WIDTH_OFF where it has the key; C3 active: "
                          "+ SINGLE_BWD_GIB_PER_MILLION_ROWS on the slope); another measured architecture: "
                          "ARCHITECTURE_CALIBRATION; else derived (derived_scale)")}

    def bind(self) -> ContextManager:
        """The forwards of one evaluation use these decisions (``saved_recompute.bound_admission``)."""
        from fastmd.engine.saved_recompute import bound_admission
        return bound_admission(self)


class MemoryAdmission:
    """Per-evaluator admission state and decision record, with the model's ``MemoryPredictor``.

    The predictor's attributes read through the admission (MatRIS's ``recompute``, ``single_bwd_decision``,
    ``gated_tail_decision``, ``memory``, ``threshold``, ...); ``bind()`` binds the predictor itself, whose
    decisions ``saved_recompute``'s readers then read.
    """

    def __init__(self, *, device, predictor: MemoryPredictor | None = None, **k6):
        """``predictor``: the model's; without one, MatRIS's ``K6MemoryPredictor`` from the keywords ``k6``
        (``engine``, ``profile``, ``checkpoint``, ``recompute``, ``mixed_width``, ``single_bwd``, ``gated_tail``,
        ``memory``, ``policy``, ``epoch``)."""
        if predictor is None:
            predictor = K6MemoryPredictor(**k6)
        elif k6:
            raise TypeError(f"MemoryAdmission takes a predictor or K6MemoryPredictor's keywords, not both: "
                            f"{sorted(k6)}")
        elif not isinstance(predictor, MemoryPredictor):
            raise TypeError(f"MemoryAdmission: {type(predictor).__name__} is not a MemoryPredictor")
        self.predictor = predictor
        self.device = torch.device(device)
        self.baseline = None
        self.decisions = []
        # True once a check admitted the requested mode (auto: some mode). A
        # rejected check is recorded in ``decisions`` too, so "no decisions"
        # must not stand for "not yet admitted": a caller that catches the
        # MemoryAdmissionError and retries is checked again.
        self.admitted = False

    def __getattr__(self, name):
        # Only names the admission lacks (never its own, e.g. a property that raised): the predictor's.
        predictor = self.__dict__.get("predictor")
        if predictor is None or hasattr(type(self), name):
            raise AttributeError(name)
        return getattr(predictor, name)

    @property
    def calibrated(self) -> bool:
        """Whether the automatic choices (execution="auto" preferring model_graph, C3) may rely on the
        predictions (``MemoryPredictor.calibrated``)."""
        return self.predictor.calibrated()

    def available(self) -> dict:
        """Bytes this process can still reach for the engine, and the parts."""
        free, total = torch.cuda.mem_get_info(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        if self.baseline is None:
            # Allocations before the engine's first evaluation (model, caller
            # tensors) stay; everything later is covered by the prediction.
            self.baseline = torch.cuda.memory_allocated(self.device)
        return {"available": free + reserved - self.baseline, "margin": margin_bytes(total), "free": free,
                "reserved": reserved, "baseline_allocated": self.baseline, "total": total}

    def check(self, modes, rows: int = 0, line_rows: int = 0, *, event: str, requested: str | None = None,
              context=None, shared=(), extents: dict | None = None) -> set:
        """Fitting subset of ``modes``; raises if ``requested`` or every mode is out.

        ``rows``: triplet rows (distributed: owned), ``line_rows``: the
        distributed rank's local line rows; ``extents``: the sizes another
        predictor reads instead (e.g. a declared shard's rows, ghost rows and
        edges), described in the error by its ``describe(extents)``. With a
        context of more than one rank, (predictions, memory) of every rank are
        gathered once; all ranks decide on the same data. ``shared`` modes
        allocate the same padded rows on every rank (the P>1 tier policy pads
        every rank to the global MAX key): each rank predicts them from the
        largest gathered rows and line rows, against its own budget.
        """
        predictor = self.predictor
        extents = {"rows": rows, "line_rows": line_rows} if extents is None else dict(extents)
        local = predictor.predictions(modes, extents)
        # Whether this rank's predictions may be relied on (a per-rank measured predictor may differ by rank).
        local["calibrated_modes"] = predictor.calibrated(modes)
        payload = tuple(local)  # gathered, not recorded
        local.update(**extents, **self.available())
        gathered = [local] if context is None or context.world_size == 1 else context.all_gather_object(local)
        budgets = [g["available"] - g["margin"] for g in gathered]
        calibrated = all(g["calibrated_modes"] for g in gathered)
        # The model's own decisions, every rank from the same gathered data (collective, deterministic).
        level = predictor.decide(gathered, budgets, modes=modes, requested=requested, event=event)
        gathered = [{**{k: v for k, v in g.items() if k not in payload}, "predicted": predictor.peaks(g, level)}
                    for g in gathered]
        predicted = predictor.peaks(local, level)
        if shared:
            top = {name: max(g[name] for g in gathered) for name in extents}
            gathered = [{**g, "predicted": {mode: predictor.predict(level, mode, top) if mode in shared else value
                                            for mode, value in g["predicted"].items()}} for g in gathered]
        fits = fitting_modes([g["predicted"] for g in gathered], budgets, strict=predictor.strict and requested is None)
        if requested is None and not calibrated:
            fits.discard("model_graph")  # execution="auto" on predictions some rank may not rely on: eager
        self.decisions.append(predictor.record(
            modes, event=event, requested=requested or "auto", fits=sorted(fits), shared_rows_modes=sorted(shared),
            by_rank=gathered, calibrated=all(v is not None for v in predicted.values())))
        wanted = [requested] if requested is not None else list(modes)
        if not any(mode in fits for mode in wanted):
            raise MemoryAdmissionError(self._message(requested, modes, gathered, fits, shared))
        self.admitted = True
        return fits

    def bind(self) -> ContextManager:
        """The context in which the forwards of one evaluation use this admission's decisions."""
        return self.predictor.bind()

    def _message(self, requested, modes, gathered, fits, shared=()) -> str:
        mode = requested or modes[0]
        worst = gathered[max(range(len(gathered)), key=lambda r: (gathered[r]["predicted"][mode] or 0)
                             - gathered[r]["available"] + gathered[r]["margin"])]
        what = f"execution={requested!r}" if requested is not None else f"execution='auto' ({'/'.join(modes)})"
        need = ", ".join(f"{mode} {_gib(worst['predicted'][mode])}" for mode in modes)
        describe = getattr(self.predictor, "describe", None)
        size = describe(worst) if describe is not None else (
            f"{worst['rows']:,} triplet rows" + (f", {worst['line_rows']:,} line rows" if worst["line_rows"] else ""))
        if shared:
            top = max(g["rows"] for g in gathered), max(g["line_rows"] for g in gathered)
            size += (f"; {'/'.join(sorted(shared))} padded to the global max {top[0]:,} triplet rows, "
                     f"{top[1]:,} line rows")
        other = sorted(fits - {requested}) if requested is not None else []
        hint = f"{' or '.join(other)} would fit" if other else "no mode fits: use more GPUs"
        return (f"memory admission: {what} does not fit on rank {gathered.index(worst)}: predicted peak {need} "
                f"({size}; {self.predictor.label}) > "
                f"available {_gib(worst['available'])} (free {_gib(worst['free'])} + reserved "
                f"{_gib(worst['reserved'])} - pre-run allocations) - margin {_gib(worst['margin'])}; {hint}. "
                f"{ENVIRONMENT}=off disables the check")

    def summary(self) -> dict:
        return {"enabled": True, "environment": ENVIRONMENT, "slack": self.predictor.slack,
                "margin": f"max({_MARGIN_BYTES} B, {_MARGIN_FRACTION} x device)",
                **self.predictor.summary(), "decisions": list(self.decisions)}
