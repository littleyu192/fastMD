# Matbench Discovery relaxation fixture

`wbm-1-12.initial.extxyz` is the **unrelaxed** WBM structure `wbm-1-12`,
Ag2CdI4 (7 atoms, periodic in three dimensions). It exercises internal atomic
coordinates as well as cell relaxation. It is copied byte for byte from the
published archive; no strain, displacement, replication, or conversion has been
applied. The small cell is intended for correctness and execution-path checks,
not a representative throughput benchmark or a claim about leaderboard accuracy.

Source and attribution:

- Janosh Riebesell (2023), **Matbench Discovery - Data Files**, Figshare,
  [doi:10.6084/m9.figshare.22715158](https://doi.org/10.6084/m9.figshare.22715158).
- License: [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/),
  as recorded in the [public Figshare metadata](https://api.figshare.com/v2/articles/22715158).
- WBM data: Wang, Botti and Marques, *Predicting stable crystalline compounds
  using chemical similarity*, npj Computational Materials (2021).
  [Matbench Discovery WBM provenance and processing](https://github.com/janosh/matbench-discovery/blob/main/data/wbm/readme.md).
- Requested benchmark: [Matbench Discovery, Geo Opt preset](https://matbench-discovery.materialsproject.org/?preset=Geo+Opt).
- The official [data-file registry](https://github.com/janosh/matbench-discovery/blob/main/matbench_discovery/data-files.yml)
  identifies `wbm_initial_atoms` as unrelaxed WBM structures in ASE extended XYZ.
- Archive: `2024-08-04-wbm-initial-atoms.extxyz.zip`,
  [Figshare file 48169597](https://ndownloader.figshare.com/files/48169597),
  98,164,028 bytes. Archive MD5:
  `2a292211ca6acb30ed8416178d644098`, independently verified against an existing
  local full archive as well as the publisher metadata. The fixture's original
  ZIP member bytes were also cross-checked with Python's `zipfile` reader.
- ZIP member: `wbm-1-12.extxyz`; local record starts at byte 2,918 and ends at
  byte 3,178 inclusive; deflate payload 216 bytes, extracted payload 540 bytes.
- Verified extracted CRC32: `4a94fc75`.
- Extracted SHA256: `6975404b868966a5cbc7a053f7ba12769278b009885f2e8c5df9d36bf61110d2`.
- Retrieved on 2026-10-23.

To reproduce the fixture without fetching the 98 MB archive:

```sh
python tests/data/fetch_wbm_fixture.py
```

The script requests a 3,179-byte HTTP range, validates the ZIP member name,
compression mode, CRC32, byte count and SHA256, and writes the original bytes.
An existing local ZIP can be supplied with `--archive`. No network access is
needed to read the checked-in fixture in ordinary tests.

Read the initial structure with `ase.io.read("tests/data/wbm-1-12.initial.extxyz")`.
For an ASE-versus-GPU comparison, start both runs from independent copies of this
same structure, use the same checkpoint, precision, force tolerance and cell
constraint, and compare convergence, final energy, force residuals and volume.
The fixture contains no reference energy or DFT force labels.

An executable comparison of the migrated MD `StructOptimizer` API is provided in
`validation/md_relax_migration/compare_wbm.py`:

```sh
python validation/md_relax_migration/compare_wbm.py --model-path /path/to/mace.model \
  --steps 5 --fmax 0.05 \
  --output wbm-relax.json
```

This loads one MACE adapter and reuses it for independent copies of the same
initial structure, comparing `engine="ase"` with strict `engine="gpu"` for fixed
and variable cells. The GPU path must not silently fall back to ASE. Terminal
properties come from trajectory records. The JSON reports final energy, forces,
positions and cell errors, absolute tolerances, the actual relaxation path and
construction/run timings; model loading is excluded. Default tolerances are
0.001 eV for total energy, 0.001 eV/Å for force components and 0.001 Å for position
and cell components, with no relative tolerance. Each is configurable on the CLI.
A failed numerical comparison returns a nonzero exit status.

The default five-step run checks numerical agreement and does not assert
convergence. `--full` raises the default limit to 500 steps; `--steps` can override
either limit. Both modes report the recorded step count without claiming that a
run which reached its limit converged. ASE and GPU execute in sequence in one
process, so these timings are diagnostic rather than an unbiased speed benchmark.
