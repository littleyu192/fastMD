"""Shared restart format compatibility without a GPU or model checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fastmd.execution.md_protocol import read_restart, write_restart


def _fields(ensemble: str, format_name: str) -> dict:
    """The version-1 on-disk schema, independently of FastMD's writer."""
    nhc = ensemble == "nvt_nhc"
    parameters = {
        "ensemble": ensemble, "dt_fs": 0.5, "tdamp_fs": 50.0, "taut_fs": 20.0,
        "tchain": 3, "tloop": 1, "fix_com": True, "total_steps": 20,
        "ramp": None if ensemble == "nve" else {
            "start_K": 300.0, "end_K": 600.0, "total_steps": 20, "update_interval": 4},
        "scalar_sync": "broadcast", "ke2_update": "scaled",
    }
    return {
        "format": np.array(format_name),
        "ids": np.array([0, 1], dtype=np.int64),
        "numbers": np.array([1, 8], dtype=np.int64),
        "masses": np.array([2.014, 17.999], dtype=np.float64),
        "positions": np.array([[0.123456789, 1.25, 2.5], [2.75, 3.5, 4.125]],
                              dtype=np.float64 if nhc else np.float32),
        "momenta": np.array([[0.125, -0.25, 0.5], [-0.125, 0.25, -0.5]], dtype=np.float64),
        "forces": np.array([[0.2, -0.3, 0.4], [-0.2, 0.3, -0.4]], dtype=np.float32),
        "potential_energy": np.array([-1.25], dtype=np.float32),
        "reference_energy": np.array([-0.75], dtype=np.float32),
        "cell": np.array([[8.0, 0.5, 0.0], [0.0, 9.0, 0.25], [0.0, 0.0, 10.0]], dtype=np.float64),
        "pbc": np.array([True, False, True]),
        "step": np.array(7, dtype=np.int64),
        "parameters": np.array(json.dumps(parameters)),
        "temperature_K": np.array(np.nan if ensemble == "nve" else 375.0),
        "nhc_eta": np.array([0.125, -0.25, 0.5] if nhc else [], dtype=np.float64),
        "nhc_p_eta": np.array([1.25, -2.5, 5.0] if nhc else [], dtype=np.float64),
        "offsets": np.array(json.dumps({"thermo": 257, "xyz": 1025})),
        "engine": np.array("native"),
    }


class RestartFormatTest(unittest.TestCase):
    def test_reads_both_formats_and_writes_fastmd_without_changing_state(self):
        for format_name in ("fastmd.distributed.md/1", "matris.distributed.md/1"):
            for ensemble in ("nve", "nvt", "nvt_nhc"):
                with self.subTest(format=format_name, ensemble=ensemble), tempfile.TemporaryDirectory() as tmp:
                    path, rewritten = Path(tmp) / "input.npz", Path(tmp) / "output.npz"
                    fields = _fields(ensemble, format_name)
                    np.savez(path, **fields)
                    data = read_restart(path)
                    arrays = {key: key for key in (
                        "ids", "numbers", "masses", "positions", "momenta", "forces", "cell", "pbc")}
                    arrays.update(energy="potential_energy", reference_energy="reference_energy")
                    for attribute, key in arrays.items():
                        actual = getattr(data, attribute)
                        np.testing.assert_array_equal(actual, fields[key], err_msg=key)
                        self.assertEqual(actual.dtype, fields[key].dtype, key)
                    self.assertEqual((data.step, data.engine), (7, "native"))
                    self.assertEqual(data.parameters, json.loads(str(fields["parameters"])))
                    self.assertEqual(data.offsets, {"thermo": 257, "xyz": 1025})
                    self.assertEqual(data.temperature_K, None if ensemble == "nve" else 375.0)
                    if ensemble == "nvt_nhc":
                        self.assertEqual(data.nhc_state["temperature_K"], 375.0)
                        for attribute, key in (("eta", "nhc_eta"), ("p_eta", "nhc_p_eta")):
                            np.testing.assert_array_equal(data.nhc_state[attribute], fields[key])
                            self.assertEqual(data.nhc_state[attribute].dtype, fields[key].dtype)
                    else:
                        self.assertIsNone(data.nhc_state)
                    write_restart(rewritten, data)
                    with np.load(rewritten) as saved:
                        self.assertEqual(str(saved["format"]), "fastmd.distributed.md/1")
                        self.assertEqual(set(saved.files), set(fields))
                        for key in fields.keys() - {"format"}:
                            np.testing.assert_array_equal(saved[key], fields[key], err_msg=key)
                            self.assertEqual(saved[key].dtype, fields[key].dtype, key)

    def test_legacy_files_without_engine_default_to_domain(self):
        for format_name in ("fastmd.distributed.md/1", "matris.distributed.md/1"):
            with self.subTest(format=format_name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "legacy.npz"
                fields = _fields("nvt_nhc", format_name)
                fields.pop("engine")
                np.savez(path, **fields)
                data = read_restart(path)
                self.assertEqual(data.engine, "domain")
                self.assertEqual(data.step, 7)
                np.testing.assert_array_equal(data.nhc_state["eta"], fields["nhc_eta"])

    def test_rejects_unknown_formats_before_reading_state(self):
        for format_name in ("other.distributed.md/1", "matris.distributed.md/2", "fastmd.distributed.md/2", ""):
            with self.subTest(format=format_name), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "unknown.npz"
                # No state: the format rejection must happen before any field is read.
                np.savez(path, format=np.array(format_name))
                with self.assertRaisesRegex(ValueError, "not a .* restart file"):
                    read_restart(path)


if __name__ == "__main__":
    unittest.main()
