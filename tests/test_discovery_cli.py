"""User-facing commands preserve the selected inputs and relaxation options."""
import json

import pytest

from fastmd.discovery import cli, run


def test_run_forwards_ase_and_shard_options(monkeypatch, capsys, tmp_path):
    captured = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        captured["args"] = args
        return {"failed": 0, "unconverged": 2}

    monkeypatch.setattr(run, "run_selection", fake_run)
    assert cli.main(["run", "--model", "alignn", "--output", str(tmp_path),
                     "--fixed-cell", "--cuda-graph", "true", "--optimizer", "BFGS",
                     "--fmax", ".03", "--steps", "12", "--limit", "2",
                     "--num-shards", "3", "--shard-index", "1", "--resume",
                     "--model-kwargs", '{"force_multiplier":0.1666666667}']) == 0
    config = captured["config"]
    assert (config.relax_cell, config.optimizer, config.steps, config.fmax) == (False, "BFGS", 12, .03)
    assert captured["cuda_graph"] is True
    assert (captured["num_shards"], captured["shard_index"], captured["limit"], captured["resume"]) == (3, 1, 2, True)
    assert captured["model_kwargs"]["force_multiplier"] == pytest.approx(1 / 6)
    assert json.loads(capsys.readouterr().out)["unconverged"] == 2


def test_run_exit_codes(monkeypatch, capsys, tmp_path):
    command = ["run", "--model", "matris", "--output", str(tmp_path)]
    monkeypatch.setattr(run, "run_selection", lambda *args, **kwargs: {"failed": 1})
    assert cli.main(command) == 1

    def invalid(*args, **kwargs):
        raise ValueError("invalid configuration")

    monkeypatch.setattr(run, "run_selection", invalid)
    assert cli.main(command) == 2
    assert "invalid configuration" in capsys.readouterr().err


def test_prepare_reuses_committed_selection_without_resampling(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")

    def unexpected(*args, **kwargs):
        pytest.fail("Existing selection must not be sampled again")

    def extract(args, reuse):
        assert args.manifest == manifest
        assert reuse is True
        return {"count": 21548}

    monkeypatch.setattr(cli, "sample_manifest", unexpected)
    monkeypatch.setattr(cli, "_extract", extract)
    assert cli.main(["prepare", "--manifest", str(manifest)]) == 0


@pytest.mark.parametrize("value", ["[]", "null", "not-json"])
def test_model_options_require_json_object(value, tmp_path):
    with pytest.raises(SystemExit) as exc:
        cli.main(["run", "--model", "mace", "--output", str(tmp_path), "--model-kwargs", value])
    assert exc.value.code == 2
