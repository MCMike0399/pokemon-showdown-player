import importlib.util
from pathlib import Path

import pytest


def scanner():
    file = Path(__file__).resolve().parents[1]/"scripts/public_snapshot.py"
    spec = importlib.util.spec_from_file_location("public_snapshot", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_public_snapshot_rejects_secret_and_private_artifacts(tmp_path, monkeypatch):
    module = scanner()
    monkeypatch.setattr(module, "private_values", lambda: ["sentinel-private-password"])
    (tmp_path/"README.md").write_text("sentinel-private-password")
    with pytest.raises(ValueError, match="snapshot rejected"):
        module.scan(tmp_path)
    (tmp_path/"README.md").write_text("clean public source")
    (tmp_path/"model.pt").write_text("weights")
    with pytest.raises(ValueError, match="artifact"):
        module.scan(tmp_path)
    (tmp_path/"model.pt").unlink()
    (tmp_path/".env").write_text("PS_PASSWORD=sentinel")
    with pytest.raises(ValueError, match="artifact"):
        module.scan(tmp_path)


def test_public_snapshot_has_no_history_and_scan_passes(tmp_path):
    module = scanner()
    result = module.export(tmp_path/"public")
    assert result["secret_scan"] == "passed" and not result["history_copied"]
    assert not (tmp_path/"public/.git").exists()
    assert not (tmp_path/"public/.env").exists()
    assert not (tmp_path/"public/data").exists()
    assert not (tmp_path/"public/teams.json").exists()
    assert "Agent instructions" in (tmp_path/"public/AGENTS.md").read_text()
