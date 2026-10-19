"""Unit tests for the standalone TIIAC project checker."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools import tiiac_check_demo as checker


def test_result_accumulates_points_and_caps_at_maximum():
    result = checker.Resultado("sample", "sample", 5)
    result.bien("first", 4)
    result.bien("second", 4)
    assert result.cerrar().pts == 5
    assert result.ok == ["first", "second"]


def test_structure_reports_required_files_and_python_source(tmp_path):
    for filename in ("README.md", "requirements.txt", "run.py", ".gitignore"):
        (tmp_path / filename).write_text("content", encoding="utf-8")
    source = tmp_path / "src" / "module.py"
    source.parent.mkdir()
    source.write_text("pass", encoding="utf-8")
    result = checker.check_estructura(tmp_path)
    assert result.pts == 15
    assert not result.mal


def test_dependencies_detects_unpinned_requirements(tmp_path):
    (tmp_path / "requirements.txt").write_text("torch==2.0\nnumpy\n", encoding="utf-8")
    result = checker.check_dependencias(tmp_path)
    assert result.pts == 5
    assert any("solo 1/2" in message for message in result.mal)


def test_seeds_finds_all_rng_families_and_flags_scattered_files(tmp_path):
    (tmp_path / "a.py").write_text(
        "torch.manual_seed(1)\nnp.random.seed(1)\nrandom.seed(1)", encoding="utf-8"
    )
    (tmp_path / "b.py").write_text("torch.manual_seed(2)", encoding="utf-8")
    (tmp_path / "c.py").write_text("random.seed(2)", encoding="utf-8")
    result = checker.check_seeds(tmp_path)
    assert result.pts == 12
    assert any("centralizarlas" in message for message in result.aviso)


def test_smoke_check_executes_command_and_requires_a_numeric_metric(tmp_path, monkeypatch):
    (tmp_path / "run.py").write_text("# --smoke", encoding="utf-8")

    def run(*args, **kwargs):
        assert args[0] == [checker.sys.executable, "run.py", "--smoke"]
        return subprocess.CompletedProcess(args[0], 0, stdout="val/loss: 0.42\n", stderr="")

    monkeypatch.setattr(checker.subprocess, "run", run)
    result = checker.check_smoke(tmp_path, ejecutar=True)
    assert result.pts == 25
    assert not result.mal


def test_smoke_check_reports_nonzero_exit(tmp_path, monkeypatch):
    (tmp_path / "run.py").write_text("--smoke", encoding="utf-8")
    monkeypatch.setattr(
        checker.subprocess, "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 2, stdout="", stderr="failed"),
    )
    result = checker.check_smoke(tmp_path, ejecutar=True)
    assert result.pts == 5
    assert any("codigo 2" in message for message in result.mal)


def test_readme_requires_install_data_run_and_reference_instructions(tmp_path):
    (tmp_path / "README.md").write_text(
        "x" * 410 + "\npip install project\ndataset download\npython run.py\nhttps://arxiv.org/123\n```\n",
        encoding="utf-8",
    )
    result = checker.check_readme(tmp_path)
    assert result.pts == 13
    assert not result.mal


def test_ai_usage_requires_rows_and_a_critical_decision(tmp_path):
    (tmp_path / "AI_USAGE.md").write_text(
        "| Herramienta | Uso | Decisión |\n|---|---|---|\n"
        "| Assistant | revisión | corregimos propuesta |\n",
        encoding="utf-8",
    )
    result = checker.check_ai_usage(tmp_path)
    assert result.pts == 8
    assert not result.mal


def test_tracking_accepts_wandb_code_and_project_link(tmp_path):
    source = tmp_path / "src" / "tracking.py"
    source.parent.mkdir()
    source.write_text("import wandb", encoding="utf-8")
    (tmp_path / "README.md").write_text("https://wandb.ai/team/project", encoding="utf-8")
    result = checker.check_tracking(tmp_path)
    assert result.pts == 10
    assert not result.mal


def test_report_passes_only_when_score_and_critical_checks_pass(capsys, tmp_path):
    checks = [checker.Resultado(str(i), str(i), 10, critico=(i == 0)) for i in range(10)]
    for result in checks:
        result.bien("ok", 10)
    score, passed = checker.informe(checks, tmp_path)
    assert score == 100 and passed
    assert "RESULTADO" in capsys.readouterr().out
