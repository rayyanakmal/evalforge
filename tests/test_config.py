"""
US-4: config.py — GateConfig model and load/save functions.

Tests for:
  - GateConfig Pydantic model with defaults
  - SuiteConfig and ProviderConfig models
  - load_config from valid YAML
  - load_config missing file → clear error
  - load_config invalid YAML → error
  - save_config writes valid YAML
  - Roundtrip (save then load)
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from evalforge.config import (
    GateConfig, SuiteConfig, ProviderConfig,
    load_config, save_config,
)


# ---------------------------------------------------------------------------
# Model Tests
# ---------------------------------------------------------------------------

class TestSuiteConfig:
    """Tests for SuiteConfig model."""

    def test_defaults(self):
        """SuiteConfig has correct defaults."""
        sc = SuiteConfig(path="test-suites/example/suite.yaml")
        assert sc.path == "test-suites/example/suite.yaml"
        assert sc.allowed_regression_pct == 5.0

    def test_custom_regression(self):
        """allowed_regression_pct can be customized."""
        sc = SuiteConfig(path="s.yaml", allowed_regression_pct=10.0)
        assert sc.allowed_regression_pct == 10.0

    def test_from_dict(self):
        """SuiteConfig can be created from a dict."""
        sc = SuiteConfig.model_validate({"path": "s.yaml", "allowed_regression_pct": 3.0})
        assert sc.path == "s.yaml"
        assert sc.allowed_regression_pct == 3.0


class TestProviderConfig:
    """Tests for ProviderConfig model."""

    def test_required_fields(self):
        """ProviderConfig requires provider and model."""
        pc = ProviderConfig(provider="deepseek", model="deepseek-v4-flash")
        assert pc.provider == "deepseek"
        assert pc.model == "deepseek-v4-flash"

    def test_from_dict(self):
        """ProviderConfig can be created from a dict."""
        pc = ProviderConfig.model_validate({"provider": "openai", "model": "gpt-4o"})
        assert pc.provider == "openai"
        assert pc.model == "gpt-4o"


class TestGateConfigModel:
    """Tests for GateConfig Pydantic model."""

    def test_defaults(self):
        """GateConfig has correct defaults."""
        gc = GateConfig()
        assert gc.baseline_dir == "evalforge-baselines/"
        assert gc.suites == []
        assert gc.concurrency == 10
        assert gc.judge is None
        assert gc.target is None

    def test_minimal_config(self):
        """GateConfig with just suites."""
        gc = GateConfig(
            suites=[SuiteConfig(path="s.yaml")],
        )
        assert len(gc.suites) == 1
        assert gc.suites[0].path == "s.yaml"

    def test_full_config(self):
        """GateConfig with all fields populated."""
        gc = GateConfig(
            baseline_dir="my-baselines/",
            suites=[
                SuiteConfig(path="s1.yaml", allowed_regression_pct=3.0),
                SuiteConfig(path="s2.yaml", allowed_regression_pct=10.0),
            ],
            judge=ProviderConfig(provider="deepseek", model="deepseek-v4-flash"),
            target=ProviderConfig(provider="openai", model="gpt-4o"),
            concurrency=5,
        )
        assert gc.baseline_dir == "my-baselines/"
        assert len(gc.suites) == 2
        assert gc.suites[0].allowed_regression_pct == 3.0
        assert gc.suites[1].allowed_regression_pct == 10.0
        assert gc.judge.provider == "deepseek"
        assert gc.target.model == "gpt-4o"
        assert gc.concurrency == 5

    def test_from_dict(self):
        """GateConfig can be created from a dict."""
        data = {
            "baseline_dir": "custom/",
            "suites": [{"path": "s.yaml"}],
            "concurrency": 8,
        }
        gc = GateConfig.model_validate(data)
        assert gc.baseline_dir == "custom/"
        assert len(gc.suites) == 1
        assert gc.concurrency == 8

    def test_serialize_to_dict(self):
        """GateConfig can be serialized to dict."""
        gc = GateConfig(
            suites=[SuiteConfig(path="s.yaml")],
        )
        d = gc.model_dump()
        assert d["baseline_dir"] == "evalforge-baselines/"
        assert d["suites"][0]["path"] == "s.yaml"


# ---------------------------------------------------------------------------
# load_config Tests
# ---------------------------------------------------------------------------

class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_valid_yaml(self):
        """load_config loads a valid evalforge.yaml file."""
        yaml_content = """
baseline_dir: my-baselines/
suites:
  - path: test-suites/example/suite.yaml
    allowed_regression_pct: 5.0
concurrency: 7
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            assert config.baseline_dir == "my-baselines/"
            assert len(config.suites) == 1
            assert config.suites[0].path == "test-suites/example/suite.yaml"
            assert config.suites[0].allowed_regression_pct == 5.0
            assert config.concurrency == 7
        finally:
            tmp_path.unlink()

    def test_load_with_judge_and_target(self):
        """load_config loads judge and target provider configs."""
        yaml_content = """
suites:
  - path: s.yaml
judge:
  provider: deepseek
  model: deepseek-v4-flash
target:
  provider: openai
  model: gpt-4o
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            assert config.judge is not None
            assert config.judge.provider == "deepseek"
            assert config.judge.model == "deepseek-v4-flash"
            assert config.target is not None
            assert config.target.provider == "openai"
            assert config.target.model == "gpt-4o"
        finally:
            tmp_path.unlink()

    def test_load_minimal_config(self):
        """load_config with minimal YAML (just suites)."""
        yaml_content = """
suites:
  - path: s.yaml
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_content)
            tmp_path = Path(f.name)

        try:
            config = load_config(tmp_path)
            assert config.baseline_dir == "evalforge-baselines/"  # default
            assert len(config.suites) == 1
            assert config.concurrency == 10  # default
            assert config.judge is None
            assert config.target is None
        finally:
            tmp_path.unlink()

    def test_load_missing_file_raises_clear_error(self):
        """load_config with non-existent file raises FileNotFoundError
        with a clear message including 'evalforge init'."""
        missing = Path("/nonexistent/evalforge.yaml")
        with pytest.raises(FileNotFoundError) as exc_info:
            load_config(missing)
        msg = str(exc_info.value)
        assert "No config found" in msg or "evalforge init" in msg

    def test_load_invalid_yaml_raises_error(self):
        """load_config with invalid YAML raises an error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("this: is: not: valid: yaml: {{{")
            tmp_path = Path(f.name)

        try:
            with pytest.raises((yaml.YAMLError, ValueError)):
                load_config(tmp_path)
        finally:
            tmp_path.unlink()

    def test_load_empty_file(self):
        """load_config on an empty file raises an error."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("")
            tmp_path = Path(f.name)

        try:
            with pytest.raises((ValueError, yaml.YAMLError)):
                load_config(tmp_path)
        finally:
            tmp_path.unlink()


# ---------------------------------------------------------------------------
# save_config Tests
# ---------------------------------------------------------------------------

class TestSaveConfig:
    """Tests for save_config function."""

    def test_save_creates_valid_yaml(self):
        """save_config writes valid YAML that can be read back."""
        config = GateConfig(
            baseline_dir="out/",
            suites=[
                SuiteConfig(path="s1.yaml", allowed_regression_pct=3.0),
            ],
            concurrency=5,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            tmp_path = Path(f.name)

        try:
            save_config(config, tmp_path)
            content = tmp_path.read_text()
            assert "baseline_dir" in content
            assert "out/" in content
            assert "s1.yaml" in content
            assert "allowed_regression_pct" in content

            # Should be loadable
            loaded = load_config(tmp_path)
            assert loaded.baseline_dir == "out/"
            assert loaded.suites[0].path == "s1.yaml"
            assert loaded.suites[0].allowed_regression_pct == 3.0
        finally:
            tmp_path.unlink()

    def test_save_creates_parent_dirs(self):
        """save_config creates parent directories if they don't exist."""
        config = GateConfig(suites=[SuiteConfig(path="s.yaml")])

        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "sub" / "deep" / "evalforge.yaml"
            save_config(config, nested)
            assert nested.exists()
            content = nested.read_text()
            assert "suites" in content

    def test_roundtrip_save_then_load(self):
        """save_config followed by load_config returns equivalent config."""
        original = GateConfig(
            baseline_dir="roundtrip/",
            suites=[
                SuiteConfig(path="a.yaml", allowed_regression_pct=2.0),
                SuiteConfig(path="b.yaml", allowed_regression_pct=7.5),
            ],
            judge=ProviderConfig(provider="deepseek", model="v4"),
            target=ProviderConfig(provider="openai", model="gpt-4o"),
            concurrency=3,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            tmp_path = Path(f.name)

        try:
            save_config(original, tmp_path)
            loaded = load_config(tmp_path)

            assert loaded.baseline_dir == original.baseline_dir
            assert len(loaded.suites) == len(original.suites)
            for i in range(len(original.suites)):
                assert loaded.suites[i].path == original.suites[i].path
                assert loaded.suites[i].allowed_regression_pct == pytest.approx(
                    original.suites[i].allowed_regression_pct
                )
            assert loaded.judge.provider == original.judge.provider
            assert loaded.judge.model == original.judge.model
            assert loaded.target.provider == original.target.provider
            assert loaded.target.model == original.target.model
            assert loaded.concurrency == original.concurrency
        finally:
            tmp_path.unlink()
