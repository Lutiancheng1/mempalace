"""Tests for the multi-model embedding registry (mempalace.embedding).

Coverage targets the parts that don't require sentence-transformers to be
installed: the model registry, name canonicalisation, fallback behaviour,
config plumbing, and the CLI ``models`` listing. The actual ST-backed model
loaders are smoke-tested with a mock so the suite stays fast and offline.
"""

from __future__ import annotations

from unittest import mock

import pytest

from mempalace import embedding as emb_mod
from mempalace.config import MempalaceConfig


def test_registry_has_canonical_models():
    """Every registered model has the fields the loader relies on."""
    assert "default" in emb_mod.MODEL_REGISTRY
    assert "bge-small-zh-v1.5" in emb_mod.MODEL_REGISTRY
    for key, meta in emb_mod.MODEL_REGISTRY.items():
        assert {"hf_repo", "dimension", "backend", "ef_name", "language"}.issubset(meta)
        assert meta["backend"] in {"onnx", "sentence-transformers"}
        assert isinstance(meta["dimension"], int) and meta["dimension"] > 0


def test_list_models_matches_registry():
    listed = emb_mod.list_models()
    assert len(listed) == len(emb_mod.MODEL_REGISTRY)
    keys = {m["key"] for m in listed}
    assert keys == set(emb_mod.MODEL_REGISTRY.keys())


def test_get_model_meta_unknown_raises():
    with pytest.raises(KeyError, match="Unknown embedding_model"):
        emb_mod.get_model_meta("totally-not-a-real-model-12345")


def test_get_model_meta_known_returns_dict():
    meta = emb_mod.get_model_meta("bge-small-zh-v1.5")
    assert meta["dimension"] == 512
    assert meta["language"] == "zh"
    assert meta["backend"] == "sentence-transformers"


def test_default_model_uses_onnx_backend(monkeypatch):
    """The 'default' key must keep the ONNX path so legacy palaces still load."""
    emb_mod._EF_CACHE.clear()
    monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)
    monkeypatch.delenv("MEMPALACE_EMBEDDING_DEVICE", raising=False)

    ef = emb_mod._real_get_embedding_function(model="default", device="cpu")
    assert ef.name() == "default"
    # _MempalaceONNX is the ONNX subclass — verify by checking class chain.
    assert "ONNX" in type(ef).__mro__[1].__name__


def test_unknown_model_falls_back_to_default(monkeypatch, caplog):
    """Upstream semantics: unrecognized model names land on the built-in
    MiniLM ONNX EF (same vectors, name masked to "default") rather than
    raising — a typo must never hard-fail the miner."""
    emb_mod._EF_CACHE.clear()
    emb_mod._WARNED.clear()
    monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)

    ef = emb_mod._real_get_embedding_function(model="not-a-real-model", device="cpu")

    assert ef.name() == "default"
    assert "ONNX" in type(ef).__mro__[1].__name__


def test_st_loader_called_with_repo_and_name(monkeypatch):
    """Non-default models should hit the SentenceTransformer wrapper with the
    HF repo and our spoofed ef_name (so chromadb's collection check passes)."""
    emb_mod._EF_CACHE.clear()

    fake_st_class = mock.MagicMock()
    fake_st_instance = mock.MagicMock()
    fake_st_instance.name.return_value = "bge-small-zh-v1.5"
    fake_st_class.return_value = fake_st_instance

    # Patch _build_st_ef directly: we just want to confirm it's the path
    # taken for non-default models, and that hf_repo + ef_name reach it.
    captured = {}

    def fake_build(model_key, hf_repo, ef_name, device):
        captured["args"] = (model_key, hf_repo, ef_name, device)
        return fake_st_instance

    monkeypatch.setattr(emb_mod, "_build_st_ef", fake_build)

    ef = emb_mod._real_get_embedding_function(model="bge-small-zh-v1.5", device="cpu")

    assert ef is fake_st_instance
    assert captured["args"][0] == "bge-small-zh-v1.5"
    assert captured["args"][1] == "BAAI/bge-small-zh-v1.5"
    assert captured["args"][2] == "bge-small-zh-v1.5"


def test_st_missing_falls_back_to_default(monkeypatch, caplog):
    """If sentence-transformers isn't installed, log a hint and use default."""
    emb_mod._EF_CACHE.clear()
    emb_mod._WARNED.clear()

    def boom(*a, **kw):
        raise ImportError("No module named 'sentence_transformers'")

    monkeypatch.setattr(emb_mod, "_build_st_ef", boom)

    with caplog.at_level("WARNING"):
        ef = emb_mod._real_get_embedding_function(model="bge-small-zh-v1.5", device="cpu")

    assert ef.name() == "default"
    msgs = " ".join(r.message for r in caplog.records)
    assert "sentence-transformers" in msgs or "mempalace[chinese]" in msgs


def test_describe_model_known():
    out = emb_mod.describe_model("bge-small-zh-v1.5")
    assert "512d" in out
    assert "zh" in out


def test_describe_model_unknown():
    out = emb_mod.describe_model("never-heard-of-it")
    assert "unknown" in out.lower()


def test_config_embedding_model_env_overrides_file(monkeypatch, tmp_path):
    """MEMPALACE_EMBEDDING_MODEL env var must beat config.json."""
    cfg_dir = tmp_path / ".mempalace"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(
        '{"embedding_model": "bge-base-zh-v1.5"}'
    )
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("MEMPALACE_EMBEDDING_MODEL", "multilingual-e5-small")
    # Bust the lru_cache on MempalaceConfig.
    MempalaceConfig.cache_clear() if hasattr(MempalaceConfig, "cache_clear") else None

    cfg = MempalaceConfig()
    assert cfg.embedding_model == "multilingual-e5-small"


def test_config_embedding_model_defaults_to_minilm(monkeypatch, tmp_path):
    """Upstream contract: unset embedding_model resolves to "minilm" (the
    built-in ONNX default). Registry keys resolve verbatim when configured."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("MEMPALACE_EMBEDDING_MODEL", raising=False)

    cfg = MempalaceConfig()
    assert cfg.embedding_model == "minilm"
