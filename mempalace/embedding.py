"""Embedding function factory with hardware acceleration and multi-model support.

Returns a ChromaDB-compatible embedding function bound to a user-selected
model + ONNX Runtime execution provider.

Models (env ``MEMPALACE_EMBEDDING_MODEL`` or ``embedding_model`` in
``~/.mempalace/config.json``):

* ``default`` — chromadb's all-MiniLM-L6-v2 ONNX (384d, English-centric, ~80MB)
* ``bge-small-zh-v1.5`` — BAAI BGE small Chinese (512d, ~95MB)
* ``bge-base-zh-v1.5`` — BAAI BGE base Chinese (768d, ~410MB)
* ``multilingual-e5-small`` — intfloat E5 small multilingual (384d, ~470MB)
* ``multilingual-e5-base`` — intfloat E5 base multilingual (768d, ~1.1GB)
* ``text2vec-base-chinese`` — shibing624 Chinese text2vec (768d, ~410MB)

Non-default models load via ``sentence-transformers`` (install with
``pip install sentence-transformers`` or the ``mempalace[chinese]`` /
``mempalace[multilingual]`` extras).

Switching models on an existing palace requires re-mining all drawers,
because vectors from different models live in incompatible spaces.

Devices (env ``MEMPALACE_EMBEDDING_DEVICE`` or ``embedding_device``):

* ``auto`` — prefer CUDA ▸ CoreML ▸ DirectML, fall back to CPU
* ``cpu`` — force CPU (the historical default)
* ``cuda`` — NVIDIA GPU via ``onnxruntime-gpu`` (``pip install mempalace[gpu]``)
* ``coreml`` — Apple Neural Engine (macOS)
* ``dml`` — DirectML (Windows / AMD / Intel GPUs)

Requesting an unavailable accelerator emits a warning and falls back to CPU
rather than hard-failing — mining must still work on a laptop without CUDA.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_PROVIDER_MAP = {
    "cpu": ["CPUExecutionProvider"],
    "cuda": ["CUDAExecutionProvider", "CPUExecutionProvider"],
    "coreml": ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    "dml": ["DmlExecutionProvider", "CPUExecutionProvider"],
}

_DEVICE_EXTRA = {
    "cuda": "mempalace[gpu]",
    "coreml": "mempalace[coreml]",
    "dml": "mempalace[dml]",
}

_AUTO_ORDER = [
    ("CUDAExecutionProvider", "cuda"),
    ("CoreMLExecutionProvider", "coreml"),
    ("DmlExecutionProvider", "dml"),
]

# ── Model registry ────────────────────────────────────────────────────────────
# Each entry maps a stable user-facing key to the upstream HuggingFace repo.
# ``backend`` selects the loader: ``"onnx"`` uses chromadb's bundled
# ``ONNXMiniLM_L6_V2`` (only valid for ``default``); ``"sentence-transformers"``
# uses the ST library and chromadb's SentenceTransformerEmbeddingFunction
# wrapper.
MODEL_REGISTRY: dict[str, dict] = {
    "default": {
        "hf_repo": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "backend": "onnx",
        "ef_name": "default",
        "language": "en",
        "description": "ChromaDB built-in ONNX, English-centric, ~80MB",
    },
    "bge-small-zh-v1.5": {
        "hf_repo": "BAAI/bge-small-zh-v1.5",
        "dimension": 512,
        "backend": "sentence-transformers",
        "ef_name": "bge-small-zh-v1.5",
        "language": "zh",
        "description": "BAAI BGE small Chinese, fast & accurate, ~95MB",
    },
    "bge-base-zh-v1.5": {
        "hf_repo": "BAAI/bge-base-zh-v1.5",
        "dimension": 768,
        "backend": "sentence-transformers",
        "ef_name": "bge-base-zh-v1.5",
        "language": "zh",
        "description": "BAAI BGE base Chinese, higher accuracy, ~410MB",
    },
    "multilingual-e5-small": {
        "hf_repo": "intfloat/multilingual-e5-small",
        "dimension": 384,
        "backend": "sentence-transformers",
        "ef_name": "multilingual-e5-small",
        "language": "multilingual",
        "description": "intfloat E5 small, 100+ languages, ~470MB",
    },
    "multilingual-e5-base": {
        "hf_repo": "intfloat/multilingual-e5-base",
        "dimension": 768,
        "backend": "sentence-transformers",
        "ef_name": "multilingual-e5-base",
        "language": "multilingual",
        "description": "intfloat E5 base, 100+ languages, ~1.1GB",
    },
    "text2vec-base-chinese": {
        "hf_repo": "shibing624/text2vec-base-chinese",
        "dimension": 768,
        "backend": "sentence-transformers",
        "ef_name": "text2vec-base-chinese",
        "language": "zh",
        "description": "shibing624 text2vec Chinese, ~410MB",
    },
}

_MODEL_EXTRA = {
    "bge-small-zh-v1.5": "mempalace[chinese]",
    "bge-base-zh-v1.5": "mempalace[chinese]",
    "text2vec-base-chinese": "mempalace[chinese]",
    "multilingual-e5-small": "mempalace[multilingual]",
    "multilingual-e5-base": "mempalace[multilingual]",
}

_EF_CACHE: dict = {}
_WARNED: set = set()


def list_models() -> list[dict]:
    """Return the registered models as a list of dicts (for CLI listing)."""
    return [{"key": key, **meta} for key, meta in MODEL_REGISTRY.items()]


def get_model_meta(model: str) -> dict:
    """Look up registry metadata for ``model``. Raises ``KeyError`` if unknown."""
    if model not in MODEL_REGISTRY:
        known = ", ".join(MODEL_REGISTRY.keys())
        raise KeyError(f"Unknown embedding_model {model!r}. Known: {known}")
    return MODEL_REGISTRY[model]


def _resolve_providers(device: str) -> tuple[list, str]:
    """Return ``(provider_list, effective_device)`` for ``device``.

    Falls back to CPU (with a one-shot warning) when the requested
    accelerator is not compiled into the installed ``onnxruntime``.
    """
    device = (device or "auto").strip().lower()

    try:
        import onnxruntime as ort

        available = set(ort.get_available_providers())
    except ImportError:
        return (["CPUExecutionProvider"], "cpu")

    if device == "auto":
        for provider, name in _AUTO_ORDER:
            if provider in available:
                return ([provider, "CPUExecutionProvider"], name)
        return (["CPUExecutionProvider"], "cpu")

    requested = _PROVIDER_MAP.get(device)
    if requested is None:
        if device not in _WARNED:
            logger.warning("Unknown embedding_device %r — falling back to cpu", device)
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    preferred = requested[0]
    if preferred == "CPUExecutionProvider":
        return (requested, "cpu")

    if preferred not in available:
        if device not in _WARNED:
            extra = _DEVICE_EXTRA.get(device, "the matching mempalace extra for your device")
            logger.warning(
                "embedding_device=%r requested but %s is not installed — "
                "falling back to CPU. Install %s.",
                device,
                preferred,
                extra,
            )
            _WARNED.add(device)
        return (["CPUExecutionProvider"], "cpu")

    return (requested, device)


def _build_default_ef(providers: list):
    """ONNX path: subclass ``ONNXMiniLM_L6_V2`` with name ``"default"``.

    Why the rename: ChromaDB 1.5 persists the EF identity on the collection
    and rejects reads that pass a differently-named EF. Spoofing the name to
    ``"default"`` lets one EF class serve palaces created with
    ``DefaultEmbeddingFunction`` and palaces we create ourselves, with the
    same GPU-capable ``preferred_providers``.

    Implemented via :func:`_build_ef_class` so the pre-multi-model
    monkey-patch surface stays usable.
    """
    ef_cls = _build_ef_class()
    return ef_cls(preferred_providers=providers)


def _build_ef_class():
    """Legacy hook returning the ONNX EF *class* (not an instance).

    Kept for the pre-multi-model test surface (``test_embedding.py``) which
    monkey-patches this name. New code should use :func:`_build_default_ef`.
    """
    from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

    class _MempalaceONNX(ONNXMiniLM_L6_V2):
        @staticmethod
        def name() -> str:
            return "default"

    return _MempalaceONNX


def _build_st_ef(model_key: str, hf_repo: str, ef_name: str, device: str):
    """SentenceTransformer path: wrap ST in a chromadb-compatible EF.

    ``device`` is the resolved mempalace device label (``cpu``/``coreml``/...);
    ST itself only knows about ``cpu``/``cuda``/``mps``. Apple Silicon best
    available is ``mps`` (Metal Performance Shaders), which both ST and PyTorch
    pick up automatically — we pass ``cpu`` for everything but ``cuda`` since
    chromadb's wrapper does not expose ``mps`` directly. The dominant cost is
    the model itself, not the device fence, so ``cpu`` here is fine.
    """
    try:
        from sentence_transformers import SentenceTransformer  # noqa: F401
    except ImportError as exc:
        extra = _MODEL_EXTRA.get(model_key, "sentence-transformers")
        raise ImportError(
            f"embedding_model={model_key!r} requires sentence-transformers. "
            f"Install with: pip install {extra}"
        ) from exc

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    st_device = "cuda" if device == "cuda" else "cpu"

    class _MempalaceST(SentenceTransformerEmbeddingFunction):
        _ef_name = ef_name

        @staticmethod
        def name() -> str:
            return _MempalaceST._ef_name

    return _MempalaceST(model_name=hf_repo, device=st_device)


def get_embedding_function(
    device: Optional[str] = None,
    model: Optional[str] = None,
):
    """Return a cached embedding function for ``model`` on ``device``.

    ``device=None`` reads from :class:`MempalaceConfig.embedding_device`.
    ``model=None`` reads from :class:`MempalaceConfig.embedding_model`.

    Unknown ``model`` values fall back to ``"default"`` with a one-shot
    warning so a typo or stale config never hard-fails the miner. Likewise,
    a registered model whose backend dependency is missing
    (e.g. ``sentence-transformers`` not installed) falls back to default —
    the alternative is the miner crashing mid-run on a laptop without the
    extra. Search will be wrong (vectors live in incompatible spaces) but
    that surfaces in the next run as an EF-name mismatch from chromadb,
    which is louder and easier to diagnose than an import crash.

    The returned function is shared across calls with the same resolved
    (model, provider list) so we only pay model-load cost once per process.
    """
    if device is None or model is None:
        from .config import MempalaceConfig

        cfg = MempalaceConfig()
        if device is None:
            device = cfg.embedding_device
        if model is None:
            model = cfg.embedding_model

    if model not in MODEL_REGISTRY:
        if model not in _WARNED:
            logger.warning(
                "Unknown embedding_model %r — falling back to %r. "
                "Run `mempalace models` to see registered keys.",
                model,
                "default",
            )
            _WARNED.add(model)
        model = "default"

    meta = MODEL_REGISTRY[model]
    providers, effective_device = _resolve_providers(device)
    cache_key = (model, tuple(providers))
    cached = _EF_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if meta["backend"] == "onnx":
        ef = _build_default_ef(providers)
    elif meta["backend"] == "sentence-transformers":
        try:
            ef = _build_st_ef(model, meta["hf_repo"], meta["ef_name"], effective_device)
        except ImportError as exc:
            if model not in _WARNED:
                logger.warning(
                    "embedding_model=%r needs sentence-transformers but it's "
                    "not installed (%s) — falling back to %r. Install with: "
                    "pip install %s",
                    model,
                    exc,
                    "default",
                    _MODEL_EXTRA.get(model, "sentence-transformers"),
                )
                _WARNED.add(model)
            return get_embedding_function(device=device, model="default")
    else:
        raise ValueError(f"Unknown backend {meta['backend']!r} for model {model!r}")

    _EF_CACHE[cache_key] = ef
    logger.info(
        "Embedding function initialized (model=%s backend=%s device=%s providers=%s)",
        model,
        meta["backend"],
        effective_device,
        providers,
    )
    return ef


def describe_device(device: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved device.

    Used by the miner CLI header so users can see at a glance whether GPU
    acceleration actually engaged.
    """
    if device is None:
        from .config import MempalaceConfig

        device = MempalaceConfig().embedding_device
    _, effective = _resolve_providers(device)
    return effective


def describe_model(model: Optional[str] = None) -> str:
    """Return a short human-readable label for the resolved model."""
    if model is None:
        from .config import MempalaceConfig

        model = MempalaceConfig().embedding_model
    try:
        meta = get_model_meta(model)
    except KeyError:
        return f"{model} (unknown — falling back to default at runtime)"
    return f"{model} ({meta['dimension']}d, {meta['language']})"
