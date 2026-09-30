# Fragment of mempalace.cli — executed into the package namespace.
# `mempalace models`: list the fork's registered embedding models (fork feature).


def cmd_models(args):
    """List registered embedding models and highlight the active one."""
    from ..embedding import MODEL_REGISTRY, list_models

    cfg = MempalaceConfig()
    active = cfg.embedding_model
    print()
    print("=" * 70)
    print("  MemPalace — Registered Embedding Models")
    print("=" * 70)
    print(f"  Active: {active}")
    if active not in MODEL_REGISTRY:
        print("          (built-in loader key — minilm / embeddinggemma /")
        print("           openai-compat are handled natively, not via the registry)")
    print()
    print(f"  {'KEY':<25} {'DIM':>5} {'LANG':<13} DESCRIPTION")
    print(f"  {'-' * 25} {'-' * 5} {'-' * 13} {'-' * 30}")
    for m in list_models():
        marker = "*" if m["key"] == active else " "
        print(
            f"{marker} {m['key']:<25} {m['dimension']:>5} "
            f"{m['language']:<13} {m['description']}"
        )
    print()
    print("  Switch via: MEMPALACE_EMBEDDING_MODEL=<key> mempalace ...")
    print('  Or set "embedding_model" in ~/.mempalace/config.json')
    print("  ⚠️  Switching models requires re-mining all drawers.")
    print()
