import time

import numpy as np


def encode(enc_cfg, texts, device, cache_path=None):
    if cache_path is not None and cache_path.exists():
        return np.load(cache_path), float(cache_path.with_suffix(".time").read_text())
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(enc_cfg["model_name"], device=device, trust_remote_code=enc_cfg.get("trust_remote_code", False))
    if enc_cfg.get("max_seq_length"):
        model.max_seq_length = enc_cfg["max_seq_length"]
    prefix = enc_cfg.get("prefix", "")
    t0 = time.perf_counter()
    emb = model.encode(
        [prefix + t for t in texts],
        batch_size=enc_cfg.get("batch_size", 64),
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    elapsed = time.perf_counter() - t0
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, emb)
        cache_path.with_suffix(".time").write_text(str(elapsed))
    return emb, elapsed
