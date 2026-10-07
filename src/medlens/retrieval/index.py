"""Serve-time brand retriever: ONNX encoder + precomputed brand embeddings (no torch)."""
import json
from pathlib import Path

import numpy as np

from medlens.catalog import ROOT

MODEL_DIR = ROOT / "models" / "retriever"
MAX_LEN = 48  # must match medlens.retrieval.model.MAX_LEN


def encode_ids(texts: list[str]) -> np.ndarray:
    ids = np.zeros((len(texts), MAX_LEN), dtype=np.int64)
    for i, t in enumerate(texts):
        b = np.frombuffer(t.encode("utf-8")[:MAX_LEN], dtype=np.uint8)
        ids[i, : len(b)] = b.astype(np.int64) + 1
    return ids


class BrandRetriever:
    def __init__(self, model_dir: Path = MODEL_DIR):
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
        self.index = np.load(model_dir / "index.npy").astype(np.float32)
        self.cores: list[str] = json.loads((model_dir / "cores.json").read_text())

    @classmethod
    def available(cls, model_dir: Path = MODEL_DIR) -> bool:
        return all((model_dir / f).exists() for f in ("model.onnx", "index.npy", "cores.json"))

    def embed(self, texts: list[str]) -> np.ndarray:
        return self.session.run(None, {"ids": encode_ids(texts)})[0]

    def search(self, texts: list[str], k: int = 50) -> dict[str, float]:
        """Best cosine similarity per retrieved brand core, across all query texts."""
        sims = self.embed(texts) @ self.index.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        best: dict[str, float] = {}
        for row, cols in zip(sims, top):
            for c in cols:
                core = self.cores[c]
                best[core] = max(best.get(core, -1.0), float(row[c]))
        return best
