"""Character-level (byte) transformer encoder for noisy brand-name retrieval.

Byte-level on purpose: OCR errors are character substitutions ("0"/"o",
"rn"/"m"), which subword tokenizers fragment unpredictably. The same encoder
embeds both the noisy query line and the clean brand name (siamese).
"""
import torch
import torch.nn.functional as F
from torch import nn

MAX_LEN = 48
VOCAB = 257  # 256 byte values + padding (id 0)


def encode_batch(texts: list[str], max_len: int = MAX_LEN) -> torch.Tensor:
    ids = torch.zeros(len(texts), max_len, dtype=torch.long)
    for i, t in enumerate(texts):
        b = t.encode("utf-8")[:max_len]
        ids[i, : len(b)] = torch.tensor(list(b), dtype=torch.long) + 1
    return ids


class CharEncoder(nn.Module):
    def __init__(self, dim: int = 256, layers: int = 4, heads: int = 4, out_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.config = {"dim": dim, "layers": layers, "heads": heads, "out_dim": out_dim, "dropout": dropout}
        self.tok = nn.Embedding(VOCAB, dim, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(1, MAX_LEN, dim) * 0.02)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 2, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        mask = ids != 0
        x = self.tok(ids) + self.pos[:, : ids.shape[1]]
        x = self.encoder(x, src_key_padding_mask=~mask)
        x = self.norm(x)
        m = mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(self.proj(pooled), dim=-1)
