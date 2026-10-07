"""Train the brand retriever, build its index and export to ONNX.

    # full run on a GPU workstation (~30-60 min on an RTX 2000 Ada)
    python -m medlens.retrieval.train

    # quick CPU smoke test
    python -m medlens.retrieval.train --steps 30 --batch-size 64 --eval-every 15 --eval-queries 200

Outputs (models/retriever/):
    model.onnx    encoder for CPU inference with onnxruntime (no torch at serve time)
    index.npy     float16 embeddings of every brand core in the catalog
    cores.json    brand core strings, row-aligned with index.npy
    metrics.json  training config + recall@k on seen and held-out brands
    encoder.pt    PyTorch weights (for further fine-tuning)
"""
import argparse
import json
import math
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from medlens.catalog import CATALOG, ROOT
from medlens.retrieval.data import PairSampler, is_heldout, noisy_query, unique_cores
from medlens.retrieval.model import MAX_LEN, CharEncoder, encode_batch

OUT = ROOT / "models" / "retriever"


@torch.no_grad()
def embed(model: CharEncoder, texts: list[str], device: str, batch: int = 4096) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(texts), batch):
        with torch.autocast(device_type=device.split(":")[0], dtype=torch.bfloat16, enabled=device != "cpu"):
            out.append(model(encode_batch(texts[i : i + batch]).to(device)).float())
    return torch.cat(out)


@torch.no_grad()
def evaluate(model, all_cores, index, queries: dict[str, list[tuple[str, int]]], device) -> dict:
    res = {}
    for name, pairs in queries.items():
        q = embed(model, [p[0] for p in pairs], device)
        top = (q @ index.T).topk(10, dim=1).indices.cpu()
        gold = torch.tensor([p[1] for p in pairs]).unsqueeze(1)
        hits = top == gold
        res[name] = {"recall@1": hits[:, :1].any(1).float().mean().item(), "recall@10": hits.any(1).float().mean().item()}
    return res


def export_onnx(model: CharEncoder, path) -> None:
    model = model.float().cpu().eval()
    dummy = encode_batch(["augmentin 625 duo"])
    torch.onnx.export(
        model, (dummy,), str(path), input_names=["ids"], output_names=["emb"],
        dynamic_axes={"ids": {0: "batch"}, "emb": {0: "batch"}}, opset_version=17, dynamo=False,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--temperature", type=float, default=0.05)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-queries", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)
    print(f"device: {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else ""))

    all_cores = unique_cores(pd.read_parquet(CATALOG, columns=["name_norm"]))
    core_pos = {c: i for i, c in enumerate(all_cores)}
    train_cores = [c for c in all_cores if not is_heldout(c)]
    heldout = [c for c in all_cores if is_heldout(c)]
    print(f"brands: {len(all_cores):,} total, {len(train_cores):,} train, {len(heldout):,} held out")

    rng = random.Random(args.seed + 1)  # eval noise is independent of training noise
    eval_sets = {
        name: [(noisy_query(c, rng), core_pos[c]) for c in rng.sample(pool, min(args.eval_queries, len(pool)))]
        for name, pool in (("seen", train_cores), ("heldout", heldout))
    }

    model = CharEncoder(dim=args.dim, layers=args.layers).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup = min(1000, args.steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / args.steps, 1.0)))
    )
    sampler = PairSampler(train_cores, args.batch_size, seed=args.seed)

    history, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        model.train()
        queries, targets = sampler.batch()
        q_ids, t_ids = encode_batch(queries).to(device), encode_batch(targets).to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            q, t = model(q_ids), model(t_ids)
            logits = (q @ t.T).float() / args.temperature
            labels = torch.arange(len(queries), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % 100 == 0:
            print(f"step {step:6d}  loss {loss.item():.4f}  lr {sched.get_last_lr()[0]:.2e}  {time.time() - t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            index = embed(model, all_cores, device)
            metrics = evaluate(model, all_cores, index, eval_sets, device)
            history.append({"step": step, **metrics})
            print(f"  eval @ {step}: " + "  ".join(f"{k} R@1 {v['recall@1']:.3f} R@10 {v['recall@10']:.3f}" for k, v in metrics.items()), flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": model.config}, OUT / "encoder.pt")
    index = embed(model, all_cores, device).cpu().numpy().astype(np.float16)
    np.save(OUT / "index.npy", index)
    (OUT / "cores.json").write_text(json.dumps(all_cores))
    export_onnx(model, OUT / "model.onnx")
    (OUT / "metrics.json").write_text(
        json.dumps({"args": vars(args), "max_len": MAX_LEN, "final": history[-1], "history": history}, indent=2)
    )
    print(f"saved -> {OUT.relative_to(ROOT)}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
