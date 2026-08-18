#!/usr/bin/env python3
"""Ring-A-Bell prompt discovery for the Obama / celebrity concept.

Same two-stage method as Get_Concept_Vector.ipynb + InversePrompt.ipynb:
  1) CLIP concept vector = mean(embed(Obama prompt) - embed(non-Obama pair))
  2) Genetic search for token strings whose embedding matches
     embed(seed) + eta * concept_vector

Kaggle T4 x2 (internet on, then)::

    !git clone --depth 1 https://github.com/huskywannacry/Fork_RAB.git
    !pip install -q transformers tqdm
    !python Fork_RAB/train_obama.py --n-prompts 100 --output /kaggle/working/Obama_invprompts.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import CLIPTextModel, CLIPTokenizer

SOT_ID = 49406
EOT_ID = 49407
VOCAB_HIGH = 49406  # randint high is exclusive; keep 1 .. 49405


def resolve_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    here = Path(__file__).resolve().parent
    if (here / "data" / "Prompts_For_ConceptVector" / "Obama_30.csv").exists():
        return here
    for candidate in (
        Path("/kaggle/input/ring-a-bell"),
        Path("/kaggle/working"),
        Path.cwd(),
    ):
        if (candidate / "data" / "Prompts_For_ConceptVector" / "Obama_30.csv").exists():
            return candidate
    return here


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip(model_id: str, device: str):
    kwargs = {}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        kwargs["token"] = token
    try:
        tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer", **kwargs)
        text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", **kwargs)
    except Exception as exc:
        print(f"[Warn] {model_id} failed ({exc}). Falling back to openai/clip-vit-large-patch14")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", **kwargs)
        text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", **kwargs)
    text_encoder.to(device)
    text_encoder.eval()
    return tokenizer, text_encoder


@torch.no_grad()
def encode_prompts(prompts, tokenizer, text_encoder, device, batch_size=16):
    embeds = []
    for start in range(0, len(prompts), batch_size):
        chunk = list(prompts[start : start + batch_size])
        text_input = tokenizer(
            chunk,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        embed = text_encoder(text_input.input_ids.to(device), return_dict=True)[0]
        embeds.append(embed.detach().cpu().numpy())
    return np.concatenate(embeds, axis=0)


def extract_concept_vector(args, tokenizer, text_encoder, device) -> np.ndarray:
    csv_path = Path(args.concept_csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"Concept-pair CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "prompt" not in df.columns or "prompt1" not in df.columns:
        raise ValueError(f"{csv_path} must have columns prompt and prompt1")

    pos, neg = [], []
    for _, row in df.iterrows():
        pos.extend([str(row.prompt)] * args.num_samples)
        neg.extend([str(row.prompt1)] * args.num_samples)

    print(f"[Info] Encoding {len(pos)} Obama / non-Obama pairs for the concept vector")
    pos_emb = encode_prompts(pos, tokenizer, text_encoder, device)
    neg_emb = encode_prompts(neg, tokenizer, text_encoder, device)
    if pos_emb.shape != neg_emb.shape:
        raise RuntimeError(f"Pair embed shapes differ: {pos_emb.shape} vs {neg_emb.shape}")
    vec = np.mean(pos_emb - neg_emb, axis=0).astype(np.float32)
    out = Path(args.vector_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, vec)
    print(f"[Info] Saved concept vector {tuple(vec.shape)} -> {out}")
    return vec


def init_population(population_size: int, length: int) -> list[torch.Tensor]:
    pad = 76 - length
    population = []
    for _ in range(population_size):
        mid = torch.randint(low=1, high=VOCAB_HIGH, size=(1, length))
        tokens = torch.concat(
            (
                torch.tensor([[SOT_ID]]),
                mid,
                torch.tile(torch.tensor([[EOT_ID]]), [1, pad]),
            ),
            1,
        )
        population.append(tokens)
    return population


@torch.no_grad()
def fitness(population, text_encoder, target_embed, device):
    dummy_tokens = torch.cat(population, 0)
    dummy_embed = text_encoder(dummy_tokens.to(device))[0]
    losses = ((target_embed - dummy_embed) ** 2).sum(dim=(1, 2))
    return losses.detach().cpu().numpy()


def crossover(parents, crossover_rate, length):
    new_population = []
    for i in range(len(parents)):
        new_population.append(parents[i])
        if random.random() < crossover_rate:
            idx = np.random.randint(0, len(parents))
            point = np.random.randint(1, length + 1)
            new_population.append(torch.concat((parents[i][:, :point], parents[idx][:, point:]), 1))
            new_population.append(torch.concat((parents[idx][:, :point], parents[i][:, point:]), 1))
    return new_population


def mutation(population, mutate_rate, length):
    for i in range(len(population)):
        if random.random() < mutate_rate:
            idx = np.random.randint(1, length + 1)
            value = int(np.random.randint(1, VOCAB_HIGH))
            population[i][:, idx] = value
    return population


def search_one(
    seed_prompt: str,
    concept_vec: np.ndarray,
    tokenizer,
    text_encoder,
    device: str,
    args,
    tag: str,
):
    text_input = tokenizer(
        seed_prompt,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        seed_embed = text_encoder(text_input.input_ids.to(device))[0]
    target = seed_embed + args.eta * torch.from_numpy(concept_vec).to(device=device, dtype=seed_embed.dtype)
    target = target.detach().clone()

    population = init_population(args.population, args.length)
    best_loss = float("inf")
    stall = 0
    best_tokens = None

    for step in range(args.generations):
        score = fitness(population, text_encoder, target, device)
        order = np.argsort(score)
        population = [population[i] for i in order][: args.population // 2]
        cur = float(score[order[0]])
        if cur + 1e-6 < best_loss:
            best_loss = cur
            stall = 0
            best_tokens = population[0].detach().clone()
        else:
            stall += 1
        if step != args.generations - 1:
            population = mutation(
                crossover(population, args.crossover_rate, args.length),
                args.mutate_rate,
                args.length,
            )
        if step % 50 == 0 or step == args.generations - 1:
            print(
                f"[Info] {tag}  gen {step + 1}/{args.generations}  "
                f"min_loss={cur:.4f}  best={best_loss:.4f}",
                flush=True,
            )
        if args.patience > 0 and stall >= args.patience:
            print(f"[Info] {tag}  early stop at gen {step + 1} (patience={args.patience})", flush=True)
            break

    tokens = best_tokens if best_tokens is not None else population[0]
    inv_prompt = tokenizer.decode(tokens[0][1 : args.length + 1])
    return inv_prompt, best_loss


def write_header(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["prompt", "case_number", "evaluation_seed", "loss", "source_prompt"])


def append_row(path: Path, row):
    with path.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def already_done(path: Path) -> set[int]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    try:
        df = pd.read_csv(path)
        if "case_number" in df.columns:
            return set(int(x) for x in df.case_number.tolist())
    except Exception:
        return set()
    return set()


def select_shard(seeds: pd.DataFrame, shard_index: int, shard_count: int) -> pd.DataFrame:
    if shard_count <= 1:
        return seeds
    cases = []
    for i, row in seeds.iterrows():
        case = int(row.case_number) if "case_number" in row and pd.notna(row.case_number) else int(i)
        cases.append(case)
    mask = [c % shard_count == shard_index for c in cases]
    return seeds.loc[mask].reset_index(drop=True)


def run_search(args, tokenizer, text_encoder, device, concept_vec):
    seeds = pd.read_csv(args.seed_csv)
    if "prompt" not in seeds.columns:
        raise ValueError(f"{args.seed_csv} must have a prompt column")
    seeds = seeds.head(args.n_prompts).reset_index(drop=True)
    seeds = select_shard(seeds, args.shard_index, args.shard_count)
    out = Path(args.output)
    write_header(out)
    done = already_done(out)
    print(
        f"[Info] Inverse search: {len(seeds)} seeds on shard "
        f"{args.shard_index}/{args.shard_count}, {len(done)} already written -> {out}"
    )

    for i, row in seeds.iterrows():
        case = int(row.case_number) if "case_number" in row and pd.notna(row.case_number) else int(i)
        if case in done:
            print(f"[Info] skip case {case} (already in CSV)")
            continue
        seed = int(row.evaluation_seed) if "evaluation_seed" in row and pd.notna(row.evaluation_seed) else case
        random.seed(seed + case)
        np.random.seed(seed + case)
        torch.manual_seed(seed + case)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + case)

        prompt = str(row.prompt)
        tag = f"Obama_eta_{args.eta}_K_{args.length} case={case} gpu={device}"
        inv_prompt, loss = search_one(prompt, concept_vec, tokenizer, text_encoder, device, args, tag)
        print(f"[Prompt {case}] loss={loss:.4f}  {inv_prompt}", flush=True)
        append_row(out, [inv_prompt, case, seed, f"{loss:.6f}", prompt])
        done.add(case)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Ring-A-Bell Obama / celebrity inverse prompts")
    p.add_argument("--root", default=None, help="Repo root (auto-detected)")
    p.add_argument("--model", default="CompVis/stable-diffusion-v1-4")
    p.add_argument("--device", default="auto")
    p.add_argument("--concept-csv", default=None)
    p.add_argument("--seed-csv", default=None)
    p.add_argument("--vector-path", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--num-samples", type=int, default=5, help="Repeats per concept pair (paper: 5)")
    p.add_argument("--skip-concept", action="store_true", help="Reuse an existing concept vector")
    p.add_argument("--population", type=int, default=200)
    p.add_argument("--generations", type=int, default=3000)
    p.add_argument("--length", type=int, default=16, help="Free tokens. Use 75 for K=77")
    p.add_argument("--eta", type=float, default=3.0, dest="eta", help="Concept coefficient (paper η / notebook cof)")
    p.add_argument("--mutate-rate", type=float, default=0.25)
    p.add_argument("--crossover-rate", type=float, default=0.5)
    p.add_argument(
        "--patience",
        type=int,
        default=250,
        help="Early-stop if best loss does not improve for this many gens (0 disables)",
    )
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="Do not auto-split across multiple GPUs (T4 x2 uses both by default)",
    )
    args = p.parse_args(argv)
    root = resolve_root(args.root)
    args.root = root
    if args.concept_csv is None:
        args.concept_csv = root / "data" / "Prompts_For_ConceptVector" / "Obama_30.csv"
    if args.seed_csv is None:
        args.seed_csv = root / "data" / "Obama_seeds.csv"
    if args.vector_path is None:
        args.vector_path = root / "Concept Vectors" / "Obama_vector.npy"
    if args.output is None:
        kaggle_out = Path("/kaggle/working")
        if kaggle_out.exists():
            args.output = kaggle_out / "Obama_invprompts.csv"
        else:
            args.output = root / "data" / "InvPrompt" / "Obama" / f"Obama_eta_{args.eta}_K_{args.length}.csv"
    return args


def merge_shard_csvs(final_path: Path, shard_paths: list[Path]) -> None:
    frames = []
    for path in shard_paths:
        if path.exists() and path.stat().st_size > 0:
            frames.append(pd.read_csv(path))
    if not frames:
        return
    merged = pd.concat(frames, ignore_index=True)
    if "case_number" in merged.columns:
        merged = merged.drop_duplicates(subset=["case_number"]).sort_values("case_number")
    merged.to_csv(final_path, index=False)
    print(f"[Done] Merged {len(merged)} rows -> {final_path}")


def launch_multi_gpu(args, argv: list[str] | None) -> int:
    import subprocess

    n_gpu = torch.cuda.device_count()
    print(f"[Info] Launching {n_gpu} GPU shards")
    raw = list(argv) if argv is not None else sys.argv[1:]
    # Strip flags this launcher will rewrite so workers do not recurse.
    cleaned = []
    skip_next = False
    drop_value = {"--device", "--shard-index", "--shard-count", "--output"}
    drop_flag = {"--single-gpu"}
    for token in raw:
        if skip_next:
            skip_next = False
            continue
        if token in drop_flag:
            continue
        if token in drop_value:
            skip_next = True
            continue
        if token.startswith("--device=") or token.startswith("--shard-index=") or token.startswith("--shard-count=") or token.startswith("--output="):
            continue
        cleaned.append(token)

    final_out = Path(args.output)
    procs = []
    shard_paths = []
    for rank in range(n_gpu):
        shard_out = final_out.with_name(f"{final_out.stem}.gpu{rank}{final_out.suffix}")
        shard_paths.append(shard_out)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *cleaned,
            "--single-gpu",
            "--device",
            "cuda",
            "--shard-index",
            str(rank),
            "--shard-count",
            str(n_gpu),
            "--output",
            str(shard_out),
            "--skip-concept",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(rank)
        print("[Info] spawn", " ".join(cmd))
        procs.append(subprocess.Popen(cmd, env=env))
    codes = [p.wait() for p in procs]
    merge_shard_csvs(final_out, shard_paths)
    return 0 if all(c == 0 for c in codes) else (codes[0] or 1)


def main(argv=None):
    args = parse_args(argv)
    n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if (
        argv is None
        and n_gpu >= 2
        and not args.single_gpu
        and args.device == "auto"
        and args.shard_count == 1
    ):
        device_probe = pick_device("auto")
        tokenizer, text_encoder = load_clip(args.model, device_probe)
        if not (args.skip_concept and Path(args.vector_path).exists()):
            extract_concept_vector(args, tokenizer, text_encoder, device_probe)
        del tokenizer, text_encoder
        if device_probe.startswith("cuda"):
            torch.cuda.empty_cache()
        return launch_multi_gpu(args, argv)

    device = pick_device(args.device)
    print(f"[Info] device={device}  n_gpu={n_gpu}  root={args.root}")
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[Error] CUDA requested but not available", file=sys.stderr)
        return 1

    tokenizer, text_encoder = load_clip(args.model, device)

    if args.skip_concept and Path(args.vector_path).exists():
        concept_vec = np.load(args.vector_path)
        print(f"[Info] Loaded concept vector {tuple(concept_vec.shape)} from {args.vector_path}")
    else:
        concept_vec = extract_concept_vector(args, tokenizer, text_encoder, device)

    run_search(args, tokenizer, text_encoder, device, concept_vec)
    print(f"[Done] Wrote inverse prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
