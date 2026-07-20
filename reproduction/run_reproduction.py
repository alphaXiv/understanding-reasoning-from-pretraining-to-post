#!/usr/bin/env python3
"""Bounded reproduction of the chess pretraining-to-GRPO scaling claim.

The implementation intentionally keeps the scientific variable narrow: child
branches change only ``pretrain_shards`` (a nested prefix of the authors'
public 2022 Lichess token shards). Architecture, trace-SFT, puzzle splits,
GRPO updates, and evaluation are identical.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import io
import json
import math
import os
import random
import time
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


FILES, RANKS = "abcdefgh", "12345678"
SQUARES = [f + r for f in FILES for r in RANKS]
BASE_VOCAB = ["<bos>", "<eos>", "<unk>"] + list("KQRBNP") + SQUARES
BASE_VOCAB += ["x", "=", "+", "#", "O-O", "O-O-O", ".", "..."]
VOCAB = list(dict.fromkeys(BASE_VOCAB + ["<T>", "</T>", "<sep>"]))
TOK = {token: index for index, token in enumerate(VOCAB)}
BOS, EOS, UNK = TOK["<bos>"], TOK["<eos>"], TOK["<unk>"]

PRETRAIN_REPO = "https://huggingface.co/datasets/chess-pre-to-post/pretrain_v1_20b/resolve/main"
SFT_URL = "https://huggingface.co/datasets/chess-pre-to-post/sft_v1_200m_90k/resolve/main/datapuzzle_v4_shard0-500.json?download=true"
RL_URL = "https://huggingface.co/datasets/chess-pre-to-post/chess-rl-data-balanced/resolve/main/train_v4_dataset_balanced_multi_turn.parquet?download=true"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def download(url: str, path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)


def stable_bucket(value: str, modulus: int = 10_000) -> int:
    return int(hashlib.sha256(value.encode()).hexdigest()[:12], 16) % modulus


def lan_tokens(board: chess.Board, move: chess.Move) -> list[str]:
    if board.is_castling(move):
        out = ["O-O" if chess.square_file(move.to_square) == 6 else "O-O-O"]
    else:
        piece = board.piece_at(move.from_square)
        if piece is None:
            raise ValueError(f"no piece for {move.uci()} in {board.fen()}")
        out = [piece.symbol().upper(), chess.square_name(move.from_square)]
        if board.is_capture(move):
            out.append("x")
        out.append(chess.square_name(move.to_square))
        if move.promotion:
            out += ["=", chess.piece_symbol(move.promotion).upper()]
    probe = board.copy()
    probe.push(move)
    if probe.is_checkmate():
        out.append("#")
    elif probe.is_check():
        out.append("+")
    return out


def pgn_tokens(text: str) -> tuple[list[str], chess.Board]:
    with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
        game = chess.pgn.read_game(io.StringIO(text))
    if game is None:
        return [], chess.Board()
    board, out = game.board(), []
    for move in game.mainline_moves():
        out.extend(lan_tokens(board, move))
        board.push(move)
    return out, board


def parse_lan_word(word: str) -> list[str]:
    if word in {"<T>", "</T>", "<sep>"}:
        return [word]
    if word in {"<call_env>", "<verify>"} or (word.startswith("<") and word.endswith(">")):
        return []
    if word.rstrip("+#") in {"O-O", "O-O-O"}:
        base = word.rstrip("+#")
        return [base] + ([word[-1]] if word[-1:] in "+#" else [])
    if not word or word[0] not in "KQRBNP":
        return []
    out, i = [word[0]], 1
    if i + 1 < len(word) and word[i] in FILES and word[i + 1] in RANKS:
        out.append(word[i : i + 2]); i += 2
    if i < len(word) and word[i] == "x":
        out.append("x"); i += 1
    if i + 1 < len(word) and word[i] in FILES and word[i + 1] in RANKS:
        out.append(word[i : i + 2]); i += 2
    if i < len(word) and word[i] == "=":
        out.append("="); i += 1
        if i < len(word) and word[i] in "QRBN":
            out.append(word[i]); i += 1
    if i < len(word) and word[i] in "+#":
        out.append(word[i])
    return out


def ids(tokens: list[str]) -> list[int]:
    return [TOK.get(token, UNK) for token in tokens]


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor) -> torch.Tensor:
    # x: batch, heads, time, head_dim
    t, d = x.shape[-2], x.shape[-1]
    positions = torch.arange(t, device=x.device, dtype=torch.float32)
    inv = 1.0 / (10_000 ** (torch.arange(0, d, 2, device=x.device, dtype=torch.float32) / d))
    angles = torch.outer(positions, inv)
    cos = angles.cos()[None, None].to(dtype=x.dtype)
    sin = angles.sin()[None, None].to(dtype=x.dtype)
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, width: int = 512, heads: int = 4, intermediate: int = 1536):
        super().__init__()
        self.heads, self.head_dim = heads, width // heads
        self.n1, self.n2 = RMSNorm(width), RMSNorm(width)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.gate = nn.Linear(width, intermediate, bias=False)
        self.up = nn.Linear(width, intermediate, bias=False)
        self.down = nn.Linear(intermediate, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        q, k, v = self.qkv(self.n1(x)).chunk(3, dim=-1)
        q = apply_rope(q.view(b, t, self.heads, self.head_dim).transpose(1, 2))
        k = apply_rope(k.view(b, t, self.heads, self.head_dim).transpose(1, 2))
        v = v.view(b, t, self.heads, self.head_dim).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).contiguous().view(b, t, c))
        y = self.n2(x)
        return x + self.down(F.silu(self.gate(y)) * self.up(y))


class ChessLM(nn.Module):
    def __init__(self, vocab: int = len(VOCAB), width: int = 512, layers: int = 6):
        super().__init__()
        self.embed = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList([Block(width) for _ in range(layers)])
        self.norm = RMSNorm(width)
        self.head = nn.Linear(width, vocab, bias=False)
        nn.init.normal_(self.embed.weight, std=0.02)
        nn.init.normal_(self.head.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


@dataclass
class Puzzle:
    puzzle_id: str
    prompt: list[int]
    board: chess.Board
    target: str
    rating: int


def normalized_fen(board: chess.Board) -> str:
    return " ".join(board.fen().split()[:2])


def load_puzzles(path: Path, sft_positions: set[str], train_n: int, eval_n: int) -> tuple[list[Puzzle], list[Puzzle]]:
    frame = pd.read_parquet(path)
    train_set: list[Puzzle] = []
    eval_set: list[Puzzle] = []
    for row in frame.itertuples(index=False):
        info, reward = row.extra_info, row.reward_model
        puzzle_id = str(info.get("PuzzleId", ""))
        if not puzzle_id:
            continue
        try:
            truth = ast.literal_eval(str(reward.get("ground_truth")))
            board = chess.Board(str(info["FEN"]))
            environment_move = chess.Move.from_uci(str(info["first_move_uci"]))
            if environment_move not in board.legal_moves:
                continue
            board.push(environment_move)
            target = str(truth[0])
            move = chess.Move.from_uci(target)
        except Exception:
            continue
        if len(truth) != 1 or move not in board.legal_moves or normalized_fen(board) in sft_positions:
            continue
        rating = int(info.get("Rating", row.difficulty))
        if not 800 <= rating <= 1800:
            continue
        prompt_text = str(row.prompt).split("<T>")[0].replace("*", "").strip()
        prompt_toks, _ = pgn_tokens(prompt_text)
        if len(prompt_toks) < 6:
            continue
        puzzle = Puzzle(puzzle_id, [BOS] + ids(prompt_toks), board, target, rating)
        if stable_bucket(puzzle_id) < 2000 and len(eval_set) < eval_n:
            eval_set.append(puzzle)
        elif stable_bucket(puzzle_id) >= 2000 and len(train_set) < train_n:
            train_set.append(puzzle)
        if len(train_set) == train_n and len(eval_set) == eval_n:
            break
    if len(train_set) < train_n or len(eval_set) < eval_n:
        raise RuntimeError(f"insufficient puzzles: train={len(train_set)} eval={len(eval_set)}")
    return train_set, eval_set


def load_sft(path: Path, trace_limit: int, max_len: int) -> tuple[list[tuple[list[int], list[int]]], set[str]]:
    payload = json.loads(path.read_text())
    examples, positions = [], set()
    for item in payload["results"]:
        prompt_toks, board = pgn_tokens(str(item["pgn"]))
        positions.add(normalized_fen(board))
        trace_text = item["cot_by_method"]["dfs_verifier"]["cot_format_no_labels"]
        trace_words = trace_text.split("</T>", 1)[0].strip().split()
        trace_toks: list[str] = []
        for word in trace_words:
            trace_toks.extend(parse_lan_word(word))
            if len(trace_toks) >= trace_limit - 1:
                break
        if not trace_toks or trace_toks[0] != "<T>":
            trace_toks.insert(0, "<T>")
        trace_toks = trace_toks[: trace_limit - 1] + ["</T>"]
        answer = parse_lan_word(item["cot_by_method"]["dfs_verifier"]["first_move_lan"])
        prompt_ids = [BOS] + ids(prompt_toks)
        response_ids = ids(trace_toks + answer) + [EOS]
        room = max_len - len(response_ids)
        prompt_ids = prompt_ids[-max(room, 1) :]
        examples.append((prompt_ids, response_ids))
    return examples, positions


def pad_supervised(batch: list[tuple[list[int], list[int]]], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    sequences, labels = [], []
    for prompt, response in batch:
        sequence = prompt + response
        label = [-100] * (len(prompt) - 1) + response
        sequences.append(sequence[:-1]); labels.append(label)
    width = max(map(len, sequences))
    x = torch.full((len(batch), width), BOS, dtype=torch.long, device=device)
    y = torch.full((len(batch), width), -100, dtype=torch.long, device=device)
    for i, (seq, lab) in enumerate(zip(sequences, labels)):
        x[i, : len(seq)] = torch.tensor(seq, device=device)
        y[i, : len(lab)] = torch.tensor(lab, device=device)
    return x, y


def cosine_lr(step: int, total: int, peak: float, minimum: float, warmup_ratio: float = 0.05) -> float:
    warmup = max(1, int(total * warmup_ratio))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1 + math.cos(math.pi * progress))


def validate_lm(model: nn.Module, data: np.ndarray, device: torch.device, seq: int) -> float:
    module = model
    module.eval()
    losses = torch.zeros(2, device=device)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        starts = list(range(0, min(len(data) - seq - 1, 256 * seq), seq))
        for offset in range(0, len(starts), 16):
            chunk = starts[offset : offset + 16]
            x = torch.tensor(np.stack([data[s : s + seq] for s in chunk]), dtype=torch.long, device=device)
            y = torch.tensor(np.stack([data[s + 1 : s + seq + 1] for s in chunk]), dtype=torch.long, device=device)
            loss = F.cross_entropy(module(x).flatten(0, 1), y.flatten())
            losses += torch.tensor([float(loss) * len(chunk), len(chunk)], device=device)
    module.train()
    return float((losses[0] / losses[1]).item())


def score_action_sequences(module: nn.Module, contexts: list[list[int]], actions: list[list[list[int]]], device: torch.device, max_len: int = 256) -> list[torch.Tensor]:
    flat_x, flat_y, owners = [], [], []
    for owner, (context, choices) in enumerate(zip(contexts, actions)):
        for action in choices:
            ctx = context[-max(1, max_len - len(action) - 1) :]
            seq = ctx + action
            flat_x.append(seq[:-1])
            flat_y.append(([-100] * (len(ctx) - 1)) + action)
            owners.append(owner)
    width = max(map(len, flat_x))
    x = torch.full((len(flat_x), width), BOS, dtype=torch.long, device=device)
    y = torch.full((len(flat_x), width), -100, dtype=torch.long, device=device)
    for i, (sx, sy) in enumerate(zip(flat_x, flat_y)):
        x[i, : len(sx)] = torch.tensor(sx, device=device)
        y[i, : len(sy)] = torch.tensor(sy, device=device)
    logits = module(x)
    token_lp = F.log_softmax(logits.float(), dim=-1)
    gathered = token_lp.gather(-1, y.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    scores = (gathered * (y != -100)).sum(-1)
    grouped = []
    cursor = 0
    for choices in actions:
        grouped.append(scores[cursor : cursor + len(choices)])
        cursor += len(choices)
    return grouped


def puzzle_choices(puzzles: list[Puzzle]) -> tuple[list[list[str]], list[list[list[int]]]]:
    ucis, encoded = [], []
    for puzzle in puzzles:
        moves = list(puzzle.board.legal_moves)
        ucis.append([move.uci() for move in moves])
        encoded.append([ids(lan_tokens(puzzle.board, move)) for move in moves])
    return ucis, encoded


def reasoning_contexts(module: nn.Module, puzzles: list[Puzzle], device: torch.device) -> tuple[list[list[int]], list[list[str]], list[list[list[int]]]]:
    ucis, actions = puzzle_choices(puzzles)
    base = [p.prompt + [TOK["<T>"]] for p in puzzles]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        first_scores = score_action_sequences(module, base, actions, device)
    contexts = []
    for prompt, scores, choices in zip(base, first_scores, actions):
        top = torch.topk(scores, k=min(4, len(choices))).indices.tolist()
        trace = []
        for j, choice_index in enumerate(top):
            if j:
                trace.append(TOK["<sep>"])
            trace.extend(choices[choice_index])
        contexts.append(prompt + trace + [TOK["</T>"]])
    return contexts, ucis, actions


def evaluate_pass1(model: nn.Module, puzzles: list[Puzzle], device: torch.device) -> tuple[float, int]:
    module = model
    local = puzzles
    counts = torch.zeros(2, device=device)
    module.eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for start in range(0, len(local), 16):
            batch = local[start : start + 16]
            contexts, ucis, actions = reasoning_contexts(module, batch, device)
            scores = score_action_sequences(module, contexts, actions, device)
            for puzzle, moves, move_scores in zip(batch, ucis, scores):
                counts[0] += float(moves[int(move_scores.argmax())] == puzzle.target)
                counts[1] += 1
    module.train()
    return float((counts[0] / counts[1]).item()), int(counts[1].item())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    # Gloo is used only as a CPU barrier for shared downloads. Each GPU trains an
    # independent seed; this avoids the cluster's reproducible NCCL collective
    # crash while turning an 8-GPU arm into eight paired scientific replicates.
    dist.init_process_group("gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    seed = int(cfg["seed"]) + rank
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    start_time = time.time()
    start_utc = utc_now()

    cache = Path("/tmp/pre2post_public_data")
    train_paths = []
    for index in range(int(cfg["pretrain_shards"])):
        train_paths.append(cache / f"raw.{index:04d}.npy")
    val_path = cache / "raw.0100.npy"
    sft_path, rl_path = cache / "sft.json", cache / "rl.parquet"
    if rank == 0:
        for index, path in enumerate(train_paths):
            download(f"{PRETRAIN_REPO}/shard_0000/raw.{index:04d}.npy?download=true", path)
        download(f"{PRETRAIN_REPO}/shard_0000/raw.0100.npy?download=true", val_path)
        download(SFT_URL, sft_path); download(RL_URL, rl_path)
    dist.barrier()

    train_data = np.concatenate([np.load(path) for path in train_paths])
    val_data = np.load(val_path)
    sft_examples, sft_positions = load_sft(sft_path, int(cfg["trace_token_limit"]), int(cfg["sequence_length"]))
    rl_train, rl_eval = load_puzzles(rl_path, sft_positions, int(cfg["rl_train_puzzles"]), int(cfg["eval_puzzles"]))

    model = ChessLM().to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    print(f"REPLICATE rank={rank} seed={seed} RUN_START_UTC {start_utc}", flush=True)
    print(f"REPLICATE rank={rank} CONFIG_JSON {json.dumps(cfg, sort_keys=True)}", flush=True)
    if rank == 0:
        print("DATA_SOURCES", json.dumps({"pretrain": "chess-pre-to-post/pretrain_v1_20b (2022 Lichess)", "sft": "chess-pre-to-post/sft_v1_200m_90k", "rl": "chess-pre-to-post/chess-rl-data-balanced"}), flush=True)
        print("MODEL", json.dumps({"parameters": parameter_count, "layers": 6, "width": 512, "intermediate": 1536, "heads": 4, "vocab": len(VOCAB), "context": cfg["sequence_length"]}), flush=True)
        print("COMPUTE", json.dumps({"backend": "kubernetes", "gpu_model": torch.cuda.get_device_name(0), "allocated_gpu_count": world, "independent_replicates": world}), flush=True)
        print("SPLIT", json.dumps({"sft_examples": len(sft_examples), "rl_train": len(rl_train), "eval": len(rl_eval), "eval_rule": "sha256 bucket < 2000; disjoint IDs; one-solver-move puzzles"}), flush=True)

    # One epoch over a nested prefix of official token shards.
    seq, batch_size = int(cfg["sequence_length"]), int(cfg["pretrain_batch_per_gpu"])
    nseq = (len(train_data) - 1) // seq
    local_indices = np.arange(nseq)
    rng = np.random.default_rng(seed)
    rng.shuffle(local_indices)
    pre_steps = len(local_indices) // batch_size
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["pretrain_lr"]), betas=(0.9, 0.95), weight_decay=0.1)
    model.train()
    for step in range(pre_steps):
        lr = cosine_lr(step, pre_steps, float(cfg["pretrain_lr"]), 1e-4)
        for group in optimizer.param_groups: group["lr"] = lr
        chosen = local_indices[step * batch_size : (step + 1) * batch_size]
        x = torch.tensor(np.stack([train_data[i * seq : i * seq + seq] for i in chosen]), dtype=torch.long, device=device)
        y = torch.tensor(np.stack([train_data[i * seq + 1 : i * seq + seq + 1] for i in chosen]), dtype=torch.long, device=device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 50 == 0 or step + 1 == pre_steps:
            print(f"REPLICATE rank={rank} PRETRAIN step={step+1} total={pre_steps} loss={float(loss):.6f} lr={lr:.8g}", flush=True)
    pretrain_val = validate_lm(model, val_data, device, seq)
    print(f"REPLICATE rank={rank} PRETRAIN_FINAL tokens={len(train_data)} steps={pre_steps} val_loss={pretrain_val:.8f}", flush=True)

    # Matched Stockfish-verified trace SFT.
    sft_steps, sft_batch = int(cfg["sft_steps"]), int(cfg["sft_batch_per_gpu"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["sft_lr"]), betas=(0.9, 0.95), weight_decay=0.01)
    for step in range(sft_steps):
        lr = cosine_lr(step, sft_steps, float(cfg["sft_lr"]), 1e-5, 0.01)
        for group in optimizer.param_groups: group["lr"] = lr
        rr = random.Random(seed + rank * 100_000 + step)
        batch = [sft_examples[rr.randrange(len(sft_examples))] for _ in range(sft_batch)]
        x, y = pad_supervised(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).flatten(0, 1), y.flatten(), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        if step % 20 == 0 or step + 1 == sft_steps:
            print(f"REPLICATE rank={rank} SFT step={step+1} total={sft_steps} loss={float(loss):.6f} lr={lr:.8g}", flush=True)

    # Frozen SFT reference and matched GRPO.
    reference = ChessLM().to(device)
    reference.load_state_dict(model.state_dict()); reference.eval()
    for parameter in reference.parameters(): parameter.requires_grad_(False)
    rl_steps = int(cfg["rl_steps"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["rl_lr"]), betas=(0.9, 0.95), weight_decay=0.0)
    eval_updates = set(map(int, cfg["eval_updates"]))
    trajectory = []
    if 0 in eval_updates:
        score, count = evaluate_pass1(model, rl_eval, device)
        trajectory.append({"update": 0, "pass_at_1": score, "n": count})
        print(f"REPLICATE rank={rank} EVAL update=0 pass_at_1={score:.8f} n={count}", flush=True)

    prompts_per_rank = int(cfg["rl_prompts_per_gpu"])
    group_size = int(cfg["rl_group_size"])
    clip_eps, kl_beta = float(cfg["rl_clip_ratio"]), float(cfg["rl_kl_beta"])
    for step in range(1, rl_steps + 1):
        batch = [rl_train[(step * world * prompts_per_rank + rank * prompts_per_rank + j) % len(rl_train)] for j in range(prompts_per_rank)]
        module = model
        module.eval()
        contexts, move_names, actions = reasoning_contexts(module, batch, device)
        module.train()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            old_scores = score_action_sequences(module, contexts, actions, device)
            ref_scores = score_action_sequences(reference, contexts, actions, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            new_scores = score_action_sequences(model, contexts, actions, device)
            losses, rewards_seen = [], []
            for puzzle, names, old, ref, new in zip(batch, move_names, old_scores, ref_scores, new_scores):
                old_lp, ref_lp, new_lp = map(lambda z: F.log_softmax(z.float(), dim=0), (old, ref, new))
                sampled = torch.multinomial(old_lp.exp(), group_size, replacement=True)
                rewards = torch.tensor([float(names[int(i)] == puzzle.target) for i in sampled], device=device)
                advantage = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-6)
                ratio = (new_lp[sampled] - old_lp[sampled]).exp()
                surrogate = torch.minimum(ratio * advantage, ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantage)
                kl = (new_lp.exp() * (new_lp - ref_lp)).sum()
                losses.append(-surrogate.mean() + kl_beta * kl)
                rewards_seen.append(rewards.mean())
            rl_loss = torch.stack(losses).mean()
        rl_loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        reward_tensor = torch.stack(rewards_seen).mean().detach()
        if step % 5 == 0 or step == 1:
            print(f"REPLICATE rank={rank} GRPO update={step} loss={float(rl_loss):.8f} sampled_reward={float(reward_tensor):.8f}", flush=True)
        if step in eval_updates:
            score, count = evaluate_pass1(model, rl_eval, device)
            trajectory.append({"update": step, "pass_at_1": score, "n": count})
            print(f"REPLICATE rank={rank} EVAL update={step} pass_at_1={score:.8f} n={count}", flush=True)

    end_utc, elapsed = utc_now(), time.time() - start_time
    xs = np.log10(np.array([max(point["update"], 1) for point in trajectory], dtype=float))
    ys = np.array([point["pass_at_1"] for point in trajectory], dtype=float)
    fit_mask = np.array([point["update"] > 0 for point in trajectory])
    slope = float(np.polyfit(xs[fit_mask], ys[fit_mask], 1)[0]) if fit_mask.sum() >= 2 else float("nan")
    result = {
        "arm": cfg["arm"], "replicate_rank": rank, "seed": seed,
        "pretraining_tokens": int(len(train_data)), "pretraining_steps": pre_steps,
        "pretraining_validation_loss": pretrain_val, "sft_steps": sft_steps, "rl_steps": rl_steps,
        "trajectory": trajectory, "rl_slope_per_log10_update": slope,
        "final_pass_at_1": trajectory[-1]["pass_at_1"], "eval_puzzles": len(rl_eval),
        "backend": "kubernetes", "gpu_model": torch.cuda.get_device_name(0), "gpu_count": 1,
        "allocated_gpu_count": world, "start_utc": start_utc, "end_utc": end_utc,
        "elapsed_seconds": elapsed,
    }
    print("RESULT_JSON", json.dumps(result, sort_keys=True), flush=True)
    print(f"REPLICATE rank={rank} RUN_END_UTC {end_utc} ELAPSED_SECONDS {elapsed:.3f}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
