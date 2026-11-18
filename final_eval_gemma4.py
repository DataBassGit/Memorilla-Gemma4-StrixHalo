# Save as final_eval_gemma4.py in C:\AI\MemoryLLM (next to apply_gemma4_patch.py)
r"""
Final evaluation for Memorilla + Gemma 4 26B-A4B on the Strix Halo, through the INT4 decoder.

No vLLM and no second encoder model: everything runs through load_gemma4_text_decoder
(the same frozen INT4 weights the module was trained against) with KV-cached greedy decoding.

Three conditions per row, all with the SAME question text and question embedding:
  true      the row's own documents
  shuffled  another collection's documents (documents AND padding mask move together)
  none      a prompt with no memory placeholders at all (true closed-book)

Scoring
  rows with `choices`   -> likelihood scoring: each choice is scored as the answer, lowest
                           mean token loss wins. Gives accuracy without an encoder.
  rows without choices  -> greedy generation, scored with exact match, token F1 and ROUGE-L.

Every difference is reported with a paired bootstrap 95% interval.

Usage (PowerShell):
  python final_eval_gemma4.py --checkpoint .\checkpoints\stage3_gemma4\epoch-02 `
      --data_repo .\data_10pct --benchmark all --split test --n 200

NOT YET RUN AGAINST REAL WEIGHTS. Read the [SelfCheck] block it prints first; it shows the
exact prompt tail, the first token ids and where the memory tokens sit.
"""

import argparse
import json
import math
import random
import re
import string
import time
from collections import Counter
from pathlib import Path

import apply_gemma4_patch

apply_gemma4_patch.install_runtime_monkeypatch()


def _disable_resume_manager() -> None:
    rm = apply_gemma4_patch.RESUME_MANAGER
    rm.output_dir = None
    rm.save_every_steps = 0
    rm.save_every_seconds = 0
    rm.pending_state = None
    rm.reset = lambda *a, **kw: None
    rm.save_checkpoint = lambda *a, **kw: None


_disable_resume_manager()

import torch
import torch.nn.functional as F

from memorilla.benchmarks import get_benchmarks
from memorilla.data import (
    ANSWER_COLUMN,
    CHOICES_COLUMN,
    COLLECTION_COLUMN,
    DEFAULT_SYSTEM_PROMPT,
    EVAL_COLUMNS,
    QUESTION_COLUMN,
    apply_chat_template,
    build_messages,
    ensure_data,
)
from memorilla.embeddings import ensure_embeddings, load_split_with_embeddings
from memorilla.llm import MemoryLLM, inject_memory
from memorilla.memory import MemoryModule
from memorilla.paths import DATA_DIR, DATA_REPO, DEFAULT_ENCODER, resolve

THOUGHT_STUB = "<|channel>thought\n<channel|>"
END_OF_TURN = "<turn|>"
CONDITIONS = ("true", "shuffled", "none")


# ----------------------------------------------------------------------------- text metrics
def _normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(pred: str, gold: str) -> float:
    return float(_normalize(pred) == _normalize(gold))


def token_f1(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p), common / len(g)
    return 2 * precision * recall / (precision + recall)


def rouge_l(pred: str, gold: str) -> float:
    p, g = _normalize(pred).split(), _normalize(gold).split()
    if not p or not g:
        return 0.0
    prev = [0] * (len(g) + 1)
    for tok in p:
        cur = [0]
        for j, gt in enumerate(g, start=1):
            cur.append(prev[j - 1] + 1 if tok == gt else max(prev[j], cur[j - 1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    precision, recall = lcs / len(p), lcs / len(g)
    return 2 * precision * recall / (precision + recall)


def clean_generation(text: str) -> str:
    """Strip thought-channel residue: leading 'thought' lines (the model's own empty thought
    block, whose special tokens are removed by decoding) and anything after a later marker."""
    text = text.strip()
    while True:
        stripped = re.sub(r"^thought[ \t]*(\n|$)", "", text).lstrip()
        if stripped == text:
            break
        text = stripped
    text = re.split(r"\n\s*thought\s*(\n|$)", text)[0]
    return text.strip()


def paired_bootstrap(a: list[float], b: list[float], rounds: int = 2000, seed: int = 23) -> tuple[float, float, float]:
    """Mean of (a - b) with a 95% paired bootstrap interval."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    n = len(diffs)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return sum(diffs) / n, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


# ----------------------------------------------------------------------------- model wrapper
class Gemma4MemoryEvaluator:
    def __init__(self, decoder: str, checkpoint: Path, prompt_style: str, cache_dir: str | None):
        self.memory = MemoryModule.from_pretrained(checkpoint)
        self.llm = MemoryLLM(decoder, self.memory, attn_implementation="sdpa", cache_dir=cache_dir)
        self.dec = self.llm.text_model.eval()
        self.tok = self.llm.tokenizer
        self.memory_token_id = self.llm.memory_token_id
        self.device = next(self.dec.parameters()).device
        self.memory.to(device=self.device, dtype=torch.bfloat16).eval()
        self.backbone = self.dec.model
        self.lm_head = self.dec.lm_head
        self.embed = self.dec.get_input_embeddings()
        text_cfg = getattr(self.dec.config, "text_config", self.dec.config)
        self.softcap = getattr(text_cfg, "final_logit_softcapping", None)
        self.prompt_style = prompt_style
        self.num_memories = self.memory.num_memories
        eot = self.tok.convert_tokens_to_ids(END_OF_TURN)
        self.eot_id = eot if isinstance(eot, int) and eot >= 0 else None
        self.stop_ids = {i for i in (self.tok.eos_token_id, self.eot_id) if i is not None}

    # -- prompts -----------------------------------------------------------------------
    def render(self, question: str, system_prompt: str, suffix: str, with_memory: bool) -> str:
        if with_memory:
            messages = build_messages(question, system_prompt, self.num_memories, suffix)
        else:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(question).strip() + suffix},
            ]
        prompt = apply_chat_template(self.tok, messages, add_generation_prompt=True)
        # "train": answer follows <|turn>model\n directly, which is how the collator rendered
        # training text for Gemma 4. "inference": keep the template's empty thought channel.
        if self.prompt_style == "train" and prompt.endswith(THOUGHT_STUB):
            prompt = prompt[: -len(THOUGHT_STUB)]
        return prompt

    def encode(self, prompts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        # Same tokenizer call as MemoryCollator (default special tokens, left padding)
        enc = self.tok(prompts, padding=True, padding_side="left", return_tensors="pt")
        return enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)

    @torch.inference_mode()
    def prompt_embeds(
        self,
        prompts: list[str],
        doc_embeds: torch.Tensor | None,
        doc_mask: torch.Tensor | None,
        question_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ids, mask = self.encode(prompts)
        embeds = self.embed(ids)
        if doc_embeds is not None:
            counts = (ids == self.memory_token_id).sum(dim=1)
            if not bool((counts == self.num_memories).all()):
                raise RuntimeError(f"Expected {self.num_memories} memory placeholders per row, found {counts.tolist()}")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                mem = self.memory(doc_embeds, doc_mask, question_embeds)
            embeds = inject_memory(embeds, mem, ids, self.memory_token_id)
        return embeds, mask, ids

    # -- decoder passes ------------------------------------------------------------------
    def _backbone(self, embeds, mask, past=None, use_cache=False):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
            out = self.backbone(
                inputs_embeds=embeds, attention_mask=mask, past_key_values=past, use_cache=use_cache, return_dict=True
            )
        return out[0], getattr(out, "past_key_values", None)

    def _logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden).float()
        if self.softcap is not None:
            logits = torch.tanh(logits / self.softcap) * self.softcap
        return logits

    @torch.inference_mode()
    def generate(self, embeds: torch.Tensor, mask: torch.Tensor, max_new_tokens: int) -> list[str]:
        batch = embeds.size(0)
        done = [False] * batch
        out_ids: list[list[int]] = [[] for _ in range(batch)]
        past, cur = None, embeds
        for _ in range(max_new_tokens):
            hidden, past = self._backbone(cur, mask, past=past, use_cache=True)
            nxt = self._logits(hidden[:, -1, :]).argmax(dim=-1)
            for b, tok_id in enumerate(nxt.tolist()):
                if done[b]:
                    continue
                if tok_id in self.stop_ids:
                    done[b] = True
                else:
                    out_ids[b].append(tok_id)
            if all(done):
                break
            cur = self.embed(nxt.unsqueeze(1))
            mask = torch.cat([mask, torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)], dim=1)
        return [clean_generation(self.tok.decode(ids, skip_special_tokens=True)) for ids in out_ids]

    @torch.inference_mode()
    def choice_losses(self, prompt_embeds: torch.Tensor, choices: list[str]) -> list[float]:
        """Mean token loss of each choice (plus end-of-turn) as the answer to ONE unpadded prompt."""
        prompt_len = prompt_embeds.size(1)
        tails = []
        for choice in choices:
            ids = self.tok.encode(str(choice).strip(), add_special_tokens=False)
            if self.eot_id is not None:
                ids = ids + [self.eot_id]
            tails.append(ids)
        max_tail = max(len(t) for t in tails)
        pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        tail_ids = torch.full((len(tails), max_tail), pad_id, dtype=torch.long, device=self.device)
        tail_mask = torch.zeros((len(tails), max_tail), dtype=torch.long, device=self.device)
        for i, t in enumerate(tails):
            tail_ids[i, : len(t)] = torch.tensor(t, device=self.device)
            tail_mask[i, : len(t)] = 1
        embeds = torch.cat([prompt_embeds.expand(len(tails), -1, -1), self.embed(tail_ids)], dim=1)
        mask = torch.cat(
            [torch.ones((len(tails), prompt_len), dtype=torch.long, device=self.device), tail_mask], dim=1
        )
        hidden, _ = self._backbone(embeds, mask, use_cache=False)
        # position p predicts token p+1; choice tokens start at prompt_len
        hidden = hidden[:, prompt_len - 1 : prompt_len - 1 + max_tail, :]
        losses = []
        for i, t in enumerate(tails):
            logits = self._logits(hidden[i, : len(t), :])
            losses.append(float(F.cross_entropy(logits, tail_ids[i, : len(t)], reduction="mean").item()))
        return losses


# ----------------------------------------------------------------------------- data helpers
def gather_documents(store, collection_ids: list, device) -> tuple[torch.Tensor, torch.Tensor]:
    rows = [store.documents(cid) for cid in collection_ids]
    max_docs = max(r.shape[0] for r in rows)
    embeds = torch.zeros(len(rows), max_docs, rows[0].shape[-1], dtype=torch.bfloat16)
    mask = torch.ones(len(rows), max_docs, dtype=torch.bool)
    for i, r in enumerate(rows):
        embeds[i, : r.shape[0]] = r.to(torch.bfloat16)
        mask[i, : r.shape[0]] = False
    return embeds.to(device), mask.to(device)


def shuffled_partners(collection_ids: list) -> list[int]:
    """For each row, the index of a row from a DIFFERENT collection (next one along, wrapping)."""
    n = len(collection_ids)
    partners = []
    for i in range(n):
        j = (i + 1) % n
        while j != i and collection_ids[j] == collection_ids[i]:
            j = (j + 1) % n
        partners.append(j)
    return partners


def row_choices(row: dict) -> list[str]:
    choices = row.get(CHOICES_COLUMN)
    if choices is None:
        return []
    choices = [str(c).strip() for c in list(choices) if c is not None and str(c).strip()]
    return choices if len(choices) >= 2 else []


# ----------------------------------------------------------------------------- evaluation
def self_check(ev: Gemma4MemoryEvaluator, prompt: str, embeds: torch.Tensor, ids: torch.Tensor) -> dict:
    row_ids = ids[0]
    mem_pos = (row_ids == ev.memory_token_id).nonzero().flatten()
    real = row_ids != (ev.tok.pad_token_id if ev.tok.pad_token_id is not None else -1)
    text_pos = (real & (row_ids != ev.memory_token_id)).nonzero().flatten()
    norms = embeds[0].float().norm(dim=-1)
    info = {
        "prompt_style": ev.prompt_style,
        "prompt_tail": prompt[-90:],
        "first_token_ids": row_ids[real][:4].tolist(),
        "bos_id": ev.tok.bos_token_id,
        "memory_positions": [int(mem_pos[0]), int(mem_pos[-1])] if mem_pos.numel() else [],
        "memory_token_norm_mean": round(float(norms[mem_pos].mean()), 2) if mem_pos.numel() else None,
        "text_token_norm_median": round(float(norms[text_pos].median()), 2),
        "text_token_norm_mean": round(float(norms[text_pos].mean()), 2),
    }
    print("\n[SelfCheck]")
    for key, value in info.items():
        print(f"  {key:24s}: {value!r}")
    return info


def evaluate_benchmark(ev: Gemma4MemoryEvaluator, bench, args) -> dict:
    data_repo, data_dir = args.data_repo, args.data_dir
    root = ensure_data(bench.config, [bench.split], repo=data_repo, data_dir=data_dir)
    ensure_embeddings(bench.config, [bench.split], repo=data_repo, data_dir=data_dir, encoder=args.encoder)
    dataset, store = load_split_with_embeddings(bench.config, bench.split, root, EVAL_COLUMNS, args.encoder)

    rng = random.Random(args.seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    indices = sorted(indices[: args.n])
    rows = [dataset[i] for i in indices]
    collection_ids = [r[COLLECTION_COLUMN] for r in rows]
    partners = shuffled_partners(collection_ids)

    system_prompt = bench.system_prompt if args.protocol == "benchmark" else DEFAULT_SYSTEM_PROMPT
    suffix = (bench.suffix or "") if args.protocol == "benchmark" else ""
    use_choices = bool(row_choices(rows[0]))
    max_new = args.max_new_tokens or bench.max_new_tokens

    records: list[dict] = []
    check_info = None
    t0 = time.monotonic()
    step = 1 if use_choices else args.batch_size
    for start in range(0, len(rows), step):
        chunk = list(range(start, min(start + step, len(rows))))
        questions = [rows[i][QUESTION_COLUMN] for i in chunk]
        q_embeds = torch.stack([store.question(indices[i]) for i in chunk]).to(ev.device, torch.bfloat16)
        mem_prompts = [ev.render(q, system_prompt, suffix, with_memory=True) for q in questions]
        none_prompts = [ev.render(q, system_prompt, suffix, with_memory=False) for q in questions]

        inputs = {}
        docs, mask = gather_documents(store, [collection_ids[i] for i in chunk], ev.device)
        inputs["true"] = ev.prompt_embeds(mem_prompts, docs, mask, q_embeds)
        docs_s, mask_s = gather_documents(store, [collection_ids[partners[i]] for i in chunk], ev.device)
        inputs["shuffled"] = ev.prompt_embeds(mem_prompts, docs_s, mask_s, q_embeds)
        inputs["none"] = ev.prompt_embeds(none_prompts, None, None, None)

        if check_info is None:
            check_info = self_check(ev, mem_prompts[0], inputs["true"][0], inputs["true"][2])

        if use_choices:
            i = chunk[0]
            choices = row_choices(rows[i])
            gold = str(rows[i][ANSWER_COLUMN]).strip()
            if gold not in choices:
                continue
            rec = {"row": indices[i], "gold": gold}
            for cond in CONDITIONS:
                losses = ev.choice_losses(inputs[cond][0], choices)
                pick = choices[min(range(len(choices)), key=losses.__getitem__)]
                rec[cond] = {"pred": pick, "accuracy": float(pick == gold), "gold_loss": losses[choices.index(gold)]}
            records.append(rec)
        else:
            generations = {c: ev.generate(inputs[c][0], inputs[c][1], max_new) for c in CONDITIONS}
            for k, i in enumerate(chunk):
                gold = str(rows[i][ANSWER_COLUMN]).strip()
                rec = {"row": indices[i], "gold": gold}
                for cond in CONDITIONS:
                    pred = generations[cond][k]
                    rec[cond] = {
                        "pred": pred,
                        "exact_match": exact_match(pred, gold),
                        "f1": token_f1(pred, gold),
                        "rougeL": rouge_l(pred, gold),
                    }
                records.append(rec)
        if (start // step) % 10 == 0:
            print(f"  [{bench.name}] {min(start + step, len(rows))}/{len(rows)} rows, {time.monotonic() - t0:.0f}s")

    metrics = ["accuracy"] if use_choices else ["exact_match", "f1", "rougeL"]
    summary: dict = {"n": len(records), "scoring": "choice_likelihood" if use_choices else "generation"}
    for metric in metrics:
        per_cond = {c: [r[c][metric] for r in records] for c in CONDITIONS}
        entry = {c: round(100 * sum(v) / max(1, len(v)), 2) for c, v in per_cond.items()}
        for name, other in (("true_minus_shuffled", "shuffled"), ("true_minus_none", "none")):
            mean, lo, hi = paired_bootstrap(per_cond["true"], per_cond[other])
            entry[name] = {"mean": round(100 * mean, 2), "ci95": [round(100 * lo, 2), round(100 * hi, 2)]}
        summary[metric] = entry
        d1, d2 = entry["true_minus_shuffled"], entry["true_minus_none"]
        print(
            f"  [{bench.name:12s}] {metric:11s} true {entry['true']:6.2f} | shuffled {entry['shuffled']:6.2f} | "
            f"none {entry['none']:6.2f} | true-shuf {d1['mean']:+.2f} {d1['ci95']} | true-none {d2['mean']:+.2f} {d2['ci95']}"
        )
    return {"benchmark": bench.name, "split": bench.split, "self_check": check_info, "summary": summary, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory or memory.pt")
    parser.add_argument("--decoder", default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="Only names the embedding folder; not loaded")
    parser.add_argument("--benchmark", default="all")
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=200, help="Rows per benchmark")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=None)
    parser.add_argument("--prompt_style", choices=("train", "inference"), default="train")
    parser.add_argument("--protocol", choices=("train", "benchmark"), default="train",
                        help="train: the system prompt training used; benchmark: upstream's per-benchmark prompt and suffix")
    parser.add_argument("--data_repo", default=DATA_REPO)
    parser.add_argument("--data_dir", default=str(DATA_DIR))
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output_dir", default="results/final_eval_gemma4")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    benches = get_benchmarks(args.benchmark)
    if args.split != "test":
        from dataclasses import replace
        benches = [replace(b, split=args.split) for b in benches]

    ev = Gemma4MemoryEvaluator(args.decoder, resolve(args.checkpoint), args.prompt_style, args.cache_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report = {"checkpoint": str(args.checkpoint), "args": vars(args), "benchmarks": {}}
    for bench in benches:
        print(f"\n=== {bench.name} ({bench.split}) ===")
        try:
            result = evaluate_benchmark(ev, bench, args)
        except Exception as exc:  # keep going so one bad benchmark does not lose the rest
            print(f"  [{bench.name}] FAILED: {type(exc).__name__}: {exc}")
            report["benchmarks"][bench.name] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        with (out_dir / f"{bench.name}.jsonl").open("w", encoding="utf-8") as handle:
            for rec in result.pop("records"):
                handle.write(json.dumps(rec, ensure_ascii=False) + "\n")
        report["benchmarks"][bench.name] = result
        with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved {out_dir / 'report.json'}")


if __name__ == "__main__":
    main()
