# Save as stanford_eval_gemma4.py in C:\AI\MemoryLLM (next to apply_gemma4_patch.py)
r"""
Stanford-protocol evaluation for a Memorilla memory module, without vLLM.

It follows upstream's evaluate.py and evaluate_baselines.py (snap-stanford/memorilla) step by step and
replaces only the two things that need vLLM:

  generation   upstream: vLLM, greedy, prompt embeddings
               here:     the decoder you trained against (INT4 Gemma 4 through apply_gemma4_patch),
                         greedy, KV-cached, same prompts built by upstream's own MemoryCollator
  scoring      upstream: Qwen3-Embedding-4B served by vLLM
               here:     the same Qwen3-Embedding-4B through plain transformers
                         (last-token pooling, L2-normalised, no instruction prefix)

Everything else is upstream's code, imported and called as is: the benchmark registry (system prompt,
answer-format suffix, token limits), the collator, clean_generation and compute_metrics.

Conditions
  memory       Stanford's evaluation. This is the number that compares with their table.
  shuffled     Same prompt and question embedding, another collection's documents. Not in the paper;
               it is the control that says how much of the score comes from the documents.
  closed_book  Stanford's closed-book baseline: no memory tokens, the baseline's own prompt and suffix.

Two steps, because the decoder and the scoring encoder do not need to be loaded together:

  python stanford_eval_gemma4.py generate --checkpoint .\checkpoints\stage3_gemma4\epoch-04 `
      --data_repo .\data_10pct --benchmark all --n 500 --output_dir results\stanford_ep04
  python stanford_eval_gemma4.py score --output_dir results\stanford_ep04

`generate` is resumable: rerun the same command and it skips rows already written. Rows are taken in a
fixed seeded order, so a later run with a larger --n (or --n 0 for the whole split) extends the earlier one.

The same two commands evaluate Stanford's own released checkpoint, which calibrates this script against
their published numbers on your hardware:

  hf download memorilla/Memorilla-Qwen3-8B --include "stage3/*" --local-dir .\checkpoints\memorilla_qwen3
  python stanford_eval_gemma4.py generate --decoder Qwen/Qwen3-8B `
      --checkpoint .\checkpoints\memorilla_qwen3\stage3 --data_repo .\data_10pct --benchmark all --n 500 `
      --output_dir results\stanford_qwen3_stage3

Tested end to end on a tiny random decoder and encoder (resume, scoring on known answers, cached decoding
against transformers' own greedy generate). NOT YET RUN AGAINST THE REAL GEMMA 4 OR QWEN3 WEIGHTS: read the
[Setup], [SelfCheck], [Sample] and [EncoderCheck] lines before trusting a number.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import signal
import sys
import time
from itertools import chain
from pathlib import Path

# Same guard as apply_gemma4_patch.py: a broken torchaudio binary must not break `import transformers`.
try:
    import torchaudio  # noqa: F401
except (ImportError, OSError):
    _orig_find_spec = importlib.util.find_spec

    def _safe_find_spec(name, *args, **kwargs):
        if name == "torchaudio" or name.startswith("torchaudio."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = _safe_find_spec

SCRIPT_VERSION = "1.0"
CONDITIONS = ("memory", "shuffled", "closed_book")
THOUGHT_STUB = "<|channel>thought\n<channel|>"
EXTRA_STOP_TOKENS = ("<turn|>", "<end_of_turn>", "<|im_end|>", "<|endoftext|>")
RUN_FILE = "run.json"
BENCHMARK_FILE = "benchmark.json"
ENCODER_CHECK_FILE = "encoder_check.json"
SUMMARY_FILE = "summary.json"
ENCODER_CHECK_ROWS = 16


# ----------------------------------------------------------------------------- small helpers
def seeded_order(num_rows: int, seed: int) -> list[int]:
    """Fixed permutation of a split's rows; the first n of it are the rows evaluated with --n n."""
    order = list(range(num_rows))
    random.Random(seed).shuffle(order)
    return order


def shuffle_partners(order: list[int], collection_ids: list) -> dict[int, int]:
    """For every row, another row from a DIFFERENT collection (the next one along the seeded order).

    Defined over the whole split, so a row's partner does not depend on --n.
    """
    total = len(order)
    partners: dict[int, int] = {}
    for position, row in enumerate(order):
        probe = (position + 1) % total
        while probe != position and collection_ids[order[probe]] == collection_ids[row]:
            probe = (probe + 1) % total
        partners[row] = order[probe]
    return partners


def strip_thought_residue(text: str) -> str:
    """Remove what is left of a Gemma thought channel once its special tokens are dropped by decoding."""
    text = text.strip()
    while True:
        stripped = re.sub(r"^thought[ \t]*(\n|$)", "", text).lstrip()
        if stripped == text:
            break
        text = stripped
    text = re.split(r"\n\s*thought\s*(\n|$)", text)[0]
    return text.strip()


def paired_bootstrap(a: list[float], b: list[float], rounds: int = 2000, seed: int = 23) -> tuple[float, float, float]:
    """Mean of (a - b) and its 95% paired bootstrap interval."""
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    count = len(diffs)
    if count == 0:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(count)] for _ in range(count)) / count for _ in range(rounds))
    return sum(diffs) / count, means[int(0.025 * rounds)], means[int(0.975 * rounds) - 1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(path: Path) -> dict[int, dict]:
    """Read a predictions file; a torn last line from an interrupted run is ignored."""
    records: dict[int, dict] = {}
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                records[int(record["row"])] = record
            except (ValueError, KeyError, TypeError):
                continue
    return records


def open_for_append(path: Path):
    """Open a predictions file for appending, starting on a fresh line even after an interrupted write."""
    needs_newline = False
    if path.exists() and path.stat().st_size > 0:
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            needs_newline = handle.read(1) != b"\n"
    handle = path.open("a", encoding="utf-8")
    if needs_newline:
        handle.write("\n")
    return handle


def write_json(path: Path, payload) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(temporary, path)


def format_seconds(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


# ----------------------------------------------------------------------------- generate: setup
def install_gemma4_patch(required: bool):
    """Install the Gemma 4 runtime patch if it is next to this script, with its checkpoint saving switched off."""
    try:
        import apply_gemma4_patch
    except ImportError as error:
        if required:
            raise SystemExit(
                "apply_gemma4_patch.py must be importable (run from C:\\AI\\MemoryLLM) to load the INT4 Gemma 4 decoder."
            ) from error
        print(f"[Setup] apply_gemma4_patch not found ({error}); running unpatched.")
        return None

    from transformers import AutoConfig

    genuine_config_loader = AutoConfig.from_pretrained
    apply_gemma4_patch.install_runtime_monkeypatch()

    # The patch's AutoConfig hook assumes a config comes back, but transformers also calls it expecting
    # (config, unused_kwargs) when it loads any model without an explicit config. That breaks every
    # non-Gemma decoder. Same aliasing as the patch, tuple-safe.
    alias = getattr(apply_gemma4_patch, "_alias_gemma4_text_config_attrs", None)
    if alias is not None and getattr(genuine_config_loader, "__func__", None) is not None:

        @classmethod
        def _tuple_safe_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
            result = genuine_config_loader.__func__(cls, pretrained_model_name_or_path, *args, **kwargs)
            alias(result[0] if isinstance(result, tuple) else result)
            return result

        AutoConfig.from_pretrained = _tuple_safe_from_pretrained

    manager = apply_gemma4_patch.RESUME_MANAGER
    manager.output_dir = None
    manager.save_every_steps = 0
    manager.save_every_seconds = 0
    manager.pending_state = None
    manager.reset = lambda *a, **kw: None
    manager.save_checkpoint = lambda *a, **kw: None
    return apply_gemma4_patch


class Evaluator:
    """The frozen decoder plus the memory module, with greedy KV-cached generation from prompt embeddings."""

    def __init__(self, decoder: str, checkpoint: Path, cache_dir: str | None):
        import torch
        from transformers import AutoConfig

        from memorilla.llm import MemoryLLM, inject_memory
        from memorilla.memory import MemoryModule
        from memorilla.utils import load_memory_state

        self.torch = torch
        self.inject_memory = inject_memory
        self.memory = MemoryModule.from_pretrained(checkpoint)

        config = AutoConfig.from_pretrained(decoder, cache_dir=cache_dir)
        hidden = getattr(config, "hidden_size", None) or getattr(getattr(config, "text_config", None), "hidden_size", None)
        if hidden is not None and self.memory.config.output_dim != hidden:
            raise SystemExit(
                f"The checkpoint produces {self.memory.config.output_dim}-dimensional memory tokens but {decoder} has "
                f"hidden size {hidden}; pass the decoder the checkpoint was trained with (--decoder)."
            )

        self.llm = MemoryLLM(decoder, self.memory, attn_implementation="sdpa", cache_dir=cache_dir)
        self.dec = self.llm.text_model.eval()
        if next(self.dec.parameters()).device.type == "cpu" and torch.cuda.is_available():
            self.dec.to("cuda:0")
        self.device = next(self.dec.parameters()).device
        self.tok = self.llm.tokenizer
        self.memory_token_id = self.llm.memory_token_id
        self.memory.to(device=self.device, dtype=torch.bfloat16).eval()
        self.num_memories = self.memory.num_memories
        self.backbone = getattr(self.dec, "model", None)
        if self.backbone is None:
            self.backbone = self.dec.base_model
        self.lm_head = self.dec.lm_head
        self.embed = self.dec.get_input_embeddings()
        text_config = getattr(self.dec.config, "text_config", self.dec.config)
        self.softcap = getattr(text_config, "final_logit_softcapping", None)
        self.autocast = self.device.type == "cuda"
        self.kv_cache = True

        vocab = self.tok.get_vocab()
        self.has_thought_channel = "<|channel>" in vocab and "<channel|>" in vocab
        stop_ids: set[int] = set()
        generation_eos = getattr(getattr(self.dec, "generation_config", None), "eos_token_id", None)
        if isinstance(generation_eos, int):
            stop_ids.add(generation_eos)
        elif isinstance(generation_eos, (list, tuple)):
            stop_ids.update(int(token_id) for token_id in generation_eos)
        if self.tok.eos_token_id is not None:
            stop_ids.add(int(self.tok.eos_token_id))
        stop_ids.update(int(vocab[name]) for name in EXTRA_STOP_TOKENS if name in vocab)
        self.stop_ids = stop_ids

        # The patch installs a Ctrl+C handler that announces a checkpoint save; evaluation saves nothing.
        try:
            signal.signal(signal.SIGINT, signal.default_int_handler)
        except ValueError:
            pass

        # The module must hold exactly the weights in the checkpoint file.
        state = load_memory_state(checkpoint)
        key = "memory_queries" if "memory_queries" in state else next(k for k in state if "memory_projection" not in k)
        loaded = dict(self.memory.named_parameters()).get(key)
        if loaded is None:
            loaded = self.memory.state_dict()[key]
        loaded = loaded.detach().cpu()
        difference = float((loaded.float() - state[key].to(loaded.dtype).float()).abs().max())
        if difference > 1e-6:
            raise SystemExit(f"Loaded memory weights differ from {checkpoint} ({key}: max diff {difference:.3g}).")
        print(f"[Setup] Memory weights match the checkpoint file ({key}, max diff {difference:.1g}).")
        print(
            f"[Setup] decoder={decoder} | device={self.device} | memory tokens={self.num_memories} | "
            f"stop ids={sorted(self.stop_ids)} | thought channel={self.has_thought_channel}"
        )

    # -- prompt embeddings ---------------------------------------------------------------
    def memory_prompt_embeds(self, input_ids, doc_embeds, doc_padding_mask, question_embeds):
        torch = self.torch
        with torch.inference_mode():
            input_ids = input_ids.to(self.device)
            counts = (input_ids == self.memory_token_id).sum(dim=1)
            if not bool((counts == self.num_memories).all()):
                raise RuntimeError(f"Expected {self.num_memories} memory placeholders per row, found {counts.tolist()}.")
            embeds = self.embed(input_ids)
            # On the GPU this is the path training used: bfloat16 inputs under autocast. Without autocast
            # (CPU), the inputs must simply match the module's own dtype.
            input_dtype = torch.bfloat16 if self.autocast else next(self.memory.parameters()).dtype
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast):
                memory_tokens = self.memory(
                    doc_embeds.to(self.device, dtype=input_dtype),
                    doc_padding_mask.to(self.device),
                    question_embeds.to(self.device, dtype=input_dtype),
                )
            return self.inject_memory(embeds, memory_tokens, input_ids, self.memory_token_id)

    def text_prompt_embeds(self, input_ids):
        with self.torch.inference_mode():
            return self.embed(input_ids.to(self.device))

    def closed_book_prompt(self, question, system_prompt: str, suffix: str) -> str:
        """Upstream's closed-book baseline prompt, ending the way the decoder's own template ends it."""
        from memorilla.data import apply_chat_template

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question) + suffix},
        ]
        prompt = apply_chat_template(self.tok, messages, add_generation_prompt=True)
        # The patch strips Gemma's empty thought channel from generation prompts so that training text lines up.
        # An untuned decoder should see its native prompt, so the baseline puts it back.
        if self.has_thought_channel and not prompt.endswith(THOUGHT_STUB):
            prompt += THOUGHT_STUB
        return prompt

    # -- decoding --------------------------------------------------------------------------
    def _backbone(self, embeds, mask, past=None, use_cache=False):
        torch = self.torch
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=self.autocast):
            out = self.backbone(
                inputs_embeds=embeds, attention_mask=mask, past_key_values=past, use_cache=use_cache, return_dict=True
            )
        return out[0], getattr(out, "past_key_values", None)

    def _logits(self, hidden):
        logits = self.lm_head(hidden).float()
        if self.softcap is not None:
            logits = self.torch.tanh(logits / self.softcap) * self.softcap
        return logits

    def generate(self, embeds, mask, max_new_tokens: int) -> list[str]:
        """Greedy decoding for a left-padded batch of prompt embeddings; returns the raw decoded texts."""
        torch = self.torch
        with torch.inference_mode():
            mask = mask.to(self.device)
            batch = embeds.size(0)
            done = [False] * batch
            out_ids: list[list[int]] = [[] for _ in range(batch)]
            past, current, generated = None, embeds, []
            for _ in range(max_new_tokens):
                if self.kv_cache:
                    hidden, past = self._backbone(current, mask, past=past, use_cache=True)
                    if past is None:
                        print("[Generate] The decoder returned no KV cache; falling back to full re-forward (slow).")
                        self.kv_cache = False
                else:
                    hidden, _ = self._backbone(torch.cat([embeds, *generated], dim=1), mask, use_cache=False)
                next_ids = self._logits(hidden[:, -1, :]).argmax(dim=-1)
                for index, token_id in enumerate(next_ids.tolist()):
                    if done[index]:
                        continue
                    if token_id in self.stop_ids:
                        done[index] = True
                    else:
                        out_ids[index].append(token_id)
                if all(done):
                    break
                current = self.embed(next_ids.unsqueeze(1))
                generated.append(current)
                mask = torch.cat([mask, torch.ones((batch, 1), dtype=mask.dtype, device=mask.device)], dim=1)
            return [self.tok.decode(ids, skip_special_tokens=True) for ids in out_ids]


def print_self_check(ev: Evaluator, name: str, input_ids, embeds) -> dict:
    """Show what the decoder actually receives for the first row of a benchmark."""
    row_ids = input_ids[0].to(ev.device)
    pad_id = ev.tok.pad_token_id if ev.tok.pad_token_id is not None else -1
    real = row_ids != pad_id
    memory_positions = (row_ids == ev.memory_token_id).nonzero().flatten()
    text_positions = (real & (row_ids != ev.memory_token_id)).nonzero().flatten()
    norms = embeds[0].float().norm(dim=-1)
    info = {
        "prompt_tail": ev.tok.decode(row_ids[real][-24:].tolist()),
        "first_token_ids": row_ids[real][:4].tolist(),
        "bos_id": ev.tok.bos_token_id,
        "memory_positions": [int(memory_positions[0]), int(memory_positions[-1])] if memory_positions.numel() else [],
        "memory_token_norm_mean": round(float(norms[memory_positions].mean()), 2) if memory_positions.numel() else None,
        "text_token_norm_median": round(float(norms[text_positions].median()), 2),
    }
    print(f"\n[SelfCheck] {name}")
    for key, value in info.items():
        print(f"  {key:24s}: {value!r}")
    return info


# ----------------------------------------------------------------------------- generate: main
def run_benchmark(ev: Evaluator, bench, args, out_dir: Path, clean_generation) -> None:
    from memorilla.data import (
        ANSWER_COLUMN,
        CHOICES_COLUMN,
        COLLECTION_COLUMN,
        DEFAULT_SYSTEM_PROMPT,
        EVAL_COLUMNS,
        HARD_COLUMN,
        QUESTION_COLUMN,
        MemoryCollator,
        MemoryDataset,
        ensure_data,
    )
    from memorilla.embeddings import ensure_embeddings, load_split_with_embeddings

    torch = ev.torch
    root = ensure_data(bench.config, [bench.split], repo=args.data_repo, data_dir=args.data_dir)
    ensure_embeddings(bench.config, [bench.split], repo=args.data_repo, data_dir=args.data_dir, encoder=args.encoder)
    dataset, store = load_split_with_embeddings(bench.config, bench.split, root, EVAL_COLUMNS, args.encoder)

    order = seeded_order(len(dataset), args.seed)
    rows = order if args.n <= 0 else order[: args.n]
    collection_ids = list(dataset[COLLECTION_COLUMN])
    partners = shuffle_partners(order, collection_ids)
    items = MemoryDataset([(bench.config, bench.split, dataset)])

    system_prompt = DEFAULT_SYSTEM_PROMPT if args.train_prompt else bench.system_prompt
    suffix = "" if args.train_prompt else (bench.suffix or "")
    collator = MemoryCollator(
        ev.tok,
        {(bench.config, bench.split): store},
        num_memories=ev.num_memories,
        max_length=bench.max_length,
        system_prompt=system_prompt,
        suffix=suffix,
        train=False,
    )

    folder = out_dir / bench.name
    folder.mkdir(parents=True, exist_ok=True)
    write_json(
        folder / BENCHMARK_FILE,
        {
            "name": bench.name,
            "config": bench.config,
            "split": bench.split,
            "scoring": bench.scoring,
            "metrics": list(bench.metrics),
            "rows_in_split": len(dataset),
            "max_new_tokens": bench.max_new_tokens,
            "system_prompt": system_prompt,
            "suffix": suffix,
            "baseline_system_prompt": bench.baseline.system_prompt if bench.baseline else None,
            "baseline_suffix": bench.baseline.suffix if bench.baseline else None,
            "baseline_max_new_tokens": bench.baseline.max_new_tokens if bench.baseline else None,
        },
    )

    encoder_check = out_dir / ENCODER_CHECK_FILE
    if not encoder_check.exists():
        probe = order[:ENCODER_CHECK_ROWS]
        write_json(
            encoder_check,
            {
                "encoder": args.encoder,
                "source": f"{bench.config}/{bench.split}",
                "texts": [str(dataset[row][QUESTION_COLUMN]).strip() for row in probe],
                "embeddings": [store.question(row).float().tolist() for row in probe],
            },
        )

    print(f"\n=== {bench.name} ({bench.split}, {len(rows)} of {len(dataset)} rows) ===")
    for condition in args.conditions:
        if condition == "closed_book" and bench.baseline is None:
            print(f"  [{bench.name}] no baseline protocol registered; skipping closed_book.")
            continue
        path = folder / f"{condition}.jsonl"
        done = read_records(path)
        todo = [row for row in rows if row not in done]
        if not todo:
            print(f"  [{bench.name} | {condition}] all {len(rows)} rows already written.")
            continue
        budget = bench.baseline.max_new_tokens if condition == "closed_book" else bench.max_new_tokens
        started = time.monotonic()
        shown_check = False
        with open_for_append(path) as handle:
            for start in range(0, len(todo), args.batch_size):
                chunk = todo[start : start + args.batch_size]
                batch_items = [items[row] for row in chunk]
                if condition == "closed_book":
                    prompts = [
                        ev.closed_book_prompt(item[QUESTION_COLUMN], bench.baseline.system_prompt, bench.baseline.suffix)
                        for item in batch_items
                    ]
                    encoded = ev.tok(
                        prompts, padding=True, padding_side="left", add_special_tokens=False, return_tensors="pt"
                    )
                    embeds = ev.text_prompt_embeds(encoded["input_ids"])
                    attention_mask = encoded["attention_mask"]
                    if not shown_check:
                        print(f"\n[SelfCheck] {bench.name} closed_book prompt tail: {prompts[0][-110:]!r}")
                else:
                    batch = collator(batch_items)
                    doc_embeds, doc_padding_mask = batch["doc_embeds"], batch["doc_padding_mask"]
                    if condition == "shuffled":
                        doc_embeds, doc_padding_mask = collator._documents([items[partners[row]] for row in chunk])
                        doc_embeds = doc_embeds.to(torch.bfloat16)
                    embeds = ev.memory_prompt_embeds(
                        batch["input_ids"], doc_embeds, doc_padding_mask, batch["question_embeds"]
                    )
                    attention_mask = batch["attention_mask"]
                    if not shown_check and condition == "memory":
                        print_self_check(ev, bench.name, batch["input_ids"], embeds)

                raw_texts = ev.generate(embeds, attention_mask, budget)
                for row, item, raw in zip(chunk, batch_items, raw_texts, strict=True):
                    record = {
                        "row": row,
                        "collection_id": int(item[COLLECTION_COLUMN]),
                        "question": item[QUESTION_COLUMN],
                        "answer": item[ANSWER_COLUMN],
                        "prediction": strip_thought_residue(clean_generation(raw)),
                    }
                    if item.get(CHOICES_COLUMN) is not None:
                        record["choices"] = [str(choice) for choice in item[CHOICES_COLUMN]]
                    if item.get(HARD_COLUMN) is not None:
                        record["hard"] = bool(item[HARD_COLUMN])
                    if condition == "shuffled":
                        record["partner_row"] = partners[row]
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()

                if not shown_check:
                    shown_check = True
                    for item, raw in list(zip(batch_items, raw_texts, strict=True))[:2]:
                        print(f"  [Sample {bench.name} | {condition}]")
                        print(f"    question  : {str(item[QUESTION_COLUMN]).strip()[:110]!r}")
                        print(f"    reference : {str(item[ANSWER_COLUMN]).strip()[:110]!r}")
                        print(f"    prediction: {strip_thought_residue(clean_generation(raw))[:160]!r}")

                finished = start + len(chunk)
                if (start // args.batch_size) % 10 == 0 or finished == len(todo):
                    elapsed = time.monotonic() - started
                    remaining = elapsed / finished * (len(todo) - finished)
                    print(
                        f"  [{bench.name} | {condition}] {finished}/{len(todo)} rows | "
                        f"{elapsed / finished:.1f}s/row | elapsed {format_seconds(elapsed)} | left {format_seconds(remaining)}"
                    )


def cmd_generate(args) -> None:
    # A training stage directory in the environment would make the patch look for a run to resume.
    for variable in ("MEMORILLA_STAGE_DIR", "MEMORILLA_OUTPUT_DIR"):
        os.environ.pop(variable, None)
    install_gemma4_patch(required="gemma-4" in args.decoder.lower())

    import torch
    from dataclasses import replace

    from memorilla.benchmarks import get_benchmarks
    from memorilla.paths import resolve

    try:
        from memorilla.metrics import clean_generation
    except Exception as error:  # metrics needs nltk and rouge-score; generation does not
        print(f"[Setup] memorilla.metrics not importable ({error}); using a local copy of clean_generation.")
        chat_tokens = re.compile(r"<\|(?:im_end|im_start|endoftext|im_sep)\|>")

        def clean_generation(text: str) -> str:
            return chat_tokens.sub("", text).strip()

    torch.manual_seed(args.seed)
    names = list(dict.fromkeys(args.benchmark))
    benches = []
    for name in names:
        benches.extend(bench for bench in get_benchmarks(name) if bench.name not in {b.name for b in benches})
    if args.split != "test":
        benches = [replace(bench, name=f"{bench.name}-{args.split}", split=args.split) for bench in benches]

    checkpoint = resolve(args.checkpoint)
    checkpoint_file = checkpoint / "memory.pt" if checkpoint.is_dir() else checkpoint
    if not checkpoint_file.exists():
        raise SystemExit(f"No memory.pt at {checkpoint_file}.")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run = {
        "script_version": SCRIPT_VERSION,
        "checkpoint": str(checkpoint_file),
        "checkpoint_sha256": sha256_file(checkpoint_file),
        "decoder": args.decoder,
        "encoder": args.encoder,
        "seed": args.seed,
        "train_prompt": bool(args.train_prompt),
    }
    print(f"[Setup] checkpoint {checkpoint_file} | sha256 {run['checkpoint_sha256'][:16]}...")
    run_path = out_dir / RUN_FILE
    if run_path.exists():
        with run_path.open(encoding="utf-8") as handle:
            previous = json.load(handle)
        changed = [
            key
            for key in ("checkpoint_sha256", "decoder", "seed", "train_prompt")
            if previous.get(key) != run[key]
        ]
        if changed and not args.force:
            raise SystemExit(
                f"{out_dir} already holds results made with a different {', '.join(changed)}. "
                "Use a new --output_dir (or --force to mix them)."
            )
    write_json(run_path, run)

    ev = Evaluator(args.decoder, checkpoint, args.cache_dir)
    for bench in benches:
        try:
            run_benchmark(ev, bench, args, out_dir, clean_generation)
        except KeyboardInterrupt:
            print("\nInterrupted. Rows written so far are kept; rerun the same command to continue.")
            raise SystemExit(130) from None
        except Exception as error:  # keep going so one bad benchmark does not lose the rest
            print(f"  [{bench.name}] FAILED: {type(error).__name__}: {error}")
    print(f"\nGeneration finished. Next: python stanford_eval_gemma4.py score --output_dir {out_dir}")


# ----------------------------------------------------------------------------- score
class HFEncoder:
    """Qwen3-Embedding through transformers: last-token pooling, L2-normalised, no instruction prefix."""

    def __init__(self, name: str, cache_dir: str | None, batch_size: int, max_length: int = 8192):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left", cache_dir=cache_dir)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        try:
            model = AutoModel.from_pretrained(name, dtype=dtype, cache_dir=cache_dir)
        except TypeError:  # older transformers name the argument torch_dtype
            model = AutoModel.from_pretrained(name, torch_dtype=dtype, cache_dir=cache_dir)
        self.model = model.to(self.device).eval()
        self.batch_size = batch_size
        self.max_length = max_length
        self.append_eos = False
        self.cache: dict[str, "object"] = {}
        print(f"[Score] Loaded encoder {name} on {self.device} ({dtype}).")

    def embed(self, texts: list[str]):
        """Return a float16 array [len(texts), D], like upstream's embed_texts."""
        import numpy as np

        torch = self.torch
        missing = sorted({text for text in texts if text not in self.cache}, key=len)
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                chunk = missing[start : start + self.batch_size]
                suffix = self.tok.eos_token if self.append_eos and self.tok.eos_token else ""
                encoded = self.tok(
                    [(text if text else " ") + suffix for text in chunk],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"]
                if bool((mask[:, -1] == 1).all()):  # left padded: the last position is every row's last token
                    pooled = hidden[:, -1]
                else:
                    last = mask.sum(dim=1) - 1
                    pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last]
                pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1).cpu().numpy().astype(np.float16)
                for text, vector in zip(chunk, pooled, strict=True):
                    self.cache[text] = vector
        if not texts:
            return np.zeros((0, 0), dtype=np.float16)
        return np.stack([self.cache[text] for text in texts])


def check_encoder(encoder: HFEncoder, out_dir: Path) -> dict | None:
    """Re-embed questions whose vLLM embeddings ship with the data; they must come out the same.

    If they do not, try once more with an explicit end-of-text token (last-token pooling reads that position)
    and keep whichever way reproduces the stored vectors.
    """
    import numpy as np

    path = out_dir / ENCODER_CHECK_FILE
    if not path.exists():
        print("[EncoderCheck] no encoder_check.json in the output folder; skipping.")
        return None
    with path.open(encoding="utf-8") as handle:
        probe = json.load(handle)
    stored = np.asarray(probe["embeddings"], dtype=np.float32)
    stored /= np.linalg.norm(stored, axis=1, keepdims=True) + 1e-8

    def measure() -> dict:
        fresh = encoder.embed(probe["texts"]).astype(np.float32)
        if stored.shape != fresh.shape:
            return {"ok": False, "note": f"stored embeddings are {stored.shape}, this encoder gives {fresh.shape}"}
        fresh /= np.linalg.norm(fresh, axis=1, keepdims=True) + 1e-8
        cosines = (stored * fresh).sum(axis=1)
        mean, low = round(float(cosines.mean()), 4), round(float(cosines.min()), 4)
        # Numerical noise between two bfloat16 runtimes stays far above these; a wrong pooling position or a
        # missing end-of-text token lands far below.
        return {"mean_cosine": mean, "min_cosine": low, "ok": mean >= 0.99 and low >= 0.95}

    result = measure()
    if not result["ok"] and "note" not in result and encoder.tok.eos_token:
        first = dict(result)
        encoder.cache.clear()
        encoder.append_eos = True
        retry = measure()
        if retry["ok"]:
            print(
                f"[EncoderCheck] default tokenisation gave min cosine {first['min_cosine']:.4f}; with an explicit "
                f"{encoder.tok.eos_token!r} appended it is {retry['min_cosine']:.4f}. Using the explicit token."
            )
            result = retry
        else:
            encoder.append_eos = False
            encoder.cache.clear()
    result["append_eos"] = encoder.append_eos
    if result["ok"]:
        print(
            f"[EncoderCheck] {len(probe['texts'])} stored questions ({probe.get('source')}): mean cosine "
            f"{result['mean_cosine']:.4f}, min {result['min_cosine']:.4f} -> matches the stored vLLM embeddings"
        )
    else:
        detail = result.get("note") or f"mean cosine {result['mean_cosine']:.4f}, min {result['min_cosine']:.4f}"
        print(f"[EncoderCheck] WARNING: this encoder does NOT reproduce the stored embeddings ({detail}).")
        print("[EncoderCheck] Choice accuracies below are not comparable with Stanford's until this is resolved.")
    return result


def per_row_scores(scoring, predictions, references, choices, encoder, M, B) -> dict[str, list[float | None]]:
    """Per-row values of the metrics that have them, for paired intervals. None = row not scored."""
    scores: dict[str, list[float | None]] = {}
    if scoring == B.CHOICE:
        nonempty = [[choice for choice in row if choice.strip()] for row in choices]
        vectors = encoder.embed([p.strip() for p in predictions] + list(chain.from_iterable(nonempty)))
        choice_vectors, offset = [], len(predictions)
        for row in nonempty:
            choice_vectors.append(vectors[offset : offset + len(row)])
            offset += len(row)
        selected = M.choice_predictions(vectors[: len(predictions)], choice_vectors, choices)
        scores[B.ACCURACY] = [
            float(M.choice_index(row, reference) == index)
            for row, reference, index in zip(choices, references, selected, strict=True)
        ]
    elif scoring == B.POLARITY:
        values: list[float | None] = []
        for prediction, reference in zip(predictions, references, strict=True):
            gold = M.first_polarity(reference)
            values.append(None if gold is None else float(M.first_polarity(prediction) == gold))
        scores[B.ACCURACY] = values
    else:
        from rouge_score import rouge_scorer

        if scoring == B.EXACT_MATCH:
            scores[B.EXACT_MATCH] = [
                float(M.normalize_answer(p) == M.normalize_answer(r)) for p, r in zip(predictions, references, strict=True)
            ]
        scores[B.F1] = [M.token_f1([p], [r]) for p, r in zip(predictions, references, strict=True)]
        scorer = rouge_scorer.RougeScorer([B.ROUGEL], use_stemmer=True)
        scores[B.ROUGEL] = [
            float(scorer.score(r, p.strip())[B.ROUGEL].fmeasure) for p, r in zip(predictions, references, strict=True)
        ]
    return scores


def cmd_score(args) -> None:
    from memorilla import benchmarks as B
    from memorilla import metrics as M

    out_dir = Path(args.output_dir)
    folders = sorted(path for path in out_dir.iterdir() if path.is_dir() and (path / BENCHMARK_FILE).exists())
    if not folders:
        raise SystemExit(f"No benchmark folders with {BENCHMARK_FILE} under {out_dir}. Run `generate` first.")
    run = {}
    if (out_dir / RUN_FILE).exists():
        with (out_dir / RUN_FILE).open(encoding="utf-8") as handle:
            run = json.load(handle)

    metas = {}
    for folder in folders:
        with (folder / BENCHMARK_FILE).open(encoding="utf-8") as handle:
            metas[folder.name] = json.load(handle)

    encoder, encoder_report = None, None
    if any(meta["scoring"] == B.CHOICE for meta in metas.values()):
        encoder = HFEncoder(args.encoder or run.get("encoder") or "Qwen/Qwen3-Embedding-4B", args.cache_dir, args.batch_size)
        encoder_report = check_encoder(encoder, out_dir)
        M.embed_texts = lambda engine, texts, chunk_size=None: engine.embed(list(texts))

    meteor_ok = True
    try:
        M.ensure_nltk_resources()
    except Exception as error:
        meteor_ok = False
        print(f"[Score] METEOR skipped: {error}")

    summary = {"run": run, "encoder_check": encoder_report, "benchmarks": {}}
    print()
    for name, meta in metas.items():
        try:
            scoring = meta["scoring"]
            records = {c: read_records(out_dir / name / f"{c}.jsonl") for c in CONDITIONS}
            records = {c: r for c, r in records.items() if r}
            if not records:
                continue
            entry = {"scoring": scoring, "split": meta["split"], "rows_in_split": meta.get("rows_in_split"), "conditions": {}}
            row_scores: dict[str, dict[str, dict[int, float]]] = {}
            for condition, by_row in records.items():
                rows = sorted(by_row)
                predictions = [str(by_row[row]["prediction"]) for row in rows]
                references = [str(by_row[row]["answer"]).strip() for row in rows]
                choices = [list(by_row[row]["choices"]) for row in rows] if scoring == B.CHOICE else None
                hard = [bool(by_row[row].get("hard")) for row in rows]
                metrics = [metric for metric in meta["metrics"] if meteor_ok or metric != B.METEOR]
                if scoring == B.CHOICE:
                    metrics = [B.ACCURACY, B.ACCURACY_HARD] if any(hard) else [B.ACCURACY]
                official = M.compute_metrics(
                    scoring, metrics, predictions, references, choices, hard if any(hard) else None, encoder
                )
                entry["conditions"][condition] = {
                    "num_examples": len(rows),
                    **{metric: round(100 * value, 2) for metric, value in official.items()},
                }
                per_row = per_row_scores(scoring, predictions, references, choices, encoder, M, B)
                row_scores[condition] = {
                    metric: {row: value for row, value in zip(rows, values, strict=True) if value is not None}
                    for metric, values in per_row.items()
                }

            entry["differences"] = {}
            if "memory" in row_scores:
                for other in ("shuffled", "closed_book"):
                    if other not in row_scores:
                        continue
                    for metric in row_scores["memory"]:
                        common = sorted(set(row_scores["memory"][metric]) & set(row_scores[other].get(metric, {})))
                        if not common:
                            continue
                        mean, low, high = paired_bootstrap(
                            [row_scores["memory"][metric][row] for row in common],
                            [row_scores[other][metric][row] for row in common],
                        )
                        entry["differences"][f"memory_minus_{other}:{metric}"] = {
                            "rows": len(common),
                            "mean": round(100 * mean, 2),
                            "ci95": [round(100 * low, 2), round(100 * high, 2)],
                        }
            summary["benchmarks"][name] = entry

            shown = list(dict.fromkeys(m for c in entry["conditions"].values() for m in c if m != "num_examples"))
            for metric in shown:
                cells = []
                for condition in CONDITIONS:
                    values = entry["conditions"].get(condition)
                    if values is not None and metric in values:
                        cells.append(f"{condition} {values[metric]:6.2f} (n={values['num_examples']})")
                line = f"[{name:12s}] {metric:13s} " + " | ".join(cells)
                for other, label in (("shuffled", "mem-shuf"), ("closed_book", "mem-closed")):
                    difference = entry["differences"].get(f"memory_minus_{other}:{metric}")
                    if difference is not None:
                        line += f" | {label} {difference['mean']:+.2f} {difference['ci95']} (n={difference['rows']})"
                print(line)
        except Exception as error:  # e.g. a reference that is not among its row's choices
            print(f"[{name:12s}] SCORING FAILED: {type(error).__name__}: {error}")
            summary["benchmarks"][name] = {"error": f"{type(error).__name__}: {error}"}

    write_json(out_dir / SUMMARY_FILE, summary)
    print(f"\nSaved {out_dir / SUMMARY_FILE}")


# ----------------------------------------------------------------------------- command line
def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="Generate answers with the decoder and the memory module.")
    generate.add_argument("--checkpoint", required=True, help="Checkpoint directory or memory.pt.")
    generate.add_argument("--decoder", default="google/gemma-4-26B-A4B-it", help="Decoder the checkpoint was trained with.")
    generate.add_argument("--encoder", default="Qwen/Qwen3-Embedding-4B", help="Names the stored embeddings; not loaded here.")
    generate.add_argument("--benchmark", nargs="+", default=["all"], help="Benchmark names, or all.")
    generate.add_argument("--split", default="test")
    generate.add_argument("--n", type=int, default=500, help="Rows per benchmark; 0 evaluates the whole split.")
    generate.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    generate.add_argument("--batch_size", type=int, default=8)
    generate.add_argument("--seed", type=int, default=23)
    generate.add_argument("--train_prompt", action="store_true",
                          help="Use the training system prompt and no suffix instead of the benchmark's (not Stanford's protocol).")
    generate.add_argument("--data_repo", default=os.environ.get("MEMORILLA_DATA_REPO", "memorilla/Memorilla-Data"))
    generate.add_argument("--data_dir", default=os.environ.get("MEMORILLA_DATA_DIR", "data"))
    generate.add_argument("--cache_dir", default=os.environ.get("MEMORILLA_MODEL_CACHE") or None)
    generate.add_argument("--output_dir", required=True)
    generate.add_argument("--force", action="store_true", help="Write into a folder made with other settings.")
    generate.set_defaults(function=cmd_generate)

    score = commands.add_parser("score", help="Score generated answers with upstream's metrics.")
    score.add_argument("--output_dir", required=True, help="The folder `generate` wrote to.")
    score.add_argument("--encoder", default=None, help="Scoring encoder (default: the one recorded by generate).")
    score.add_argument("--batch_size", type=int, default=16)
    score.add_argument("--cache_dir", default=os.environ.get("MEMORILLA_MODEL_CACHE") or None)
    score.set_defaults(function=cmd_score)

    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
