# Save as eval_controls_gemma4.py in C:\AI\MemoryLLM
"""
Pre-Evaluation & Retrieval Control Suite for Memorilla + Gemma 4 26B-A4B (INT4 MoE).

Proves whether the Perceiver MemoryModule is performing genuine document recall
vs. static soft-prompt formatting by evaluating every Stage 3 dataset across 4 conditions:
  1. True Docs      : Actual retrieved document embeddings for each question.
  2. Shuffled Docs  : Document embeddings circularly shifted across the batch (idx -> idx+1),
                      preserving exact memory-token norms & task formatting while removing true facts.
  3. Gaussian Noise : Random normal vectors matching the batch's document embedding mean & std.
  4. No Memory      : Pure frozen Gemma 4 26B-A4B decoder with the 16 memory tokens stripped.

Also performs:
  - Token Norm Audit: Measures L2 norm & std of the 16 memory tokens vs. native Gemma 4 text tokens
                      (with and without the sqrt(2816) hook).
  - Live KV-Cached Generation: Decodes side-by-side answers (True Docs vs. Shuffled Docs vs. No Memory)
                               without requiring vLLM.
"""

import argparse
import collections
import contextlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import apply_gemma4_patch

# Install runtime patches first so Gemma 4 26B loads in INT4 and custom hooks are available
apply_gemma4_patch.install_runtime_monkeypatch()

# Disable CheckpointResumeManager side-effects so pre-eval NEVER overwrites training state
# or fast-forwards DataLoaders when train.py calls install_runtime_monkeypatch()
def _disable_resume_manager() -> None:
    rm = apply_gemma4_patch.RESUME_MANAGER
    rm.output_dir = None
    rm.save_every_steps = 0
    rm.save_every_seconds = 0
    rm.pending_state = None
    rm.resume_epoch = 0
    rm.resume_micro_step = 0
    rm.reset = lambda *a, **kw: None
    rm.save_checkpoint = lambda *a, **kw: None

_disable_resume_manager()

import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Subset
from transformers import AutoTokenizer


STAGE3_DATASETS = [
    "triviaqa",
    "factkg",
    "narrativeqa",
    "pv4",
    "pmv2",
    "pubmedqa",
    "lamp4",
    "lamp7",
]


def discover_default_checkpoints(stage_dir: str = "checkpoints/stage3_gemma4") -> list[tuple[str, Path]]:
    """
    Finds available Stage 3 checkpoints on disk in chronological order.
    """
    root = Path(stage_dir)
    found: list[tuple[str, Path]] = []

    warmstart = Path("checkpoints/gemma4_warmstart/memory.pt")
    if warmstart.exists():
        found.append(("warmstart", warmstart))

    stage2 = Path("checkpoints/stage2_gemma4/memory.pt")
    if stage2.exists():
        found.append(("stage2_final", stage2))

    if root.exists():
        for ep_dir in sorted(root.glob("epoch-*")):
            mem_pt = ep_dir / "memory.pt"
            if mem_pt.exists():
                found.append((ep_dir.name, mem_pt))

        resume_pt = root / "resume_latest" / "memory.pt"
        if resume_pt.exists():
            found.append(("resume_latest", resume_pt))

    return found


def load_memory_weights_into_module(memory_module: nn.Module, ckpt_path: Path) -> None:
    """
    Loads a `memory.pt` or `trainer_state.pt` file cleanly into `pl_module.memory`.
    The `_unscale_on_load` pre-hook in `apply_gemma4_patch.py` automatically handles
    whether `sqrt(2816)` was folded into `memory.pt` on save.
    """
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "memory_state_dict" in raw:
        state_dict = dict(raw["memory_state_dict"])
    elif isinstance(raw, dict) and "state_dict" in raw:
        state_dict = {
            k.removeprefix("memory."): v
            for k, v in raw["state_dict"].items()
            if k.startswith("memory.")
        }
    else:
        state_dict = dict(raw)

    missing, unexpected = memory_module.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [Warn] Missing keys when loading {ckpt_path}: {missing[:5]}")
    if unexpected:
        print(f"  [Warn] Unexpected keys when loading {ckpt_path}: {unexpected[:5]}")


def capture_memorilla_components(
    config_path: str,
    init_memory_path: str,
    decoder_name: str = "google/gemma-4-26B-A4B-it",
    max_length: int = 1024,
):
    """
    Runs `train.py` initialization up to `Trainer.fit()` and intercepts the fully
    constructed `pl_module`, `train_dataloader`, and `val_dataloader` without training.
    """
    captured: dict[str, Any] = {}
    orig_fit = pl.Trainer.fit

    def _intercept_fit(self, model, train_dataloaders=None, val_dataloaders=None, datamodule=None, **kwargs):
        captured["trainer"] = self
        captured["pl_module"] = model
        captured["train_dataloaders"] = train_dataloaders
        captured["val_dataloaders"] = val_dataloaders
        captured["datamodule"] = datamodule
        # Attach trainer reference & disable Lightning self.log() outside active fit loop
        model._trainer = self
        model.log = lambda *a, **kw: None
        model.log_dict = lambda *a, **kw: None
        if hasattr(model, "setup"):
            try:
                model.setup("fit")
            except Exception as exc:
                print(f"[PreEval] Note: model.setup('fit') returned: {exc}")
        if datamodule is not None and hasattr(datamodule, "setup"):
            try:
                datamodule.setup("fit")
            except Exception as exc:
                print(f"[PreEval] Note: datamodule.setup('fit') returned: {exc}")
        print("[PreEval] Intercepted initialized LightningModule and DataLoaders from train.py.")
        return None

    pl.Trainer.fit = _intercept_fit
    orig_argv = list(sys.argv)
    try:
        sys.argv = [
            "train.py",
            "--config",
            config_path,
            "--decoder",
            decoder_name,
            "--init_memory",
            str(init_memory_path),
            "--max_length",
            str(max_length),
            "--devices",
            "1",
            "--attn_implementation",
            "sdpa",
        ]
        _disable_resume_manager()

        import runpy
        runpy.run_path("train.py", run_name="__main__")
    finally:
        pl.Trainer.fit = orig_fit
        sys.argv = orig_argv
        _disable_resume_manager()

    if "pl_module" not in captured:
        raise RuntimeError("Failed to intercept pl_module from train.py.")
    return captured


def extract_per_dataset_loaders(
    captured: dict[str, Any],
    batch_size: int = 8,
    max_batches: int = 4,
    seed: int = 42,
) -> dict[str, tuple[str, DataLoader]]:
    """
    Unwraps validation and training DataLoaders (whether returned as a list of DataLoaders,
    a ConcatDataset, a custom multi-dataset wrapper, or per-dataset factory calls) so we can
    evaluate each of the 8 Stage 3 datasets individually.
    """
    pl_module = captured["pl_module"]
    datamodule = captured.get("datamodule")
    val_dl = captured.get("val_dataloaders")
    train_dl = captured.get("train_dataloaders")

    if val_dl is None:
        for src in (datamodule, pl_module):
            if src is not None and hasattr(src, "val_dataloader"):
                try:
                    val_dl = src.val_dataloader()
                    break
                except Exception as exc:
                    print(f"[PreEval] Warning calling {type(src).__name__}.val_dataloader(): {exc}")

    if train_dl is None:
        for src in (datamodule, pl_module):
            if src is not None and hasattr(src, "train_dataloader"):
                try:
                    train_dl = src.train_dataloader()
                    break
                except Exception as exc:
                    print(f"[PreEval] Warning calling {type(src).__name__}.train_dataloader(): {exc}")

    def _infer_subdataset_name(ds_obj, idx: int, split_hint: str) -> str:
        # Recursively inspect object and its attributes for a known Stage 3 dataset name
        visited = set()
        queue = [ds_obj]
        while queue:
            cur = queue.pop(0)
            if id(cur) in visited or cur is None:
                continue
            visited.add(id(cur))
            for attr in ("name", "config", "dataset_name", "subset", "task", "path", "data_dir", "root"):
                val = getattr(cur, attr, None)
                if isinstance(val, (str, Path)):
                    s = str(val).lower()
                    for known in STAGE3_DATASETS:
                        if known in s:
                            return known
            if hasattr(cur, "__dict__"):
                for k, v in cur.__dict__.items():
                    if isinstance(v, (str, Path)):
                        s = str(v).lower()
                        for known in STAGE3_DATASETS:
                            if known in s:
                                return known
                    elif not isinstance(v, (int, float, bool, torch.Tensor, list, dict, tuple, set)):
                        queue.append(v)

        val_order = ["pv4", "pmv2", "factkg", "triviaqa", "narrativeqa"]
        if split_hint == "val" and idx < len(val_order):
            return val_order[idx]
        if idx < len(STAGE3_DATASETS):
            return STAGE3_DATASETS[idx]
        return f"{split_hint}_ds_{idx}"

    def _build_subset_loader(sub_ds, collate_fn, seed_val: int) -> DataLoader:
        n_total = len(sub_ds)
        target_items = batch_size * max_batches
        g = torch.Generator().manual_seed(seed_val)
        perm = torch.randperm(n_total, generator=g)[: min(n_total, target_items)].tolist()
        sub = Subset(sub_ds, perm)
        return DataLoader(sub, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    def _slice_concat_by_known_counts(ds, collate_fn, split_tag: str, base_seed: int, out_dict: dict) -> bool:
        # Check if ds is a merged Stage 3 dataset matching known per-dataset split sizes
        val_counts = [
            ("pv4", 1755),
            ("pmv2", 2059),
            ("factkg", 13266),
            ("triviaqa", 17444),
            ("narrativeqa", 3461),
        ]
        train_counts = [
            ("pv4", 2500),
            ("pmv2", 2500),
            ("factkg", 8636),
            ("triviaqa", 15910),
            ("narrativeqa", 12331),
            ("pubmedqa", 900),
            ("lamp4", 2500),
            ("lamp7", 2500),
        ]
        expected = val_counts if split_tag == "val" else train_counts
        total_expected = sum(c for _, c in expected)
        if abs(len(ds) - total_expected) > 50:
            return False

        offset = 0
        for idx, (name, count) in enumerate(expected):
            end = min(len(ds), offset + count)
            if end > offset and name not in out_dict:
                sub_range = Subset(ds, list(range(offset, end)))
                out_dict[name] = (split_tag, _build_subset_loader(sub_range, collate_fn, base_seed + idx))
            offset = end
        return True

    def _unpack_dl_into_datasets(dl_obj, split_tag: str, base_seed: int, out_dict: dict):
        if dl_obj is None:
            return
        # Case 1: List or tuple of DataLoaders
        if isinstance(dl_obj, (list, tuple)):
            for i, sub_dl in enumerate(dl_obj):
                if isinstance(sub_dl, DataLoader):
                    ds = sub_dl.dataset
                    if hasattr(ds, "datasets") and isinstance(ds.datasets, (list, tuple)):
                        for j, inner_ds in enumerate(ds.datasets):
                            name = _infer_subdataset_name(inner_ds, j, split_tag)
                            if name not in out_dict:
                                out_dict[name] = (split_tag, _build_subset_loader(inner_ds, sub_dl.collate_fn, base_seed + j))
                    else:
                        name = _infer_subdataset_name(ds, i, split_tag)
                        if name not in out_dict:
                            out_dict[name] = (split_tag, _build_subset_loader(ds, sub_dl.collate_fn, base_seed + i))
            return

        # Case 2: Single DataLoader wrapping a ConcatDataset or merged Stage 3 dataset
        if isinstance(dl_obj, DataLoader):
            ds = dl_obj.dataset
            collate_fn = dl_obj.collate_fn
            if hasattr(ds, "datasets") and isinstance(ds.datasets, (list, tuple)):
                for i, inner_ds in enumerate(ds.datasets):
                    name = _infer_subdataset_name(inner_ds, i, split_tag)
                    if name not in out_dict:
                        out_dict[name] = (split_tag, _build_subset_loader(inner_ds, collate_fn, base_seed + i))
            elif not _slice_concat_by_known_counts(ds, collate_fn, split_tag, base_seed, out_dict):
                name = f"combined_{split_tag}"
                if name not in out_dict:
                    out_dict[name] = (split_tag, _build_subset_loader(ds, collate_fn, base_seed))

    per_ds_loaders: dict[str, tuple[str, DataLoader]] = {}
    _unpack_dl_into_datasets(val_dl, "val", seed, per_ds_loaders)
    _unpack_dl_into_datasets(train_dl, "train-holdout", seed + 100, per_ds_loaders)

    return per_ds_loaders


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: move_batch_to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [move_batch_to_device(v, device) for v in batch]
    if isinstance(batch, tuple):
        return tuple(move_batch_to_device(v, device) for v in batch)
    return batch


def find_decoder_and_memory(pl_module: nn.Module) -> tuple[nn.Module, nn.Module]:
    llm_wrapper = getattr(pl_module, "llm", None)
    memory_mod = getattr(pl_module, "memory", None)
    if memory_mod is None and llm_wrapper is not None:
        memory_mod = getattr(llm_wrapper, "memory", None)

    decoder_mod = None
    if llm_wrapper is not None and hasattr(llm_wrapper, "text_model"):
        decoder_mod = llm_wrapper.text_model
    else:
        for attr in ("text_model", "model", "decoder", "language_model"):
            cand = getattr(pl_module, attr, None)
            if isinstance(cand, nn.Module) and cand is not memory_mod:
                decoder_mod = cand
                break

    if memory_mod is None or decoder_mod is None:
        raise RuntimeError("Could not locate `memory` and `text_model` submodules on pl_module.")
    return decoder_mod, memory_mod


class ControlInterceptor:
    """
    Context-controlled interceptor that hooks `MemoryModule.forward` and `Decoder.forward`
    to evaluate exact per-example and per-token loss across:
      - mode="true"      : Real document embeddings
      - mode="shuffled"  : Document embeddings (and masks) rolled by +1 across batch dim
      - mode="noise"     : Document embeddings replaced with matched Gaussian noise
      - mode="no_memory" : 16 soft memory tokens masked out before entering Gemma 4 Layer 0
    """

    def __init__(self, decoder: nn.Module, memory: nn.Module, num_memories: int = 16):
        self.decoder = decoder
        self.memory = memory
        self.num_memories = num_memories
        self.mode = "true"
        self.current_batch_labels: torch.Tensor | None = None
        self.last_mem_out: torch.Tensor | None = None
        self.last_mem_mask: torch.Tensor | None = None
        self.last_per_example_loss: torch.Tensor | None = None
        self.last_per_example_tokens: torch.Tensor | None = None
        self.last_batch_token_loss: float = float("nan")
        self.last_captured_inputs_embeds: torch.Tensor | None = None
        self.last_captured_attention_mask: torch.Tensor | None = None
        self.last_captured_labels: torch.Tensor | None = None
        self.norm_stats: dict[str, float] = {}

    def _transform_doc_inputs(self, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        if self.mode == "true":
            return args, kwargs

        def _roll_or_noise_doc(obj: Any, is_mask: bool = False) -> Any:
            if isinstance(obj, list) and len(obj) > 1:
                if self.mode == "shuffled":
                    return obj[1:] + obj[:1]
                if self.mode == "noise" and not is_mask:
                    out_list = []
                    for t in obj:
                        if isinstance(t, torch.Tensor) and t.is_floating_point():
                            std = t.float().std().clamp(min=1e-4)
                            mean = t.float().mean()
                            out_list.append((torch.randn_like(t.float()) * std + mean).to(t.dtype))
                        else:
                            out_list.append(t)
                    return out_list
            elif isinstance(obj, torch.Tensor) and obj.dim() >= 2 and obj.shape[0] > 1:
                if self.mode == "shuffled":
                    return torch.roll(obj, shifts=1, dims=0)
                if self.mode == "noise" and not is_mask and obj.is_floating_point():
                    std = obj.float().std().clamp(min=1e-4)
                    mean = obj.float().mean()
                    return (torch.randn_like(obj.float()) * std + mean).to(obj.dtype)
            return obj

        new_args = list(args)
        new_kwargs = dict(kwargs)

        # First positional argument in MemoryModule.forward is always document embeddings
        if len(new_args) >= 1:
            new_args[0] = _roll_or_noise_doc(new_args[0], is_mask=False)
        # If a 3rd positional arg is a document mask, roll it alongside doc_embeds
        if len(new_args) >= 3 and self.mode == "shuffled":
            new_args[2] = _roll_or_noise_doc(new_args[2], is_mask=True)

        for k in list(new_kwargs.keys()):
            kl = k.lower()
            if any(tag in kl for tag in ("doc", "context", "passage", "memory_input")):
                new_kwargs[k] = _roll_or_noise_doc(new_kwargs[k], is_mask=("mask" in kl))
            elif "mask" in kl and self.mode == "shuffled":
                new_kwargs[k] = _roll_or_noise_doc(new_kwargs[k], is_mask=True)

        return tuple(new_args), new_kwargs

    @staticmethod
    def _locate_memory_positions(inputs_embeds: torch.Tensor, mem_out: torch.Tensor | None, num_memories: int) -> torch.Tensor:
        """
        Finds the exact boolean mask `[B, Seq]` where `memorilla` placed the `num_memories`
        soft memory tokens inside `inputs_embeds` (which sit after the chat prefix rather than 0..15).
        """
        B, S, D = inputs_embeds.shape
        mem_pos_mask = torch.zeros((B, S), dtype=torch.bool, device=inputs_embeds.device)
        if isinstance(mem_out, torch.Tensor) and mem_out.shape[0] == B and mem_out.shape[-1] == D:
            # Match the first memory slot vector per batch item against inputs_embeds
            first_slot = mem_out[:, 0:1, :].float()  # [B, 1, D]
            diffs = (inputs_embeds.float() - first_slot).abs().amax(dim=-1)  # [B, S]
            start_indices = torch.argmin(diffs, dim=-1)  # [B]
            for b in range(B):
                st = int(start_indices[b].item())
                if diffs[b, st].item() < 1e-2 and st + num_memories <= S:
                    mem_pos_mask[b, st : st + num_memories] = True
                else:
                    # Fallback: top-M highest L2-norm positions if exact match wasn't contiguous
                    norms = inputs_embeds[b].float().norm(dim=-1)
                    top_idx = torch.topk(norms, k=min(num_memories, S), largest=True).indices
                    mem_pos_mask[b, top_idx] = True
        return mem_pos_mask

    @contextlib.contextmanager
    def active(self, mode: str):
        self.mode = mode
        orig_mem_fwd = self.memory.forward

        def _hooked_mem_fwd(*args, **kwargs):
            import inspect
            sig = inspect.signature(orig_mem_fwd)
            bound = sig.bind_partial(*args, **kwargs)

            def _is_question_param(name: str, val: torch.Tensor) -> bool:
                n = name.lower()
                if "question" in n or "query" in n or n in ("q", "q_embeds", "query_embeds"):
                    return True
                if val.dim() == 2 and val.dtype.is_floating_point:
                    return True
                return False

            def _transform_param(name: str, x):
                if not torch.is_tensor(x) or x.dim() < 2:
                    return x
                if _is_question_param(name, x):
                    return x  # Keep question_embeds fixed to the current example!
                if self.mode == "shuffled":
                    # Roll BOTH doc_embeds (float) AND doc_mask (bool/int) together across batch dim 0
                    if x.size(0) > 1:
                        return torch.roll(x, shifts=1, dims=0)
                    elif x.dtype.is_floating_point:
                        return torch.roll(x, shifts=1, dims=1) if x.size(1) > 1 else x.flip(-1)
                    return x
                elif self.mode == "noise":
                    if x.dtype.is_floating_point:
                        return torch.randn_like(x) * x.std(dim=-1, keepdim=True).clamp(min=1e-4) + x.mean(dim=-1, keepdim=True)
                    return x
                return x

            if self.mode in ("shuffled", "noise"):
                for k, v in list(bound.arguments.items()):
                    bound.arguments[k] = _transform_param(k, v)
                t_args, t_kwargs = bound.args, bound.kwargs
            else:
                t_args, t_kwargs = args, kwargs

            mem_out = orig_mem_fwd(*t_args, **t_kwargs)
            # Fallback: if doc_embeds was a packed 2D tensor with cu_seqlens where batch dim wasn't rolled,
            # ensure shuffled/noise mode also perturbs the resulting [B, 16, 2816] memory tokens
            if isinstance(mem_out, torch.Tensor) and mem_out.dim() == 3 and mem_out.shape[0] > 1:
                if self.mode == "shuffled" and len(args) > 0 and isinstance(args[0], torch.Tensor) and args[0].dim() == 2:
                    mem_out = torch.roll(mem_out, shifts=1, dims=0)
            if isinstance(mem_out, torch.Tensor):
                self.last_mem_out = mem_out.detach()
            return mem_out

        def _hooked_dec_fwd(*args, **kwargs):
            inputs_embeds = kwargs.get("inputs_embeds")
            attention_mask = kwargs.get("attention_mask")
            raw_labels = kwargs.get("labels")
            if raw_labels is None:
                raw_labels = self.current_batch_labels
            M = self.num_memories

            aligned_labels: torch.Tensor | None = None
            if isinstance(inputs_embeds, torch.Tensor) and isinstance(raw_labels, torch.Tensor):
                raw_labels = raw_labels.to(inputs_embeds.device)
                if raw_labels.shape[1] + M == inputs_embeds.shape[1]:
                    pad = torch.full(
                        (raw_labels.shape[0], M),
                        -100,
                        dtype=raw_labels.dtype,
                        device=raw_labels.device,
                    )
                    aligned_labels = torch.cat([pad, raw_labels], dim=1)
                else:
                    aligned_labels = raw_labels

            if isinstance(inputs_embeds, torch.Tensor):
                mem_pos_mask = self._locate_memory_positions(inputs_embeds, self.last_mem_out, M)
                self.last_mem_mask = mem_pos_mask.detach()
                self.last_captured_inputs_embeds = inputs_embeds.detach()
                if isinstance(attention_mask, torch.Tensor):
                    self.last_captured_attention_mask = attention_mask.detach()
                if isinstance(aligned_labels, torch.Tensor):
                    self.last_captured_labels = aligned_labels.detach()

                # Record exact memory-token (from MemoryModule) vs pure text-token norm audit on first True pass
                if not self.norm_stats and self.mode == "true" and self.last_mem_out is not None:
                    mem_f = self.last_mem_out.float()
                    mem_l2 = mem_f.norm(dim=-1).mean().item()
                    mem_std = mem_f.std().item()
                    mem_unscaled_l2 = mem_l2 / math.sqrt(2816.0)

                    txt_mask = ~mem_pos_mask
                    if isinstance(attention_mask, torch.Tensor) and attention_mask.shape == mem_pos_mask.shape:
                        txt_mask = txt_mask & attention_mask.bool()
                    valid_txt = inputs_embeds.float()[txt_mask]
                    txt_l2 = valid_txt.norm(dim=-1).mean().item() if valid_txt.numel() > 0 else 1.0
                    txt_std = valid_txt.std().item() if valid_txt.numel() > 0 else 1.0
                    self.norm_stats = {
                        "memory_token_l2_norm": round(mem_l2, 2),
                        "memory_token_unscaled_l2_norm": round(mem_unscaled_l2, 2),
                        "memory_token_elem_std": round(mem_std, 4),
                        "text_token_l2_norm": round(txt_l2, 2),
                        "text_token_elem_std": round(txt_std, 4),
                        "l2_norm_ratio_mem_over_text": round(mem_l2 / max(1e-6, txt_l2), 2),
                        "l2_norm_ratio_unscaled_over_text": round(mem_unscaled_l2 / max(1e-6, txt_l2), 2),
                    }

            eval_labels = aligned_labels
            # Mask out the exact 16 memory token positions in "no_memory" mode (preserving chat prefix!)
            if self.mode == "no_memory" and isinstance(inputs_embeds, torch.Tensor) and self.last_mem_mask is not None:
                kwargs = dict(kwargs)
                masked_embeds = inputs_embeds.clone()
                masked_embeds[self.last_mem_mask] = 0.0
                kwargs["inputs_embeds"] = masked_embeds
                if isinstance(attention_mask, torch.Tensor) and attention_mask.shape == self.last_mem_mask.shape:
                    masked_attn = attention_mask.clone()
                    masked_attn[self.last_mem_mask] = 0
                    kwargs["attention_mask"] = masked_attn

            out = orig_dec_fwd(*args, **kwargs)

            # Compute exact per-example and token-weighted cross-entropy loss from logits & eval_labels
            logits = getattr(out, "logits", None)
            if isinstance(logits, torch.Tensor) and isinstance(eval_labels, torch.Tensor):
                min_seq = min(logits.shape[1], eval_labels.shape[1])
                shift_logits = logits[:, : min_seq - 1, :].float()
                shift_labels = eval_labels[:, 1:min_seq]
                B, S, V = shift_logits.shape
                token_losses = F.cross_entropy(
                    shift_logits.reshape(-1, V),
                    shift_labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                ).view(B, S)
                valid_mask = (shift_labels != -100).float()
                tokens_per_ex = valid_mask.sum(dim=1).clamp(min=1.0)
                loss_per_ex = (token_losses * valid_mask).sum(dim=1) / tokens_per_ex
                total_valid_tokens = valid_mask.sum().clamp(min=1.0)
                batch_token_loss = ((token_losses * valid_mask).sum() / total_valid_tokens).item()

                self.last_per_example_loss = loss_per_ex.detach().cpu()
                self.last_per_example_tokens = valid_mask.sum(dim=1).detach().cpu()
                self.last_batch_token_loss = batch_token_loss
            elif hasattr(out, "loss") and out.loss is not None:
                val = float(out.loss.detach().cpu().item())
                self.last_batch_token_loss = val
                self.last_per_example_loss = torch.tensor([val])
                self.last_per_example_tokens = torch.tensor([1.0])

            return out

        self.memory.forward = _hooked_mem_fwd
        self.decoder.forward = _hooked_dec_fwd
        try:
            yield
        finally:
            self.memory.forward = orig_mem_fwd
            self.decoder.forward = orig_dec_fwd


def run_batch_step(pl_module: nn.Module, batch: Any) -> None:
    """
    Executes a single forward pass on `pl_module` using `validation_step` or `training_step`
    under bfloat16 autocast.
    """
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        if hasattr(pl_module, "validation_step"):
            try:
                pl_module.validation_step(batch, 0)
                return
            except Exception as exc:
                if not getattr(run_batch_step, "_warned_val", False):
                    print(f"  [Note] validation_step fallback to training_step ({exc})")
                    run_batch_step._warned_val = True
        if hasattr(pl_module, "training_step"):
            pl_module.training_step(batch, 0)
        else:
            pl_module(batch)


@torch.inference_mode()
def greedy_decode_from_embeds(
    decoder: nn.Module,
    tokenizer: Any,
    prompt_embeds: torch.Tensor,
    prompt_attn_mask: torch.Tensor | None = None,
    max_new_tokens: int = 28,
) -> str:
    """
    Runs KV-cached greedy decoding directly from `inputs_embeds` (`[1, Seq, 2816]`)
    on `load_gemma4_text_decoder` without needing vLLM.
    """
    device = prompt_embeds.device
    embed_layer = decoder.get_input_embeddings()
    eos_id = getattr(tokenizer, "eos_token_id", None)

    # Temporarily enable KV caching on decoder config
    cfg = getattr(decoder, "config", None)
    orig_cache = getattr(cfg, "use_cache", False) if cfg else False
    if cfg is not None:
        cfg.use_cache = True
        if hasattr(cfg, "text_config"):
            cfg.text_config.use_cache = True

    generated_ids: list[int] = []
    past_key_values = None
    cur_embeds = prompt_embeds
    seq_len = prompt_embeds.shape[1]
    if prompt_attn_mask is not None and prompt_attn_mask.shape[1] == seq_len:
        attn_mask = prompt_attn_mask.to(dtype=torch.long, device=device)
    else:
        attn_mask = torch.ones((1, seq_len), dtype=torch.long, device=device)

    try:
        for _ in range(max_new_tokens):
            out = decoder(
                inputs_embeds=cur_embeds,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            next_token_logits = out.logits[:, -1, :]
            next_id = int(torch.argmax(next_token_logits, dim=-1).item())
            if eos_id is not None and next_id == eos_id:
                break
            generated_ids.append(next_id)
            past_key_values = getattr(out, "past_key_values", None)

            next_id_tensor = torch.tensor([[next_id]], dtype=torch.long, device=device)
            cur_embeds = embed_layer(next_id_tensor)
            attn_mask = torch.cat(
                [attn_mask, torch.ones((1, 1), dtype=torch.long, device=device)], dim=1
            )
    finally:
        if cfg is not None:
            cfg.use_cache = orig_cache
            if hasattr(cfg, "text_config"):
                cfg.text_config.use_cache = orig_cache

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def evaluate_checkpoint_controls(
    ckpt_label: str,
    ckpt_path: Path,
    pl_module: nn.Module,
    decoder: nn.Module,
    memory: nn.Module,
    per_ds_loaders: dict[str, tuple[str, DataLoader]],
    tokenizer: Any,
    max_batches_per_ds: int = 4,
    num_gen_samples: int = 4,
) -> dict[str, Any]:
    print(f"\n{'=' * 92}")
    print(f" EVALUATING CHECKPOINT: {ckpt_label} ({ckpt_path})")
    print(f"{'=' * 92}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    load_memory_weights_into_module(memory, ckpt_path)
    memory.to(device=device, dtype=torch.bfloat16)
    pl_module.eval()
    decoder.eval()
    memory.eval()

    interceptor = ControlInterceptor(decoder=decoder, memory=memory, num_memories=16)

    dataset_results: dict[str, dict[str, Any]] = {}
    generation_samples: list[dict[str, str]] = []

    total_tokens_all = 0.0
    weighted_sum = {"true": 0.0, "shuffled": 0.0, "noise": 0.0, "no_memory": 0.0}
    total_examples_all = 0
    total_wins_vs_shuffled = 0
    total_wins_vs_nomem = 0

    for ds_name, (split_tag, loader) in per_ds_loaders.items():
        ds_tokens = 0.0
        ds_loss_sums = {"true": 0.0, "shuffled": 0.0, "noise": 0.0, "no_memory": 0.0}
        ds_examples = 0
        ds_wins_shuffled = 0
        ds_wins_nomem = 0
        t_ds = time.monotonic()

        for b_idx, raw_batch in enumerate(loader):
            if b_idx >= max_batches_per_ds:
                break
            batch = move_batch_to_device(raw_batch, device)
            interceptor.current_batch_labels = batch.get("labels") if isinstance(batch, dict) else None

            batch_ex_losses: dict[str, torch.Tensor] = {}
            batch_ex_tokens: torch.Tensor | None = None
            captured_true_embeds: torch.Tensor | None = None
            captured_shuf_embeds: torch.Tensor | None = None
            captured_mem_mask: torch.Tensor | None = None
            captured_labels: torch.Tensor | None = None

            for mode in ("true", "shuffled", "noise", "no_memory"):
                with interceptor.active(mode):
                    run_batch_step(pl_module, batch)
                if interceptor.last_per_example_loss is not None and interceptor.last_per_example_tokens is not None:
                    batch_ex_losses[mode] = interceptor.last_per_example_loss
                    if mode == "true":
                        batch_ex_tokens = interceptor.last_per_example_tokens
                        captured_true_embeds = interceptor.last_captured_inputs_embeds
                        captured_mem_mask = interceptor.last_mem_mask
                        captured_labels = interceptor.last_captured_labels
                    elif mode == "shuffled":
                        captured_shuf_embeds = interceptor.last_captured_inputs_embeds

            if batch_ex_tokens is None or "true" not in batch_ex_losses:
                continue

            n_tok = float(batch_ex_tokens.sum().item())
            ds_tokens += n_tok
            total_tokens_all += n_tok

            for mode in ("true", "shuffled", "noise", "no_memory"):
                if mode in batch_ex_losses:
                    tok_weighted = float((batch_ex_losses[mode] * batch_ex_tokens).sum().item())
                    ds_loss_sums[mode] += tok_weighted
                    weighted_sum[mode] += tok_weighted

            # Per-example head-to-head win rate (True Docs < Shuffled Docs)
            true_vec = batch_ex_losses["true"]
            shuf_vec = batch_ex_losses["shuffled"]
            nomem_vec = batch_ex_losses["no_memory"]
            b_size = int(true_vec.numel())
            ds_examples += b_size
            total_examples_all += b_size

            wins_s = int((true_vec < shuf_vec).sum().item())
            wins_n = int((true_vec < nomem_vec).sum().item())
            ds_wins_shuffled += wins_s
            ds_wins_nomem += wins_n
            total_wins_vs_shuffled += wins_s
            total_wins_vs_nomem += wins_n

            # Run a side-by-side generation sample on the first batch of selected datasets
            if (
                b_idx == 0
                and len(generation_samples) < num_gen_samples
                and captured_true_embeds is not None
                and captured_shuf_embeds is not None
                and captured_labels is not None
                and tokenizer is not None
            ):
                lbl_0 = captured_labels[0]
                valid_pos = torch.nonzero(lbl_0 != -100, as_tuple=False).flatten()
                if valid_pos.numel() > 0:
                    ans_start = int(valid_pos[0].item())
                    gt_ids = lbl_0[valid_pos].tolist()
                    gt_text = tokenizer.decode(gt_ids, skip_special_tokens=True).strip()

                    true_prompt = captured_true_embeds[0:1, :ans_start, :]
                    shuf_prompt = captured_shuf_embeds[0:1, :ans_start, :]
                    nomem_prompt = true_prompt.clone()
                    nomem_mask = torch.ones((1, ans_start), dtype=torch.long, device=true_prompt.device)
                    if captured_mem_mask is not None:
                        m0 = captured_mem_mask[0:1, :ans_start]
                        nomem_prompt[m0] = 0.0
                        nomem_mask[m0] = 0

                    gen_true = greedy_decode_from_embeds(decoder, tokenizer, true_prompt, max_new_tokens=24)
                    gen_shuf = greedy_decode_from_embeds(decoder, tokenizer, shuf_prompt, max_new_tokens=24)
                    gen_nomem = greedy_decode_from_embeds(
                        decoder, tokenizer, nomem_prompt, prompt_attn_mask=nomem_mask, max_new_tokens=24
                    )

                    generation_samples.append(
                        {
                            "dataset": ds_name,
                            "ground_truth": gt_text[:120],
                            "gen_true_docs": gen_true[:120],
                            "gen_shuffled_docs": gen_shuf[:120],
                            "gen_no_memory": gen_nomem[:120],
                            "ex0_loss_true": round(float(true_vec[0].item()), 4),
                            "ex0_loss_shuffled": round(float(shuf_vec[0].item()), 4),
                        }
                    )

        if ds_tokens > 0:
            l_true = ds_loss_sums["true"] / ds_tokens
            l_shuf = ds_loss_sums["shuffled"] / ds_tokens
            l_noise = ds_loss_sums["noise"] / ds_tokens
            l_nomem = ds_loss_sums["no_memory"] / ds_tokens
            recall_delta = l_shuf - l_true
            format_delta = l_nomem - l_shuf
            win_pct = (ds_wins_shuffled / max(1, ds_examples)) * 100.0
            dt = time.monotonic() - t_ds

            dataset_results[ds_name] = {
                "split": split_tag,
                "examples": ds_examples,
                "answer_tokens": int(ds_tokens),
                "loss_true_docs": round(l_true, 4),
                "ppl_true_docs": round(math.exp(min(20.0, l_true)), 2),
                "loss_shuffled_docs": round(l_shuf, 4),
                "loss_gaussian_noise": round(l_noise, 4),
                "loss_no_memory": round(l_nomem, 4),
                "recall_gain_vs_shuffled": round(recall_delta, 4),
                "format_gain_vs_nomem": round(format_delta, 4),
                "win_rate_vs_shuffled_pct": round(win_pct, 1),
                "elapsed_s": round(dt, 1),
            }
            print(
                f"  [{ds_name:11s} | {split_tag:13s}] "
                f"True: {l_true:.4f} ({math.exp(min(20.0, l_true)):5.2f}x) | "
                f"Shuf: {l_shuf:.4f} | Noise: {l_noise:.4f} | NoMem: {l_nomem:.4f} | "
                f"Recall Δ: {recall_delta:+.4f} | WinRate: {win_pct:5.1f}% ({ds_examples} ex, {dt:.1f}s)"
            )

    overall_true = weighted_sum["true"] / max(1.0, total_tokens_all)
    overall_shuf = weighted_sum["shuffled"] / max(1.0, total_tokens_all)
    overall_noise = weighted_sum["noise"] / max(1.0, total_tokens_all)
    overall_nomem = weighted_sum["no_memory"] / max(1.0, total_tokens_all)
    overall_win_pct = (total_wins_vs_shuffled / max(1, total_examples_all)) * 100.0

    # Print Norm Audit
    if interceptor.norm_stats:
        ns = interceptor.norm_stats
        print(f"\n  [Layer-0 Embedding & Memory Norm Audit]")
        print(
            f"    Memory Tokens (with √2816) : L2 Norm = {ns['memory_token_l2_norm']:.2f} | "
            f"Elem Std = {ns['memory_token_elem_std']:.4f} | Ratio = {ns['l2_norm_ratio_mem_over_text']}x"
        )
        print(
            f"    Memory Tokens (unscaled 1x): L2 Norm = {ns['memory_token_unscaled_l2_norm']:.2f} | "
            f"Ratio (Unscaled/Text) = {ns['l2_norm_ratio_unscaled_over_text']}x"
        )
        print(
            f"    Gemma 4 Text Embeddings    : L2 Norm = {ns['text_token_l2_norm']:.2f} | "
            f"Elem Std = {ns['text_token_elem_std']:.4f}"
        )

    # Print Overall Summary
    print(f"\n  [Overall Control Summary Across {total_examples_all} Examples ({int(total_tokens_all)} Answer Tokens)]")
    print(f"    1. True Docs Loss      : {overall_true:.4f} (PPL: {math.exp(min(20.0, overall_true)):.2f}x)")
    print(f"    2. Shuffled Docs Loss  : {overall_shuf:.4f} (PPL: {math.exp(min(20.0, overall_shuf)):.2f}x)")
    print(f"    3. Gaussian Noise Loss : {overall_noise:.4f} (PPL: {math.exp(min(20.0, overall_noise)):.2f}x)")
    print(f"    4. No Memory (0 Slots) : {overall_nomem:.4f} (PPL: {math.exp(min(20.0, overall_nomem)):.2f}x)")
    print(
        f"    => Pure Document Recall Gain (Shuffled - True) : {overall_shuf - overall_true:+.4f} "
        f"(Per-Example Recall Win Rate: {overall_win_pct:.1f}%)"
    )
    print(
        f"    => Task Formatting Gain (NoMemory - Shuffled)  : {overall_nomem - overall_shuf:+.4f}"
    )

    if generation_samples:
        print(f"\n  [Live KV-Cached Autoregressive Generation Samples]")
        for s in generation_samples:
            print(f"    Dataset: {s['dataset']} (Ex0 Loss True={s['ex0_loss_true']} vs Shuf={s['ex0_loss_shuffled']})")
            print(f"      Ground Truth  : {s['ground_truth']}")
            print(f"      True Memory   : {s['gen_true_docs']}")
            print(f"      Shuffled Docs : {s['gen_shuffled_docs']}")
            print(f"      No Memory     : {s['gen_no_memory']}")

    return {
        "checkpoint": ckpt_label,
        "path": str(ckpt_path),
        "norm_audit": interceptor.norm_stats,
        "overall": {
            "total_examples": total_examples_all,
            "total_answer_tokens": int(total_tokens_all),
            "loss_true_docs": round(overall_true, 4),
            "loss_shuffled_docs": round(overall_shuf, 4),
            "loss_gaussian_noise": round(overall_noise, 4),
            "loss_no_memory": round(overall_nomem, 4),
            "pure_recall_gain": round(overall_shuf - overall_true, 4),
            "task_format_gain": round(overall_nomem - overall_shuf, 4),
            "win_rate_vs_shuffled_pct": round(overall_win_pct, 1),
        },
        "per_dataset": dataset_results,
        "generation_samples": generation_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Run Memorilla + Gemma 4 26B Pre-Eval & Retrieval Controls")
    parser.add_argument("--config", type=str, default="configs/stage3_gemma4_10pct.yaml")
    parser.add_argument("--checkpoints", nargs="*", default=None, help="Explicit memory.pt paths to evaluate")
    parser.add_argument("--compare_epochs", action="store_true", help="Evaluate all saved Stage 3 epochs in one pass")
    parser.add_argument("--batches_per_dataset", type=int, default=4, help="Number of 8-example batches per dataset")
    parser.add_argument("--fast", action="store_true", help="Quick smoke-test mode (1 batch per dataset, ~2.5 mins)")
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--output_json", type=str, default="checkpoints/stage3_gemma4/pre_eval_report.json")
    args = parser.parse_args()

    if args.fast:
        args.batches_per_dataset = 1

    discovered = discover_default_checkpoints()
    if args.checkpoints:
        ckpts_to_run = [(Path(p).parent.name or Path(p).stem, Path(p)) for p in args.checkpoints]
    elif args.compare_epochs:
        ckpts_to_run = [item for item in discovered if item[0].startswith("epoch-") or item[0] == "resume_latest"]
    else:
        # Default: evaluate the newest epoch-* checkpoint (and resume_latest if newer)
        epoch_ckpts = [item for item in discovered if item[0].startswith("epoch-")]
        ckpts_to_run = [epoch_ckpts[-1]] if epoch_ckpts else discovered[-1:]
        if any(item[0] == "resume_latest" for item in discovered):
            latest_resume = [item for item in discovered if item[0] == "resume_latest"][-1]
            if not ckpts_to_run or ckpts_to_run[-1][1] != latest_resume[1]:
                ckpts_to_run.append(latest_resume)

    if not ckpts_to_run:
        raise FileNotFoundError("No memory.pt checkpoints found in checkpoints/stage3_gemma4.")

    print("[PreEval] Checkpoints queued for evaluation:")
    for label, path in ckpts_to_run:
        print(f"  - {label:15s}: {path}")

    # Resolve a valid existing memory.pt for train.py's initial _warm_start
    init_mem = ckpts_to_run[0][1]
    stage2_mem = Path("checkpoints/stage2_gemma4/memory.pt")
    if stage2_mem.exists():
        init_mem = stage2_mem

    t0 = time.monotonic()
    captured = capture_memorilla_components(
        args.config,
        init_memory_path=str(init_mem),
        max_length=args.max_length,
    )
    pl_module = captured["pl_module"]
    decoder, memory = find_decoder_and_memory(pl_module)

    tokenizer = getattr(pl_module, "tokenizer", None)
    if tokenizer is None:
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
        except Exception:
            tokenizer = None

    per_ds_loaders = extract_per_dataset_loaders(
        captured,
        batch_size=8,
        max_batches=args.batches_per_dataset,
    )
    print(f"[PreEval] Prepared individual evaluation loaders for {len(per_ds_loaders)} datasets: {list(per_ds_loaders.keys())}")

    all_reports = []
    for label, path in ckpts_to_run:
        rep = evaluate_checkpoint_controls(
            ckpt_label=label,
            ckpt_path=path,
            pl_module=pl_module,
            decoder=decoder,
            memory=memory,
            per_ds_loaders=per_ds_loaders,
            tokenizer=tokenizer,
            max_batches_per_ds=args.batches_per_dataset,
        )
        all_reports.append(rep)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_elapsed_s": round(time.monotonic() - t0, 1),
                "reports": all_reports,
            },
            f,
            indent=2,
        )
    print(f"\n[PreEval] Saved complete control report to {out_path} (Total time: {time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()