[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-md.svg)](https://huggingface.co/DataPhreak/Memorilla-Gemma4-26B-A4B)

# Memorilla for Gemma 4 on a Strix Halo

An unofficial port of Stanford SNAP's [Memorilla](https://github.com/snap-stanford/memorilla) memory module to a frozen, 4-bit **Gemma 4 26B-A4B** decoder, trained on a single AMD Strix Halo workstation with about a tenth of the original training data.

Memorilla gives a frozen LLM a memory. A document collection is embedded once, and a small trained module turns the question plus those embeddings into 16 soft tokens that go in front of the prompt. The LLM itself is never changed. Stanford trained their module for Qwen3-8B at full precision on four H100s.

This repo tests whether the idea holds up when three things change at once:

| | Stanford | This port |
| --- | --- | --- |
| Decoder | Qwen3-8B, bfloat16 | Gemma 4 26B-A4B (mixture of experts), routed experts quantized to 4 bits |
| Hardware | 4 x H100 80 GB | 1 x AMD Strix Halo (Radeon 8060S, 64 GB shared memory, Windows ROCm) |
| Training data | Full three-stage curriculum | 10% of each large training set, warm-started from Stanford's Stage 2 module |

It does hold up. Sixteen tokens raise the frozen 4-bit decoder's score on all eight of Stanford's benchmarks. The run also includes a control that Stanford's results table does not have, and that control changes how three of the eight results should be read. Both are below.

> This is an independent project. It is not affiliated with or endorsed by Stanford SNAP or Google.
> Weights are published on [Hugging Face](https://huggingface.co/DataPhreak/Memorilla-Gemma4-26B-A4B).

## Results

All numbers are for one checkpoint (Stage 3, last epoch) under Stanford's evaluation protocol: their per-benchmark prompts, greedy decoding, and their own metric code. Each benchmark is scored on 500 test rows drawn with a fixed seed. For TriviaQA (500 rows) and PubMedQA (100 rows) that is the whole test set.

Every row was run three ways:

- **Closed book.** The decoder alone, with no memory tokens, using upstream's closed-book baseline prompt.
- **Wrong documents.** Memory tokens built from the right question and *another collection's* documents (`shuffled` in the scripts).
- **Memory.** Memory tokens built from the right question and the right documents.

| Benchmark | Metric | Closed book | Wrong documents | Memory | Memory − wrong documents [95% CI] | Memory − closed book [95% CI] |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FactKG | accuracy | 67.0 | 69.4 | **79.8** | **+10.4** [6.0, 14.8] | +12.8 [8.6, 17.2] |
| TriviaQA | exact match | 54.0 | 56.2 | **63.2** | **+7.0** [4.2, 10.0] | +9.2 [5.8, 12.6] |
| LaMP-4 | ROUGE-L | 5.51 | 13.62 | **16.96** | **+3.34** [2.37, 4.37] | +11.45 [10.38, 12.53] |
| NarrativeQA | F1 | 7.07 | 14.20 | **16.62** | **+2.43** [0.84, 4.09] | +9.56 [7.60, 11.50] |
| LaMP-7 | ROUGE-L | 37.76 | 44.96 | **46.24** | +1.28 [−0.12, 2.72] | +8.47 [7.00, 9.92] |
| PV4 | accuracy | 61.4 | 65.8 | **67.4** | +1.6 [−2.4, 5.6] | +6.0 [1.0, 11.4] |
| PMv2 | accuracy | 28.0 | **34.6** | 33.8 | −0.8 [−5.0, 3.4] | +5.8 [1.4, 10.4] |
| PubMedQA | accuracy | 23.0 | **51.0** | 46.0 | −5.0 [−14.0, 4.0] | +23.0 [9.0, 36.0] |

Intervals are paired bootstrap over the same rows. A bold difference has an interval that excludes zero.

The raw outputs are in [`results/stanford_ep04`](./results/stanford_ep04). `summary.json` holds every number in these tables, and the per-row answer files for all three conditions let you re-score the run without generating anything: `python stanford_eval_gemma4.py score --output_dir results\stanford_ep04`.

**1. The tokens help everywhere.** Memory beats closed book on all eight benchmarks, and every one of those intervals excludes zero. The decoder is frozen and quantized; the only thing added is 16 input tokens. PubMedQA doubles and LaMP-4 triples.

**2. On four benchmarks the documents demonstrably matter.** FactKG, TriviaQA, LaMP-4 and NarrativeQA all score higher with the right documents than with someone else's. LaMP-7 is borderline: the gap is positive on F1 (+1.42 [0.08, 2.78]) but its ROUGE-L interval touches zero.

**3. On three benchmarks, wrong documents work as well as right ones.** On PV4, PMv2 and PubMedQA the gain over closed book is real, but this run cannot credit it to the documents. The wrong-document tokens still carry the right question, re-encoded by a module trained on these tasks. Whether the lift comes from that, from the answer format the tokens induce, or from something else is an open question. We have not run the experiment that would separate them (see [Open questions](#limitations-and-open-questions)).

One piece of the PubMedQA jump is format. Stanford's rule scores the first word of the answer, and 18% of closed-book answers did not open with yes, no or maybe. With memory tokens, every answer did. That cannot be the whole story: even if all 18% had been right, closed book would reach 41, still under both memory conditions.

### Compared with Stanford's Qwen3-8B results

Stanford's figures are from Table 11 of the [paper](https://arxiv.org/abs/2610.21375) (v3).

| Benchmark | Metric | Stanford closed book | Stanford Memorilla | Stanford gain | Ours closed book | Ours memory | Our gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TriviaQA | exact match | 36.00 | 57.00 | +21.00 | 54.0 | 63.2 | +9.2 |
| PV4 | accuracy | 58.30 | 66.65 | +8.35 | 61.4 | 67.4 | +6.0 |
| PV4, hard subset | accuracy | 31.96 | 41.97 | +10.01 | 41.67 | 48.61 | +6.94 |
| NarrativeQA | F1 | 9.87 | 16.56 | +6.69 | 7.07 | 16.62 | +9.56 |
| FactKG | accuracy | 65.97 | 84.33 | +18.36 | 67.0 | 79.8 | +12.8 |
| LaMP-7 | ROUGE-L | 43.13 | 48.37 | +5.24 | 37.76 | 46.24 | +8.47 |
| LaMP-4 | ROUGE-L | 12.24 | 18.15 | +5.91 | 5.51 | 16.96 | +11.45 |
| PubMedQA | accuracy | 21.00 | 59.00 | +38.00 | 23.0 | 46.0 | +23.0 |
| PMv2 | accuracy | 25.64 | 44.60 | +18.96 | 28.0 | 33.8 | +5.8 |

In absolute score this checkpoint is level with or above Stanford's on TriviaQA, PV4 and NarrativeQA, a few points below on FactKG, LaMP-4 and LaMP-7, and clearly below on PubMedQA and PMv2. Measured as gain over each decoder's own closed-book score, it gains more than Stanford's on NarrativeQA, LaMP-4 and LaMP-7 and less on the other five.

The two columns are not a like-for-like comparison:

- **Different decoders.** Gemma 4 26B-A4B knows more trivia than Qwen3-8B (54 against 36 closed book on TriviaQA), which is the likely reason the TriviaQA score here is higher. On the three free-text generation benchmarks it starts lower.
- **Sample size.** Our numbers are 500 rows per benchmark; Stanford's are full test sets. Expect about ±4 points of noise on our accuracies and ±10 on PubMedQA. The PV4 hard subset is only 72 of our rows.
- **Epoch choice.** We report the last Stage 3 epoch for every benchmark. Upstream's training guide recommends choosing the epoch per benchmark on validation data.
- **Personalization training.** Upstream's guide adds a further stage that continues training on PV4 or PMv2 alone, and upstream publishes separate `pv4/` and `pmv2/` checkpoints. This port has no such stage; its PV4 and PMv2 numbers come from the shared Stage 3 module.
- **Harness.** Upstream evaluates with vLLM. This setup has no vLLM, so generation here is a reimplementation of the same protocol on plain `transformers`. Scoring calls upstream's metric code unchanged, and our scoring encoder reproduces upstream's stored embeddings at cosine 0.9998. Stanford's own checkpoint has not been run through this harness.

### A control the original results should include

Stanford's Table 11 compares Memorilla with closed book, retrieval, full context and other memory systems. It has no row where the module is given the *wrong* documents. The nearest thing in the paper is one sentence in Appendix E.1: replacing the document embeddings with Gaussian noise costs Memorilla 5.93 points, which Appendix E.4 attributes to FactKG. It sits among ablations that use a smaller decoder, and no such check is reported for the other seven benchmarks.

Without that row, "gain over closed book" bundles two things: what the documents contribute, and everything else 16 trained tokens do to a frozen model. In this port the second part is the entire measured gain on PMv2 and PubMedQA, and more than two thirds of it on PV4, LaMP-4, NarrativeQA and LaMP-7:

| Benchmark | Gain over closed book | Of which survives with wrong documents |
| --- | ---: | ---: |
| FactKG | +12.8 | +2.4 |
| TriviaQA | +9.2 | +2.2 |
| LaMP-4 (ROUGE-L) | +11.45 | +8.11 |
| NarrativeQA (F1) | +9.56 | +7.13 |
| LaMP-7 (ROUGE-L) | +8.47 | +7.20 |
| PV4 | +6.0 | +4.4 |
| PMv2 | +5.8 | +6.6 |
| PubMedQA | +23.0 | +28.0 |

We do not know whether Stanford's module behaves the same way. It was trained on ten times the data, with a personalization stage, for a different decoder. But the question is cheap to answer on data-center hardware and slow on ours, so we are asking rather than running it: **the published Qwen3-8B checkpoints should be benchmarked with shuffled documents.** The evaluation script in this repo already supports that (see [Running Stanford's checkpoint](#running-stanfords-checkpoint-through-the-same-controls)). If you run it, please open an issue with the numbers.

### A second scorer for PMv2

Stanford scores PMv2 by embedding the free-text answer and taking the nearest of four answer choices. To check that this was not hiding a document effect, the same answers were re-scored by a judge: the plain frozen Gemma decoder, shown the answer and the choices but neither the question nor the reference, judged once per rotation of the choices.

| PMv2 | Stanford's scorer | Gemma judge |
| --- | ---: | ---: |
| Memory | 33.8 | 32.2 |
| Wrong documents | 34.6 | 34.4 |
| Memory − wrong documents [95% CI] | −0.8 [−5.0, 3.4] | −2.2 [−6.4, 1.8] |

The two scorers reach the same totals but pick the same choice on only 66–70% of rows, and only 21–24% of rows are right under both. Many PMv2 answers match none of the four choices well, so row-level scoring of this benchmark is noisy for any model. A blind re-read of 60 of the disagreement rows by a third model (Claude) sided with the judge on 36, with Stanford's scorer on 17, and with neither on 7.

## The checkpoint

<!-- TODO: add the Hugging Face link once the checkpoint is published. -->
Hugging Face: to be linked with the first release.

- **What it is.** A Memorilla memory module (about 190M parameters, 16 memory tokens) for `google/gemma-4-26B-A4B-it`, reading document embeddings from `Qwen/Qwen3-Embedding-4B`.
- **Which one.** Stage 3, `epoch-04`. `memory.pt` SHA-256: `1ffad0557d43796255a25c2a03620e4159542d8073b0627123dbafb605ff472a`.
- **Scale.** Gemma multiplies its token embeddings by √2816. The saved `memory.pt` has that factor folded into the output projection, so it produces correctly scaled tokens without this repo's patch. With the patch installed, the factor is divided back out on load and applied in a forward hook.
- **Trained against 4-bit weights.** The module only ever saw the decoder with its routed experts quantized to 4 bits. It has not been evaluated against the full-precision decoder.

## How the port works

Everything Gemma-specific lives in [`apply_gemma4_patch.py`](./apply_gemma4_patch.py), which patches upstream's code at import time. `train.py` and `evaluate.py` are upstream's files with a three-line import added.

1. **New output projection.** [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py) takes Stanford's Stage 2 module, keeps its 183M-parameter Perceiver, and replaces the 2560 → 4096 projection with a freshly initialized 2560 → 2816 one (7.2M parameters) for Gemma's hidden size.
2. **Embedding scale kept out of the optimizer.** The projection is initialized at ordinary variance and its output is multiplied by √2816 in a forward hook. Baking the factor into the weights would make AdamW's weight decay and step size behave very differently for that one layer.
3. **4-bit routed experts.** About 91% of Gemma 4 26B-A4B's weights sit in fused 3-D expert tensors that `bitsandbytes` skips. The patch quantizes them itself (uniform 4-bit, one scale per group of 64 weights, two weights packed per byte), which takes the experts from 42.5 GB to 11.3 GB. Attention, the shared experts and the embeddings stay in bfloat16. The vision tower is dropped.
4. **Batched expert dispatch.** Tokens are grouped by active expert and run through two batched matrix multiplies per layer in place of a Python loop over experts.
5. **Gradient checkpointing that works on a frozen decoder.** Upstream puts the decoder in eval mode, which disables Hugging Face's checkpointing. The patch checkpoints every decoder layer whenever gradients are on and turns the KV cache off during training.
6. **Chat-template fix.** Gemma 4's template appends an empty thought channel to generation prompts. Upstream's collator counted those four tokens as part of the prompt, which masked the first four tokens of every answer out of the loss. The patch strips the stub so labels line up.
7. **Pause and resume.** Checkpoints are written every 50 steps, every 15 minutes and on Ctrl+C, and training resumes mid-epoch from the last one.

### Memory use while training

| | Measured | Notes |
| --- | --- | --- |
| Decoder on GPU | 16.4 GB | 11.3 GB of 4-bit experts plus about 5.1 GB in bfloat16 |
| PyTorch peak | 23.8–25.8 GB | Decoder, memory module, optimizer state, one layer of dequantized experts |
| Windows dedicated GPU memory | 43.5 of 63.8 GB | |
| Windows shared GPU memory | 0.4 of 47.6 GB | No spill into system RAM |

The document encoder is not loaded during training; upstream ships precomputed embeddings.

## Training

| Stage | Data | What trains | Epochs | Learning rate | Batch × accumulation |
| --- | --- | --- | ---: | ---: | --- |
| 1 | `enwiki`, 10% | Projection only, Perceiver frozen | 1 | 5e-4 | 8 × 16 |
| 2 | Upstream's ten-dataset mixture, subsampled | Whole module | 1 | 1.5e-4 | 8 × 16 |
| 3 | The eight benchmarks' training sets, subsampled | Whole module | 5 | 1e-4 | 8 × 16 |

**Subsampling.** [`subsample_data.py`](./subsample_data.py) keeps `max(2,500 rows, 10%)` of each training file and leaves validation and test sets whole. Small training sets are therefore kept in full or nearly so. Stage 3 trained on 47,777 rows: FactKG 8,636, TriviaQA 15,910, NarrativeQA 12,331, PubMedQA 900, and 2,500 each for PV4, PMv2, LaMP-4 and LaMP-7.

**Warm start.** "A tenth of the data" describes what this run trained on. The Perceiver started from Stanford's fully trained Stage 2 weights, so the result also rests on their training.

Two things about this particular run that a reader should know:

- **Stages 1 and 2 were trained before the chat-template fix** (item 6 above), so the first four tokens of each answer were not supervised in those stages. Stage 3 was restarted from the Stage 2 checkpoint with the fix in place, and the published checkpoint comes from that clean run.
- **The clean Stage 3 run holds the memory module's weights in float32** under bfloat16 mixed precision, so that small updates are not rounded away.

## Run it yourself

Setup, environment variables and the three training commands are in [`example.bat.txt`](./example.bat.txt) (Windows PowerShell). In outline:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install --no-deps git+https://github.com/snap-stanford/memorilla.git
uv pip install -r requires.txt

python apply_gemma4_patch.py --memorilla_root .
python subsample_data.py --data_dir .\data_10pct --ratio 0.10 --seed 42
python prepare_gemma4_weights.py --output_dir .\checkpoints\gemma4_warmstart
# then Stages 1-3 with train.py, as in example.bat.txt
```

### Evaluation

Upstream's `evaluate.py` needs vLLM. Use [`stanford_eval_gemma4.py`](./stanford_eval_gemma4.py), which follows the same protocol on plain `transformers`, adds the two control conditions, and is resumable.

```powershell
# Generate answers: memory, wrong documents, closed book
python stanford_eval_gemma4.py generate --checkpoint .\checkpoints\stage3_gemma4\epoch-04 `
    --data_repo .\data_10pct --benchmark all --n 500 `
    --conditions memory shuffled closed_book --output_dir results\stanford_ep04

# Optional second scorer: the frozen decoder as judge
python stanford_eval_gemma4.py judge --output_dir results\stanford_ep04

# Score with upstream's metrics; writes summary.json
python stanford_eval_gemma4.py score --output_dir results\stanford_ep04
```

`--n 0` runs the full test sets.

### Running Stanford's checkpoint through the same controls

```powershell
hf download memorilla/Memorilla-Qwen3-8B --include "stage3/*" --local-dir .\checkpoints\memorilla_qwen3
python stanford_eval_gemma4.py generate --decoder Qwen/Qwen3-8B `
    --checkpoint .\checkpoints\memorilla_qwen3\stage3 `
    --data_repo .\data_10pct --benchmark all --n 0 `
    --conditions memory shuffled closed_book --output_dir results\stanford_qwen3
python stanford_eval_gemma4.py score --output_dir results\stanford_qwen3
```

We have not run this. The script was developed and used with the Gemma decoder; its Qwen path is only covered by small-model tests.

## Limitations and open questions

- **What the tokens carry when the documents are wrong.** The missing experiment is a fourth condition: tokens built from the wrong question, or from noise at the same norm. That would separate the question's contribution from the mere presence of trained tokens.
- **Personalization.** No document effect was measured on PV4 or PMv2. The run had 2,500 training rows for each, no personalization stage, and test users it never saw in training. With 500 test rows, an effect of a few points could also be present and undetected.
- **Sample sizes.** 500 rows per benchmark, one seed, one checkpoint.
- **Token scale.** The memory tokens this module produces have roughly 90 times the norm of Gemma's text embeddings. Training without the √2816 factor was not tried.
- **Full-precision decoder.** Untested.
- **No calibration run.** Stanford's checkpoint has not been evaluated in this harness, so differences between the two harnesses are not ruled out.

## Files

| File | Purpose |
| --- | --- |
| [`apply_gemma4_patch.py`](./apply_gemma4_patch.py) | Gemma 4 loader, 4-bit experts, scale hooks, chat-template fix, save and resume |
| [`prepare_gemma4_weights.py`](./prepare_gemma4_weights.py) | Builds the warm-start checkpoint from Stanford's Stage 2 module |
| [`subsample_data.py`](./subsample_data.py) | Subsamples training data and aligns the precomputed embeddings |
| [`configs/`](./configs) | Stage 1–3 recipes |
| [`example.bat.txt`](./example.bat.txt) | End-to-end PowerShell workflow |
| [`stanford_eval_gemma4.py`](./stanford_eval_gemma4.py) | Stanford-protocol evaluation with controls and the judge |
| [`results/stanford_ep04/`](./results/stanford_ep04) | Outputs of the published evaluation: summary, run settings, per-row answers, judge verdicts |
| [`final_eval_gemma4.py`](./final_eval_gemma4.py) | Earlier likelihood-based evaluation used during training |
| [`train.py`](./train.py), [`evaluate.py`](./evaluate.py) | Upstream entry points with the patch import added |
| [`memorilla_studio.html`](./memorilla_studio.html) | Training dashboard |

## Credits and license

Memorilla is the work of Fallahpour, Idehpour, Kothapalli, Mounier, Wu, Pardis and Leskovec at Stanford SNAP. Their [code](https://github.com/snap-stanford/memorilla) is released under Apache 2.0, and `train.py` and `evaluate.py` here are modified copies of their files. Their [data and checkpoints](https://huggingface.co/memorilla) are on Hugging Face; each dataset keeps its source license. The per-row files under `results/` quote test questions and reference answers from those datasets and stay under the same licenses. [Gemma 4 26B-A4B](https://huggingface.co/google/gemma-4-26B-A4B-it) and [Qwen3-Embedding-4B](https://huggingface.co/Qwen/Qwen3-Embedding-4B) are used under the licenses on their model pages.

```bibtex
@article{fallahpour2026memorilla,
  title={Memorilla: Latent Semantic Memory for LLMs},
  author={Fallahpour, Adibvafa and Idehpour, Parsa and Kothapalli, Vignesh and Mounier, Nikita and Wu, Shirley and Pardis, Shayan and Leskovec, Jure},
  journal={arXiv preprint arXiv:2610.21375},
  year={2026}
}
```
