---
license: apache-2.0
language:
- en
base_model:
- memorilla/Memorilla-Qwen3-8B
base_model_relation: finetune
datasets:
- memorilla/Memorilla-Data
tags:
- memory
- long-context
- memorilla
- soft-prompt
- gemma4
- arxiv:2610.21375
---

# Memorilla-Gemma4-26B-A4B

A [Memorilla](https://github.com/snap-stanford/memorilla) memory module for a frozen **Gemma 4 26B-A4B** decoder. It turns a question and a collection of document embeddings into 16 soft tokens that are placed in front of the prompt. The decoder's own weights are never changed.

This is an unofficial port of Stanford SNAP's work. Their module was trained for Qwen3-8B at full precision on four H100s. This one was trained against Gemma 4 with its routed experts quantized to 4 bits, on a single AMD Strix Halo workstation, with about a tenth of the training data.

Sixteen tokens raise the frozen 4-bit decoder's score on all eight of Stanford's benchmarks. On four of them the gain demonstrably comes from the documents; on three, tokens built from the wrong documents work just as well. The results section shows both.

Code, training recipe and full write-up: [DataBassGit/Memorilla-Gemma4-StrixHalo](https://github.com/DataBassGit/Memorilla-Gemma4-StrixHalo).

> Independent project. Not affiliated with or endorsed by Stanford SNAP or Google.

## At a glance

| | |
| --- | --- |
| What this repo holds | `memory.pt` and `config.json` for one memory module (about 190M parameters) |
| Decoder it was trained for | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it), frozen, routed experts in 4-bit |
| Document and question encoder | [`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B), frozen |
| Output | 16 memory tokens of width 2816 per question |
| Started from | The Stage 2 module of [`memorilla/Memorilla-Qwen3-8B`](https://huggingface.co/memorilla/Memorilla-Qwen3-8B), with a new output projection |
| Checkpoint | Stage 3, `epoch-04` |
| `memory.pt` SHA-256 | `1ffad0557d43796255a25c2a03620e4159542d8073b0627123dbafb605ff472a` |

The decoder and the encoder are not included. Download them from their own pages.

## Results

One checkpoint, evaluated under Stanford's protocol: their per-benchmark prompts, greedy decoding, and their metric code. Each benchmark is scored on 500 test rows drawn with a fixed seed; for TriviaQA (500 rows) and PubMedQA (100 rows) that is the whole test set.

Every row was run three ways:

- **Closed book.** The decoder alone, with no memory tokens.
- **Wrong documents.** Memory tokens built from the right question and another collection's documents.
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

- **The tokens help everywhere.** Memory beats closed book on all eight benchmarks, with every interval clear of zero. PubMedQA doubles and LaMP-4 triples, on a frozen, quantized decoder.
- **On four benchmarks the documents demonstrably matter.** FactKG, TriviaQA, LaMP-4 and NarrativeQA score higher with the right documents than with someone else's. LaMP-7 is borderline.
- **On three benchmarks, wrong documents work as well as right ones.** On PV4, PMv2 and PubMedQA the gain over closed book is real but cannot be credited to the documents. The wrong-document tokens still carry the right question. What produces the lift there is an open question.

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

This is not a like-for-like comparison. The decoders differ, our numbers are 500-row samples against their full test sets (about ±4 points on accuracies, ±10 on PubMedQA, and only 72 rows in the PV4 hard subset), we report the last epoch where upstream recommends choosing per benchmark, upstream has a personalization stage this port lacks, and our generation loop reimplements their protocol without vLLM. The [GitHub README](https://github.com/DataBassGit/Memorilla-Gemma4-StrixHalo#compared-with-stanfords-qwen3-8b-results) goes through each point.

### How much of the gain needs the right documents

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

Stanford's results table has no wrong-document row, so for their checkpoints this split is unknown. The paper's nearest check is one appendix sentence about replacing document embeddings with Gaussian noise on FactKG. We think the published Qwen3-8B checkpoints should be benchmarked with shuffled documents, and the evaluation script in the GitHub repo can do it.

### A second scorer for PMv2

Stanford scores PMv2 by embedding the answer and taking the nearest of four choices. The same answers were re-scored by the plain frozen Gemma decoder acting as judge.

| PMv2 | Stanford's scorer | Gemma judge |
| --- | ---: | ---: |
| Memory | 33.8 | 32.2 |
| Wrong documents | 34.6 | 34.4 |
| Memory − wrong documents [95% CI] | −0.8 [−5.0, 3.4] | −2.2 [−6.4, 1.8] |

Both scorers agree that the right documents add nothing measurable on PMv2. They pick the same choice on only 66–70% of rows, so row-level scoring of this benchmark is noisy for any model.

## How to use it

There is no standalone inference server for this checkpoint yet. What exists and has been run is the training and evaluation code in the [GitHub repo](https://github.com/DataBassGit/Memorilla-Gemma4-StrixHalo), which also contains the loader that puts Gemma 4's routed experts into 4 bits.

Download the module:

```bash
hf download DataPhreak/Memorilla-Gemma4-26B-A4B --local-dir checkpoints/memorilla_gemma4
```

Reproduce the evaluation (from the GitHub repo, after its setup steps):

```bash
python stanford_eval_gemma4.py generate --checkpoint checkpoints/memorilla_gemma4 \
    --data_repo ./data_10pct --benchmark all --n 500 \
    --conditions memory shuffled closed_book --output_dir results/stanford_ep04
python stanford_eval_gemma4.py score --output_dir results/stanford_ep04
```

Load the module on its own with upstream's API:

```python
from memorilla.memory import MemoryModule

memory = MemoryModule.from_pretrained("checkpoints/memorilla_gemma4")
# doc_embeds:       [batch, num_docs, 2560]  Qwen3-Embedding-4B vectors, one per document chunk
# doc_padding_mask: [batch, num_docs]        True at padding positions, or None
# question_embeds:  [batch, 2560]
tokens = memory(doc_embeds, doc_padding_mask, question_embeds)   # [batch, 16, 2816]
```

**Scale.** Gemma multiplies its token embeddings by √2816. The saved weights have that factor folded into the output projection, so the tokens come out at the scale Gemma expects with no extra step. If the GitHub repo's patch is installed, it divides the factor back out on load and applies it in a forward hook; the result is the same.

**Precision.** The module was trained only against the 4-bit decoder. It has not been evaluated with the full-precision decoder.

## Training

| Stage | Data | What trains | Epochs | Learning rate |
| --- | --- | --- | ---: | ---: |
| 1 | `enwiki`, 10% | Output projection only | 1 | 5e-4 |
| 2 | Upstream's ten-dataset mixture, subsampled | Whole module | 1 | 1.5e-4 |
| 3 | The eight benchmarks' training sets, subsampled | Whole module | 5 | 1e-4 |

- **Data.** Each training file keeps `max(2,500 rows, 10%)`; validation and test sets are untouched. Stage 3 trained on 47,777 rows. All data comes from [`memorilla/Memorilla-Data`](https://huggingface.co/datasets/memorilla/Memorilla-Data).
- **Warm start.** The Perceiver (183M parameters) started from Stanford's fully trained Stage 2 weights. Only the 2560 → 2816 output projection (7.2M parameters) started from scratch. "A tenth of the data" describes this run, not everything the module has learned from.
- **Hardware.** One AMD Strix Halo (Radeon 8060S, 64 GB shared memory) on Windows ROCm. Peak PyTorch memory was 24–26 GB.
- **Known wrinkle.** Stages 1 and 2 ran before a chat-template fix, so the first four tokens of each answer were not supervised in those stages. Stage 3 was restarted from the Stage 2 checkpoint with the fix, and this checkpoint comes from that clean run.

## Intended use and limitations

This is a research checkpoint for studying latent memory on a quantized, non-Qwen decoder. It is not a tested product component.

- **Do not rely on it to recall facts about a user.** On both personalization benchmarks, the right user history and a stranger's scored the same.
- **Small evaluation.** 500 rows per benchmark, one seed, one checkpoint.
- **Unexplained gains.** We have not run the control that would show what the tokens carry when the documents are wrong (tokens built from the wrong question, or from noise).
- **Token scale.** The memory tokens have roughly 90 times the norm of Gemma's text embeddings. Training without the √2816 factor was not tried.
- **No calibration run.** Stanford's checkpoint has not been evaluated in our harness, so harness differences are not ruled out.
- **Inherited behaviour.** Answers come from Gemma 4 and carry its biases and failure modes, along with those of the benchmark training data.

## Credits and license

Memorilla is the work of Fallahpour, Idehpour, Kothapalli, Mounier, Wu, Pardis and Leskovec at Stanford SNAP. Their code and checkpoints are released under Apache 2.0, and this module, which starts from their Stage 2 weights, is released under the same license. Gemma 4 and Qwen3-Embedding-4B are used under the licenses on their model pages. Each dataset in Memorilla-Data keeps its source license.

```bibtex
@article{fallahpour2026memorilla,
  title={Memorilla: Latent Semantic Memory for LLMs},
  author={Fallahpour, Adibvafa and Idehpour, Parsa and Kothapalli, Vignesh and Mounier, Nikita and Wu, Shirley and Pardis, Shayan and Leskovec, Jure},
  journal={arXiv preprint arXiv:2610.21375},
  year={2026}
}
```
