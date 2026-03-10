# MA-EgoQA: Multi-Agent Egocentric Video Question Answering

[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/KangsanKim71/MA-EgoQA)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ma-egoqa.github.io)

**Kangsan Kim<sup>1</sup>, Yanlai Yang<sup>2</sup>, Suji Kim<sup>1,3</sup>, Woongyeong Yeo<sup>1</sup>, Youngwan Lee<sup>1,4</sup>, Mengye Ren<sup>2†</sup>, Sung Ju Hwang<sup>1,5†</sup>**

<sup>1</sup>KAIST &nbsp; <sup>2</sup>New York University &nbsp; <sup>3</sup>Samsung Electronics &nbsp; <sup>4</sup>ETRI &nbsp; <sup>5</sup>DeepAuto.ai

<sup>†</sup>Equal advising

---

## Overview

**MA-EgoQA** is the first benchmark for question answering over multiple long-horizon egocentric video streams from embodied agents. As intelligent agents increasingly assist our physical activities, understanding events collectively observed by multiple agents becomes essential — yet remains largely unexplored.

MA-EgoQA is built on the [EgoLife](https://egolife-dataset.github.io/) dataset, where **6 people** lived together for **7 days** wearing egocentric cameras, resulting in **266 hours** of multi-agent video. Every question requires reasoning across **more than two agents' observations**.

<p align="center">
  <img src="assets/concept_figure.png" width="80%" alt="MA-EgoQA Concept Figure"/>
</p>

---

## Benchmark

### Five Question Categories

| Category | Abbr. | Description |
|---|---|---|
| Social Interaction | SI | Localizing conversations and group behaviors across video streams |
| Task Coordination | TC | How agents divide roles and collaborate toward shared goals |
| Theory of Mind | ToM | Reasoning about agents' beliefs, intentions, and mental states |
| Temporal Reasoning | TR | Concurrency and ordering of events across agents' timelines |
| Environmental Interaction | EI | Tracking distributed object usage across agents |

### Statistics

| | MA-EgoQA |
|---|---|
| Total QA pairs | **1,741** |
| Video sources | 6 agents × 7 days |
| Total duration | 266 hours |
| Question format | Multiple choice (5 options) |
| Min. agents required | 2+ per question |

### Comparison with Related Benchmarks

| Dataset | # QA | Duration | Egocentric | Days-Long | Cross-Video | Theory-of-Mind |
|---|---|---|:---:|:---:|:---:|:---:|
| EgoSchema | 5,063 | 180 sec | ✓ | ✗ | ✗ | ✗ |
| EgoLifeQA | 6,000 | 44 hour | ✓ | ✓ | ✗ | ✗ |
| EgoToM | 1,039 | 300 sec | ✓ | ✗ | ✗ | ✓ |
| MuMA-ToM | 900 | 36 sec | ✗ | ✗ | ✗ | ✓ |
| EgoExoLearn | 2,200 | 13 min | ✓ | ✗ | ✓ | ✗ |
| **MA-EgoQA (Ours)** | **1,741** | **266 hour** | ✓ | ✓ | ✓ | ✓ |

MA-EgoQA is the **only benchmark** covering all four dimensions simultaneously.

---

## EgoMAS Baseline

We propose **EgoMAS** (Egocentric Multi-Agent System), a training-free baseline that addresses the unique challenges of multi-agent egocentric reasoning.

### Key Components

**1. Event-based Shared Memory**
Every 10 minutes, a centralized manager integrates individual agent captions into structured **4W1H** records:
- **When** — timestamp
- **What** — event description
- **Where** — location
- **Who** — agents involved
- **How** — manner of action

**2. Agent-wise Dynamic Retrieval**
Given a query, EgoMAS performs two-stage retrieval:
1. **System-level**: top-*n* shared memory entries via BM25
2. **Agent-level**: dynamically generated per-agent sub-queries with relevance filtering

This uses only **4–7k tokens** per query, compared to 128k–1M for non-retrieval baselines.

---

## Results

| Model | Context | SI | TC | ToM | TR | EI | Avg. |
|---|---|---|---|---|---|---|---|
| Random | — | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 | 20.0 |
| Gemini-2.5-Flash | 1M | 41.2 | 36.4 | 24.3 | 46.6 | 34.0 | 36.9 |
| GPT-5 | 272k | 36.2 | 33.9 | 22.6 | 39.7 | 38.7 | 34.8 |
| BM25 + Qwen3VL-8B | 8.1k | **44.7** | 37.6 | 30.2 | 33.5 | 30.6 | 36.0 |
| WorldMM-8B | 4.1k | 29.3 | 34.5 | 21.7 | 25.1 | 22.6 | 27.6 |
| **EgoMAS (Gemini-2.5-Flash)** | **4.6k** | 41.5 | **41.3** | **33.6** | 39.4 | **48.2** | **41.4** |
| **EgoMAS (Qwen3VL-8B-Thinking)** | **5.4k** | 38.0 | 39.9 | 28.9 | **47.4** | 44.9 | 40.3 |
| Oracle (Gemini-2.5-Flash) | — | 86.4 | 97.6 | 70.2 | 88.2 | 81.3 | 83.8 |

**Key findings:**
- Even the strongest baseline (Gemini-2.5-Flash with 1M context) achieves only **36.9%** — barely above random
- EgoMAS with an 8B model **outperforms** both Gemini-2.5-Flash and GPT-5 baselines
- The large gap to Oracle (~83.8%) highlights significant room for future improvement

---

## Citation

```bibtex
@inproceedings{kim2026maegoqa,
  title     = {Multi-Agent Egocentric Video Question Answering},
  author    = {Kim, Kangsan and Yang, Yanlai and Kim, Suji and Yeo, Woongyeong and
               Lee, Youngwan and Ren, Mengye and Hwang, Sung Ju},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026},
  url       = {https://github.com/KangsanKim07/MA-EgoQA}
}
```
