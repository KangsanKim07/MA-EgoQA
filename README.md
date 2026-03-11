# MA-EgoQA: Multi-Agent Egocentric Video Question Answering

[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://ma-egoqa.github.io)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/KangsanKim71/MA-EgoQA)

## Overview

**MA-EgoQA** is the first benchmark for question answering over multiple long-horizon egocentric video streams from embodied agents. As intelligent agents increasingly assist our physical activities, understanding events collectively observed by multiple agents becomes essential — yet remains largely unexplored.

MA-EgoQA is built on the [EgoLife](https://egolife-dataset.github.io/) dataset, where **6 people** lived together for **7 days** wearing egocentric cameras, resulting in **266 hours** of multi-agent video. Every question requires reasoning across **more than two agents' observations**.

<p align="center">
  <img src="assets/concept_figure.png" width="80%" alt="MA-EgoQA Concept Figure"/>
</p>

---

## MA-EgoQA Benchmark

### Five Question Categories

| Category | Abbr. | Description |
|---|---|---|
| Social Interaction | SI | Localizing conversations and group behaviors across video streams |
| Task Coordination | TC | How agents divide roles and collaborate toward shared goals |
| Theory of Mind | ToM | Reasoning about agents' beliefs, intentions, and mental states |
| Temporal Reasoning | TR | Concurrency and ordering of events across agents' timelines |
| Environmental Interaction | EI | Tracking distributed object usage across agents |

<p align="center">
  <img src="assets/examples.png" width="80%" alt="MA-EgoQA QA Examples"/>
</p>

---

## EgoMAS Baseline

We propose **EgoMAS** (Egocentric Multi-Agent System), a training-free baseline that addresses the unique challenges of multi-agent egocentric reasoning.

<p align="center">
  <img src="assets/egomas.png" width="80%" alt="EgoMAS Method Figure"/>
</p>

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
