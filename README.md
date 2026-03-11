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

## Run EgoMAS

### 1. Clone the Repository and Install
```sh
git clone https://github.com/KangsanKim07/MA-EgoQA.git
cd MA-EgoQA
pip install -r requirements.txt
```

### 2. Download the MA-EgoQA dataset from HuggingFace to the `data/` directory
```sh
huggingface-cli download KangsanKim71/MA-EgoQA --local-dir data --repo-type dataset
```

### 3. Construct Event-based Shared Memory with 10-min Window
```sh
python egomas/src/construct_shared_memory.py
```

### 4. Index Captions with BM25
```sh
python index_bm25.py
```

### 5. Inference EgoMAS with Agent-wise Dynamic Retrieval
```sh
python inference_egomas.py  # Multi-process
```
You can run EgoMAS with a single process by run:
```sh
python inference_egomas_singleproc.py
```

---

## Citation

```bibtex
@misc{kim2026maegoqa,
    title={MA-EgoQA: Question Answering over Egocentric Videos from Multiple Embodied Agents}, 
    author={Kangsan Kim and Yanlai Yang and Suji Kim and Woongyeong Yeo and Youngwan Lee and Mengye Ren and Sung Ju Hwang},
    year={2026},
    eprint={2603.09827},
    archivePrefix={arXiv},
    primaryClass={cs.CV},
    url={https://arxiv.org/abs/2603.09827}, 
}
```
