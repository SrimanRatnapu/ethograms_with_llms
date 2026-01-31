# LLM Ethogram Evaluation

Code accompanying the paper **"Evaluating Large Multimodal Language Models for Automated Ethogram Generation from Public Zoo Footage"**.

## Overview
This repository documents an evaluation pipeline comparing LLM-generated ethogram labels against human annotations.

**It accompanies the arXiv preprint “Evaluating Large Multimodal Language Models for Automated Ethogram Generation from Public Zoo Footage,” which describes the experimental design and results.**

Key design choices:
- Focal-animal sampling
- Coarse-grained behavior taxonomy
- Fixed 10-second windows
- Coverage-driven clip selection (evaluation-focused; not prevalence estimation)
- Frame-sampling ablation (1 / 3 / 5 frames)

## What is included
- Analysis scripts for agreement metrics, per-class precision/recall/F1, and confusion matrices
- Scripts to (a) slice YouTube videos into 10-second subclips and (b) run LLM labeling
- Frozen LLM prompt used in experiments

## What is NOT included
- Raw video files are not redistributed.
- Full human annotation files are not redistributed in this public package.

## Running analysis (example)
```bash
pip install -r requirements.txt
python scripts/analyze_ethogram.py --human1 HumanAnnotator1.csv --human2 HumanAnnotator2.csv --llm labels_5f.csv --out_dir out
```

## Data provenance
A `video_manifest.csv` (URLs + timestamps) should be stored privately by the authors. If you choose to publish it, place it under `data/`.
