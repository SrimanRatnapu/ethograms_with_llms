#!/usr/bin/env python3
"""
Ethogram evaluation script

Inputs:
  --human1 HumanAnnotator1.csv
  --human2 HumanAnnotator2.csv
  --llm    LLM_labels.csv

Outputs (in --out_dir, default ./ethogram_analysis):
  - table_human_human_agreement.csv
  - table_llm_metrics_all.csv
  - table_llm_metrics_filtered.csv
  - confusion_matrix_filtered.csv
  - merged_interval_labels.csv
  - summary.csv
  - f1_by_behavior.png
  - confusion_matrix.png

Notes:
- Assumes "filename" is the join key across files (your 10s subclip mp4 name).
- Human consensus is defined only where annotators agree; disagreements are excluded from consensus-based eval.
- "OutOfFrame" and "Uncertain" are excluded in the filtered subset by default.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
import matplotlib.pyplot as plt

EXCLUDE_DEFAULT = {"OutOfFrame", "Uncertain"}

def norm_label(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    s = s.replace("Object Interaction", "ObjectInteraction").replace("Objectinteraction", "ObjectInteraction")
    s = s.replace("Out Of Frame", "OutOfFrame").replace("OutofFrame", "OutOfFrame")
    mapping = {
        "resting":"Resting",
        "locomotion":"Locomotion",
        "feeding":"Feeding",
        "social":"Social",
        "objectinteraction":"ObjectInteraction",
        "outofframe":"OutOfFrame",
        "uncertain":"Uncertain",
        # If any legacy label appears, map here:
        "playing":"ObjectInteraction",
    }
    key = re.sub(r"[^a-zA-Z]", "", s).lower()
    return mapping.get(key, s)

def compute_metrics(df, labels):
    y_true = df["consensus"].tolist()
    y_pred = df["llm"].tolist()
    acc = accuracy_score(y_true, y_pred)
    pr, rc, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    macro_f1 = float(np.mean(f1)) if len(f1) else float("nan")
    return acc, pr, rc, f1, sup, macro_f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human1", required=True)
    ap.add_argument("--human2", required=True)
    ap.add_argument("--llm", required=True)
    ap.add_argument("--out_dir", default="ethogram_analysis")
    ap.add_argument("--exclude", default="OutOfFrame,Uncertain",
                   help="Comma-separated labels to exclude for filtered eval (default: OutOfFrame,Uncertain)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exclude = {s.strip() for s in args.exclude.split(",") if s.strip()}

    h1 = pd.read_csv(args.human1)
    h2 = pd.read_csv(args.human2)
    llm = pd.read_csv(args.llm)

    for df in (h1, h2, llm):
        df["filename"] = df["filename"].astype(str).str.strip()
        df["behavior_label"] = df["behavior_label"].apply(norm_label)

    hh = (
        h1[["filename", "behavior_label"]].rename(columns={"behavior_label":"h1"})
        .merge(
            h2[["filename", "behavior_label"]].rename(columns={"behavior_label":"h2"}),
            on="filename",
            how="inner",
        )
    )

    total_hh = len(hh)
    agree_mask = hh["h1"] == hh["h2"]
    agree_n = int(agree_mask.sum())
    disagree_n = int((~agree_mask).sum())

    hh_kappa = cohen_kappa_score(hh["h1"], hh["h2"])

    hh["consensus"] = np.where(agree_mask, hh["h1"], np.nan)

    m = hh.merge(
        llm[["filename", "behavior_label", "confidence", "notes"]].rename(columns={"behavior_label":"llm"}),
        on="filename",
        how="left",
    )

    missing_llm = int(m["llm"].isna().sum())

    m_eval = m.dropna(subset=["consensus", "llm"]).copy()
    m_eval_filt = m_eval[~m_eval["consensus"].isin(exclude)].copy()

    labels_all = sorted(set(m_eval["consensus"]) | set(m_eval["llm"]))
    labels_main = [l for l in sorted(set(m_eval_filt["consensus"])) if l not in exclude]

    acc_all, pr_all, rc_all, f1_all, sup_all, macro_f1_all = compute_metrics(m_eval, labels_all)
    acc_main, pr_main, rc_main, f1_main, sup_main, macro_f1_main = compute_metrics(m_eval_filt, labels_main)

    tbl_hh = pd.DataFrame({
        "metric":["n_intervals","n_agree","n_disagree","kappa"],
        "value":[total_hh, agree_n, disagree_n, hh_kappa]
    })
    tbl_llm_all = pd.DataFrame({
        "label": labels_all,
        "support_true": sup_all,
        "precision": pr_all,
        "recall": rc_all,
        "f1": f1_all,
    })
    tbl_llm_main = pd.DataFrame({
        "label": labels_main,
        "support_true": sup_main,
        "precision": pr_main,
        "recall": rc_main,
        "f1": f1_main,
    })
    summary = pd.DataFrame({
        "subset":[
            "All consensus intervals (incl excluded labels)",
            f"Filtered (exclude {', '.join(sorted(exclude))})"
        ],
        "n":[len(m_eval), len(m_eval_filt)],
        "accuracy":[acc_all, acc_main],
        "macro_f1":[macro_f1_all, macro_f1_main],
        "missing_llm_labels":[missing_llm, missing_llm],
    })

    if len(m_eval_filt):
        cm = confusion_matrix(m_eval_filt["consensus"], m_eval_filt["llm"], labels=labels_main)
        cm_df = pd.DataFrame(cm, index=[f"true_{l}" for l in labels_main], columns=[f"pred_{l}" for l in labels_main])
    else:
        cm = None
        cm_df = pd.DataFrame()

    # Write outputs
    tbl_hh.to_csv(out_dir/"table_human_human_agreement.csv", index=False)
    tbl_llm_all.to_csv(out_dir/"table_llm_metrics_all.csv", index=False)
    tbl_llm_main.to_csv(out_dir/"table_llm_metrics_filtered.csv", index=False)
    summary.to_csv(out_dir/"summary.csv", index=False)
    m.to_csv(out_dir/"merged_interval_labels.csv", index=False)
    cm_df.to_csv(out_dir/"confusion_matrix_filtered.csv", index=True)

    # Figures
    if len(tbl_llm_main):
        plt.figure(figsize=(8,4))
        plt.bar(tbl_llm_main["label"], tbl_llm_main["f1"])
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("F1 score")
        plt.title("LLM vs Human Consensus: F1 by Behavior (Filtered)")
        plt.tight_layout()
        plt.savefig(out_dir/"f1_by_behavior.png", dpi=200)
        plt.close()

    if cm is not None and len(labels_main):
        plt.figure(figsize=(7,6))
        plt.imshow(cm, interpolation="nearest")
        plt.xticks(range(len(labels_main)), labels_main, rotation=30, ha="right")
        plt.yticks(range(len(labels_main)), labels_main)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix (Filtered)")
        plt.colorbar()
        plt.tight_layout()
        plt.savefig(out_dir/"confusion_matrix.png", dpi=200)
        plt.close()

    print("Wrote outputs to:", out_dir.resolve())
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
