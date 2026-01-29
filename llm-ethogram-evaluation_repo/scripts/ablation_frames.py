#!/usr/bin/env python3
import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

EXCLUDE = {"OutOfFrame","Uncertain"}

def norm_label(x):
    if pd.isna(x): return np.nan
    s = str(x).strip()
    s = s.replace("Object Interaction","ObjectInteraction").replace("Objectinteraction","ObjectInteraction")
    s = s.replace("Out Of Frame","OutOfFrame").replace("OutofFrame","OutOfFrame")
    mapping = {
        "resting":"Resting","locomotion":"Locomotion","feeding":"Feeding","social":"Social",
        "objectinteraction":"ObjectInteraction","outofframe":"OutOfFrame","uncertain":"Uncertain",
        "playing":"ObjectInteraction",
    }
    key = re.sub(r"[^a-zA-Z]", "", s).lower()
    return mapping.get(key, s)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human1", required=True)
    ap.add_argument("--human2", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    args = ap.parse_args()

    h1 = pd.read_csv(args.human1); h2 = pd.read_csv(args.human2)
    for df in (h1,h2):
        df["filename"] = df["filename"].astype(str).str.strip()
        df["behavior_label"] = df["behavior_label"].apply(norm_label)

    hh = (h1[["filename","behavior_label"]].rename(columns={"behavior_label":"h1"})
          .merge(h2[["filename","behavior_label"]].rename(columns={"behavior_label":"h2"}), on="filename", how="inner"))
    hh["consensus"] = np.where(hh["h1"]==hh["h2"], hh["h1"], np.nan)

    out_rows = []
    for p in args.labels:
        llm = pd.read_csv(p)
        llm["filename"] = llm["filename"].astype(str).str.strip()
        llm["behavior_label"] = llm["behavior_label"].apply(norm_label)

        m = (hh.merge(llm[["filename","behavior_label"]].rename(columns={"behavior_label":"llm"}),
                      on="filename", how="left")
               .dropna(subset=["consensus","llm"]))
        m = m[~m["consensus"].isin(EXCLUDE)]
        y_true = m["consensus"].values
        y_pred = m["llm"].values
        acc = accuracy_score(y_true, y_pred)
        pr, rc, f1, sup = precision_recall_fscore_support(y_true, y_pred, zero_division=0)
        macro_f1 = float(np.mean(f1)) if len(f1) else float("nan")
        out_rows.append((Path(p).name, len(m), acc, macro_f1))

    out = pd.DataFrame(out_rows, columns=["labels_file","N","Accuracy","MacroF1"])
    print(out.to_string(index=False))

if __name__ == "__main__":
    main()
