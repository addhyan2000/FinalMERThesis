"""Collect every LOSO + baseline artifact into one JSON for figure generation."""
import csv
import json
import os
import pathlib
import subprocess
import collections

ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
SP = os.path.dirname(os.path.abspath(__file__))  # data.json lives beside this script

# Canonical config ordering / short IDs
ORDER = [
    ("C1", "config_1_pure_base", False, False, False, False, "I"),
    ("C2", "config_2_temporal_only", False, False, False, True, "I"),
    ("C3", "config_3_spatial_only", False, False, True, False, "I"),
    ("C9", "config_9_permutation", False, False, True, True, "Other"),
    ("C5", "config_5_attention_base", False, True, True, False, "II"),
    ("C6", "config_6_full_stage2_noevm", False, True, True, True, "III"),
    ("C4", "config_4_motion_amp_base", True, False, False, False, "II"),
    ("C12", "config_12_permutation", True, False, False, True, "Other"),
    ("C13", "config_13_permutation", True, False, True, False, "Other"),
    ("C7", "config_7_full_no_attention", True, False, True, True, "III"),
    ("C16", "config_16_permutation", True, True, True, False, "Other"),
    ("C8", "config_8_proposed_unified", True, True, True, True, "IV"),
]
PREFIX2ID = {p: i for i, p, *_ in ORDER}


def cid(config_name):
    for i, p, *_ in ORDER:
        if config_name.startswith(p):
            return i
    return None


def load_dir(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub, "final_results.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        k = cid(j["config_name"])
        if k:
            out[k] = j
    return out


def load_branch(branch, root):
    out = {}
    files = subprocess.run(["git", "ls-tree", "-r", "--name-only", branch, "--",
                            f"Ablation_Study/{root}"], capture_output=True, text=True,
                           cwd=ROOT).stdout.split()
    for f in files:
        if not f.endswith("final_results.json"):
            continue
        txt = subprocess.run(["git", "show", f"{branch}:{f}"], capture_output=True,
                             text=True, cwd=ROOT).stdout
        try:
            j = json.loads(txt)
        except Exception:
            continue
        k = cid(j["config_name"])
        if k:
            out[k] = j
    return out


def hw(d):
    """Parse train time + VRAM out of configuration_summary.txt."""
    out = {}
    if not os.path.isdir(d):
        return out
    for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub, "configuration_summary.txt")
        if not os.path.exists(p):
            continue
        k = cid(sub)
        if not k:
            continue
        t = v = None
        for line in open(p):
            if "Total Train Time" in line:
                t = float(line.split(":")[1].strip().split()[0])
            if "Peak VRAM" in line:
                v = float(line.split(":")[1].strip().split()[0])
        out[k] = {"train_sec": t, "vram_mb": v}
    return out


def curves(d):
    out = {}
    if not os.path.isdir(d):
        return out
    for sub in sorted(os.listdir(d)):
        p = os.path.join(d, sub, "training_metrics.csv")
        if not os.path.exists(p):
            continue
        k = cid(sub)
        if not k:
            continue
        out[k] = list(csv.DictReader(open(p)))
    return out


data = {"order": ORDER}

# ---- primary: this branch, full LOSO 25 folds -------------------------------
data["loso25"] = load_dir(f"{ROOT}/Ablation_Study/results")
data["loso25_replicate"] = load_dir(f"{ROOT}/Ablation_Study/results_individual")
data["loso25_hw"] = hw(f"{ROOT}/Ablation_Study/results")
data["loso25_curves"] = curves(f"{ROOT}/Ablation_Study/results")

# ---- baselines in the working tree -----------------------------------------
data["weekend_holdout52"] = load_dir(f"{ROOT}/Ablation_Study/results_weekend/holdout")
data["weekend_loso139_partial"] = load_dir(f"{ROOT}/Ablation_Study/results_weekend/loso")

# ---- baselines from sibling branches ---------------------------------------
data["holdout39"] = load_branch("holdout-all", "results")
data["holdout39_individual6"] = load_branch("holdout-all", "results_individual")
data["pilot_loso5"] = load_branch("loso-handle", "results")
data["loso20"] = load_branch("new_gui_loso_holdout", "results_weekend")

# ---- literature -----------------------------------------------------------
data["literature"] = list(csv.DictReader(
    open(f"{ROOT}/Ablation_Study/literature_baselines.csv", encoding="utf-8-sig")))

# ---- per-subject fold composition -----------------------------------------
rows = list(csv.DictReader(open(f"{ROOT}/Processed_Data/master_thesis_labels.csv",
                                encoding="utf-8-sig")))
sel = [r for r in rows
       if r["Unified_Emotion"] in ("Negative", "Positive", "Surprise")
       and r["Expression_Type"] == "micro-expression" and r["Dataset"] == "CASME_II"]
bysub = collections.defaultdict(lambda: collections.Counter())
for r in sel:
    bysub[int(r["Subject_ID"])][r["Unified_Emotion"]] += 1
data["folds"] = [{"subject": s,
                  "Negative": bysub[s]["Negative"],
                  "Positive": bysub[s]["Positive"],
                  "Surprise": bysub[s]["Surprise"]} for s in sorted(bysub)]
data["raw_emotion_counts"] = dict(collections.Counter(r["Raw_Emotion"] for r in rows))
data["grouped_counts"] = dict(collections.Counter(r["Unified_Emotion"] for r in rows))

json.dump(data, open(f"{SP}/data.json", "w"), indent=1)

# ---- console sanity report -------------------------------------------------
def pm(j):
    f = j["metrics"]["per_class_f1"]
    return sum(f) / len(f)


print("regime                     n_configs")
for k in ["loso25", "loso25_replicate", "weekend_holdout52", "weekend_loso139_partial",
          "holdout39", "holdout39_individual6", "pilot_loso5", "loso20"]:
    print(f"  {k:26s} {len(data[k])}")

print("\nFULL LOSO 25-fold (this branch), ranked by pooled macro F1")
for k, j in sorted(data["loso25"].items(), key=lambda kv: -pm(kv[1])):
    m = j["metrics"]
    print(f"  {k:4s} foldacc={m['accuracy']:.4f} pooledacc={m['micro_f1']:.4f} "
          f"foldmF1={m['macro_f1']:.4f} pooledmF1={pm(j):.4f} N={m['num_samples']}")

# macro-F1 ceiling under mean-of-folds averaging
ceil = 0.0
for f in data["folds"]:
    k = sum(1 for c in ("Negative", "Positive", "Surprise") if f[c] > 0)
    ceil += k / 3.0
ceil /= len(data["folds"])
print(f"\nmean-of-folds macro-F1 structural ceiling = {ceil:.4f}")
print(f"n folds = {len(data['folds'])}, total clips = {sum(f['Negative']+f['Positive']+f['Surprise'] for f in data['folds'])}")
