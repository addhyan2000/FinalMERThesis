# Initial Version Guide — MER_Client_GPU

Main runbook for the **client GPU package** (`MER_Client_GPU`).

For scope and checklist, see `CLIENT_INITIAL_VERSION.md`. For the GUI, see `JUNE19_GUI.md`.

---

## Setup

```powershell
cd MER_Client_GPU
.\setup_gpu.ps1
python tools/check_environment.py
```

Expected: `CUDA available: True`

---

## Smoke tests (no dataset)

```powershell
python tools/smoke_step1_cpu.py
python tools/smoke_step2_cpu.py
python tools/smoke_ablation_cpu.py
```

Or launch the GUI: `python tools/run_gui.py`

---

## Real data

Place in `DATASETS/CASME II/`:
- `CASME2-coding-20140508.xlsx`
- `Cropped/` and/or `Video/` folders

### Step 1 — metadata

```powershell
python Stage1_DataPipeline/main_step1.py --dataset_mode casme2_only --casme2_media_mode both --force
```

### Step 2 — tensors

```powershell
python Stage1_DataPipeline/main_step2.py --force --max_workers 8 --output_subdir tensors --dataset_filter CASME_II --expression_filter micro-expression
python Stage1_DataPipeline/main_step2.py --force --max_workers 8 --output_subdir tensors_raw --dataset_filter CASME_II --expression_filter micro-expression
```

---

## Ablation (GPU)

Default — proposed model:

```powershell
python tools/run_ablation_gpu.py --label_mode grouped --epochs 5 --configs config_8_proposed_unified
```

All 12 valid configs:

```powershell
python tools/run_ablation_gpu.py --label_mode grouped --epochs 30
```

LOSO pilot (5 subjects):

```powershell
python tools/run_ablation_gpu.py --protocol loso --loso_max_folds 5 --epochs 50 --configs config_8_proposed_unified
```

Full LOSO (all subjects):

```powershell
python tools/run_ablation_gpu.py --protocol loso --full_loso --epochs 60 --configs config_8_proposed_unified
```

Individual labels:

```powershell
python tools/run_ablation_gpu.py --label_mode individual --epochs 5 --configs config_8_proposed_unified --output_root Ablation_Study/results_individual
```

Plots:

```powershell
python tools/plot_ablation_results.py
python tools/compare_with_literature.py
```

---

## Results

- `Ablation_Study/results/summary.csv`
- Per-config folders under `Ablation_Study/results/`
- `final_results.json`, `confusion_matrix.npy`, `training_metrics.csv`

---

## Package for redistribution

```powershell
.\BUILD_CLIENT_GPU.ps1
```

Creates `MER_Client_GPU.zip` in the parent folder (clean, no stale Processed_Data).

---

## More docs

| File | Purpose |
|---|---|
| `GPU_MODE.md` | Install, troubleshooting |
| `JUNE19_GPU_TEST.md` | June 19 verification checklist |
| `JUNE19_GUI.md` | GUI options (LOSO, both media, configs) |
