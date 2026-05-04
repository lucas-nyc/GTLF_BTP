# GTLF_BTP
Graph–Tabular Latent Fusion for Non‑Contact Body Temperature Prediction from Thermal Facial Landmarks

# Installing Dependencies

This repository is built using:
```text
Python: 3.8.2
NumPy: 1.24.4
PyTorch: 2.4.1+cu124
CUDA: 12.4
tqdm: 4.67.1
```

Install `requirements.txt` using

```bash
pip install -r requirements.txt 
```

# Dataset and Pre-trained Models Access
Please download the GTLF_BTP dataset and place it in `dataset/` folder:

https://compvis.site.hw.ac.uk/dataset/

```text
GTLF_BTP/
├─ dataset/
│  ├─ complete/
│  └─ imputed/
│    └─ CMILK/
│      └─ MNAR/
│      └─ MCAR/
│  └─ missing/
│    └─ MNAR/
│    └─ MCAR/
```

Please download the pre-trained models and place it in `save/` folder:

```text
GTLF_BTP/
├─ save/
│  ├─ baseline/
│  ├─ gtlf/
```

# Full run

Use the command below for a full run: baseline, fusion, statistical analysis, and visualization.
 
```powershell
python run.py --all
```

# Example runs
```powershell
python run.py --tasks baseline fusion --baseline-args "--no-cv" --fusion-args "--epochs 50 --no-cv"
python run.py --tasks stats --baseline-csv "out/run001/eval_per_set.csv" --fusion-csv "out/run002/eval_per_set.csv"
python run.py --tasks visualization --visualization-args "top3 out/run001 out/run002 --imputation-method CMILK"
python run.py --tasks visualization --visualization-args "baseline-vs-fused --baseline-summary out/run001 --fuse-summary out/run002 --imputation-method CMILK"
python run.py --tasks visualization --visualization-args "significance-share out/run002/analysis_vs_run001"
python run.py --tasks fusion --fusion-args "--save-pretrained"
python run.py --tasks fusion --eval-only --gtlf-save-dir "save/gtlf"
```

# Referencing
Please cite the following when using our data or code:
