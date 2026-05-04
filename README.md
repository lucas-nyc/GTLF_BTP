# GTLF_BTP
Graph–Tabular Latent Fusion for Non‑Contact Body Temperature Prediction from Thermal Facial Landmarks

# Installing Dependencies
Install `requirements.txt` using
```bash
pip install -r requirements.txt 
```

# Dataset Access
Please download the GTLF_BTP dataset and place in "dataset" folder:

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

## Paths

By default, `config/config.py` reads datasets from `dataset/` inside this package:

# Full run

- `run.py`
  Unified runner for baseline, fusion, statistical analysis, visualization, or any combination.

# Example runs
```powershell
pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt
python run.py --all
python run.py --tasks baseline fusion --baseline-args "--no-cv" --fusion-args "--epochs 50 --no-cv"
python run.py --tasks stats --baseline-csv "out/run001/eval_per_set.csv" --fusion-csv "out/run002/eval_per_set.csv"
python run.py --tasks visualization --visualization-args "top3 out/run001 out/run002 --imputation-method CMILK"
python run.py --tasks visualization --visualization-args "baseline-vs-fused --baseline-summary out/run001 --fuse-summary out/run002 --imputation-method CMILK"
python run.py --tasks visualization --visualization-args "significance-share out/run002/analysis_vs_run001"
```


# Referencing
Please cite the following when using our data or code:
