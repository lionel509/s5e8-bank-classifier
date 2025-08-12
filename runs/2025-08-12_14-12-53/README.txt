This folder contains outputs for a LightGBM run.
- metrics.csv: per-fold best AUC + OOF AUC
- oof_predictions.csv: OOF probabilities with IDs and targets  
- submission.csv: ready for Kaggle submit
- run_metadata.json: detailed run configuration and results
- logs/training.log: detailed training logs
- figs/*: overall figures (reserved for future ROC plots)

Run Details:
- Algorithm: LightGBM (with Web Dashboard)
- CV Folds: 3
- Random Seed: 2025
- Total Features: 16 (9 categorical, 7 numerical)
- Final OOF AUC: 0.968591
- Mean CV AUC: 0.968599 ± 0.000170
- Training Time: 77.0s (1.3 minutes)
- Web Dashboard: http://127.0.0.1:8765
- Timestamp: 2025-08-12_14-12-53
