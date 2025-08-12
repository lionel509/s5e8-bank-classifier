This folder contains outputs for a single run.
- oof_predictions.csv: OOF probabilities with IDs and targets
- submission.csv: ready for Kaggle submit
- folds/*: per-fold model.pth, training logs, and plots (ROC, losses, AUC)
- figs/*: overall figures (OOF ROC, hist)
- logs/live_status.csv and logs/status.json: live tracking outputs
