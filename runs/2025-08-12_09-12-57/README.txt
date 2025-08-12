This folder contains outputs for a single run.
- metrics.csv: per-fold best AUC + OOF AUC
- oof_predictions.csv: OOF probabilities with IDs and targets
- submission.csv: ready for Kaggle submit
- folds/*: per-fold model.pth, training logs, and plots (ROC, losses, AUC)
- figs/*: overall figures (OOF ROC, hist)
- models/*, logs/*, artifacts/*: reserved for extras
