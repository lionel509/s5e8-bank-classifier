Stacking Pipeline v5 Results
============================

This folder contains outputs from a fast stacking ensemble pipeline optimized for Apple M2.

Pipeline:
1. Neural Network (Model A) - warm-start from checkpoint if available
2. CatBoost (Model B) - CPU-optimized for stability  
3. LightGBM Meta-Model - uses OOF predictions from Models A & B

Results:
- Neural Network OOF AUC: 0.958445
- CatBoost OOF AUC: 0.965924
- Meta-Model OOF AUC: 0.965950

Performance Summary:
- Neural Network CV: 0.958521 ± 0.000286
- CatBoost CV: 0.965928 ± 0.000481
- Total Runtime: 691.3s (11.5 minutes)

Configuration:
- CV Folds: 5
- Random Seed: 2025
- Total Features: 16 (7 numerical, 9 categorical)
- Target Rate: 0.1207

Files:
- oof_predictions.csv: OOF predictions for all models
- submission.csv: Final meta-model predictions for test set
- metrics.csv: Per-fold and overall AUC scores
- figs/: ROC curves and performance comparison plots
- logs/: Detailed training logs

Optimization Features:
- Neural Network: 7 epochs with early stopping
- CatBoost: 750 iterations with early stopping
- Meta-Model: 300 estimators, 8 leaves
- MPS acceleration: True
- Warm-start: Yes

Timestamp: 2025-08-12_14-41-51
