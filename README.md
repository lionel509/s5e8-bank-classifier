# s5e8-bank-classifier

Binary‑classification model for predicting client outcomes using the Kaggle Playground Series S5E8 bank dataset. 

## v5 Fast Stacking Pipeline

The project now features a **fast stacking pipeline** optimized for Apple M2 with approximately 30-minute runtime:

### Pipeline Structure:
1. **Neural Network (Model A)** - PyTorch MLP with warm-start capability from previous checkpoints, MPS acceleration
2. **CatBoost (Model B)** - Gradient boosting optimized for CPU with early stopping  
3. **LightGBM Meta-Model** - Uses out-of-fold (OOF) predictions from Models A & B as features

### Key Features:
- **Speed Optimized**: Reduced epochs/iterations, warm-starting, efficient meta-model parameters
- **Apple M2 Compatible**: MPS acceleration for neural network, CPU-optimized for other models
- **Comprehensive Outputs**: All results saved in `runs/<timestamp>/stacking/`
- **Detailed Logging**: Training progress and metrics stored in `logs/` subfolder
- **Rich Visualizations**: ROC curves, performance comparisons in `figs/` subfolder
- **OOF AUC Reporting**: Individual model performance plus final stacked result

### Output Structure:
```
runs/<timestamp>/stacking/
├── oof_predictions.csv      # Out-of-fold predictions for all models
├── submission.csv           # Final meta-model predictions  
├── metrics.csv             # Per-fold and overall AUC scores
├── README.txt              # Detailed run summary
├── run_metadata.json       # Complete configuration and results
├── logs/
│   └── training.log        # Detailed training logs
└── figs/
    ├── model_comparison.png # ROC curves and performance bars
    └── cv_performance.png   # Cross-validation results by fold
```

This approach maintains strong predictive performance while significantly reducing training time through intelligent model selection and parameter tuning.
