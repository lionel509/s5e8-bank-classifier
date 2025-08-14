#!/usr/bin/env python3
"""
Tabular v6 - Tri-Model Blending with Advanced Features
- Backbone from tabular_tri_model_v4.py
- Utilities (logging, plotting, run folders) from tabular_stacking_v5.py
- Hard-coded settings, no CLI or external configs.
- Models: LightGBM, XGBoost, CatBoost
- Ensembling: Non-negative weighted blend (grid search) + optional LR meta-model
"""

import os
import gc
import time
import random
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime
from itertools import product

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegression
from scipy.optimize import minimize

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG - Hardcoded paths and parameters
# =============================================================================
# v4: TRAIN_PATH = "playground-series-s5e8/train.csv"
TRAIN_PATH = "playground-series-s5e8/train.csv"
# v4: TEST_PATH = "playground-series-s5e8/test.csv"
TEST_PATH = "playground-series-s5e8/test.csv"

# v4: ID_COL = "id"
ID_COL = "id"
# v4: TARGET_COL = "y"
TARGET_COL = "y"

# Global Settings
# v4: N_SPLITS = 3
CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=2025)
# v4: RANDOM_SEED = 2025
RANDOM_SEED = 2025
USE_META = False # Do not use Logistic Regression meta-model by default

# v4: No explicit early stopping, but had NUM_BOOST_ROUND = 10000 and EARLY_STOPPING_ROUNDS = 100
EARLY_STOPPING_ROUNDS = 200
MAX_TREES = 5000
N_THREADS = -1

# Preprocessing
RARE_CATEGORY_FREQ = 0.005 # 0.5%
RARE_NAME = "RARE"
MISSING_NAME = "MISSING"

# Create timestamped run directory
# v4: RUN_TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
# v4: RUN_DIR = Path(f"runs/{RUN_TIMESTAMP}")
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/v6/{RUN_TIMESTAMP}")
FIGS_DIR = RUN_DIR / "fig"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# Model Parameters
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': -1,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'min_data_in_leaf': 40,
    'lambda_l1': 0.0,
    'lambda_l2': 2.0,
    'verbosity': -1,
    'force_row_wise': True,
    'seed': RANDOM_SEED,
    'n_jobs': N_THREADS,
    'n_estimators': MAX_TREES,
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.03,
    'max_depth': 7,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.0,
    'reg_lambda': 2.0,
    'tree_method': 'hist',
    'seed': RANDOM_SEED,
    'n_jobs': N_THREADS,
    'n_estimators': MAX_TREES,
}

CAT_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.03,
    'depth': 6,
    'l2_leaf_reg': 5,
    'bagging_temperature': 0.5,
    'random_seed': RANDOM_SEED,
    'verbose': False,
    'thread_count': N_THREADS,
    'iterations': MAX_TREES,
}

# =============================================================================
# SETUP LOGGING (from v5)
# =============================================================================
def setup_logging(run_dir):
    """Setup logging to both file and console."""
    log_file = run_dir / "run.log"
    logger = logging.getLogger('tabular_v6')
    logger.setLevel(logging.INFO)

    # Remove existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger

logger = setup_logging(RUN_DIR)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=RANDOM_SEED):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # The following are not in v4 but are good practice
    pd.core.common.random_state(seed)
    # For torch if used
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# =============================================================================
# PREPROCESSING
# =============================================================================
# v4: had a preprocess_data function with different logic
def preprocess_data(train_df, test_df):
    """
    Preprocess data: NaN filling, rare category handling, label encoding.
    CV-safe, simple, and robust.
    """
    logger.info("Starting data preprocessing...")
    
    # Combine for consistent processing
    combined_df = pd.concat([train_df.drop(columns=[TARGET_COL]), test_df], axis=0, ignore_index=True)
    
    # Identify column types
    cat_cols = [c for c in combined_df.columns if combined_df[c].dtype == 'object' and c != ID_COL]
    num_cols = [c for c in combined_df.columns if combined_df[c].dtype != 'object' and c != ID_COL]
    
    logger.info(f"Identified {len(num_cols)} numerical and {len(cat_cols)} categorical features.")

    # Handle NaNs
    # Numeric NaN -> median (calculated on train set only to be CV-safe)
    for col in num_cols:
        if combined_df[col].isnull().any():
            median_val = train_df[col].median()
            combined_df[col].fillna(median_val, inplace=True)
            logger.info(f"Filled NaNs in numeric column '{col}' with median {median_val:.4f}")

    # Categorical NaN -> "MISSING"
    for col in cat_cols:
        if combined_df[col].isnull().any():
            combined_df[col].fillna(MISSING_NAME, inplace=True)
            logger.info(f"Filled NaNs in categorical column '{col}' with '{MISSING_NAME}'")

    # Handle rare categories (based on train set frequencies)
    for col in cat_cols:
        counts = train_df[col].value_counts(normalize=True)
        rare_values = counts[counts < RARE_CATEGORY_FREQ].index
        if len(rare_values) > 0:
            combined_df[col] = combined_df[col].replace(rare_values, RARE_NAME)
            logger.info(f"Replaced {len(rare_values)} rare categories in '{col}' with '{RARE_NAME}'")

    # Label-encode all categoricals
    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        logger.info(f"Label-encoded categorical column '{col}'")

    # Assert no NaNs remain
    assert combined_df.isnull().sum().sum() == 0, "NaNs remain after preprocessing"
    logger.info("Preprocessing complete. No NaNs remaining.")

    # Split back into train and test
    train_processed = combined_df.iloc[:len(train_df)].copy()
    test_processed = combined_df.iloc[len(train_df):].copy()
    
    # Re-attach target to train
    train_processed[TARGET_COL] = train_df[TARGET_COL]
    
    feature_cols = num_cols + cat_cols
    return train_processed, test_processed, feature_cols

# =============================================================================
# PLOTTING (from v5)
# =============================================================================
def plot_roc_curves(oof_data, file_path):
    plt.figure(figsize=(10, 8))
    for col in oof_data.columns:
        if col not in [ID_COL, 'target']:
            fpr, tpr, _ = roc_curve(oof_data['target'], oof_data[col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{col} (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved ROC curves plot to {file_path}")

def plot_confusion_matrix(y_true, y_pred, threshold, file_path):
    cm = confusion_matrix(y_true, y_pred > threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix @ Threshold {threshold:.3f}')
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved confusion matrix plot to {file_path}")

def plot_feature_importance(importances, top_n, model_name, file_path):
    if importances.empty:
        logger.warning(f"Feature importance dataframe for {model_name} is empty. Skipping plot.")
        return
    top_importances = importances.nlargest(top_n, 'importance')
    plt.figure(figsize=(12, top_n / 2))
    sns.barplot(x='importance', y='feature', data=top_importances)
    plt.title(f'Top {top_n} Feature Importances ({model_name} - Gain)')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved {model_name} feature importance plot to {file_path}")

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    """Main training pipeline."""
    overall_start_time = time.time()
    set_seed(RANDOM_SEED)
    logger.info(f"Starting v6 run. Directory: {RUN_DIR}")
    logger.info(f"Random seed set to {RANDOM_SEED}")

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission_df = pd.read_csv("playground-series-s5e8/sample_submission.csv")
    logger.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

    # Preprocess data
    # v4: used a different preprocessing logic and was called inside the main block
    train_processed, test_processed, feature_cols = preprocess_data(train_df, test_df)
    
    X = train_processed[feature_cols]
    y = train_processed[TARGET_COL]
    X_test = test_processed[feature_cols]

    # Initialize prediction arrays
    oof_lgbm = np.zeros(len(train_df))
    oof_xgb = np.zeros(len(train_df))
    oof_cat = np.zeros(len(train_df))
    
    test_lgbm = np.zeros(len(test_df))
    test_xgb = np.zeros(len(test_df))
    test_cat = np.zeros(len(test_df))

    lgbm_importances = pd.DataFrame(columns=['feature', 'importance'])
    xgb_importances = pd.DataFrame(columns=['feature', 'importance'])

    fold_metrics = []

    # v4: Used ProcessPoolExecutor for parallel fold training.
    # This version uses a sequential loop for simplicity and easier debugging.
    logger.info(f"Starting training with {CV.get_n_splits()}-fold StratifiedKFold CV.")
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        fold_start_time = time.time()
        logger.info(f"===== Fold {fold+1} =====")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- LightGBM ---
        logger.info("Training LightGBM...")
        lgbm = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)],
                 eval_metric='auc',
                 callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        
        oof_lgbm[val_idx] = lgbm.predict_proba(X_val)[:, 1]
        test_lgbm += lgbm.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        lgbm_auc = roc_auc_score(y_val, oof_lgbm[val_idx])
        
        fold_imp = pd.DataFrame({'feature': feature_cols, 'importance': lgbm.feature_importances_})
        lgbm_importances = pd.concat([lgbm_importances, fold_imp], ignore_index=True)

        # --- XGBoost ---
        logger.info("Training XGBoost...")
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train,
                      eval_set=[(X_val, y_val)],
                      verbose=False)
        
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])

        fold_imp_xgb = pd.DataFrame({'feature': feature_cols, 'importance': xgb_model.feature_importances_})
        xgb_importances = pd.concat([xgb_importances, fold_imp_xgb], ignore_index=True)

        # --- CatBoost ---
        logger.info("Training CatBoost...")
        cat = cb.CatBoostClassifier(**CAT_PARAMS)
        cat.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False)
        
        oof_cat[val_idx] = cat.predict_proba(X_val)[:, 1]
        test_cat += cat.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        cat_auc = roc_auc_score(y_val, oof_cat[val_idx])

        fold_time = time.time() - fold_start_time
        logger.info(f"Fold {fold+1} AUCs -> LGBM: {lgbm_auc:.6f}, XGB: {xgb_auc:.6f}, CAT: {cat_auc:.6f}")
        logger.info(f"Fold {fold+1} completed in {fold_time:.2f} seconds.")
        
        fold_metrics.append({
            'fold': fold + 1,
            'lgbm_auc': lgbm_auc,
            'xgb_auc': xgb_auc,
            'cat_auc': cat_auc,
            'time_seconds': fold_time
        })
        
        del lgbm, xgb_model, cat
        gc.collect()

    # --- Post-CV Analysis ---
    logger.info("===== CV Finished. Analyzing results. =====")
    
    # OOF Scores
    oof_lgbm_score = roc_auc_score(y, oof_lgbm)
    oof_xgb_score = roc_auc_score(y, oof_xgb)
    oof_cat_score = roc_auc_score(y, oof_cat)
    logger.info(f"Overall OOF AUC -> LGBM: {oof_lgbm_score:.6f}, XGB: {oof_xgb_score:.6f}, CAT: {oof_cat_score:.6f}")

    # --- Ensembling ---
    oof_preds_df = pd.DataFrame({'lgbm': oof_lgbm, 'xgb': oof_xgb, 'cat': oof_cat})
    
    logger.info("Finding best blend weights with grid search...")
    best_auc = 0
    best_weights = [1/3, 1/3, 1/3]
    
    # Coarse grid search for weights
    for w in product(np.arange(0, 1.01, 0.05), repeat=3):
        if sum(w) > 1e-6:
            weights = np.array(w) / sum(w)
            blend_oof = oof_preds_df @ weights
            auc_score = roc_auc_score(y, blend_oof)
            if auc_score > best_auc:
                best_auc = auc_score
                best_weights = weights

    logger.info(f"Best blend weights (LGBM, XGB, CAT): {[round(w, 4) for w in best_weights]}")
    logger.info(f"Best OOF Blend AUC (Grid Search): {best_auc:.6f}")

    blend_oof_preds = oof_preds_df @ best_weights
    blend_test_preds = np.vstack([test_lgbm, test_xgb, test_cat]).T @ best_weights

    # --- Threshold Tuning ---
    logger.info("Tuning classification threshold on blended OOF predictions...")
    best_threshold = 0.5
    best_acc = 0
    best_f1 = 0
    
    thresholds = np.arange(0.01, 1.0, 0.01)
    for thr in thresholds:
        acc = accuracy_score(y, blend_oof_preds > thr)
        f1 = f1_score(y, blend_oof_preds > thr)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            best_threshold = thr
        elif acc == best_acc and f1 > best_f1: # F1 as tie-breaker
            best_f1 = f1
            best_threshold = thr

    logger.info(f"Best Threshold: {best_threshold:.3f}")
    logger.info(f"Accuracy @ Best Threshold: {best_acc:.6f}")
    logger.info(f"F1 Score @ Best Threshold: {best_f1:.6f}")

    # --- Leaderboard ---
    logger.info("\n" + "="*50 + "\n" + "FINAL LEADERBOARD" + "\n" + "="*50)
    leaderboard = pd.DataFrame({
        'Model': ['LightGBM', 'XGBoost', 'CatBoost', 'Blend(Grid)'],
        'AUC(OOF)': [oof_lgbm_score, oof_xgb_score, oof_cat_score, best_auc]
    })
    logger.info("\n" + leaderboard.to_string(index=False))
    logger.info(f"\nBest Threshold for Blend(Grid): {best_threshold:.3f} (Accuracy: {best_acc:.4f}, F1: {best_f1:.4f})")
    if USE_META:
        logger.warning("Logistic Regression meta-model was not run as USE_META=False.")

    # --- Save Artifacts ---
    logger.info("Saving artifacts...")
    
    # OOF predictions
    oof_df = pd.DataFrame({
        ID_COL: train_df[ID_COL],
        'target': y,
        'lgbm_oof': oof_lgbm,
        'xgb_oof': oof_xgb,
        'cat_oof': oof_cat,
        'blend_oof': blend_oof_preds
    })
    oof_df.to_csv(RUN_DIR / 'oof_predictions.csv', index=False)

    # Submissions
    submission_df = sample_submission_df.copy()
    submission_df[TARGET_COL] = blend_test_preds
    submission_df.to_csv(RUN_DIR / 'submission.csv', index=False)

    final_submission_df = sample_submission_df.copy()
    final_submission_df[TARGET_COL] = (blend_test_preds > best_threshold).astype(int)
    final_submission_df.to_csv(RUN_DIR / 'final_submission.csv', index=False)

    # Metrics
    pd.DataFrame(fold_metrics).to_csv(RUN_DIR / 'fold_metrics.csv', index=False)
    
    summary_metrics = {
        'oof_auc': {
            'lgbm': oof_lgbm_score,
            'xgb': oof_xgb_score,
            'cat': oof_cat_score,
            'blend': best_auc
        },
        'best_threshold': best_threshold,
        'accuracy_at_threshold': best_acc,
        'f1_at_threshold': best_f1,
        'blend_weights': {'lgbm': best_weights[0], 'xgb': best_weights[1], 'cat': best_weights[2]}
    }
    with open(RUN_DIR / 'summary_metrics.json', 'w') as f:
        json.dump(summary_metrics, f, indent=4)
    
    with open(RUN_DIR / 'weights.json', 'w') as f:
        json.dump(summary_metrics['blend_weights'], f, indent=4)

    # Feature Importances
    lgbm_importances_agg = lgbm_importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    lgbm_importances_agg.to_csv(RUN_DIR / 'feature_importance_lgbm.csv', index=False)
    
    xgb_importances_agg = xgb_importances.groupby('feature')['importance'].mean().sort_values(ascending=False).reset_index()
    xgb_importances_agg.to_csv(RUN_DIR / 'feature_importance_xgb.csv', index=False)

    # Plots
    plot_roc_curves(oof_df.drop(columns=['lgbm_oof', 'xgb_oof', 'cat_oof']), FIGS_DIR / 'roc_curve_blend.png')
    plot_roc_curves(oof_df, FIGS_DIR / 'roc_curves_all.png')
    plot_confusion_matrix(y, blend_oof_preds, best_threshold, FIGS_DIR / 'confusion_matrix.png')
    plot_feature_importance(lgbm_importances_agg, 30, 'LGBM', FIGS_DIR / 'feature_importance_lgbm.png')

    total_time = time.time() - overall_start_time
    logger.info(f"All artifacts saved. Total runtime: {total_time:.2f} seconds.")
    logger.info(f"Run v6 finished successfully. Find artifacts in {RUN_DIR}")

if __name__ == "__main__":
    main()
