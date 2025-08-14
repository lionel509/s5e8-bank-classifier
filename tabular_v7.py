import os
import gc
import time
import random
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.linear_model import LogisticRegressionCV

import lightgbm as lgb
import xgboost as xgb
import catboost as cb

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIG - Hardcoded paths and parameters
# =============================================================================
TRAIN_PATH = "playground-series-s5e8/train.csv"
TEST_PATH = "playground-series-s5e8/test.csv"
SAMPLE_SUBMISSION_PATH = "playground-series-s5e8/sample_submission.csv"

ID_COL = "id"
TARGET_COL = "y"

# Global Settings
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025) # Increased folds for stability
RANDOM_SEED = 2025

# Model Training Settings
EARLY_STOPPING_ROUNDS = 150
MAX_TREES = 10000 # Increased max trees
N_THREADS = -1

# Create timestamped run directory
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/v7/{RUN_TIMESTAMP}")
FIGS_DIR = RUN_DIR / "fig"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# Model Parameters (Slightly more aggressive)
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 48,
    'max_depth': 7,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.7,
    'bagging_freq': 1,
    'min_data_in_leaf': 30,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'verbosity': -1,
    'seed': RANDOM_SEED,
    'n_jobs': N_THREADS,
    'n_estimators': MAX_TREES,
}

XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.02,
    'max_depth': 6,
    'min_child_weight': 1,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'tree_method': 'hist',
    'seed': RANDOM_SEED,
    'n_jobs': N_THREADS,
    'n_estimators': MAX_TREES,
}

CAT_PARAMS = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'learning_rate': 0.02,
    'depth': 6,
    'l2_leaf_reg': 3,
    'bagging_temperature': 0.7,
    'random_seed': RANDOM_SEED,
    'verbose': False,
    'thread_count': N_THREADS,
    'iterations': MAX_TREES,
}

# =============================================================================
# SETUP LOGGING
# =============================================================================
def setup_logging(run_dir):
    log_file = run_dir / "run.log"
    logger = logging.getLogger('tabular_v7')
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    for handler in [file_handler, console_handler]:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logging(RUN_DIR)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
def feature_engineer(df):
    logger.info("Starting advanced feature engineering...")
    
    # Binary feature for previous contact
    df['was_contacted'] = (df['pdays'] != -1).astype(int)

    # Interaction features
    df['balance_per_age'] = df['balance'] / (df['age'] + 1e-6)
    df['duration_per_campaign'] = df['duration'] / (df['campaign'] + 1e-6)
    df['balance_per_duration'] = df['balance'] / (df['duration'] + 1e-6)
    
    # Polynomial features for key numerical columns
    for col in ['age', 'balance', 'duration']:
        df[f'{col}_sq'] = df[col]**2

    # Cyclical features for time
    month_map = {
        'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
    }
    # It's possible the month column is already numeric in some cases, so handle that
    if df['month'].dtype == 'object':
        df['month'] = df['month'].str.lower().map(month_map).fillna(0).astype(int)

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12.0)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12.0)
    df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31.0)
    df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31.0)
    
    logger.info(f"Created {len(df.columns)} total features.")
    return df

# =============================================================================
# PLOTTING
# =============================================================================
def plot_roc_curves(oof_data, file_path):
    plt.figure(figsize=(10, 8))
    for col in oof_data.columns:
        if col not in [ID_COL, 'target']:
            fpr, tpr, _ = roc_curve(oof_data['target'], oof_data[col])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{col} (AUC = {roc_auc:.5f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curves - OOF')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curves plot to {file_path}")

def plot_confusion_matrix(y_true, y_pred, threshold, file_path):
    cm = confusion_matrix(y_true, y_pred > threshold)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.title(f'Confusion Matrix @ Threshold {threshold:.3f}')
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {file_path}")

# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    overall_start_time = time.time()
    set_seed(RANDOM_SEED)
    logger.info(f"Starting v7 run. Directory: {RUN_DIR}")

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Pre-computation for feature engineering
    y = train_df[TARGET_COL]
    train_ids = train_df[ID_COL]
    test_ids = test_df[ID_COL]
    
    # Drop unnecessary columns
    train_df = train_df.drop(columns=[ID_COL, TARGET_COL])
    test_df = test_df.drop(columns=[ID_COL])

    # Feature Engineering
    train_df = feature_engineer(train_df)
    test_df = feature_engineer(test_df)

    # Label Encoding for object types
    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    feature_cols = train_df.columns.tolist()
    X = train_df
    X_test = test_df

    # Initialize OOF and test prediction arrays for Level 0 models
    oof_lgbm = np.zeros(len(X))
    oof_xgb = np.zeros(len(X))
    oof_cat = np.zeros(len(X))
    test_lgbm = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))

    fold_metrics = []

    logger.info(f"Starting Level 0 training with {CV.get_n_splits()}-fold CV.")
    for fold, (train_idx, val_idx) in enumerate(tqdm(CV.split(X, y), total=CV.get_n_splits(), desc="Training Folds")):
        fold_start_time = time.time()
        logger.info(f"===== Fold {fold+1} =====")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- LightGBM ---
        lgbm_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        oof_lgbm[val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
        test_lgbm += lgbm_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()

        # --- XGBoost ---
        xgb_model = xgb.XGBClassifier(**XGB_PARAMS)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_xgb[val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_xgb += xgb_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()

        # --- CatBoost ---
        cat_model = cb.CatBoostClassifier(**CAT_PARAMS)
        cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        oof_cat[val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_cat += cat_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()

        lgbm_auc = roc_auc_score(y_val, oof_lgbm[val_idx])
        xgb_auc = roc_auc_score(y_val, oof_xgb[val_idx])
        cat_auc = roc_auc_score(y_val, oof_cat[val_idx])
        logger.info(f"Fold {fold+1} AUCs -> LGBM: {lgbm_auc:.5f}, XGB: {xgb_auc:.5f}, CAT: {cat_auc:.5f}")
        fold_metrics.append({'fold': fold + 1, 'lgbm_auc': lgbm_auc, 'xgb_auc': xgb_auc, 'cat_auc': cat_auc})
        del lgbm_model, xgb_model, cat_model; gc.collect()

    # --- Level 0 Analysis ---
    logger.info("===== Level 0 CV Finished. Analyzing results. =====")
    oof_lgbm_score = roc_auc_score(y, oof_lgbm)
    oof_xgb_score = roc_auc_score(y, oof_xgb)
    oof_cat_score = roc_auc_score(y, oof_cat)
    logger.info(f"Overall OOF AUC -> LGBM: {oof_lgbm_score:.5f}, XGB: {oof_xgb_score:.5f}, CAT: {oof_cat_score:.5f}")

    # --- Level 1 Stacking ---
    logger.info("===== Starting Level 1 Stacking =====")
    
    # Create Level 1 training data from OOF predictions
    X_train_l1 = pd.DataFrame({'lgbm': oof_lgbm, 'xgb': oof_xgb, 'cat': oof_cat})
    X_test_l1 = pd.DataFrame({'lgbm': test_lgbm, 'xgb': test_xgb, 'cat': test_cat})

    # Train Logistic Regression meta-model
    logger.info("Training LogisticRegressionCV meta-model...")
    meta_model = LogisticRegressionCV(Cs=10, cv=5, scoring='roc_auc', penalty='l2', solver='liblinear', random_state=RANDOM_SEED)
    meta_model.fit(X_train_l1, y)

    # Get final predictions
    stacked_oof_preds = meta_model.predict_proba(X_train_l1)[:, 1]
    stacked_test_preds = meta_model.predict_proba(X_test_l1)[:, 1]

    stacked_auc = roc_auc_score(y, stacked_oof_preds)
    logger.info(f"Stacked Ensemble OOF AUC: {stacked_auc:.5f}")

    # --- Threshold Tuning ---
    logger.info("Tuning classification threshold on stacked OOF predictions...")
    best_threshold, best_acc, best_f1 = 0.5, 0, 0
    for thr in np.arange(0.05, 0.95, 0.01):
        acc = accuracy_score(y, stacked_oof_preds > thr)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1_score(y, stacked_oof_preds > thr)
            best_threshold = thr
    logger.info(f"Best Threshold: {best_threshold:.3f} -> Accuracy: {best_acc:.5f}, F1: {best_f1:.5f}")

    # --- Save Artifacts ---
    logger.info("Saving artifacts...")
    oof_df = pd.DataFrame({
        ID_COL: train_ids, 'target': y, 'lgbm': oof_lgbm, 'xgb': oof_xgb, 'cat': oof_cat, 'stacked': stacked_oof_preds
    })
    oof_df.to_csv(RUN_DIR / 'oof_predictions.csv', index=False)

    submission_df = sample_submission_df.copy()
    submission_df[TARGET_COL] = stacked_test_preds
    submission_df.to_csv(RUN_DIR / 'submission.csv', index=False)

    pd.DataFrame(fold_metrics).to_csv(RUN_DIR / 'fold_metrics.csv', index=False)
    
    summary_metrics = {
        'level0_oof_auc': {'lgbm': oof_lgbm_score, 'xgb': oof_xgb_score, 'cat': oof_cat_score},
        'level1_stacked_auc': stacked_auc,
        'best_threshold': best_threshold,
        'accuracy_at_threshold': best_acc,
        'f1_at_threshold': best_f1,
        'meta_model_coef': meta_model.coef_[0].tolist()
    }
    with open(RUN_DIR / 'summary_metrics.json', 'w') as f: json.dump(summary_metrics, f, indent=4)

    # --- Plots ---
    plot_roc_curves(oof_df.drop(columns=['lgbm', 'xgb', 'cat']), FIGS_DIR / 'roc_curve_stacked.png')
    plot_roc_curves(oof_df, FIGS_DIR / 'roc_curves_all.png')
    plot_confusion_matrix(y, stacked_oof_preds, best_threshold, FIGS_DIR / 'confusion_matrix.png')

    total_time = time.time() - overall_start_time
    logger.info(f"Run v7 finished successfully in {total_time:.2f} seconds. Find artifacts in {RUN_DIR}")

if __name__ == "__main__":
    main()
