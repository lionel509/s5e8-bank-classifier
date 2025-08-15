#!/usr/bin/env python3
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
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import optuna

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
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=2025)
RANDOM_SEED = 2025
N_THREADS = -1

# Optuna Settings
N_TRIALS = 30  # Number of Optuna trials for each model
OPTUNA_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED) # Faster CV for tuning

# Model Training Settings
EARLY_STOPPING_ROUNDS = 150
MAX_TREES = 10000

# Create timestamped run directory
RUN_TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/v9/{RUN_TIMESTAMP}")
FIGS_DIR = RUN_DIR / "fig"
RUN_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(exist_ok=True)

# =============================================================================
# OPTUNA HYPERPARAMETER TUNING
# =============================================================================

def objective_lgbm(trial, X, y, pbar):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_estimators': MAX_TREES,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbosity': -1,
        'seed': RANDOM_SEED,
        'n_jobs': N_THREADS,
    }
    
    scores = []
    for fold, (train_idx, val_idx) in enumerate(OPTUNA_CV.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))
    
    pbar.update(1)
    return np.mean(scores)

def objective_xgb(trial, X, y, pbar):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'n_estimators': MAX_TREES,
        'eta': trial.suggest_float('eta', 1e-3, 0.1, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'tree_method': 'hist',
        'seed': RANDOM_SEED,
        'n_jobs': N_THREADS,
    }

    scores = []
    for fold, (train_idx, val_idx) in enumerate(OPTUNA_CV.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    pbar.update(1)
    return np.mean(scores)

def objective_cat(trial, X, y, pbar):
    params = {
        'loss_function': 'Logloss',
        'eval_metric': 'AUC',
        'iterations': MAX_TREES,
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1, log=True),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0),
        'random_strength': trial.suggest_float('random_strength', 1e-8, 10.0, log=True),
        'random_seed': RANDOM_SEED,
        'verbose': False,
        'thread_count': N_THREADS,
    }

    scores = []
    for fold, (train_idx, val_idx) in enumerate(OPTUNA_CV.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train, 
                  eval_set=[(X_val, y_val)], 
                  early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
                  verbose=False)
        
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    pbar.update(1)
    return np.mean(scores)

def objective_rf(trial, X, y, pbar):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        'n_jobs': N_THREADS,
    }

    model = RandomForestClassifier(**params)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(OPTUNA_CV.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    pbar.update(1)
    return np.mean(scores)

def objective_et(trial, X, y, pbar):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 1000, step=100),
        'max_depth': trial.suggest_int('max_depth', 5, 50),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
        'max_features': trial.suggest_float('max_features', 0.5, 1.0),
        'class_weight': 'balanced',
        'random_state': RANDOM_SEED,
        'n_jobs': N_THREADS,
    }

    model = ExtraTreesClassifier(**params)
    scores = []
    for fold, (train_idx, val_idx) in enumerate(OPTUNA_CV.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict_proba(X_val)[:, 1]
        scores.append(roc_auc_score(y_val, preds))

    pbar.update(1)
    return np.mean(scores)

# =============================================================================
# SETUP LOGGING
# =============================================================================
def setup_logging(run_dir):
    log_file = run_dir / "run.log"
    logger = logging.getLogger('tabular_v9')
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

def plot_feature_importances(importances, feature_names, model_name, file_path):
    mean_importances = np.mean(importances, axis=0)
    df = pd.DataFrame({'feature': feature_names, 'importance': mean_importances})
    df = df.sort_values('importance', ascending=False).head(30)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=df)
    plt.title(f'Top 30 Feature Importances - {model_name.upper()}')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved feature importances plot for {model_name} to {file_path}")

def plot_oof_distributions(oof_df, file_path):
    num_models = len(oof_df.columns) - 2 # exclude id and target
    plt.figure(figsize=(15, 4 * ((num_models + 1) // 2)))
    for i, col in enumerate([c for c in oof_df.columns if c not in [ID_COL, 'target']]):
        plt.subplot(3, 3, i + 1)
        sns.histplot(oof_df[col], bins=50, kde=True)
        plt.title(f'OOF Prediction Distribution - {col.upper()}')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved OOF distributions plot to {file_path}")

def plot_oof_correlation(oof_df, file_path):
    oof_preds = oof_df.drop(columns=[ID_COL, 'target'])
    corr = oof_preds.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.3f', cmap='viridis')
    plt.title('Correlation Heatmap of OOF Predictions')
    plt.tight_layout()
    plt.savefig(file_path, dpi=300)
    plt.close()
    logger.info(f"Saved OOF correlation heatmap to {file_path}")

def save_optuna_plots(study, model_name, figs_dir):
    try:
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image(figs_dir / f"optuna_history_{model_name}.png")
        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image(figs_dir / f"optuna_param_importances_{model_name}.png")
        logger.info(f"Saved Optuna plots for {model_name}")
    except (ImportError, ValueError) as e:
        logger.warning(f"Could not save Optuna plots for {model_name}: {e}. Try `pip install plotly`.")


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================
def main():
    overall_start_time = time.time()
    set_seed(RANDOM_SEED)
    logger.info(f"Starting v9 run. Directory: {RUN_DIR}")

    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)

    # Pre-computation for feature engineering
    y = train_df[TARGET_COL]
    train_ids = train_df[ID_COL]
    test_ids = test_df[ID_COL]
    
    train_df = train_df.drop(columns=[ID_COL, TARGET_COL])
    test_df = test_df.drop(columns=[ID_COL])

    # Feature Engineering & Label Encoding
    train_df = feature_engineer(train_df)
    test_df = feature_engineer(test_df)

    cat_cols = [c for c in train_df.columns if train_df[c].dtype == 'object']
    for col in cat_cols:
        le = LabelEncoder()
        train_df[col] = le.fit_transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    X = train_df
    X_test = test_df

    # =========================================================================
    # STAGE 1: HYPERPARAMETER TUNING
    # =========================================================================
    logger.info("===== STAGE 1: Hyperparameter Tuning with Optuna =====")
    
    objectives = {
        "lgbm": objective_lgbm,
        "xgb": objective_xgb,
        "cat": objective_cat,
        "rf": objective_rf,
        "et": objective_et
    }
    
    best_params = {}
    for model_name, objective_func in objectives.items():
        logger.info(f"--- Tuning {model_name.upper()} ---")
        study_name = f"{model_name}_study"
        storage_name = f"sqlite:///{RUN_DIR}/{study_name}.db"
        study = optuna.create_study(direction='maximize', study_name=study_name, storage=storage_name, load_if_exists=True)
        with tqdm(total=N_TRIALS, desc=f"Tuning {model_name.upper()}") as pbar:
            study.optimize(lambda trial: objective_func(trial, X, y, pbar), n_trials=N_TRIALS)
        best_params[model_name] = study.best_params
        logger.info(f"Best AUC for {model_name.upper()}: {study.best_value:.5f}")
        logger.info(f"Best params: {study.best_params}")
        save_optuna_plots(study, model_name, FIGS_DIR)

    # Save best parameters
    with open(RUN_DIR / 'best_params.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    logger.info(f"Saved best hyperparameters to {RUN_DIR / 'best_params.json'}")

    # =========================================================================
    # STAGE 2: FINAL MODEL TRAINING
    # =========================================================================
    logger.info("===== STAGE 2: Training Final Models with Best Hyperparameters =====")

    # Initialize OOF and test prediction arrays
    oof_preds = {model: np.zeros(len(X)) for model in best_params}
    test_preds = {model: np.zeros(len(X_test)) for model in best_params}
    fold_metrics = []
    feature_importances = {model: [] for model in best_params}

    pbar = tqdm(total=CV.get_n_splits() * len(best_params), desc="Training Final Models")
    for fold, (train_idx, val_idx) in enumerate(CV.split(X, y)):
        logger.info(f"===== Fold {fold+1} =====")
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # --- LGBM ---
        lgbm_params = best_params['lgbm']
        lgbm_params.update({'n_estimators': MAX_TREES, 'seed': RANDOM_SEED, 'n_jobs': N_THREADS, 'verbosity': -1})
        lgbm_model = lgb.LGBMClassifier(**lgbm_params)
        lgbm_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
        oof_preds['lgbm'][val_idx] = lgbm_model.predict_proba(X_val)[:, 1]
        test_preds['lgbm'] += lgbm_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        feature_importances['lgbm'].append(lgbm_model.feature_importances_)
        pbar.update(1)

        # --- XGBoost ---
        xgb_params = best_params['xgb']
        xgb_params.update({'n_estimators': MAX_TREES, 'seed': RANDOM_SEED, 'n_jobs': N_THREADS})
        xgb_model = xgb.XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        oof_preds['xgb'][val_idx] = xgb_model.predict_proba(X_val)[:, 1]
        test_preds['xgb'] += xgb_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        feature_importances['xgb'].append(xgb_model.feature_importances_)
        pbar.update(1)

        # --- CatBoost ---
        cat_params = best_params['cat']
        cat_params.update({'iterations': MAX_TREES, 'random_seed': RANDOM_SEED, 'thread_count': N_THREADS, 'verbose': False})
        cat_model = cb.CatBoostClassifier(**cat_params)
        cat_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
        oof_preds['cat'][val_idx] = cat_model.predict_proba(X_val)[:, 1]
        test_preds['cat'] += cat_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        feature_importances['cat'].append(cat_model.feature_importances_)
        pbar.update(1)

        # --- RandomForest ---
        rf_params = best_params['rf']
        rf_params.update({'random_state': RANDOM_SEED, 'n_jobs': N_THREADS, 'class_weight': 'balanced'})
        rf_model = RandomForestClassifier(**rf_params)
        rf_model.fit(X_train, y_train)
        oof_preds['rf'][val_idx] = rf_model.predict_proba(X_val)[:, 1]
        test_preds['rf'] += rf_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        feature_importances['rf'].append(rf_model.feature_importances_)
        pbar.update(1)

        # --- ExtraTrees ---
        et_params = best_params['et']
        et_params.update({'random_state': RANDOM_SEED, 'n_jobs': N_THREADS, 'class_weight': 'balanced'})
        et_model = ExtraTreesClassifier(**et_params)
        et_model.fit(X_train, y_train)
        oof_preds['et'][val_idx] = et_model.predict_proba(X_val)[:, 1]
        test_preds['et'] += et_model.predict_proba(X_test)[:, 1] / CV.get_n_splits()
        feature_importances['et'].append(et_model.feature_importances_)
        pbar.update(1)

        fold_auc = {name: roc_auc_score(y_val, oof_preds[name][val_idx]) for name in best_params}
        fold_auc['fold'] = fold + 1
        fold_metrics.append(fold_auc)
        logger.info(f"Fold {fold+1} AUCs: " + ", ".join([f"{k.upper()}: {v:.5f}" for k, v in fold_auc.items() if k != 'fold']))
        del lgbm_model, xgb_model, cat_model, rf_model, et_model; gc.collect()
    
    pbar.close()

    # --- Level 0 Analysis ---
    logger.info("===== Level 0 CV Finished. Analyzing results. =====")
    oof_scores = {name: roc_auc_score(y, oof_preds[name]) for name in best_params}
    logger.info("Overall OOF AUCs: " + ", ".join([f"{k.upper()}: {v:.5f}" for k, v in oof_scores.items()]))

    # --- Level 1 Stacking ---
    logger.info("===== Starting Level 1 Stacking =====")
    X_train_l1 = pd.DataFrame(oof_preds)
    X_test_l1 = pd.DataFrame(test_preds)

    meta_model = LogisticRegressionCV(Cs=10, cv=5, scoring='roc_auc', penalty='l2', solver='liblinear', random_state=RANDOM_SEED)
    meta_model.fit(X_train_l1, y)

    stacked_oof_preds = meta_model.predict_proba(X_train_l1)[:, 1]
    stacked_test_preds = meta_model.predict_proba(X_test_l1)[:, 1]
    stacked_auc = roc_auc_score(y, stacked_oof_preds)
    logger.info(f"Stacked Ensemble OOF AUC: {stacked_auc:.5f}")

    # --- Threshold Tuning & Artifacts ---
    logger.info("Tuning threshold and saving artifacts...")
    best_threshold, best_acc, best_f1 = 0.5, 0, 0
    for thr in np.arange(0.05, 0.95, 0.01):
        acc = accuracy_score(y, stacked_oof_preds > thr)
        if acc > best_acc:
            best_acc = acc
            best_f1 = f1_score(y, stacked_oof_preds > thr)
            best_threshold = thr
    logger.info(f"Best Threshold: {best_threshold:.3f} -> Accuracy: {best_acc:.5f}, F1: {best_f1:.5f}")

    oof_df = pd.DataFrame(oof_preds)
    oof_df[ID_COL] = train_ids
    oof_df['target'] = y
    oof_df['stacked'] = stacked_oof_preds
    oof_df.to_csv(RUN_DIR / 'oof_predictions.csv', index=False)

    submission_df = sample_submission_df.copy()
    submission_df[TARGET_COL] = stacked_test_preds
    submission_df.to_csv(RUN_DIR / 'submission.csv', index=False)

    pd.DataFrame(fold_metrics).to_csv(RUN_DIR / 'fold_metrics.csv', index=False)
    
    summary_metrics = {
        'level0_oof_auc': oof_scores,
        'level1_stacked_auc': stacked_auc,
        'best_threshold': best_threshold,
        'accuracy_at_threshold': best_acc,
        'f1_at_threshold': best_f1,
        'meta_model_coef': meta_model.coef_[0].tolist()
    }
    with open(RUN_DIR / 'summary_metrics.json', 'w') as f: json.dump(summary_metrics, f, indent=4)

    # --- Plots ---
    plot_roc_curves(oof_df.drop(columns=['stacked', ID_COL, 'target']), FIGS_DIR / 'roc_curve_level0.png')
    plot_roc_curves(oof_df[[ID_COL, 'target', 'stacked']], FIGS_DIR / 'roc_curve_stacked.png')
    plot_confusion_matrix(y, stacked_oof_preds, best_threshold, FIGS_DIR / 'confusion_matrix.png')
    plot_oof_distributions(oof_df.drop(columns=[ID_COL, 'target']), FIGS_DIR / 'oof_distributions.png')
    plot_oof_correlation(oof_df.drop(columns=[ID_COL, 'target', 'stacked']), FIGS_DIR / 'oof_correlation.png')
    for model_name, importances in feature_importances.items():
        plot_feature_importances(importances, X.columns.tolist(), model_name, FIGS_DIR / f'feature_importance_{model_name}.png')

    total_time = time.time() - overall_start_time
    logger.info(f"Run v9 finished successfully in {total_time:.2f} seconds. Find artifacts in {RUN_DIR}")

if __name__ == "__main__":
    main()
