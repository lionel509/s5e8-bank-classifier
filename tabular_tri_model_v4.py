#!/usr/bin/env python3
"""
Tabular Tri-Model v4 - Web Dashboard with Real-Time Training Monitoring
Based on v3 preprocessing but optimized for speed with LightGBM and web interface.
"""

import os
import gc
import time
import random
import logging
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

# Web dashboard imports
from flask import Flask, render_template, jsonify
import webbrowser
from threading import Timer

# =============================================================================
# CONFIG - Hardcoded paths and parameters
# =============================================================================
TRAIN_PATH = "playground-series-s5e8/train.csv"
TEST_PATH = "playground-series-s5e8/test.csv"

ID_COL = "id"
TARGET_COL = "y"
N_SPLITS = 3
RANDOM_SEED = 2025
MIN_CAT_COUNT = 25  # Rare category threshold
RARE_NAME = "__RARE__"

# Create timestamped run directory
RUN_TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/{RUN_TIMESTAMP}")
RUN_DIR.mkdir(parents=True, exist_ok=True)

# Output paths within run directory
OOF_OUTPUT_PATH = RUN_DIR / "oof_predictions.csv"
SUBMISSION_OUTPUT_PATH = RUN_DIR / "submission.csv"
METRICS_OUTPUT_PATH = RUN_DIR / "metrics.csv"
METADATA_OUTPUT_PATH = RUN_DIR / "run_metadata.json"
README_OUTPUT_PATH = RUN_DIR / "README.txt"
FIGS_DIR = RUN_DIR / "figs"
LOGS_DIR = RUN_DIR / "logs"

# Create subdirectories
FIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Web dashboard config
WEB_PORT = 8765
AUTO_OPEN_BROWSER = True

# LightGBM parameters optimized for speed and performance
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 127,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbosity': -1,
    'n_jobs': 1,  # Single thread within each fold to avoid nested parallelism
    'force_col_wise': True,
    'deterministic': True
}

EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 10000

# =============================================================================
# GLOBAL STATE FOR WEB DASHBOARD
# =============================================================================
training_state = {
    'status': 'initializing',
    'current_fold': 0,
    'total_folds': N_SPLITS,
    'fold_results': [],
    'overall_progress': 0,
    'start_time': None,
    'elapsed_time': 0,
    'oof_auc': None,
    'mean_cv_auc': None,
    'std_cv_auc': None,
    'logs': [],
    'feature_info': {},
    'data_info': {},
    'completed': False,
    'error': None
}

# =============================================================================
# WEB DASHBOARD
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'lgbm-dashboard-2025'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LightGBM Training Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: rgba(255,255,255,0.95); border-radius: 15px; 
            padding: 30px; margin-bottom: 20px; text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 { color: #4a5568; font-size: 2.5em; margin-bottom: 10px; }
        .header p { color: #718096; font-size: 1.1em; }
        .dashboard-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }
        .card { 
            background: rgba(255,255,255,0.95); border-radius: 15px; 
            padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .card h3 { color: #2d3748; margin-bottom: 15px; font-size: 1.3em; }
        .progress-container { background: #e2e8f0; border-radius: 10px; height: 20px; overflow: hidden; }
        .progress-bar { 
            background: linear-gradient(90deg, #48bb78, #38a169); 
            height: 100%; transition: width 0.3s ease; border-radius: 10px;
        }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-label { font-weight: 600; color: #4a5568; }
        .metric-value { 
            font-weight: bold; color: #2b6cb0; font-family: 'Courier New', monospace;
        }
        .fold-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }
        .fold-card { 
            background: linear-gradient(135deg, #f7fafc, #edf2f7); 
            border-radius: 10px; padding: 15px; text-align: center; border: 2px solid #e2e8f0;
        }
        .fold-card.completed { border-color: #48bb78; background: linear-gradient(135deg, #f0fff4, #c6f6d5); }
        .fold-card.active { border-color: #3182ce; background: linear-gradient(135deg, #ebf8ff, #bee3f8); animation: pulse 2s infinite; }
        .status { 
            display: inline-block; padding: 8px 16px; border-radius: 20px; 
            font-weight: bold; text-transform: uppercase; font-size: 0.8em;
        }
        .status.running { background: #bee3f8; color: #2c5282; }
        .status.completed { background: #c6f6d5; color: #2f855a; }
        .status.error { background: #fed7d7; color: #c53030; }
        .logs { 
            background: #1a202c; color: #e2e8f0; border-radius: 10px; 
            padding: 20px; height: 300px; overflow-y: auto; font-family: 'Courier New', monospace;
            font-size: 0.9em; line-height: 1.4;
        }
        .log-entry { margin: 5px 0; }
        .log-time { color: #81c784; }
        .log-info { color: #64b5f6; }
        .log-warning { color: #ffb74d; }
        .log-error { color: #e57373; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.7; } }
        .full-width { grid-column: 1 / -1; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 LightGBM Training Dashboard</h1>
            <p>Real-time monitoring of tabular classification training</p>
            <div style="margin-top: 15px;">
                <span id="status" class="status running">Initializing</span>
            </div>
        </div>
        
        <div class="dashboard-grid">
            <div class="card">
                <h3>📊 Training Progress</h3>
                <div class="metric">
                    <span class="metric-label">Overall Progress:</span>
                    <span class="metric-value" id="overall-progress">0%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-bar" id="progress-bar" style="width: 0%"></div>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Fold:</span>
                    <span class="metric-value" id="current-fold">0 / {{ total_folds }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Elapsed Time:</span>
                    <span class="metric-value" id="elapsed-time">0s</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Model Performance</h3>
                <div class="metric">
                    <span class="metric-label">OOF AUC:</span>
                    <span class="metric-value" id="oof-auc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Mean CV AUC:</span>
                    <span class="metric-value" id="mean-cv-auc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CV Std:</span>
                    <span class="metric-value" id="cv-std">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📈 Data Information</h3>
                <div class="metric">
                    <span class="metric-label">Train Samples:</span>
                    <span class="metric-value" id="train-samples">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Test Samples:</span>
                    <span class="metric-value" id="test-samples">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Features:</span>
                    <span class="metric-value" id="total-features">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Categorical:</span>
                    <span class="metric-value" id="cat-features">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Numerical:</span>
                    <span class="metric-value" id="num-features">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ Model Configuration</h3>
                <div class="metric">
                    <span class="metric-label">Algorithm:</span>
                    <span class="metric-value">LightGBM</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CV Folds:</span>
                    <span class="metric-value">{{ total_folds }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Random Seed:</span>
                    <span class="metric-value">{{ random_seed }}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Learning Rate:</span>
                    <span class="metric-value">{{ learning_rate }}</span>
                </div>
            </div>
        </div>
        
        <div class="card full-width">
            <h3>🔄 Fold Results</h3>
            <div class="fold-results" id="fold-results">
                <!-- Fold cards will be populated by JavaScript -->
            </div>
        </div>
        
        <div class="card full-width">
            <h3>📝 Training Logs</h3>
            <div class="logs" id="logs">
                <div class="log-entry">Waiting for training to start...</div>
            </div>
        </div>
    </div>

    <script>
        let updateInterval;
        
        function updateDashboard() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    // Update status
                    const statusEl = document.getElementById('status');
                    statusEl.textContent = data.status;
                    statusEl.className = `status ${data.completed ? 'completed' : (data.error ? 'error' : 'running')}`;
                    
                    // Update progress
                    document.getElementById('overall-progress').textContent = `${Math.round(data.overall_progress)}%`;
                    document.getElementById('progress-bar').style.width = `${data.overall_progress}%`;
                    document.getElementById('current-fold').textContent = `${data.current_fold} / ${data.total_folds}`;
                    document.getElementById('elapsed-time').textContent = `${Math.round(data.elapsed_time)}s`;
                    
                    // Update performance metrics
                    document.getElementById('oof-auc').textContent = data.oof_auc ? data.oof_auc.toFixed(6) : '-';
                    document.getElementById('mean-cv-auc').textContent = data.mean_cv_auc ? data.mean_cv_auc.toFixed(6) : '-';
                    document.getElementById('cv-std').textContent = data.std_cv_auc ? data.std_cv_auc.toFixed(6) : '-';
                    
                    // Update data info
                    if (data.data_info.train_shape) {
                        document.getElementById('train-samples').textContent = data.data_info.train_shape[0];
                        document.getElementById('test-samples').textContent = data.data_info.test_shape[0];
                    }
                    if (data.feature_info.total) {
                        document.getElementById('total-features').textContent = data.feature_info.total;
                        document.getElementById('cat-features').textContent = data.feature_info.categorical;
                        document.getElementById('num-features').textContent = data.feature_info.numerical;
                    }
                    
                    // Update fold results
                    updateFoldResults(data.fold_results, data.current_fold, data.total_folds);
                    
                    // Update logs
                    updateLogs(data.logs);
                    
                    // Stop updating if completed
                    if (data.completed && updateInterval) {
                        clearInterval(updateInterval);
                    }
                })
                .catch(error => console.error('Error updating dashboard:', error));
        }
        
        function updateFoldResults(results, currentFold, totalFolds) {
            const container = document.getElementById('fold-results');
            container.innerHTML = '';
            
            for (let i = 0; i < totalFolds; i++) {
                const foldCard = document.createElement('div');
                const isCompleted = results.length > i;
                const isActive = i === currentFold && !isCompleted;
                
                foldCard.className = `fold-card ${isCompleted ? 'completed' : (isActive ? 'active' : '')}`;
                foldCard.innerHTML = `
                    <h4>Fold ${i + 1}</h4>
                    <div style="margin-top: 10px;">
                        ${isCompleted ? 
                            `<div><strong>AUC:</strong> ${results[i].auc.toFixed(6)}</div>
                             <div><strong>Time:</strong> ${Math.round(results[i].time)}s</div>` :
                            (isActive ? '<div>Training...</div>' : '<div>Pending</div>')
                        }
                    </div>
                `;
                container.appendChild(foldCard);
            }
        }
        
        function updateLogs(logs) {
            const logsEl = document.getElementById('logs');
            logsEl.innerHTML = logs.map(log => {
                const logClass = log.includes('ERROR') ? 'log-error' : 
                                log.includes('WARNING') ? 'log-warning' : 
                                log.includes('INFO') ? 'log-info' : '';
                return `<div class="log-entry ${logClass}">${log}</div>`;
            }).join('');
            logsEl.scrollTop = logsEl.scrollHeight;
        }
        
        // Start updating
        updateDashboard();
        updateInterval = setInterval(updateDashboard, 1000);
    </script>
</body>
</html>
"""

@app.route('/')
def dashboard():
    return HTML_TEMPLATE.replace('{{ total_folds }}', str(N_SPLITS)) \
                       .replace('{{ random_seed }}', str(RANDOM_SEED)) \
                       .replace('{{ learning_rate }}', str(LGBM_PARAMS['learning_rate']))

@app.route('/api/status')
def api_status():
    return jsonify(training_state)

def run_web_server():
    """Run the Flask web server in a separate thread."""
    app.run(host='127.0.0.1', port=WEB_PORT, debug=False, use_reloader=False, threaded=True)

def open_browser():
    """Open the web browser to the dashboard URL."""
    webbrowser.open(f'http://127.0.0.1:{WEB_PORT}')

def update_training_state(key, value):
    """Thread-safe update of training state."""
    training_state[key] = value
    if key == 'start_time' and value:
        training_state['elapsed_time'] = 0
    elif training_state['start_time']:
        training_state['elapsed_time'] = time.time() - training_state['start_time']

def log_message(message, level='INFO'):
    """Add a log message to the training state."""
    timestamp = datetime.now().strftime('%H:%M:%S')
    log_entry = f"[{timestamp}] {level}: {message}"
    training_state['logs'].append(log_entry)
    # Keep only last 100 log entries
    if len(training_state['logs']) > 100:
        training_state['logs'] = training_state['logs'][-100:]
    print(log_entry)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def apply_rare_encoding(series: pd.Series, min_count: int = MIN_CAT_COUNT) -> pd.Series:
    """Apply rare category encoding to reduce cardinality."""
    value_counts = series.value_counts()
    rare_categories = value_counts[value_counts < min_count].index
    return series.where(~series.isin(rare_categories), RARE_NAME)

def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Preprocess train and test data with feature engineering from v3.
    Returns processed dataframes and column lists.
    """
    log_message("Starting data preprocessing...")
    
    # Identify feature columns
    feature_cols = [c for c in train_df.columns if c not in [TARGET_COL, ID_COL]]
    
    # Validate columns exist in test
    missing_cols = [c for c in feature_cols if c not in test_df.columns]
    assert not missing_cols, f"Missing columns in test: {missing_cols}"
    
    # Identify categorical and numerical columns
    obj_cols = [c for c in feature_cols if train_df[c].dtype == 'object']
    lowcard_int_cols = [
        c for c in feature_cols 
        if str(train_df[c].dtype).startswith('int') and train_df[c].nunique() <= 30
    ]
    cat_cols = sorted(list(set(obj_cols + lowcard_int_cols)))
    num_cols = sorted([c for c in feature_cols if c not in cat_cols])
    
    log_message(f"Found {len(cat_cols)} categorical columns and {len(num_cols)} numerical columns")
    
    # Process categorical columns with rare encoding and label encoding
    train_processed = train_df.copy()
    test_processed = test_df.copy()
    
    for col in cat_cols:
        # Apply rare encoding
        train_rare = apply_rare_encoding(train_processed[col].astype(str), MIN_CAT_COUNT)
        test_rare = apply_rare_encoding(test_processed[col].astype(str), MIN_CAT_COUNT)
        
        # Fit label encoder on combined data
        le = LabelEncoder()
        combined_data = pd.concat([train_rare, test_rare], axis=0).fillna("NA")
        le.fit(combined_data)
        
        # Transform both datasets
        train_processed[col] = le.transform(train_rare.fillna("NA"))
        test_processed[col] = le.transform(test_rare.fillna("NA"))
    
    log_message("Data preprocessing completed")
    return train_processed, test_processed, cat_cols, num_cols

def train_fold_worker(args):
    """
    Worker function to train a single fold. Designed for ProcessPoolExecutor.
    Returns: (fold_idx, fold_auc, oof_preds, test_preds, training_time)
    """
    fold_idx, train_indices, val_indices, X_train, y_train, X_test, cat_cols = args
    
    # Re-initialize random seed in worker process
    set_seed(RANDOM_SEED + fold_idx)
    
    start_time = time.time()
    
    # Split data for this fold
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(
        X_fold_train, 
        label=y_fold_train, 
        categorical_feature=cat_cols,
        free_raw_data=False
    )
    val_data = lgb.Dataset(
        X_fold_val, 
        label=y_fold_val, 
        categorical_feature=cat_cols,
        reference=train_data,
        free_raw_data=False
    )
    
    # Train model
    model = lgb.train(
        params=LGBM_PARAMS,
        train_set=train_data,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[val_data],
        valid_names=['valid'],
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(0)  # Suppress training logs
        ]
    )
    
    # Get predictions and ensure they are numpy arrays
    oof_preds = np.asarray(model.predict(X_fold_val, num_iteration=model.best_iteration)).flatten()
    test_preds = np.asarray(model.predict(X_test, num_iteration=model.best_iteration)).flatten()
    
    # Calculate fold AUC
    fold_auc = float(roc_auc_score(y_fold_val, oof_preds))
    
    training_time = time.time() - start_time
    
    # Clean up to save memory
    del model, train_data, val_data
    gc.collect()
    
    return fold_idx, fold_auc, oof_preds, test_preds, training_time

def main():
    """Main training pipeline with web dashboard."""
    overall_start_time = time.time()
    
    # Start web server in background
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # Open browser after a short delay
    if AUTO_OPEN_BROWSER:
        Timer(2.0, open_browser).start()
    
    # Initialize training state
    update_training_state('status', 'loading_data')
    update_training_state('start_time', overall_start_time)
    
    log_message("=" * 60)
    log_message("Starting Tabular Tri-Model v4 - LightGBM Parallel Training")
    log_message("=" * 60)
    log_message(f"Web dashboard available at: http://127.0.0.1:{WEB_PORT}")
    
    try:
        # Set random seed
        set_seed(RANDOM_SEED)
        log_message(f"Random seed set to: {RANDOM_SEED}")
        
        # Load data
        log_message("Loading data...")
        if not os.path.exists(TRAIN_PATH):
            raise FileNotFoundError(f"Train file not found: {TRAIN_PATH}")
        if not os.path.exists(TEST_PATH):
            raise FileNotFoundError(f"Test file not found: {TEST_PATH}")
        
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        # Update data info in state
        update_training_state('data_info', {
            'train_shape': train_df.shape,
            'test_shape': test_df.shape
        })
        
        log_message(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
        
        # Preprocess data
        update_training_state('status', 'preprocessing')
        train_processed, test_processed, cat_cols, num_cols = preprocess_data(train_df, test_df)
        
        # Prepare features and target
        feature_cols = cat_cols + num_cols
        X_train = train_processed[feature_cols]
        y_train = np.array(train_processed[TARGET_COL].values, dtype=np.float32)
        X_test = test_processed[feature_cols]
        
        # Update feature info in state
        update_training_state('feature_info', {
            'total': len(feature_cols),
            'categorical': len(cat_cols),
            'numerical': len(num_cols)
        })
        
        log_message(f"Features: {len(feature_cols)} ({len(cat_cols)} categorical, {len(num_cols)} numerical)")
        log_message(f"Target distribution: {float(y_train.mean()):.4f} positive rate")
        
        # Setup cross-validation
        update_training_state('status', 'training')
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        
        # Prepare arguments for parallel processing
        fold_args = []
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
            fold_args.append((
                fold_idx, train_indices, val_indices, 
                X_train, y_train, X_test, cat_cols
            ))
        
        # Initialize prediction arrays
        oof_predictions = np.zeros(len(X_train))
        test_predictions = np.zeros((len(X_test), N_SPLITS))
        fold_aucs = []
        
        # Parallel training using ProcessPoolExecutor
        log_message(f"Starting parallel training with {N_SPLITS} folds...")
        cv_start_time = time.time()
        
        max_workers = min(N_SPLITS, os.cpu_count() or 1)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fold training jobs
            future_to_fold = {executor.submit(train_fold_worker, args): args[0] for args in fold_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_fold):
                fold_idx = future_to_fold[future]
                try:
                    fold_idx, fold_auc, oof_preds, test_preds, training_time = future.result()
                    
                    # Store predictions
                    _, val_indices = fold_args[fold_idx][1], fold_args[fold_idx][2]
                    oof_predictions[val_indices] = oof_preds
                    test_predictions[:, fold_idx] = test_preds
                    fold_aucs.append(fold_auc)
                    
                    # Update training state
                    fold_result = {'auc': fold_auc, 'time': training_time}
                    training_state['fold_results'].append(fold_result)
                    update_training_state('current_fold', len(training_state['fold_results']))
                    update_training_state('overall_progress', (len(training_state['fold_results']) / N_SPLITS) * 100)
                    
                    log_message(f"Fold {fold_idx + 1}/{N_SPLITS} completed - AUC: {fold_auc:.6f} - Time: {training_time:.1f}s")
                    
                except Exception as exc:
                    log_message(f"Fold {fold_idx} generated an exception: {exc}", 'ERROR')
                    update_training_state('error', str(exc))
                    raise
        
        cv_time = time.time() - cv_start_time
        
        # Calculate overall OOF AUC
        oof_auc = float(roc_auc_score(y_train, oof_predictions))
        mean_fold_auc = np.mean(fold_aucs)
        std_fold_auc = np.std(fold_aucs)
        
        # Update final metrics
        update_training_state('oof_auc', oof_auc)
        update_training_state('mean_cv_auc', mean_fold_auc)
        update_training_state('std_cv_auc', std_fold_auc)
        
        log_message("=" * 60)
        log_message("CROSS-VALIDATION RESULTS")
        log_message("=" * 60)
        for i, auc in enumerate(fold_aucs):
            log_message(f"Fold {i + 1}: {auc:.6f}")
        log_message(f"Mean CV AUC: {mean_fold_auc:.6f} ± {std_fold_auc:.6f}")
        log_message(f"OOF AUC: {oof_auc:.6f}")
        log_message(f"CV Training Time: {cv_time:.1f}s")
        
        # Create final test predictions (mean of all folds)
        final_test_predictions = test_predictions.mean(axis=1)
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            ID_COL: train_processed[ID_COL],
            'oof': oof_predictions,
            TARGET_COL: y_train
        })
        oof_df.to_csv(OOF_OUTPUT_PATH, index=False)
        log_message(f"OOF predictions saved to: {OOF_OUTPUT_PATH}")
        
        # Save submission
        submission_df = pd.DataFrame({
            ID_COL: test_processed[ID_COL],
            TARGET_COL: final_test_predictions
        })
        submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        log_message(f"Submission saved to: {SUBMISSION_OUTPUT_PATH}")
        
        # Calculate total time
        total_time = time.time() - overall_start_time
        
        # Save metrics.csv (following the existing run format)
        metrics_data = []
        for i, auc in enumerate(fold_aucs):
            metrics_data.append({'fold': i, 'best_val_auc': auc})
        metrics_data.append({'fold': 'OOF', 'best_val_auc': oof_auc})
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
        log_message(f"Metrics saved to: {METRICS_OUTPUT_PATH}")
        
        # Save README.txt (following the existing run format)
        readme_content = f"""This folder contains outputs for a LightGBM run.
- metrics.csv: per-fold best AUC + OOF AUC
- oof_predictions.csv: OOF probabilities with IDs and targets  
- submission.csv: ready for Kaggle submit
- run_metadata.json: detailed run configuration and results
- logs/training.log: detailed training logs
- figs/*: overall figures (reserved for future ROC plots)

Run Details:
- Algorithm: LightGBM (with Web Dashboard)
- CV Folds: {N_SPLITS}
- Random Seed: {RANDOM_SEED}
- Total Features: {len(feature_cols)} ({len(cat_cols)} categorical, {len(num_cols)} numerical)
- Final OOF AUC: {oof_auc:.6f}
- Mean CV AUC: {mean_fold_auc:.6f} ± {std_fold_auc:.6f}
- Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)
- Web Dashboard: http://127.0.0.1:{WEB_PORT}
- Timestamp: {RUN_TIMESTAMP}
"""
        with open(README_OUTPUT_PATH, 'w') as f:
            f.write(readme_content)
        log_message(f"README saved to: {README_OUTPUT_PATH}")
        
        # Final summary
        total_time = time.time() - overall_start_time
        log_message("=" * 60)
        log_message("TRAINING COMPLETED")
        log_message("=" * 60)
        log_message(f"Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        log_message(f"Final OOF AUC: {oof_auc:.6f}")
        log_message(f"Submission shape: {submission_df.shape}")
        
        # Save run metadata
        metadata = {
            'oof_auc': float(oof_auc),
            'mean_cv_auc': float(mean_fold_auc),
            'std_cv_auc': float(std_fold_auc),
            'fold_aucs': [float(auc) for auc in fold_aucs],
            'n_splits': N_SPLITS,
            'random_seed': RANDOM_SEED,
            'total_time_seconds': total_time,
            'cv_time_seconds': cv_time,
            'lgbm_params': LGBM_PARAMS,
            'feature_counts': {
                'total': len(feature_cols),
                'categorical': len(cat_cols),
                'numerical': len(num_cols)
            }
        }
        
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        log_message(f"Run metadata saved to: {METADATA_OUTPUT_PATH}")
        log_message("🎉 Training completed successfully!")
        
        # Mark as completed
        update_training_state('status', 'completed')
        update_training_state('completed', True)
        
    except Exception as e:
        log_message(f"Training failed with error: {str(e)}", 'ERROR')
        update_training_state('error', str(e))
        update_training_state('status', 'error')
        raise
    
    # Keep the web server running for a while so user can see final results
    log_message("Web dashboard will remain available. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log_message("Shutting down...")

if __name__ == "__main__":
    main()
