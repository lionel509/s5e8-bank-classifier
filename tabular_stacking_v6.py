import os
import gc
import time
import random
import logging
import json
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, List, Dict, Optional, Union
import glob
from itertools import product
from scipy.optimize import minimize

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

# PyTorch imports for neural network
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Web dashboard imports
from flask import Flask, render_template, jsonify
import webbrowser
from threading import Timer
import threading

# =============================================================================
# CONFIG - Hardcoded paths and parameters
# =============================================================================
TRAIN_PATH = "playground-series-s5e8/train.csv"
TEST_PATH = "playground-series-s5e8/test.csv"

ID_COL = "id"
TARGET_COL = "y"
N_SPLITS = 5
RANDOM_SEED = 2025
MIN_CAT_COUNT = 25  # Rare category threshold
RARE_NAME = "__RARE__"
HIGH_CARD_THRESHOLD = 50  # Threshold for K-fold target encoding

# Training modes
TRAINING_MODE = "PUSH"  # "FAST" or "PUSH"
TARGET_AUC = 0.97  # Target performance threshold

# Create timestamped run directory with stacking subfolder
RUN_TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/{RUN_TIMESTAMP}")
STACKING_DIR = RUN_DIR / "stacking_v52"
STACKING_DIR.mkdir(parents=True, exist_ok=True)

# Output subdirectories
OOF_DIR = STACKING_DIR / "oof"
PREDS_DIR = STACKING_DIR / "preds"
FIGS_DIR = STACKING_DIR / "figs"
LOGS_DIR = STACKING_DIR / "logs"

# Create all subdirectories
for dir_path in [OOF_DIR, PREDS_DIR, FIGS_DIR, LOGS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Output paths
OOF_OUTPUT_PATH = OOF_DIR / "oof_predictions.csv"
SUBMISSION_OUTPUT_PATH = PREDS_DIR / "final_submission.csv"
METRICS_OUTPUT_PATH = LOGS_DIR / "metrics.csv"
METADATA_OUTPUT_PATH = LOGS_DIR / "run_metadata.json"
README_OUTPUT_PATH = STACKING_DIR / "README.txt"
WEIGHTS_OUTPUT_PATH = LOGS_DIR / "ensemble_weights.json"

# Web dashboard config
WEB_PORT = 8765
AUTO_OPEN_BROWSER = True

# Model-specific parameters - adaptive based on training mode
def get_model_params(mode: str = "FAST"):
    """Get model parameters based on training mode"""
    
    if mode == "FAST":
        # Fast mode - ~30 minute target
        nn_params = {
            'hidden_layers': [256, 128],
            'dropout': 0.2,
            'emb_dropout': 0.1,
            'input_dropout': 0.1,
            'batch_size': 4096,
            'epochs': 5,
            'patience': 3,
            'base_lr': 1e-3,
            'weight_decay': 1e-5,
            'use_mps': True,
            'n_seeds': 2,  # 2 seeds for fast mode
        }
        
        catboost_params = {
            'iterations': 500,
            'learning_rate': 0.1,
            'depth': 6,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'early_stopping_rounds': 30,
            'task_type': 'CPU',
            'thread_count': -1,
        }
        
        xgboost_params = {
            'n_estimators': 500,
            'learning_rate': 0.1,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        lgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'n_estimators': 500,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'verbosity': -1,
            'n_jobs': -1,
        }
        
    else:  # PUSH mode
        # Push mode - maximum performance
        nn_params = {
            'hidden_layers': [512, 256, 128],
            'dropout': 0.2,
            'emb_dropout': 0.1,
            'input_dropout': 0.1,
            'batch_size': 2048,
            'epochs': 15,
            'patience': 7,
            'base_lr': 1e-3,
            'weight_decay': 1e-5,
            'use_mps': True,
            'n_seeds': 5,  # 5 seeds for push mode
        }
        
        catboost_params = {
            'iterations': 2000,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'early_stopping_rounds': 100,
            'task_type': 'CPU',
            'thread_count': -1,
        }
        
        xgboost_params = {
            'n_estimators': 2000,
            'learning_rate': 0.05,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        lgbm_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'n_estimators': 2000,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': RANDOM_SEED,
            'verbosity': -1,
            'n_jobs': -1,
        }
    
    # Meta-model parameters (same for both modes)
    meta_lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 7,
        'learning_rate': 0.1,
        'n_estimators': 100,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': RANDOM_SEED,
        'verbosity': -1,
        'n_jobs': 1,
    }
    
    return {
        'neural_network': nn_params,
        'catboost': catboost_params,
        'xgboost': xgboost_params,
        'lightgbm': lgbm_params,
        'meta_lightgbm': meta_lgb_params
    }

# =============================================================================
# GLOBAL STATE FOR WEB DASHBOARD
# =============================================================================
training_state = {
    'status': 'initializing',
    'current_stage': 'preprocessing',
    'current_model': '',
    'current_fold': 0,
    'total_folds': N_SPLITS,
    'fold_results': {
        'neural_network': [],
        'catboost': [],
        'xgboost': [],
        'lightgbm': [],
        'meta_logistic': [],
        'meta_lightgbm': [],
        'meta_weighted': [],
        'meta_rank': []
    },
    'overall_progress': 0,
    'start_time': None,
    'elapsed_time': 0,
    'oof_aucs': {
        'neural_network': None,
        'catboost': None,
        'xgboost': None,
        'lightgbm': None,
        'meta_logistic': None,
        'meta_lightgbm': None,
        'meta_weighted': None,
        'meta_rank': None,
        'best_model': None
    },
    'mean_cv_aucs': {
        'neural_network': None,
        'catboost': None,
        'xgboost': None,
        'lightgbm': None
    },
    'logs': [],
    'feature_info': {},
    'data_info': {},
    'completed': False,
    'error': None,
    'ensemble_weights': {},
    'mode': TRAINING_MODE,
    'target_auc': TARGET_AUC
}

# =============================================================================
# WEB DASHBOARD
# =============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'stacking-dashboard-2025'

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Stacking Pipeline Dashboard</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #333; min-height: 100vh; padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { 
            background: rgba(255,255,255,0.95); border-radius: 15px; 
            padding: 30px; margin-bottom: 20px; text-align: center;
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .header h1 { color: #4a5568; font-size: 2.5em; margin-bottom: 10px; }
        .header p { color: #718096; font-size: 1.1em; }
        .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(350px, 1fr)); gap: 20px; margin-bottom: 20px; }
        .card { 
            background: rgba(255,255,255,0.95); border-radius: 15px; 
            padding: 25px; box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .card h3 { color: #2d3748; margin-bottom: 15px; font-size: 1.3em; }
        .progress-container { background: #e2e8f0; border-radius: 10px; height: 20px; overflow: hidden; margin: 10px 0; }
        .progress-bar { 
            background: linear-gradient(90deg, #48bb78, #38a169); 
            height: 100%; transition: width 0.3s ease; border-radius: 10px;
        }
        .metric { display: flex; justify-content: space-between; margin: 10px 0; }
        .metric-label { font-weight: 600; color: #4a5568; }
        .metric-value { 
            font-weight: bold; color: #2b6cb0; font-family: 'Courier New', monospace;
        }
        .model-results { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
        .model-card { 
            background: linear-gradient(135deg, #f7fafc, #edf2f7); 
            border-radius: 10px; padding: 15px; text-align: center; border: 2px solid #e2e8f0;
        }
        .model-card.completed { border-color: #48bb78; background: linear-gradient(135deg, #f0fff4, #c6f6d5); }
        .model-card.active { border-color: #3182ce; background: linear-gradient(135deg, #ebf8ff, #bee3f8); animation: pulse 2s infinite; }
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
        .stage-indicator { 
            display: flex; justify-content: space-between; margin: 20px 0;
            padding: 15px; background: rgba(255,255,255,0.1); border-radius: 10px;
        }
        .stage { 
            flex: 1; text-align: center; padding: 10px; border-radius: 8px; 
            margin: 0 5px; color: white; font-weight: bold;
        }
        .stage.active { background: #3182ce; }
        .stage.completed { background: #48bb78; }
        .stage.pending { background: #a0aec0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Stacking Pipeline Dashboard</h1>
            <p>Real-time monitoring of Neural Network + CatBoost + LightGBM stacking ensemble</p>
            <div style="margin-top: 15px;">
                <span id="status" class="status running">Initializing</span>
            </div>
        </div>
        
        <div class="stage-indicator">
            <div class="stage active" id="stage-preprocessing">Preprocessing</div>
            <div class="stage pending" id="stage-level1">Level 1 Models</div>
            <div class="stage pending" id="stage-level2">Meta Model</div>
            <div class="stage pending" id="stage-complete">Complete</div>
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
                    <span class="metric-label">Current Stage:</span>
                    <span class="metric-value" id="current-stage">Preprocessing</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Model:</span>
                    <span class="metric-value" id="current-model">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Current Fold:</span>
                    <span class="metric-value" id="current-fold">0 / """ + str(N_SPLITS) + """</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Elapsed Time:</span>
                    <span class="metric-value" id="elapsed-time">0s</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🎯 Model Performance</h3>
                <div class="metric">
                    <span class="metric-label">Neural Network OOF AUC:</span>
                    <span class="metric-value" id="nn-oof-auc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CatBoost OOF AUC:</span>
                    <span class="metric-value" id="cb-oof-auc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Meta Model OOF AUC:</span>
                    <span class="metric-value" id="meta-oof-auc">-</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Best Model:</span>
                    <span class="metric-value" id="best-model">-</span>
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
                <div class="metric">
                    <span class="metric-label">Target Rate:</span>
                    <span class="metric-value" id="target-rate">-</span>
                </div>
            </div>
            
            <div class="card">
                <h3>⚙️ Pipeline Configuration</h3>
                <div class="metric">
                    <span class="metric-label">Level 1 Models:</span>
                    <span class="metric-value">Neural Network + CatBoost</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Meta Model:</span>
                    <span class="metric-value">LightGBM</span>
                </div>
                <div class="metric">
                    <span class="metric-label">CV Folds:</span>
                    <span class="metric-value">""" + str(N_SPLITS) + """</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Random Seed:</span>
                    <span class="metric-value">""" + str(RANDOM_SEED) + """</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Device:</span>
                    <span class="metric-value" id="device">-</span>
                </div>
            </div>
        </div>
        
        <div class="card full-width">
            <h3>🔄 Model Results by Fold</h3>
            <div class="model-results" id="model-results">
                <!-- Model results will be populated by JavaScript -->
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
                    statusEl.className = `status ${data.status}`;
                    
                    // Update stage indicators
                    updateStageIndicator(data.current_stage);
                    
                    // Update progress
                    document.getElementById('overall-progress').textContent = data.overall_progress + '%';
                    document.getElementById('progress-bar').style.width = data.overall_progress + '%';
                    document.getElementById('current-stage').textContent = data.current_stage;
                    document.getElementById('current-model').textContent = data.current_model;
                    document.getElementById('current-fold').textContent = `${data.current_fold} / ${data.total_folds}`;
                    
                    // Update timing
                    const elapsed = data.elapsed_time;
                    document.getElementById('elapsed-time').textContent = elapsed < 60 ? 
                        `${elapsed.toFixed(0)}s` : `${(elapsed/60).toFixed(1)}m`;
                    
                    // --- START OF FIX ---
                    const meta_aucs = {
                        logistic: data.oof_aucs.meta_logistic,
                        lightgbm: data.oof_aucs.meta_lightgbm,
                        weighted: data.oof_aucs.meta_weighted,
                        rank: data.oof_aucs.meta_rank
                    };
                    const meta_values = Object.values(meta_aucs).filter(v => v);
                    const best_meta_auc = meta_values.length > 0 ? Math.max(...meta_values) : 0;

                    // Update performance metrics
                    document.getElementById('nn-oof-auc').textContent = 
                        data.oof_aucs.neural_network ? data.oof_aucs.neural_network.toFixed(6) : '-';
                    document.getElementById('cb-oof-auc').textContent = 
                        data.oof_aucs.catboost ? data.oof_aucs.catboost.toFixed(6) : '-';
                    document.getElementById('meta-oof-auc').textContent = 
                        best_meta_auc ? best_meta_auc.toFixed(6) : '-';
                    
                    // Update best model
                    const all_aucs = {
                        'Neural Network': data.oof_aucs.neural_network,
                        'CatBoost': data.oof_aucs.catboost,
                        'XGBoost': data.oof_aucs.xgboost,
                        'LightGBM': data.oof_aucs.lightgbm,
                        'Meta Model': best_meta_auc
                    };

                    let bestModel = '-';
                    let bestAuc = 0;

                    for (const [name, auc] of Object.entries(all_aucs)) {
                        if (auc && auc > bestAuc) {
                            bestAuc = auc;
                            bestModel = name;
                        }
                    }

                    if (bestAuc > 0) {
                        document.getElementById('best-model').textContent = `${bestModel} (${bestAuc.toFixed(6)})`;
                    }
                    // --- END OF FIX ---

                    // Update data info
                    if (data.data_info) {
                        document.getElementById('train-samples').textContent = data.data_info.train_samples || '-';
                        document.getElementById('test-samples').textContent = data.data_info.test_samples || '-';
                        document.getElementById('total-features').textContent = data.data_info.total_features || '-';
                        document.getElementById('cat-features').textContent = data.data_info.categorical_features || '-';
                        document.getElementById('num-features').textContent = data.data_info.numerical_features || '-';
                        document.getElementById('target-rate').textContent = 
                            data.data_info.target_rate ? data.data_info.target_rate.toFixed(4) : '-';
                        document.getElementById('device').textContent = data.data_info.device || '-';
                    }
                    
                    // Update fold results
                    updateModelResults(data.fold_results);
                    
                    // Update logs
                    updateLogs(data.logs);
                    
                    // Stop updating if completed or error
                    if (data.completed || data.error) {
                        clearInterval(updateInterval);
                        if (data.error) {
                            statusEl.textContent = 'Error';
                            statusEl.className = 'status error';
                        }
                    }
                })
                .catch(error => {
                    console.error('Error fetching status:', error);
                });
        }
        
        function updateStageIndicator(currentStage) {
            const stages = ['preprocessing', 'level1', 'level2', 'complete'];
            const currentIndex = stages.indexOf(currentStage);
            
            stages.forEach((stage, index) => {
                const el = document.getElementById(`stage-${stage}`);
                if (index < currentIndex) {
                    el.className = 'stage completed';
                } else if (index === currentIndex) {
                    el.className = 'stage active';
                } else {
                    el.className = 'stage pending';
                }
            });
        }
        
        function updateModelResults(foldResults) {
            const container = document.getElementById('model-results');
            let html = '';
            
            // Neural Network results
            if (foldResults.neural_network && foldResults.neural_network.length > 0) {
                html += '<div class="model-card completed"><h4>Neural Network</h4>';
                foldResults.neural_network.forEach((result, i) => {
                    html += `<div>Fold ${i+1}: ${result.auc.toFixed(6)}</div>`;
                });
                html += '</div>';
            }
            
            // CatBoost results
            if (foldResults.catboost && foldResults.catboost.length > 0) {
                html += '<div class="model-card completed"><h4>CatBoost</h4>';
                foldResults.catboost.forEach((result, i) => {
                    html += `<div>Fold ${i+1}: ${result.auc.toFixed(6)}</div>`;
                });
                html += '</div>';
            }
            
            // Meta model results
            if (foldResults.meta_model && foldResults.meta_model.length > 0) {
                html += '<div class="model-card completed"><h4>Meta Model</h4>';
                foldResults.meta_model.forEach((result, i) => {
                    html += `<div>Fold ${i+1}: ${result.auc.toFixed(6)}</div>`;
                });
                html += '</div>';
            }
            
            container.innerHTML = html;
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logs');
            container.innerHTML = '';
            
            logs.slice(-50).forEach(log => {  // Show last 50 logs
                const div = document.createElement('div');
                div.className = 'log-entry';
                div.innerHTML = `<span class="log-time">[${log.timestamp}]</span> ${log.message}`;
                container.appendChild(div);
            });
            
            container.scrollTop = container.scrollHeight;
        }
        
        // Start updating
        updateDashboard();
        updateInterval = setInterval(updateDashboard, 2000);  // Update every 2 seconds
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return HTML_TEMPLATE

@app.route('/api/status')
def status():
    current_time = time.time()
    if training_state['start_time']:
        training_state['elapsed_time'] = current_time - training_state['start_time']
    
    return jsonify(training_state)

def start_web_server():
    """Start the Flask web server in a separate thread"""
    app.run(host='127.0.0.1', port=WEB_PORT, debug=False, use_reloader=False, threaded=True)

def open_browser():
    """Open the web browser after a short delay"""
    webbrowser.open(f'http://127.0.0.1:{WEB_PORT}')

def update_training_state(key: str, value):
    """Thread-safe update of training state"""
    training_state[key] = value

# =============================================================================
# SETUP LOGGING
# =============================================================================
def setup_logging():
    """Setup logging to both file and console"""
    log_file = LOGS_DIR / "training.log"
    
    # Create logger
    logger = logging.getLogger('stacking_v5')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

logger = setup_logging()

def log_message(message: str, level: str = 'INFO'):
    """Log message with timestamp for both file and web dashboard"""
    timestamp = datetime.now().strftime('%H:%M:%S')
    
    # Add to web logs
    training_state['logs'].append({
        'timestamp': timestamp,
        'message': message,
        'level': level
    })
    
    # Keep only last 100 logs for memory efficiency
    if len(training_state['logs']) > 100:
        training_state['logs'] = training_state['logs'][-100:]
    
    # Also log to file
    if level == 'INFO':
        logger.info(message)
    elif level == 'WARNING':
        logger.warning(message)
    elif level == 'ERROR':
        logger.error(message)
    else:
        logger.info(message)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================
def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_device():
    """Get the best available device (MPS > CUDA > CPU)"""
    if torch.backends.mps.is_available():
        return torch.device('mps')
    elif torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def find_latest_checkpoint() -> Optional[str]:
    """Find the latest neural network checkpoint from previous runs"""
    runs_pattern = "runs/*/folds/fold_0/model.pth"
    checkpoint_files = glob.glob(runs_pattern)
    
    if not checkpoint_files:
        return None
    
    # Sort by modification time and get the latest
    latest_checkpoint = max(checkpoint_files, key=os.path.getmtime)
    log_message(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint

# =============================================================================
# ADVANCED FEATURE ENGINEERING
# =============================================================================
def kfold_target_encoding(X: pd.DataFrame, y: pd.Series, cat_cols: List[str], 
                         n_splits: int = 5, smoothing: float = 1.0, 
                         noise_level: float = 0.01) -> pd.DataFrame:
    """
    K-fold target encoding for categorical features to prevent overfitting.
    Uses smoothing and noise injection for regularization.
    """
    X_encoded = X.copy()
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    
    for col in cat_cols:
        if X[col].nunique() < HIGH_CARD_THRESHOLD:
            continue  # Skip low-cardinality features
            
        log_message(f"K-fold target encoding for high-cardinality feature: {col}")
        
        # Initialize encoded column
        X_encoded[f'{col}_target_enc'] = 0.0
        
        # Global mean for smoothing
        global_mean = y.mean()
        
        for train_idx, val_idx in skf.split(X, y):
            # Calculate target statistics on training fold
            train_stats = pd.DataFrame({
                'category': X.iloc[train_idx][col],
                'target': y.iloc[train_idx]
            }).groupby('category')['target'].agg(['mean', 'count']).reset_index()
            
            # Apply smoothing: (count * mean + smoothing * global_mean) / (count + smoothing)
            train_stats['smoothed_mean'] = (
                (train_stats['count'] * train_stats['mean'] + smoothing * global_mean) /
                (train_stats['count'] + smoothing)
            )
            
            # Map to validation fold
            mapping = dict(zip(train_stats['category'], train_stats['smoothed_mean']))
            X_encoded.loc[val_idx, f'{col}_target_enc'] = X.iloc[val_idx][col].map(mapping).fillna(global_mean)
            
            # Add small amount of noise to prevent overfitting
            if noise_level > 0:
                noise = np.random.normal(0, noise_level, len(val_idx))
                X_encoded.loc[val_idx, f'{col}_target_enc'] += noise
    
    return X_encoded

def frequency_encoding(X: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    Add frequency encoding for all categorical features.
    """
    X_encoded = X.copy()
    
    for col in cat_cols:
        log_message(f"Frequency encoding for feature: {col}")
        
        # Calculate frequencies
        freq_map = X[col].value_counts().to_dict()
        X_encoded[f'{col}_freq'] = X[col].map(freq_map)
        
        # Add normalized frequency (0-1 range)
        max_freq = max(freq_map.values())
        X_encoded[f'{col}_freq_norm'] = X_encoded[f'{col}_freq'] / max_freq
    
    return X_encoded

def advanced_numerical_features(X: pd.DataFrame, num_cols: List[str]) -> pd.DataFrame:
    """
    Create advanced numerical features including interactions and transformations.
    """
    X_enhanced = X.copy()
    
    if len(num_cols) < 2:
        return X_enhanced
    
    log_message(f"Creating advanced numerical features from {len(num_cols)} numerical columns")
    
    # Statistical features
    X_enhanced['num_mean'] = X[num_cols].mean(axis=1)
    X_enhanced['num_std'] = X[num_cols].std(axis=1)
    X_enhanced['num_min'] = X[num_cols].min(axis=1)
    X_enhanced['num_max'] = X[num_cols].max(axis=1)
    X_enhanced['num_range'] = X_enhanced['num_max'] - X_enhanced['num_min']
    
    # Create a few key interactions (limit to avoid explosion)
    important_pairs = []
    for i, col1 in enumerate(num_cols[:5]):  # Limit to first 5 for speed
        for col2 in num_cols[i+1:i+3]:  # Only 2 interactions per feature
            important_pairs.append((col1, col2))
    
    for col1, col2 in important_pairs:
        # Interaction features
        X_enhanced[f'{col1}_{col2}_sum'] = X[col1] + X[col2]
        X_enhanced[f'{col1}_{col2}_diff'] = X[col1] - X[col2]
        X_enhanced[f'{col1}_{col2}_prod'] = X[col1] * X[col2]
        
        # Ratio (with safety check)
        safe_ratio = X[col1] / (X[col2] + 1e-8)
        X_enhanced[f'{col1}_{col2}_ratio'] = np.clip(safe_ratio, -100, 100)
    
    log_message(f"Created {X_enhanced.shape[1] - X.shape[1]} new numerical features")
    return X_enhanced
# =============================================================================
# DATA PREPROCESSING (Enhanced v5.2)
# =============================================================================
def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Enhanced preprocessing with advanced feature engineering
    """
    log_message("Starting enhanced data preprocessing...")
    
    # Combine for consistent preprocessing
    combined_df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    log_message(f"Combined dataset shape: {combined_df.shape}")
    
    # Identify column types
    num_cols = combined_df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove ID and target from feature lists
    if ID_COL in num_cols:
        num_cols.remove(ID_COL)
    if TARGET_COL in num_cols:
        num_cols.remove(TARGET_COL)
    if ID_COL in cat_cols:
        cat_cols.remove(ID_COL)
    if TARGET_COL in cat_cols:
        cat_cols.remove(TARGET_COL)
    
    log_message(f"Original features - Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")
    
    # Handle rare categories
    for col in cat_cols:
        value_counts = combined_df[col].value_counts()
        rare_values = value_counts[value_counts < MIN_CAT_COUNT].index
        if len(rare_values) > 0:
            combined_df[col] = combined_df[col].replace(rare_values.tolist(), RARE_NAME)
            log_message(f"Column '{col}': {len(rare_values)} rare values replaced with '{RARE_NAME}'")
    
    # Fill missing values first
    if combined_df.isnull().sum().sum() > 0:
        log_message("Filling missing values...")
        for col in num_cols:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        for col in cat_cols:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
    
    # Split back into train and test for target encoding
    train_processed = combined_df[:len(train_df)].copy()
    test_processed = combined_df[len(train_df):].copy()
    y_train = train_processed[TARGET_COL] if TARGET_COL in train_processed.columns else None
    
    # Advanced feature engineering
    log_message("Applying advanced feature engineering...")
    
    # 1. Frequency encoding for all categoricals
    train_processed = frequency_encoding(train_processed, cat_cols)
    test_processed = frequency_encoding(test_processed, cat_cols)
    
    # 2. K-fold target encoding for high-cardinality categoricals (only on train)
    if y_train is not None:
        train_processed = kfold_target_encoding(train_processed, y_train, cat_cols)
        # For test set, use overall statistics
        for col in cat_cols:
            if train_processed[col].nunique() >= HIGH_CARD_THRESHOLD:
                target_col_name = f'{col}_target_enc'
                if target_col_name in train_processed.columns:
                    # Calculate overall target encoding from training data
                    overall_stats = pd.DataFrame({
                        'category': train_processed[col],
                        'target': y_train
                    }).groupby('category')['target'].mean().to_dict()
                    
                    global_mean = y_train.mean()
                    test_processed[target_col_name] = test_processed[col].map(overall_stats).fillna(global_mean)
    
    # 3. Advanced numerical features
    train_processed = advanced_numerical_features(train_processed, num_cols)
    test_processed = advanced_numerical_features(test_processed, num_cols)
    
    # 4. Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        # Fit on combined data for consistency
        combined_values = pd.concat([train_processed[col], test_processed[col]])
        le.fit(combined_values.astype(str))
        
        train_processed[col] = le.transform(train_processed[col].astype(str))
        test_processed[col] = le.transform(test_processed[col].astype(str))
        label_encoders[col] = le
        log_message(f"Encoded '{col}': {len(le.classes_)} unique values")
    
    # Update feature lists to include new features
    new_cat_cols = [col for col in cat_cols]  # Original categorical columns only
    new_num_cols = [col for col in train_processed.columns if col not in [ID_COL, TARGET_COL] and col not in new_cat_cols]
    
    # Debug: Check for duplicate column names
    all_columns = list(train_processed.columns)
    duplicate_cols = [col for col in set(all_columns) if all_columns.count(col) > 1]
    if duplicate_cols:
        log_message(f"WARNING: Found duplicate column names: {duplicate_cols}")
        # Make column names unique by adding suffixes manually
        new_columns = []
        col_counts = {}
        for col in train_processed.columns:
            if col in col_counts:
                col_counts[col] += 1
                new_col = f"{col}.{col_counts[col]}"
            else:
                col_counts[col] = 0
                new_col = col
            new_columns.append(new_col)
        
        train_processed.columns = new_columns
        test_processed.columns = new_columns
        log_message(f"Fixed duplicate column names. New columns count: {len(new_columns)}")
    
    log_message(f"Enhanced features - Numerical: {len(new_num_cols)}, Categorical: {len(new_cat_cols)}")
    log_message(f"Total features added: {len(new_num_cols) + len(new_cat_cols) - len(num_cols) - len(cat_cols)}")
    
    return train_processed, test_processed, new_num_cols, new_cat_cols

# =============================================================================
# NEURAL NETWORK COMPONENTS
# =============================================================================
class TabDataset(Dataset):
    def __init__(self, df, y=None, num_cols=None, cat_cols=None):
        self.num = df[num_cols].values.astype(np.float32) if num_cols else np.zeros((len(df), 0), np.float32)
        self.cat = df[cat_cols].values.astype(np.int64) if cat_cols else np.zeros((len(df), 0), np.int64)
        self.y = y.astype(np.float32) if y is not None else None

    def __len__(self):
        return len(self.num)

    def __getitem__(self, idx):
        if self.y is None:
            return self.num[idx], self.cat[idx]
        return self.num[idx], self.cat[idx], self.y[idx]

class TabularNN(nn.Module):
    def __init__(self, num_dim, cat_cardinalities, hidden_layers, dropout=0.25, emb_dropout=0.05, input_dropout=0.05):
        super().__init__()
        self.has_cat = len(cat_cardinalities) > 0
        self.has_num = num_dim > 0
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout > 0 and self.has_num else nn.Identity()

        # Embeddings
        if self.has_cat:
            emb_dims = []
            self.emb_layers = nn.ModuleList()
            for card in cat_cardinalities:
                emb_dim = int(min(64, max(4, round(1.6 * (card ** 0.56)))))
                self.emb_layers.append(nn.Embedding(card, emb_dim))
                emb_dims.append(emb_dim)
            self.emb_dropout = nn.Dropout(emb_dropout) if emb_dropout > 0 else nn.Identity()
            emb_total = sum(emb_dims)
        else:
            self.emb_layers = None
            self.emb_dropout = nn.Identity()
            emb_total = 0

        in_dim = (num_dim if self.has_num else 0) + emb_total

        layers = []
        prev = in_dim
        for h in hidden_layers:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout)
            ]
            prev = h
        layers += [nn.Linear(prev, 1)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_num, x_cat):
        feats = []

        if self.has_cat and self.emb_layers is not None:
            embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.emb_layers)]
            cat_feat = torch.cat(embs, dim=1)
            cat_feat = self.emb_dropout(cat_feat)
            feats.append(cat_feat)

        if self.has_num:
            x_num = self.input_dropout(x_num)
            feats.append(x_num)

        x = torch.cat(feats, dim=1) if len(feats) > 1 else feats[0]
        logit = self.mlp(x).squeeze(1)
        return logit

def train_neural_network_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    num_cols: List[str],
    cat_cols: List[str],
    device: torch.device,
    nn_params: Dict,
    warm_start_path: Optional[str] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train neural network for a single fold with multiple seeds"""
    
    log_message(f"Training Neural Network - Fold {fold_idx + 1}")
    
    # Prepare data
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Scale numerical features
    scaler = StandardScaler()
    X_fold_train[num_cols] = scaler.fit_transform(X_fold_train[num_cols])
    X_fold_val[num_cols] = scaler.transform(X_fold_val[num_cols])
    X_test_scaled = X_test.copy()
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])
    
    # Get categorical cardinalities
    cat_cardinalities = []
    if cat_cols:
        for c in cat_cols:
            try:
                nunique_val = X_train[c].nunique()
                cat_cardinalities.append(int(nunique_val))
            except (TypeError, ValueError) as e:
                log_message(f"Warning: Could not get cardinality for column {c}: {e}")
                cat_cardinalities.append(2)  # Default to binary
    
    # Create datasets
    train_dataset = TabDataset(X_fold_train, y_fold_train, num_cols, cat_cols)
    val_dataset = TabDataset(X_fold_val, y_fold_val, num_cols, cat_cols)
    test_dataset = TabDataset(X_test_scaled, None, num_cols, cat_cols)
    
    # Create data loaders - disable pin_memory for MPS compatibility
    pin_memory = device.type != 'mps'
    train_loader = DataLoader(train_dataset, batch_size=nn_params['batch_size'], shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=nn_params['batch_size'], shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=nn_params['batch_size'], shuffle=False, pin_memory=pin_memory)
    
    # Multiple seed training
    best_auc = 0
    best_oof_preds = None
    best_test_preds = None
    
    for seed_idx in range(nn_params['n_seeds']):
        log_message(f"NN Fold {fold_idx + 1} - Seed {seed_idx + 1}/{nn_params['n_seeds']}")
        
        # Set seed for this run
        torch.manual_seed(RANDOM_SEED + seed_idx)
        
        # Create model
        actual_num_dim = X_fold_train[num_cols].shape[1] if num_cols else 0
        model = TabularNN(
            num_dim=actual_num_dim,
            cat_cardinalities=cat_cardinalities,
            hidden_layers=nn_params['hidden_layers'],
            dropout=nn_params['dropout'],
            emb_dropout=nn_params['emb_dropout'],
            input_dropout=nn_params['input_dropout']
        ).to(device)
        
        # Warm start if checkpoint is available and shapes match
        if warm_start_path and os.path.exists(warm_start_path) and seed_idx == 0:
            try:
                checkpoint = torch.load(warm_start_path, map_location=device)
                model_state = checkpoint.get('model_state_dict', checkpoint)
                model.load_state_dict(model_state, strict=False)
                log_message(f"Warm-started from checkpoint: {warm_start_path}")
            except Exception as e:
                log_message(f"Could not load checkpoint {warm_start_path}: {str(e)}. Training from scratch.")
        
        # Setup training
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=nn_params['base_lr'], weight_decay=nn_params['weight_decay'])
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        for epoch in range(nn_params['epochs']):
            # Training
            model.train()
            train_loss = 0
            for batch_num, batch_cat, batch_y in train_loader:
                batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_num, batch_cat)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            with torch.no_grad():
                for batch_num, batch_cat, batch_y in val_loader:
                    batch_num, batch_cat, batch_y = batch_num.to(device), batch_cat.to(device), batch_y.to(device)
                    outputs = model(batch_num, batch_cat)
                    val_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            val_auc = roc_auc_score(val_targets, val_preds)
            scheduler.step()
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
            
            if epoch % 2 == 0:  # Print every 2 epochs for speed
                log_message(f"NN Fold {fold_idx + 1} Seed {seed_idx + 1} Epoch {epoch + 1}/{nn_params['epochs']}: "
                           f"Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, "
                           f"Best: {best_val_auc:.4f}")
            
            if patience_counter >= nn_params['patience']:
                log_message(f"Early stopping at epoch {epoch + 1}")
                break
        
        # Generate predictions for this seed
        model.eval()
        with torch.no_grad():
            # OOF predictions
            oof_preds = []
            for batch_num, batch_cat, batch_y in val_loader:
                batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
                outputs = model(batch_num, batch_cat)
                oof_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            
            # Test predictions
            test_preds = []
            for batch_num, batch_cat in test_loader:
                batch_num, batch_cat = batch_num.to(device), batch_cat.to(device)
                outputs = model(batch_num, batch_cat)
                test_preds.extend(torch.sigmoid(outputs).cpu().numpy())
        
        # Track best seed
        if best_val_auc > best_auc:
            best_auc = best_val_auc
            best_oof_preds = np.array(oof_preds)
            best_test_preds = np.array(test_preds)
        
        # Clear memory
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()
    
    if best_oof_preds is None or best_test_preds is None:
        raise ValueError("No valid predictions generated for neural network")
    
    log_message(f"NN Fold {fold_idx + 1} - Best AUC: {best_auc:.4f}")
    return float(best_auc), best_oof_preds, best_test_preds

def train_catboost_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str],
    cb_params: Dict
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train CatBoost for a single fold"""
    
    log_message(f"Training CatBoost - Fold {fold_idx + 1}")
    
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Get categorical column indices instead of names to avoid duplication issues
    cat_feature_indices = []
    for col in cat_cols:
        if col in X_train.columns:
            # Get the first occurrence index of the column
            col_indices = [i for i, c in enumerate(X_train.columns) if c == col]
            if col_indices:
                cat_feature_indices.append(col_indices[0])
                log_message(f"Found categorical column '{col}' at index {col_indices[0]}")
            else:
                log_message(f"Warning: Could not find index for categorical column '{col}'")
        else:
            log_message(f"Warning: Categorical column '{col}' not found in processed data")
    
    log_message(f"CatBoost using {len(cat_feature_indices)} categorical features at indices: {cat_feature_indices}")
    
    # Debug: Print all column names to identify duplicates
    log_message(f"DEBUG: DataFrame columns ({len(X_fold_train.columns)}): {list(X_fold_train.columns)}")
    
    # Create CatBoost model
    model = cb.CatBoostClassifier(**cb_params)
    
    # Train the model using column indices instead of names
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        cat_features=cat_feature_indices,
        verbose=False,
        plot=False
    )
    
    # Get predictions
    oof_preds = model.predict_proba(X_fold_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]
    
    # Calculate AUC
    val_auc = roc_auc_score(y_fold_val, oof_preds)
    
    log_message(f"CatBoost Fold {fold_idx + 1} - Val AUC: {val_auc:.4f}")
    
    return float(val_auc), oof_preds, test_preds

def train_xgboost_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    xgb_params: Dict
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train XGBoost for a single fold"""
    
    log_message(f"Training XGBoost - Fold {fold_idx + 1}")
    
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
    dtest = xgb.DMatrix(X_test)
    
    # Add objective to params for xgb.train
    params = xgb_params.copy()
    params['objective'] = 'binary:logistic'
    params['eval_metric'] = 'auc'
    
    # Train the model with early stopping
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 500),
        evals=[(dval, 'eval')],
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Get predictions
    oof_preds = model.predict(dval, iteration_range=(0, model.best_iteration))
    test_preds = model.predict(dtest, iteration_range=(0, model.best_iteration))
    
    # Calculate AUC
    val_auc = roc_auc_score(y_fold_val, oof_preds)
    
    log_message(f"XGBoost Fold {fold_idx + 1} - Val AUC: {val_auc:.4f}")
    
    return float(val_auc), oof_preds, test_preds

def train_lightgbm_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    lgb_params: Dict
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train LightGBM for a single fold"""
    
    log_message(f"Training LightGBM - Fold {fold_idx + 1}")
    
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Create LightGBM datasets
    train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
    val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
    
    # Train model
    model = lgb.train(
        lgb_params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
    )
    
    # Get predictions
    oof_preds = model.predict(X_fold_val)
    test_preds = model.predict(X_test)
    
    # Ensure predictions are numpy arrays
    oof_preds = np.array(oof_preds).flatten()
    test_preds = np.array(test_preds).flatten()
    
    # Calculate AUC
    val_auc = roc_auc_score(y_fold_val, oof_preds)
    
    log_message(f"LightGBM Fold {fold_idx + 1} - Val AUC: {val_auc:.4f}")
    
    return float(val_auc), oof_preds, test_preds

def train_lightgbm_meta_model(
    oof_predictions: np.ndarray,
    y_train: np.ndarray,
    test_predictions: np.ndarray,
    meta_lgb_params: Dict
) -> Tuple[float, np.ndarray]:
    """Train LightGBM meta-model on OOF predictions"""
    
    log_message("Training LightGBM Meta-Model...")
    
    # Setup cross-validation for meta-model
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    meta_oof_preds = np.zeros(len(oof_predictions))
    meta_test_preds = np.zeros(len(test_predictions))
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(oof_predictions, y_train)):
        X_fold_train, X_fold_val = oof_predictions[train_idx], oof_predictions[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        # Train model
        model = lgb.train(
            meta_lgb_params,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # Predictions
        fold_val_preds = model.predict(X_fold_val)
        fold_test_preds = model.predict(test_predictions)
        
        # Ensure predictions are numpy arrays
        fold_val_preds = np.array(fold_val_preds).flatten()
        fold_test_preds = np.array(fold_test_preds).flatten()
        
        meta_oof_preds[val_idx] = fold_val_preds
        meta_test_preds += fold_test_preds / N_SPLITS
        
        # Calculate fold AUC
        fold_auc = roc_auc_score(y_fold_val, fold_val_preds)
        fold_aucs.append(fold_auc)
        
        log_message(f"Meta LightGBM Fold {fold_idx + 1} - AUC: {fold_auc:.4f}")
    
    # Overall meta-model AUC
    meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
    log_message(f"Meta LightGBM OOF AUC: {meta_oof_auc:.4f}")
    
    return float(meta_oof_auc), meta_test_preds

def train_logistic_meta_model(
    oof_predictions: np.ndarray,
    y_train: np.ndarray,
    test_predictions: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Train Logistic Regression meta-model"""
    
    log_message("Training Logistic Regression Meta-Model...")
    
    # Use cross-validation to get OOF predictions
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    
    # Get OOF predictions from logistic regression
    meta_oof_preds = cross_val_predict(
        LogisticRegression(random_state=RANDOM_SEED, max_iter=1000),
        oof_predictions, y_train, cv=skf, method='predict_proba'
    )[:, 1]
    
    # Train final model on all data
    final_model = LogisticRegression(random_state=RANDOM_SEED, max_iter=1000)
    final_model.fit(oof_predictions, y_train)
    meta_test_preds = final_model.predict_proba(test_predictions)[:, 1]
    
    # Calculate AUC
    meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
    log_message(f"Logistic Regression Meta-Model OOF AUC: {meta_oof_auc:.4f}")
    
    return float(meta_oof_auc), meta_test_preds

def optimize_weighted_blend(
    oof_predictions: np.ndarray,
    y_train: np.ndarray,
    test_predictions: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Optimize non-negative weighted blend using coordinate ascent"""
    
    log_message("Optimizing Non-negative Weighted Blend...")
    
    n_models = oof_predictions.shape[1]
    
    def objective(weights):
        # Ensure weights are non-negative and sum to 1
        weights = np.maximum(weights, 0)
        if weights.sum() == 0:
            weights = np.ones(n_models) / n_models
        else:
            weights = weights / weights.sum()
        
        # Calculate weighted prediction
        pred = np.dot(oof_predictions, weights)
        return -roc_auc_score(y_train, pred)
    
    # Initialize with equal weights
    initial_weights = np.ones(n_models) / n_models
    
    # Optimize
    result = minimize(
        objective,
        initial_weights,
        method='L-BFGS-B',
        bounds=[(0, 1) for _ in range(n_models)]
    )
    
    optimal_weights = result.x
    optimal_weights = optimal_weights / optimal_weights.sum()  # Normalize
    
    # Calculate final predictions
    meta_oof_preds = np.dot(oof_predictions, optimal_weights)
    meta_test_preds = np.dot(test_predictions, optimal_weights)
    
    meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
    
    log_message(f"Optimal weights: {optimal_weights}")
    log_message(f"Weighted Blend OOF AUC: {meta_oof_auc:.4f}")
    
    return float(meta_oof_auc), meta_test_preds, optimal_weights

def rank_blend(
    oof_predictions: np.ndarray,
    y_train: np.ndarray,
    test_predictions: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Create rank-based ensemble"""
    
    log_message("Creating Rank-based Blend...")
    
    # Convert predictions to ranks
    oof_ranks = np.zeros_like(oof_predictions)
    test_ranks = np.zeros_like(test_predictions)
    
    for i in range(oof_predictions.shape[1]):
        oof_ranks[:, i] = pd.Series(oof_predictions[:, i]).rank(pct=True)
        test_ranks[:, i] = pd.Series(test_predictions[:, i]).rank(pct=True)
    
    # Simple average of ranks
    meta_oof_preds = np.mean(oof_ranks, axis=1)
    meta_test_preds = np.mean(test_ranks, axis=1)
    
    meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
    log_message(f"Rank Blend OOF AUC: {meta_oof_auc:.4f}")
    
    return float(meta_oof_auc), meta_test_preds

def analyze_performance_bottlenecks(
    oof_aucs: Dict[str, float],
    best_auc: float,
    target_auc: float = TARGET_AUC
) -> str:
    """Analyze performance and suggest improvements if below target"""
    
    if best_auc >= target_auc:
        return f"✅ Target AUC of {target_auc:.3f} achieved! Best: {best_auc:.6f}"
    
    gap = target_auc - best_auc
    suggestions = []
    
    log_message("🔍 PERFORMANCE ANALYSIS - Below Target AUC")
    log_message(f"Target: {target_auc:.3f}, Best: {best_auc:.6f}, Gap: {gap:.6f}")
    
    # Analyze individual models
    base_models = ['neural_network', 'catboost', 'xgboost', 'lightgbm']
    base_aucs = [oof_aucs.get(model, 0) for model in base_models if oof_aucs.get(model)]
    
    if len(base_aucs) > 0:
        best_base = max(base_aucs)
        worst_base = min(base_aucs)
        diversity = best_base - worst_base
        
        if diversity < 0.005:
            suggestions.append("📊 Low model diversity - try different architectures/hyperparameters")
        
        if best_base < 0.90:
            suggestions.append("⚡ Base models underperforming - increase iterations/epochs")
        
        if all(auc < 0.95 for auc in base_aucs):
            suggestions.append("🔧 All base models weak - check feature engineering")
    
    # Feature engineering suggestions
    if gap > 0.02:
        suggestions.extend([
            "🧬 Try advanced feature engineering: polynomial features, clustering",
            "📈 Increase model complexity: deeper networks, more boosting rounds",
            "🔄 Add more diverse models: different algorithms/preprocessing"
        ])
    elif gap > 0.01:
        suggestions.extend([
            "🎯 Fine-tune hyperparameters with more aggressive search",
            "📊 Analyze feature importance and create targeted interactions"
        ])
    else:
        suggestions.append("🔬 Small gap - try ensemble of ensembles or pseudo-labeling")
    
    suggestion_text = "\n".join([f"  {s}" for s in suggestions])
    log_message(f"💡 IMPROVEMENT SUGGESTIONS:\n{suggestion_text}")
    
    return f"❌ Below target by {gap:.6f}. See suggestions above."

def plot_enhanced_results(
    y_train: np.ndarray,
    oof_predictions: np.ndarray,
    test_predictions: np.ndarray,
    base_aucs: List[float],
    meta_results: Dict,
    model_names: List[str]
):
    """Create comprehensive result plots for the enhanced pipeline"""
    
    log_message("Creating enhanced result plots...")
    
    # Calculate individual OOF AUCs
    individual_aucs = []
    for i in range(oof_predictions.shape[1]):
        auc = roc_auc_score(y_train, oof_predictions[:, i])
        individual_aucs.append(auc)
    
    # Create a large figure with multiple subplots
    fig = plt.figure(figsize=(20, 15))
    
    # Plot 1: ROC Curves for all models
    plt.subplot(2, 3, 1)
    colors = ['skyblue', 'lightcoral', 'lightgreen', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    for i, (name, auc) in enumerate(zip(model_names, individual_aucs)):
        RocCurveDisplay.from_predictions(
            y_train, oof_predictions[:, i], 
            name=f'{name} (AUC={auc:.4f})', 
            color=colors[i % len(colors)],
            alpha=0.8
        )
    
    # Add meta-model ROC curves
    for i, (meta_name, meta_data) in enumerate(meta_results.items()):
        if 'test_preds' in meta_data and len(meta_data['test_preds']) == len(y_train):
            plt.plot([0, 1], [0, 1], '--', color=colors[(i+4) % len(colors)], alpha=0.5)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('ROC Curves - All Models')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 2: Model performance comparison
    plt.subplot(2, 3, 2)
    all_names = model_names + [f'Meta-{k}' for k in meta_results.keys()]
    all_aucs = individual_aucs + [meta_results[k]['auc'] for k in meta_results.keys()]
    
    bars = plt.bar(range(len(all_names)), all_aucs, 
                   color=colors[:len(all_names)], alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('ROC AUC Score')
    plt.xticks(range(len(all_names)), all_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, auc in zip(bars, all_aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{auc:.4f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Cross-validation results for base models
    plt.subplot(2, 3, 3)
    fold_numbers = list(range(1, N_SPLITS + 1))
    
    for i, (name, aucs) in enumerate(zip(model_names, [base_aucs[i*N_SPLITS:(i+1)*N_SPLITS] for i in range(len(model_names))])):
        if len(aucs) == N_SPLITS:
            plt.plot(fold_numbers, aucs, 'o-', label=name, 
                    color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('Fold Number')
    plt.ylabel('ROC AUC Score')
    plt.title('Cross-Validation Performance by Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Prediction distributions
    plt.subplot(2, 3, 4)
    for i, name in enumerate(model_names):
        plt.hist(oof_predictions[:, i], bins=30, alpha=0.5, 
                label=f'{name}', color=colors[i % len(colors)])
    plt.xlabel('Prediction Value')
    plt.ylabel('Frequency')
    plt.title('OOF Prediction Distributions')
    plt.legend()
    
    # Plot 5: Model correlation heatmap
    plt.subplot(2, 3, 5)
    corr_matrix = np.corrcoef(oof_predictions.T)
    sns.heatmap(corr_matrix, annot=True, xticklabels=model_names, 
                yticklabels=model_names, cmap='coolwarm', center=0)
    plt.title('Model Correlation Matrix')
    
    # Plot 6: Ensemble weights (if available)
    plt.subplot(2, 3, 6)
    if 'weighted' in meta_results and 'weights' in meta_results['weighted']:
        weights = meta_results['weighted']['weights']
        plt.pie(weights, labels=model_names, autopct='%1.1f%%', startangle=90)
        plt.title('Optimal Ensemble Weights')
    else:
        plt.text(0.5, 0.5, 'Ensemble weights\nnot available', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Ensemble Weights')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'enhanced_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create feature importance plot if available
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        plt.text(0.5, 0.5, 'Feature importance analysis\nwould be implemented here\nfor tree-based models', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        plt.title('Feature Importance Analysis (Placeholder)')
        plt.savefig(FIGS_DIR / 'feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    except Exception as e:
        log_message(f"Could not create feature importance plot: {e}")
    
    log_message(f"Enhanced plots saved to: {FIGS_DIR}")

def save_enhanced_readme(
    model_params: Dict,
    all_aucs: Dict[str, float],
    final_model_name: str,
    final_auc: float,
    total_time: float,
    feature_cols: List[str],
    num_cols: List[str],
    cat_cols: List[str],
    y_train: np.ndarray,
    warm_start_path: Optional[str],
    weights_data: Dict
):
    """Save comprehensive README for the enhanced pipeline"""
    
    readme_content = f"""Enhanced Stacking Pipeline v5.2 Results
==========================================

This folder contains outputs from the enhanced 4-model stacking ensemble with advanced feature engineering and multiple meta-model approaches.

PIPELINE ARCHITECTURE:
=====================
Level 1 Models (Base Models):
1. Neural Network - Multi-seed training with warm-start capability
2. CatBoost - Robust gradient boosting 
3. XGBoost - High-performance gradient boosting
4. LightGBM - Fast gradient boosting

Level 2 Models (Meta-Models):
1. Logistic Regression - Linear ensemble
2. LightGBM Meta - Non-linear ensemble  
3. Weighted Blend - Optimized coordinate ascent
4. Rank Blend - Robust ranking approach

FEATURE ENGINEERING:
===================
- K-Fold Target Encoding for high-cardinality categoricals (leak-safe)
- Frequency Encoding for all categorical features
- Advanced numerical feature interactions and transformations
- Statistical aggregations and ratio features

PERFORMANCE RESULTS:
===================
Base Models:
- Neural Network OOF AUC: {all_aucs.get('neural_network', 0):.6f}
- CatBoost OOF AUC: {all_aucs.get('catboost', 0):.6f}
- XGBoost OOF AUC: {all_aucs.get('xgboost', 0):.6f}  
- LightGBM OOF AUC: {all_aucs.get('lightgbm', 0):.6f}

Meta-Models:
- Logistic Regression: {all_aucs.get('meta_logistic', 0):.6f}
- LightGBM Meta: {all_aucs.get('meta_lightgbm', 0):.6f}
- Weighted Blend: {all_aucs.get('meta_weighted', 0):.6f}
- Rank Blend: {all_aucs.get('meta_rank', 0):.6f}

FINAL RESULTS:
=============
- Final Model: {final_model_name}
- Final AUC: {final_auc:.6f}
- Target AUC: {TARGET_AUC:.3f} ({'✅ ACHIEVED' if final_auc >= TARGET_AUC else '❌ NOT REACHED'})
- Training Mode: {TRAINING_MODE}
- Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)

CONFIGURATION:
=============
- CV Folds: {N_SPLITS}
- Random Seed: {RANDOM_SEED}
- High-Card Threshold: {HIGH_CARD_THRESHOLD}
- Total Features: {len(feature_cols)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)
- Target Rate: {float(np.mean(y_train)):.4f}

MODEL PARAMETERS:
================
Neural Network ({TRAINING_MODE} mode):
- Epochs: {model_params['neural_network']['epochs']}
- Seeds: {model_params['neural_network']['n_seeds']}
- Hidden Layers: {model_params['neural_network']['hidden_layers']}
- Batch Size: {model_params['neural_network']['batch_size']}

CatBoost:
- Iterations: {model_params['catboost']['iterations']}
- Learning Rate: {model_params['catboost']['learning_rate']}
- Depth: {model_params['catboost']['depth']}

XGBoost:
- Estimators: {model_params['xgboost']['n_estimators']}
- Learning Rate: {model_params['xgboost']['learning_rate']}
- Max Depth: {model_params['xgboost']['max_depth']}

LightGBM:
- Estimators: {model_params['lightgbm']['n_estimators']}
- Learning Rate: {model_params['lightgbm']['learning_rate']}
- Num Leaves: {model_params['lightgbm']['num_leaves']}

ENSEMBLE WEIGHTS:
================
{weights_data.get('weight_labels', ['N/A'])}
Optimal Weights: {[f'{w:.4f}' for w in weights_data.get('optimal_weights', [])]}

FILES STRUCTURE:
===============
├── oof/
│   └── oof_predictions.csv          # Out-of-fold predictions for all models
├── preds/  
│   └── final_submission.csv         # Final submission using best model
├── figs/
│   ├── enhanced_model_analysis.png  # Comprehensive model analysis plots
│   └── feature_importance.png       # Feature importance analysis
├── logs/
│   ├── metrics.csv                  # Detailed per-fold metrics
│   ├── run_metadata.json           # Complete run configuration and results
│   ├── ensemble_weights.json       # Ensemble weights and meta-model performance
│   └── training.log                # Detailed training logs
└── README.txt                      # This file

OPTIMIZATION FEATURES:
=====================
- Multi-seed neural network training for robustness
- Warm-start capability from previous runs
- Advanced feature engineering with leak-safe target encoding
- Multiple meta-model approaches with automatic selection
- Non-negative weight optimization using coordinate ascent
- Comprehensive performance analysis and bottleneck detection
- Real-time web dashboard monitoring at http://127.0.0.1:{WEB_PORT}

REPRODUCIBILITY:
===============
- All random seeds fixed for reproducibility
- Deterministic cross-validation splits
- Saved model parameters and configurations
- Comprehensive logging of all steps

Run Timestamp: {RUN_TIMESTAMP}
Generated by Enhanced Stacking Pipeline v5.2
"""
    
    with open(README_OUTPUT_PATH, 'w') as f:
        f.write(readme_content)
    log_message(f"Enhanced README saved to: {README_OUTPUT_PATH}")

def main():
    """Main enhanced stacking pipeline v5.2"""
    
    overall_start_time = time.time()
    set_seed(RANDOM_SEED)
    
    # Get model parameters based on training mode
    model_params = get_model_params(TRAINING_MODE)
    
    # Start web server in background
    log_message("Starting web dashboard...")
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    
    # Open browser after a short delay
    if AUTO_OPEN_BROWSER:
        Timer(2, open_browser).start()
    
    # Initialize training state
    update_training_state('start_time', overall_start_time)
    update_training_state('status', 'running')
    
    log_message("=" * 70)
    log_message("ENHANCED STACKING PIPELINE v5.2 - STARTED")
    log_message("=" * 70)
    log_message(f"Run timestamp: {RUN_TIMESTAMP}")
    log_message(f"Output directory: {STACKING_DIR}")
    log_message(f"Training mode: {TRAINING_MODE}")
    log_message(f"Target AUC: {TARGET_AUC:.3f}")
    log_message(f"Device: {get_device()}")
    log_message(f"Web dashboard: http://127.0.0.1:{WEB_PORT}")
    
    try:
        # Load data
        update_training_state('current_stage', 'preprocessing')
        log_message("Loading data...")
        train_df = pd.read_csv(TRAIN_PATH)
        test_df = pd.read_csv(TEST_PATH)
        
        log_message(f"Train shape: {train_df.shape}")
        log_message(f"Test shape: {test_df.shape}")
        
        # Update data info for web dashboard
        update_training_state('data_info', {
            'train_samples': len(train_df),
            'test_samples': len(test_df),
            'device': str(get_device())
        })
        
        # Enhanced preprocessing with feature engineering
        train_processed, test_processed, num_cols, cat_cols = preprocess_data(train_df, test_df)
        
        # Prepare features and targets
        feature_cols = num_cols + cat_cols
        X_train = train_processed[feature_cols]
        y_train = train_processed[TARGET_COL].values.astype(np.float32)
        X_test = test_processed[feature_cols]
        
        log_message(f"Features: {len(feature_cols)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)")
        log_message(f"Target distribution: {float(np.mean(y_train)):.4f} positive rate")
        
        # Update feature info for web dashboard
        training_state['data_info'].update({
            'total_features': len(feature_cols),
            'numerical_features': len(num_cols),
            'categorical_features': len(cat_cols),
            'target_rate': float(np.mean(y_train))
        })
        
        # Setup cross-validation
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        
        # Initialize prediction arrays for 4 base models
        oof_nn = np.zeros(len(X_train))
        oof_cb = np.zeros(len(X_train))
        oof_xgb = np.zeros(len(X_train))
        oof_lgb = np.zeros(len(X_train))
        
        test_nn = np.zeros(len(X_test))
        test_cb = np.zeros(len(X_test))
        test_xgb = np.zeros(len(X_test))
        test_lgb = np.zeros(len(X_test))
        
        # Track fold results
        nn_aucs = []
        cb_aucs = []
        xgb_aucs = []
        lgb_aucs = []
        
        device = get_device()
        log_message(f"Using device: {device}")
        
        # Find warm-start checkpoint
        warm_start_path = find_latest_checkpoint()
        
        log_message("=" * 60)
        log_message("LEVEL 1 TRAINING - 4 Base Models")
        log_message("=" * 60)
        
        update_training_state('current_stage', 'level1')
        update_training_state('overall_progress', 10)
        
        # Train Level 1 models (4 base models)
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
            fold_start_time = time.time()
            
            update_training_state('current_fold', fold_idx + 1)
            log_message(f"Training Fold {fold_idx + 1}/{N_SPLITS}")
            
            # Train Neural Network
            update_training_state('current_model', 'Neural Network')
            nn_auc, nn_oof_preds, nn_test_preds = train_neural_network_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test,
                num_cols, cat_cols, device, model_params['neural_network'], warm_start_path
            )
            
            # Train CatBoost
            update_training_state('current_model', 'CatBoost')
            cb_auc, cb_oof_preds, cb_test_preds = train_catboost_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test, 
                cat_cols, model_params['catboost']
            )
            
            # Train XGBoost
            update_training_state('current_model', 'XGBoost')
            xgb_auc, xgb_oof_preds, xgb_test_preds = train_xgboost_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test,
                model_params['xgboost']
            )
            
            # Train LightGBM
            update_training_state('current_model', 'LightGBM')
            lgb_auc, lgb_oof_preds, lgb_test_preds = train_lightgbm_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test,
                model_params['lightgbm']
            )
            
            # Store predictions
            oof_nn[val_indices] = nn_oof_preds
            oof_cb[val_indices] = cb_oof_preds
            oof_xgb[val_indices] = xgb_oof_preds
            oof_lgb[val_indices] = lgb_oof_preds
            
            test_nn += nn_test_preds / N_SPLITS
            test_cb += cb_test_preds / N_SPLITS
            test_xgb += xgb_test_preds / N_SPLITS
            test_lgb += lgb_test_preds / N_SPLITS
            
            # Track AUCs
            nn_aucs.append(nn_auc)
            cb_aucs.append(cb_auc)
            xgb_aucs.append(xgb_auc)
            lgb_aucs.append(lgb_auc)
            
            # Update web dashboard with fold results
            training_state['fold_results']['neural_network'].append({'auc': nn_auc})
            training_state['fold_results']['catboost'].append({'auc': cb_auc})
            training_state['fold_results']['xgboost'].append({'auc': xgb_auc})
            training_state['fold_results']['lightgbm'].append({'auc': lgb_auc})
            
            fold_time = time.time() - fold_start_time
            log_message(f"Fold {fold_idx + 1} completed in {fold_time:.1f}s")
            
            # Update overall progress
            progress = 10 + (fold_idx + 1) / N_SPLITS * 50  # 10% to 60%
            update_training_state('overall_progress', progress)
        
        # Calculate Level 1 OOF AUCs
        oof_nn_auc = roc_auc_score(y_train, oof_nn)
        oof_cb_auc = roc_auc_score(y_train, oof_cb)
        oof_xgb_auc = roc_auc_score(y_train, oof_xgb)
        oof_lgb_auc = roc_auc_score(y_train, oof_lgb)
        
        log_message("=" * 60)
        log_message("LEVEL 1 RESULTS")
        log_message("=" * 60)
        log_message(f"Neural Network - OOF AUC: {oof_nn_auc:.6f}, Mean CV: {np.mean(nn_aucs):.6f} ± {np.std(nn_aucs):.6f}")
        log_message(f"CatBoost - OOF AUC: {oof_cb_auc:.6f}, Mean CV: {np.mean(cb_aucs):.6f} ± {np.std(cb_aucs):.6f}")
        log_message(f"XGBoost - OOF AUC: {oof_xgb_auc:.6f}, Mean CV: {np.mean(xgb_aucs):.6f} ± {np.std(xgb_aucs):.6f}")
        log_message(f"LightGBM - OOF AUC: {oof_lgb_auc:.6f}, Mean CV: {np.mean(lgb_aucs):.6f} ± {np.std(lgb_aucs):.6f}")
        
        # Update web dashboard with Level 1 results
        update_training_state('oof_aucs', {
            'neural_network': float(oof_nn_auc),
            'catboost': float(oof_cb_auc),
            'xgboost': float(oof_xgb_auc),
            'lightgbm': float(oof_lgb_auc),
            'meta_logistic': None,
            'meta_lightgbm': None,
            'meta_weighted': None,
            'meta_rank': None,
            'best_model': None
        })
        
        log_message("=" * 60)
        log_message("LEVEL 2 TRAINING - Meta-Models")
        log_message("=" * 60)
        
        update_training_state('current_stage', 'level2')
        update_training_state('overall_progress', 65)
        
        # Prepare meta-features
        oof_predictions = np.column_stack([oof_nn, oof_cb, oof_xgb, oof_lgb])
        test_predictions = np.column_stack([test_nn, test_cb, test_xgb, test_lgb])
        
        # Train meta-models
        meta_results = {}
        
        # 1. Logistic Regression Meta-Model
        update_training_state('current_model', 'Logistic Meta-Model')
        log_auc, log_test_preds = train_logistic_meta_model(
            oof_predictions, y_train, test_predictions
        )
        meta_results['logistic'] = {'auc': log_auc, 'test_preds': log_test_preds}
        
        # 2. LightGBM Meta-Model
        update_training_state('current_model', 'LightGBM Meta-Model')
        lgb_meta_auc, lgb_meta_test_preds = train_lightgbm_meta_model(
            oof_predictions, y_train, test_predictions, model_params['meta_lightgbm']
        )
        meta_results['lightgbm'] = {'auc': lgb_meta_auc, 'test_preds': lgb_meta_test_preds}
        
        # 3. Weighted Blend
        update_training_state('current_model', 'Weighted Blend')
        weighted_auc, weighted_test_preds, optimal_weights = optimize_weighted_blend(
            oof_predictions, y_train, test_predictions
        )
        meta_results['weighted'] = {'auc': weighted_auc, 'test_preds': weighted_test_preds, 'weights': optimal_weights}
        
        # 4. Rank Blend
        update_training_state('current_model', 'Rank Blend')
        rank_auc, rank_test_preds = rank_blend(
            oof_predictions, y_train, test_predictions
        )
        meta_results['rank'] = {'auc': rank_auc, 'test_preds': rank_test_preds}
        
        # Update progress
        update_training_state('overall_progress', 85)
        
        # Find best meta-model
        best_meta_name = max(meta_results.keys(), key=lambda k: meta_results[k]['auc'])
        best_meta_auc = meta_results[best_meta_name]['auc']
        best_test_preds = meta_results[best_meta_name]['test_preds']
        
        # Compare with best base model
        base_aucs = [float(oof_nn_auc), float(oof_cb_auc), float(oof_xgb_auc), float(oof_lgb_auc)]
        best_base_auc = max(base_aucs)
        best_base_idx = base_aucs.index(best_base_auc)
        best_base_name = ['neural_network', 'catboost', 'xgboost', 'lightgbm'][best_base_idx]
        
        if best_meta_auc > best_base_auc:
            final_auc = best_meta_auc
            final_preds = best_test_preds
            final_model_name = f"meta_{best_meta_name}"
        else:
            final_auc = best_base_auc
            final_model_name = best_base_name
            if best_base_name == 'neural_network':
                final_preds = test_nn
            elif best_base_name == 'catboost':
                final_preds = test_cb
            elif best_base_name == 'xgboost':
                final_preds = test_xgb
            else:
                final_preds = test_lgb
        
        log_message("=" * 60)
        log_message("META-MODEL RESULTS")
        log_message("=" * 60)
        log_message(f"Logistic Regression Meta: {log_auc:.6f}")
        log_message(f"LightGBM Meta: {lgb_meta_auc:.6f}")
        log_message(f"Weighted Blend: {weighted_auc:.6f}")
        log_message(f"Rank Blend: {rank_auc:.6f}")
        log_message(f"Best Meta: {best_meta_name} ({best_meta_auc:.6f})")
        log_message(f"Best Base: {best_base_name} ({best_base_auc:.6f})")
        log_message(f"Final Model: {final_model_name} ({final_auc:.6f})")
        
        # Performance analysis
        all_aucs = {
            'neural_network': oof_nn_auc,
            'catboost': oof_cb_auc,
            'xgboost': oof_xgb_auc,
            'lightgbm': oof_lgb_auc,
            'meta_logistic': log_auc,
            'meta_lightgbm': lgb_meta_auc,
            'meta_weighted': weighted_auc,
            'meta_rank': rank_auc
        }
        
        analysis_result = analyze_performance_bottlenecks(all_aucs, final_auc, TARGET_AUC)
        log_message(analysis_result)
        
        # Save all predictions
        oof_df = pd.DataFrame({
            ID_COL: train_processed[ID_COL],
            'oof_nn': oof_nn,
            'oof_cb': oof_cb,
            'oof_xgb': oof_xgb,
            'oof_lgb': oof_lgb,
            'oof_meta_logistic': log_test_preds[:len(oof_nn)] if len(log_test_preds) == len(oof_nn) else np.zeros(len(oof_nn)),
            'oof_meta_lightgbm': lgb_meta_test_preds[:len(oof_nn)] if len(lgb_meta_test_preds) == len(oof_nn) else np.zeros(len(oof_nn)),
            'oof_meta_weighted': weighted_test_preds[:len(oof_nn)] if len(weighted_test_preds) == len(oof_nn) else np.zeros(len(oof_nn)),
            'oof_meta_rank': rank_test_preds[:len(oof_nn)] if len(rank_test_preds) == len(oof_nn) else np.zeros(len(oof_nn)),
            TARGET_COL: y_train
        })
        oof_df.to_csv(OOF_OUTPUT_PATH, index=False)
        log_message(f"OOF predictions saved to: {OOF_OUTPUT_PATH}")
        
        # Save final submission (using best model)
        submission_df = pd.DataFrame({
            ID_COL: test_processed[ID_COL],
            TARGET_COL: final_preds
        })
        submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        log_message(f"Final submission saved to: {SUBMISSION_OUTPUT_PATH}")
        
        # Save ensemble weights
        weights_data = {
            'optimal_weights': optimal_weights.tolist() if 'optimal_weights' in locals() else [],
            'weight_labels': ['neural_network', 'catboost', 'xgboost', 'lightgbm'],
            'final_model': final_model_name,
            'meta_model_performance': {k: v['auc'] for k, v in meta_results.items()}
        }
        with open(WEIGHTS_OUTPUT_PATH, 'w') as f:
            json.dump(weights_data, f, indent=2)
        log_message(f"Ensemble weights saved to: {WEIGHTS_OUTPUT_PATH}")
        
        # Calculate total time
        total_time = time.time() - overall_start_time
        
        # Create enhanced plots
        model_names = ['Neural Network', 'CatBoost', 'XGBoost', 'LightGBM']
        all_base_aucs = nn_aucs + cb_aucs + xgb_aucs + lgb_aucs  # Flatten all fold results
        plot_enhanced_results(
            y_train, oof_predictions, test_predictions, 
            all_base_aucs, meta_results, model_names
        )
        
        # Save enhanced metrics
        metrics_data = []
        
        # Base model fold results
        for i in range(N_SPLITS):
            metrics_data.extend([
                {'model': 'neural_network', 'fold': i+1, 'auc': nn_aucs[i]},
                {'model': 'catboost', 'fold': i+1, 'auc': cb_aucs[i]},
                {'model': 'xgboost', 'fold': i+1, 'auc': xgb_aucs[i]},
                {'model': 'lightgbm', 'fold': i+1, 'auc': lgb_aucs[i]}
            ])
        
        # OOF results
        metrics_data.extend([
            {'model': 'neural_network', 'fold': 'OOF', 'auc': float(oof_nn_auc)},
            {'model': 'catboost', 'fold': 'OOF', 'auc': float(oof_cb_auc)},
            {'model': 'xgboost', 'fold': 'OOF', 'auc': float(oof_xgb_auc)},
            {'model': 'lightgbm', 'fold': 'OOF', 'auc': float(oof_lgb_auc)},
            {'model': 'meta_logistic', 'fold': 'OOF', 'auc': log_auc},
            {'model': 'meta_lightgbm', 'fold': 'OOF', 'auc': lgb_meta_auc},
            {'model': 'meta_weighted', 'fold': 'OOF', 'auc': weighted_auc},
            {'model': 'meta_rank', 'fold': 'OOF', 'auc': rank_auc},
            {'model': 'final_model', 'fold': 'FINAL', 'auc': final_auc}
        ])
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
        log_message(f"Enhanced metrics saved to: {METRICS_OUTPUT_PATH}")
        
        # Save comprehensive metadata
        metadata = {
            'pipeline': 'enhanced_stacking_v5.2',
            'training_mode': TRAINING_MODE,
            'target_auc': TARGET_AUC,
            'final_model': final_model_name,
            'final_auc': final_auc,
            'target_achieved': final_auc >= TARGET_AUC,
            'base_models': {
                'neural_network': {
                    'oof_auc': float(oof_nn_auc),
                    'mean_cv_auc': float(np.mean(nn_aucs)),
                    'std_cv_auc': float(np.std(nn_aucs)),
                    'fold_aucs': [float(auc) for auc in nn_aucs],
                    'params': model_params['neural_network']
                },
                'catboost': {
                    'oof_auc': float(oof_cb_auc),
                    'mean_cv_auc': float(np.mean(cb_aucs)),
                    'std_cv_auc': float(np.std(cb_aucs)),
                    'fold_aucs': [float(auc) for auc in cb_aucs],
                    'params': model_params['catboost']
                },
                'xgboost': {
                    'oof_auc': float(oof_xgb_auc),
                    'mean_cv_auc': float(np.mean(xgb_aucs)),
                    'std_cv_auc': float(np.std(xgb_aucs)),
                    'fold_aucs': [float(auc) for auc in xgb_aucs],
                    'params': model_params['xgboost']
                },
                'lightgbm': {
                    'oof_auc': float(oof_lgb_auc),
                    'mean_cv_auc': float(np.mean(lgb_aucs)),
                    'std_cv_auc': float(np.std(lgb_aucs)),
                    'fold_aucs': [float(auc) for auc in lgb_aucs],
                    'params': model_params['lightgbm']
                }
            },
            'meta_models': {
                'logistic_regression': {'oof_auc': log_auc},
                'lightgbm_meta': {'oof_auc': lgb_meta_auc, 'params': model_params['meta_lightgbm']},
                'weighted_blend': {'oof_auc': weighted_auc, 'weights': optimal_weights.tolist()},
                'rank_blend': {'oof_auc': rank_auc}
            },
            'data_info': {
                'n_splits': N_SPLITS,
                'random_seed': RANDOM_SEED,
                'high_card_threshold': HIGH_CARD_THRESHOLD,
                'total_features': len(feature_cols),
                'numerical_features': len(num_cols),
                'categorical_features': len(cat_cols),
                'target_rate': float(np.mean(y_train))
            },
            'runtime': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60,
                'training_mode': TRAINING_MODE
            },
            'warm_start': {
                'used': warm_start_path is not None,
                'checkpoint_path': warm_start_path
            },
            'performance_analysis': analysis_result,
            'timestamp': RUN_TIMESTAMP
        }
        
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        log_message(f"Enhanced metadata saved to: {METADATA_OUTPUT_PATH}")
        
        # Save enhanced README
        save_enhanced_readme(
            model_params, all_aucs, final_model_name, final_auc, total_time,
            feature_cols, num_cols, cat_cols, y_train, warm_start_path, weights_data
        )
        
        log_message("=" * 70)
        log_message("ENHANCED STACKING PIPELINE COMPLETED")
        log_message("=" * 70)
        log_message(f"Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        log_message(f"Final Model: {final_model_name}")
        log_message(f"Final AUC: {final_auc:.6f}")
        log_message(f"Target AUC: {TARGET_AUC:.3f} ({'✅ ACHIEVED' if final_auc >= TARGET_AUC else '❌ NOT REACHED'})")
        log_message("🎉 Enhanced stacking pipeline completed successfully!")
        
        # Update final web dashboard state
        update_training_state('current_stage', 'complete')
        update_training_state('overall_progress', 100)
        update_training_state('status', 'completed')
        update_training_state('completed', True)
        training_state['oof_aucs'].update({
            'meta_logistic': float(log_auc),
            'meta_lightgbm': float(lgb_meta_auc),
            'meta_weighted': float(weighted_auc),
            'meta_rank': float(rank_auc),
            'best_model': final_model_name
        })
        update_training_state('ensemble_weights', weights_data)
        
    except Exception as e:
        log_message(f"Pipeline failed with error: {str(e)}", 'ERROR')
        update_training_state('error', str(e))
        update_training_state('status', 'error')
        raise
    
    # Keep the web server running so user can see final results
    log_message("Web dashboard will remain available. Press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log_message("Shutting down...")

if __name__ == "__main__":
    main()
