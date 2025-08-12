#!/usr/bin/env python3
"""
Tabular Stacking v5 - Fast Ensemble Pipeline with Neural Network, CatBoost, and LightGBM Meta-Model
Based on v5 preprocessing but optimized for speed with stacking ensemble approach.

Pipeline:
1. Neural Network (Model A) - warm-start from latest checkpoint if available
2. CatBoost (Model B) - GPU disabled if unavailable, MPS-compatible settings
3. LightGBM Meta-Model - uses OOF predictions from Models A & B

Optimized for Apple M2 with ~30 minute runtime target.
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
from typing import Tuple, List, Dict, Optional
import glob

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import lightgbm as lgb
import catboost as cb

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

# Create timestamped run directory with stacking subfolder
RUN_TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
RUN_DIR = Path(f"runs/{RUN_TIMESTAMP}")
STACKING_DIR = RUN_DIR / "stacking"
STACKING_DIR.mkdir(parents=True, exist_ok=True)

# Output paths within stacking directory
OOF_OUTPUT_PATH = STACKING_DIR / "oof_predictions.csv"
SUBMISSION_OUTPUT_PATH = STACKING_DIR / "submission.csv"
METRICS_OUTPUT_PATH = STACKING_DIR / "metrics.csv"
METADATA_OUTPUT_PATH = STACKING_DIR / "run_metadata.json"
README_OUTPUT_PATH = STACKING_DIR / "README.txt"
FIGS_DIR = STACKING_DIR / "figs"
LOGS_DIR = STACKING_DIR / "logs"

# Create subdirectories
FIGS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Web dashboard config
WEB_PORT = 8765
AUTO_OPEN_BROWSER = True

# Model-specific parameters
# Neural Network (Model A) - optimized for speed
NN_PARAMS = {
    'hidden_layers': [512, 256, 128],
    'dropout': 0.2,
    'emb_dropout': 0.1,
    'input_dropout': 0.1,
    'batch_size': 4096,
    'epochs': 7,  # Reduced for speed
    'patience': 5,  # Reduced for speed
    'base_lr': 1e-3,
    'weight_decay': 1e-5,
    'use_mps': True,  # Apple M2 acceleration
}

# CatBoost (Model B) - optimized for speed
CATBOOST_PARAMS = {
    'iterations': 750,  # Reduced for speed
    'learning_rate': 0.1,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': RANDOM_SEED,
    'verbose': False,
    'early_stopping_rounds': 50,
    'task_type': 'CPU',  # Force CPU to avoid GPU issues
    'thread_count': -1,
    'bootstrap_type': 'Bernoulli',
    'subsample': 0.8,
}

# LightGBM Meta-Model (Level 2) - small and fast
LGBM_META_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'num_leaves': 8,  # Small for meta-model
    'learning_rate': 0.05,
    'n_estimators': 300,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_child_samples': 20,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'random_state': RANDOM_SEED,
    'verbosity': -1,
    'n_jobs': 1,
    'force_col_wise': True,
    'deterministic': True
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
        'meta_model': []
    },
    'overall_progress': 0,
    'start_time': None,
    'elapsed_time': 0,
    'oof_aucs': {
        'neural_network': None,
        'catboost': None,
        'meta_model': None
    },
    'mean_cv_aucs': {
        'neural_network': None,
        'catboost': None
    },
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
                    
                    // Update performance metrics
                    document.getElementById('nn-oof-auc').textContent = 
                        data.oof_aucs.neural_network ? data.oof_aucs.neural_network.toFixed(6) : '-';
                    document.getElementById('cb-oof-auc').textContent = 
                        data.oof_aucs.catboost ? data.oof_aucs.catboost.toFixed(6) : '-';
                    document.getElementById('meta-oof-auc').textContent = 
                        data.oof_aucs.meta_model ? data.oof_aucs.meta_model.toFixed(6) : '-';
                    
                    // Update best model
                    if (data.oof_aucs.meta_model) {
                        const aucs = data.oof_aucs;
                        const bestAuc = Math.max(aucs.neural_network || 0, aucs.catboost || 0, aucs.meta_model || 0);
                        const bestModel = bestAuc === aucs.meta_model ? 'Meta Model' :
                                         bestAuc === aucs.neural_network ? 'Neural Network' : 'CatBoost';
                        document.getElementById('best-model').textContent = `${bestModel} (${bestAuc.toFixed(6)})`;
                    }
                    
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
            if (foldResults.neural_network.length > 0) {
                html += '<div class="model-card completed"><h4>Neural Network</h4>';
                foldResults.neural_network.forEach((result, i) => {
                    html += `<div>Fold ${i+1}: ${result.auc.toFixed(6)}</div>`;
                });
                html += '</div>';
            }
            
            // CatBoost results
            if (foldResults.catboost.length > 0) {
                html += '<div class="model-card completed"><h4>CatBoost</h4>';
                foldResults.catboost.forEach((result, i) => {
                    html += `<div>Fold ${i+1}: ${result.auc.toFixed(6)}</div>`;
                });
                html += '</div>';
            }
            
            // Meta model results
            if (foldResults.meta_model.length > 0) {
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
    if torch.backends.mps.is_available() and NN_PARAMS['use_mps']:
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
# DATA PREPROCESSING (From v5)
# =============================================================================
def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Preprocess training and test data using v5 preprocessing logic
    """
    log_message("Starting data preprocessing...")
    
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
    
    log_message(f"Numerical columns: {len(num_cols)}")
    log_message(f"Categorical columns: {len(cat_cols)}")
    
    # Handle rare categories
    for col in cat_cols:
        value_counts = combined_df[col].value_counts()
        rare_values = value_counts[value_counts < MIN_CAT_COUNT].index
        if len(rare_values) > 0:
            combined_df[col] = combined_df[col].replace(rare_values.tolist(), RARE_NAME)
            log_message(f"Column '{col}': {len(rare_values)} rare values replaced with '{RARE_NAME}'")
    
    # Encode categorical variables
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        combined_df[col] = le.fit_transform(combined_df[col].astype(str))
        label_encoders[col] = le
        log_message(f"Encoded '{col}': {len(le.classes_)} unique values")
    
    # Fill missing values
    if combined_df.isnull().sum().sum() > 0:
        log_message("Filling missing values...")
        for col in num_cols:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].median())
        
        for col in cat_cols:
            if combined_df[col].isnull().sum() > 0:
                combined_df[col] = combined_df[col].fillna(combined_df[col].mode()[0])
    
    # Split back into train and test
    train_processed = combined_df[:len(train_df)].copy()
    test_processed = combined_df[len(train_df):].copy()
    
    log_message(f"Processed train shape: {train_processed.shape}")
    log_message(f"Processed test shape: {test_processed.shape}")
    
    return train_processed, test_processed, num_cols, cat_cols

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
    warm_start_path: Optional[str] = None
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train neural network for a single fold"""
    
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
    cat_cardinalities = [int(X_train[c].nunique()) for c in cat_cols] if cat_cols else []
    
    # Create datasets
    train_dataset = TabDataset(X_fold_train, y_fold_train, num_cols, cat_cols)
    val_dataset = TabDataset(X_fold_val, y_fold_val, num_cols, cat_cols)
    test_dataset = TabDataset(X_test_scaled, None, num_cols, cat_cols)
    
    # Create data loaders - disable pin_memory for MPS compatibility
    pin_memory = device.type != 'mps'
    train_loader = DataLoader(train_dataset, batch_size=NN_PARAMS['batch_size'], shuffle=True, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=NN_PARAMS['batch_size'], shuffle=False, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=NN_PARAMS['batch_size'], shuffle=False, pin_memory=pin_memory)
    
    # Create model
    model = TabularNN(
        num_dim=len(num_cols),
        cat_cardinalities=cat_cardinalities,
        hidden_layers=NN_PARAMS['hidden_layers'],
        dropout=NN_PARAMS['dropout'],
        emb_dropout=NN_PARAMS['emb_dropout'],
        input_dropout=NN_PARAMS['input_dropout']
    ).to(device)
    
    # Warm start if checkpoint is available and shapes match
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            checkpoint = torch.load(warm_start_path, map_location=device)
            
            # Check if the model architecture matches
            model_state = checkpoint.get('model_state_dict', checkpoint)
            model.load_state_dict(model_state, strict=False)  # Allow partial loading
            log_message(f"Warm-started from checkpoint: {warm_start_path}")
        except Exception as e:
            log_message(f"Could not load checkpoint {warm_start_path}: {str(e)}. Training from scratch.")
    
    # Setup training
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=NN_PARAMS['base_lr'], weight_decay=NN_PARAMS['weight_decay'])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    
    for epoch in range(NN_PARAMS['epochs']):
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
            log_message(f"NN Fold {fold_idx + 1} Epoch {epoch + 1}/{NN_PARAMS['epochs']}: "
                       f"Train Loss: {train_loss/len(train_loader):.4f}, Val AUC: {val_auc:.4f}, "
                       f"Best: {best_val_auc:.4f}")
        
        if patience_counter >= NN_PARAMS['patience']:
            log_message(f"Early stopping at epoch {epoch + 1}")
            break
    
    # Generate OOF and test predictions
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
    
    return float(best_val_auc), np.array(oof_preds), np.array(test_preds)

def train_catboost_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    cat_cols: List[str]
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Train CatBoost for a single fold"""
    
    log_message(f"Training CatBoost - Fold {fold_idx + 1}")
    
    X_fold_train = X_train.iloc[train_indices]
    y_fold_train = y_train[train_indices]
    X_fold_val = X_train.iloc[val_indices]
    y_fold_val = y_train[val_indices]
    
    # Create CatBoost model
    model = cb.CatBoostClassifier(**CATBOOST_PARAMS)
    
    # Train the model
    model.fit(
        X_fold_train, y_fold_train,
        eval_set=(X_fold_val, y_fold_val),
        cat_features=cat_cols,
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

def train_lightgbm_meta_model(
    oof_nn: np.ndarray,
    oof_cb: np.ndarray,
    y_train: np.ndarray,
    test_nn: np.ndarray,
    test_cb: np.ndarray
) -> Tuple[float, np.ndarray]:
    """Train LightGBM meta-model on OOF predictions"""
    
    log_message("Training LightGBM Meta-Model...")
    
    # Create meta-features
    X_meta = np.column_stack([oof_nn, oof_cb])
    X_test_meta = np.column_stack([test_nn, test_cb])
    
    # Setup cross-validation for meta-model
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
    meta_oof_preds = np.zeros(len(X_meta))
    meta_test_preds = np.zeros(len(X_test_meta))
    fold_aucs = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X_meta, y_train)):
        X_fold_train, X_fold_val = X_meta[train_idx], X_meta[val_idx]
        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
        
        # Create LightGBM datasets
        train_data = lgb.Dataset(X_fold_train, label=y_fold_train)
        val_data = lgb.Dataset(X_fold_val, label=y_fold_val)
        
        # Train model
        model = lgb.train(
            LGBM_META_PARAMS,
            train_data,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        # Predictions
        fold_val_preds = model.predict(X_fold_val)
        fold_test_preds = model.predict(X_test_meta)
        
        # Ensure predictions are numpy arrays
        fold_val_preds = np.array(fold_val_preds).flatten()
        fold_test_preds = np.array(fold_test_preds).flatten()
        
        meta_oof_preds[val_idx] = fold_val_preds
        meta_test_preds += fold_test_preds / N_SPLITS
        
        # Calculate fold AUC
        fold_auc = roc_auc_score(y_fold_val, fold_val_preds)
        fold_aucs.append(fold_auc)
        
        log_message(f"Meta-model Fold {fold_idx + 1} - AUC: {fold_auc:.4f}")
    
    # Overall meta-model AUC
    meta_oof_auc = roc_auc_score(y_train, meta_oof_preds)
    log_message(f"Meta-model OOF AUC: {meta_oof_auc:.4f}")
    
    return float(meta_oof_auc), meta_test_preds

def plot_results(
    y_train: np.ndarray,
    oof_nn: np.ndarray,
    oof_cb: np.ndarray,
    meta_test_preds: np.ndarray,
    nn_aucs: List[float],
    cb_aucs: List[float],
    meta_auc: float
):
    """Create and save result plots"""
    
    log_message("Creating result plots...")
    
    # Calculate OOF AUCs for plotting
    oof_nn_auc = roc_auc_score(y_train, oof_nn)
    oof_cb_auc = roc_auc_score(y_train, oof_cb)
    
    # Plot 1: ROC Curves for Level 1 models
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    RocCurveDisplay.from_predictions(y_train, oof_nn, name=f'Neural Network (AUC={oof_nn_auc:.4f})')
    RocCurveDisplay.from_predictions(y_train, oof_cb, name=f'CatBoost (AUC={oof_cb_auc:.4f})', ax=plt.gca())
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.title('Level 1 Models - ROC Curves')
    plt.legend()
    
    # Plot 2: Model performance comparison
    plt.subplot(1, 2, 2)
    models = ['Neural Network', 'CatBoost', 'Meta-Model']
    aucs = [oof_nn_auc, oof_cb_auc, meta_auc]
    colors = ['skyblue', 'lightcoral', 'lightgreen']
    
    bars = plt.bar(models, aucs, color=colors, alpha=0.7)
    plt.title('Model Performance Comparison')
    plt.ylabel('ROC AUC Score')
    plt.ylim(0.5, max(aucs) + 0.05)
    
    # Add value labels on bars
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{auc:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Cross-validation results
    plt.figure(figsize=(10, 6))
    
    fold_numbers = list(range(1, N_SPLITS + 1))
    
    plt.plot(fold_numbers, nn_aucs, 'o-', label='Neural Network', color='skyblue', linewidth=2, markersize=6)
    plt.plot(fold_numbers, cb_aucs, 's-', label='CatBoost', color='lightcoral', linewidth=2, markersize=6)
    
    plt.axhline(y=float(oof_nn_auc), color='skyblue', linestyle='--', alpha=0.7, label=f'NN OOF AUC: {oof_nn_auc:.4f}')
    plt.axhline(y=float(oof_cb_auc), color='lightcoral', linestyle='--', alpha=0.7, label=f'CB OOF AUC: {oof_cb_auc:.4f}')
    plt.axhline(y=float(meta_auc), color='lightgreen', linestyle='-', linewidth=2, label=f'Meta AUC: {meta_auc:.4f}')
    
    plt.xlabel('Fold Number')
    plt.ylabel('ROC AUC Score')
    plt.title('Cross-Validation Performance by Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGS_DIR / 'cv_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    log_message(f"Plots saved to: {FIGS_DIR}")

def main():
    """Main stacking pipeline"""
    
    overall_start_time = time.time()
    set_seed(RANDOM_SEED)
    
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
    
    log_message("=" * 60)
    log_message("STACKING PIPELINE v5 - STARTED")
    log_message("=" * 60)
    log_message(f"Run timestamp: {RUN_TIMESTAMP}")
    log_message(f"Output directory: {STACKING_DIR}")
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
        
        # Preprocess data
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
        
        # Initialize prediction arrays
        oof_nn = np.zeros(len(X_train))
        oof_cb = np.zeros(len(X_train))
        test_nn = np.zeros(len(X_test))
        test_cb = np.zeros(len(X_test))
        
        nn_aucs = []
        cb_aucs = []
        
        device = get_device()
        log_message(f"Using device: {device}")
        
        # Find warm-start checkpoint
        warm_start_path = find_latest_checkpoint()
        
        log_message("=" * 50)
        log_message("LEVEL 1 TRAINING - Neural Network & CatBoost")
        log_message("=" * 50)
        
        update_training_state('current_stage', 'level1')
        update_training_state('overall_progress', 10)
        
        # Train Level 1 models
        for fold_idx, (train_indices, val_indices) in enumerate(skf.split(X_train, y_train)):
            fold_start_time = time.time()
            
            update_training_state('current_fold', fold_idx + 1)
            log_message(f"Training Fold {fold_idx + 1}/{N_SPLITS}")
            
            # Train Neural Network (Model A)
            update_training_state('current_model', 'Neural Network')
            nn_auc, nn_oof_preds, nn_test_preds = train_neural_network_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test,
                num_cols, cat_cols, device, warm_start_path
            )
            
            # Train CatBoost (Model B)
            update_training_state('current_model', 'CatBoost')
            cb_auc, cb_oof_preds, cb_test_preds = train_catboost_fold(
                fold_idx, train_indices, val_indices, X_train, y_train, X_test, cat_cols
            )
            
            # Store predictions
            oof_nn[val_indices] = nn_oof_preds
            oof_cb[val_indices] = cb_oof_preds
            test_nn += nn_test_preds / N_SPLITS
            test_cb += cb_test_preds / N_SPLITS
            
            nn_aucs.append(nn_auc)
            cb_aucs.append(cb_auc)
            
            # Update web dashboard with fold results
            training_state['fold_results']['neural_network'].append({'auc': nn_auc})
            training_state['fold_results']['catboost'].append({'auc': cb_auc})
            
            fold_time = time.time() - fold_start_time
            log_message(f"Fold {fold_idx + 1} completed in {fold_time:.1f}s")
            
            # Update overall progress
            progress = 10 + (fold_idx + 1) / N_SPLITS * 60  # 10% to 70%
            update_training_state('overall_progress', progress)
        
        # Calculate Level 1 OOF AUCs
        oof_nn_auc = roc_auc_score(y_train, oof_nn)
        oof_cb_auc = roc_auc_score(y_train, oof_cb)
        
        log_message("=" * 50)
        log_message("LEVEL 1 RESULTS")
        log_message("=" * 50)
        log_message(f"Neural Network - OOF AUC: {oof_nn_auc:.6f}, Mean CV: {np.mean(nn_aucs):.6f} ± {np.std(nn_aucs):.6f}")
        log_message(f"CatBoost - OOF AUC: {oof_cb_auc:.6f}, Mean CV: {np.mean(cb_aucs):.6f} ± {np.std(cb_aucs):.6f}")
        
        # Update web dashboard with Level 1 results
        update_training_state('oof_aucs', {
            'neural_network': float(oof_nn_auc),
            'catboost': float(oof_cb_auc),
            'meta_model': None
        })
        update_training_state('mean_cv_aucs', {
            'neural_network': float(np.mean(nn_aucs)),
            'catboost': float(np.mean(cb_aucs))
        })
        
        log_message("=" * 50)
        log_message("LEVEL 2 TRAINING - LightGBM Meta-Model")
        log_message("=" * 50)
        
        update_training_state('current_stage', 'level2')
        update_training_state('current_model', 'LightGBM Meta-Model')
        update_training_state('overall_progress', 75)
        
        # Train Level 2 meta-model
        meta_auc, meta_test_preds = train_lightgbm_meta_model(
            oof_nn, oof_cb, y_train, test_nn, test_cb
        )
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            ID_COL: train_processed[ID_COL],
            'oof_nn': oof_nn,
            'oof_cb': oof_cb,
            'oof_meta': meta_test_preds[:len(oof_nn)] if len(meta_test_preds) == len(oof_nn) else np.zeros(len(oof_nn)),  # Placeholder
            TARGET_COL: y_train
        })
        oof_df.to_csv(OOF_OUTPUT_PATH, index=False)
        log_message(f"OOF predictions saved to: {OOF_OUTPUT_PATH}")
        
        # Save final submission (using meta-model predictions)
        submission_df = pd.DataFrame({
            ID_COL: test_processed[ID_COL],
            TARGET_COL: meta_test_preds
        })
        submission_df.to_csv(SUBMISSION_OUTPUT_PATH, index=False)
        log_message(f"Submission saved to: {SUBMISSION_OUTPUT_PATH}")
        
        # Save metrics
        metrics_data = []
        for i in range(N_SPLITS):
            metrics_data.append({'model': 'neural_network', 'fold': i, 'auc': nn_aucs[i]})
            metrics_data.append({'model': 'catboost', 'fold': i, 'auc': cb_aucs[i]})
        
        metrics_data.extend([
            {'model': 'neural_network', 'fold': 'OOF', 'auc': oof_nn_auc},
            {'model': 'catboost', 'fold': 'OOF', 'auc': oof_cb_auc},
            {'model': 'meta_model', 'fold': 'OOF', 'auc': meta_auc}
        ])
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(METRICS_OUTPUT_PATH, index=False)
        log_message(f"Metrics saved to: {METRICS_OUTPUT_PATH}")
        
        # Create plots
        plot_results(y_train, oof_nn, oof_cb, meta_test_preds, nn_aucs, cb_aucs, meta_auc)
        
        # Calculate total time
        total_time = time.time() - overall_start_time
        
        # Save README
        readme_content = f"""Stacking Pipeline v5 Results
============================

This folder contains outputs from a fast stacking ensemble pipeline optimized for Apple M2.

Pipeline:
1. Neural Network (Model A) - warm-start from checkpoint if available
2. CatBoost (Model B) - CPU-optimized for stability  
3. LightGBM Meta-Model - uses OOF predictions from Models A & B

Results:
- Neural Network OOF AUC: {oof_nn_auc:.6f}
- CatBoost OOF AUC: {oof_cb_auc:.6f}
- Meta-Model OOF AUC: {meta_auc:.6f}

Performance Summary:
- Neural Network CV: {np.mean(nn_aucs):.6f} ± {np.std(nn_aucs):.6f}
- CatBoost CV: {np.mean(cb_aucs):.6f} ± {np.std(cb_aucs):.6f}
- Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)

Configuration:
- CV Folds: {N_SPLITS}
- Random Seed: {RANDOM_SEED}
- Total Features: {len(feature_cols)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)
- Target Rate: {float(np.mean(y_train)):.4f}

Files:
- oof_predictions.csv: OOF predictions for all models
- submission.csv: Final meta-model predictions for test set
- metrics.csv: Per-fold and overall AUC scores
- figs/: ROC curves and performance comparison plots
- logs/: Detailed training logs

Optimization Features:
- Neural Network: {NN_PARAMS['epochs']} epochs with early stopping
- CatBoost: {CATBOOST_PARAMS['iterations']} iterations with early stopping
- Meta-Model: {LGBM_META_PARAMS['n_estimators']} estimators, {LGBM_META_PARAMS['num_leaves']} leaves
- MPS acceleration: {NN_PARAMS['use_mps']}
- Warm-start: {"Yes" if warm_start_path else "No"}

Timestamp: {RUN_TIMESTAMP}
"""
        
        with open(README_OUTPUT_PATH, 'w') as f:
            f.write(readme_content)
        log_message(f"README saved to: {README_OUTPUT_PATH}")
        
        # Save metadata
        metadata = {
            'pipeline': 'stacking_v5',
            'models': {
                'neural_network': {
                    'oof_auc': float(oof_nn_auc),
                    'mean_cv_auc': float(np.mean(nn_aucs)),
                    'std_cv_auc': float(np.std(nn_aucs)),
                    'fold_aucs': [float(auc) for auc in nn_aucs],
                    'params': NN_PARAMS
                },
                'catboost': {
                    'oof_auc': float(oof_cb_auc),
                    'mean_cv_auc': float(np.mean(cb_aucs)),
                    'std_cv_auc': float(np.std(cb_aucs)),
                    'fold_aucs': [float(auc) for auc in cb_aucs],
                    'params': CATBOOST_PARAMS
                },
                'meta_model': {
                    'oof_auc': float(meta_auc),
                    'params': LGBM_META_PARAMS
                }
            },
            'data_info': {
                'n_splits': N_SPLITS,
                'random_seed': RANDOM_SEED,
                'total_features': len(feature_cols),
                'numerical_features': len(num_cols),
                'categorical_features': len(cat_cols),
                'target_rate': float(np.mean(y_train))
            },
            'runtime': {
                'total_time_seconds': total_time,
                'total_time_minutes': total_time / 60
            },
            'warm_start': {
                'used': warm_start_path is not None,
                'checkpoint_path': warm_start_path
            },
            'timestamp': RUN_TIMESTAMP
        }
        
        with open(METADATA_OUTPUT_PATH, 'w') as f:
            json.dump(metadata, f, indent=2)
        log_message(f"Metadata saved to: {METADATA_OUTPUT_PATH}")
        
        update_training_state('overall_progress', 95)
        
        # Final summary
        log_message("=" * 60)
        log_message("STACKING PIPELINE COMPLETED")
        log_message("=" * 60)
        log_message(f"Total Runtime: {total_time:.1f}s ({total_time/60:.1f} minutes)")
        log_message(f"Neural Network OOF AUC: {oof_nn_auc:.6f}")
        log_message(f"CatBoost OOF AUC: {oof_cb_auc:.6f}")
        log_message(f"Meta-Model OOF AUC: {meta_auc:.6f}")
        log_message(f"Final submission shape: {submission_df.shape}")
        log_message("🎉 Stacking pipeline completed successfully!")
        
        # Update final web dashboard state
        update_training_state('current_stage', 'complete')
        update_training_state('overall_progress', 100)
        update_training_state('status', 'completed')
        update_training_state('completed', True)
        training_state['oof_aucs']['meta_model'] = float(meta_auc)
        
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
