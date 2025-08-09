# 🧠 Machine Learning Models & Tools Reference

## 📋 Table of Contents
- [Supervised Learning Models](#supervised-learning-models)
- [Unsupervised Learning Models](#unsupervised-learning-models)
- [Model Utilities & Enhancements](#model-utilities--enhancements)
- [AutoML Frameworks](#automl-frameworks)
- [Deep Learning Frameworks](#deep-learning-frameworks)
- [Visualization & Monitoring](#visualization--monitoring)

---

## 🧠 Supervised Learning Models

### 🔹 Classification & Regression

#### 🔸 Linear Models
- **Logistic Regression** (binary classification)
- **Linear Regression** (regression)
- **Ridge / Lasso / ElasticNet** (regularized regression)

#### 🔸 Tree-Based Models
- **Decision Tree**
- **Random Forest**
- **Extra Trees**
- **Gradient Boosting Machine (GBM)**
- **XGBoost**
- **LightGBM**
- **CatBoost**

#### 🔸 Neural Networks (Deep Learning)
- **Feedforward Neural Network (MLP)** 
- **TabNet** (deep learning for tabular data)
- **Deep & Wide Models**
- **Autoencoders** (for unsupervised pretraining / denoising)

#### 🔸 Kernel-Based Models
- **Support Vector Machine (SVM)**
- **SVR** (for regression)

#### 🔸 Probabilistic Models
- **Naive Bayes** (Gaussian, Multinomial)
- **Bayesian Ridge**
- **Gaussian Processes**

#### 🔸 k-Nearest Neighbors
- **KNN Classifier**
- **KNN Regressor**

---

## 🧪 Unsupervised Learning Models
*(Less relevant for classification, but helpful for feature engineering)*

- **K-Means Clustering**
- **DBSCAN**
- **PCA** (Principal Component Analysis)
- **t-SNE / UMAP** (for visualization)
- **Autoencoders** (for representation learning)

---

## 🛠 Model Utilities & Enhancements

### 🔸 Meta-Models (Ensemble Methods)
- **Stacking** – combine predictions from many models
- **Blending** – weighted average of multiple models
- **Voting Classifier** – majority vote from base models

### 🔸 Feature Engineering Tools
- **Polynomial Features**
- **One-Hot Encoding / Target Encoding**
- **Feature Selection** (e.g., SelectKBest, SHAP)

### 🔸 Validation Tools
- **Train/Validation Split**
- **K-Fold Cross-Validation**
- **Stratified K-Fold**
- **TimeSeriesSplit** (for time-series data)

---

## 🤖 AutoML Frameworks
*(Automate model selection + hyperparameter tuning)*

- **AutoGluon** (by AWS, tabular-focused)
- **H2O AutoML**
- **TPOT** (genetic programming)
- **Google AutoML Tables**
- **MLJAR**
- **FLAML** (fast lightweight AutoML)
- **Auto-sklearn**

---

## 🧠 Deep Learning Frameworks

- **TensorFlow / Keras**
- **PyTorch**
- **JAX** (experimental, fast gradient engine)

---

## 📊 Visualization & Monitoring

- **SHAP / LIME** – interpretability
- **TensorBoard / WandB** – tracking experiments
- **Matplotlib / Seaborn / Plotly** – plotting

---

## 💡 Quick Tips for Bank Classification

For your bank classification project, consider these approaches:

1. **Start Simple**: Begin with Logistic Regression or Random Forest
2. **Feature Engineering**: Use techniques like one-hot encoding for categorical variables
3. **Ensemble Methods**: Try combining multiple models with voting or stacking
4. **Hyperparameter Tuning**: Use Grid Search or Random Search to optimize
5. **Cross-Validation**: Use Stratified K-Fold to ensure balanced splits
6. **Model Interpretation**: Use SHAP values to understand feature importance

### Recommended Model Pipeline:
1. **Baseline**: Logistic Regression
2. **Tree-based**: Random Forest or XGBoost
3. **Neural Network**: Your current MLP approach
4. **Ensemble**: Combine the best performing models
