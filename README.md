# Retention in Gamified Learning: Analyzing Duolingo User Engagement

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A machine learning analysis of user engagement patterns in Duolingo using the 2018 SLAM dataset to predict retention and identify distinct user profiles.

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Dataset](#dataset)
- [Methods](#methods)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [User Profiles](#user-profiles)
- [Contributors](#contributors)
- [References](#references)
- [License](#license)

## 🎯 Overview

This project analyzes user engagement patterns in Duolingo using machine learning techniques to understand what drives user retention in gamified language learning platforms. We analyzed 5,520 users' first 30 days of activity from the 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM) dataset.

Our research reveals that **engagement frequency and consistency significantly outweigh performance accuracy** in predicting user retention, challenging purely competence-based models of learning.

### Research Question

What behavioral patterns predict user retention in language learning apps, and how can we segment users to develop targeted retention strategies?

## 🔍 Key Findings

1. **Behavioral metrics > Performance metrics**: Engagement frequency (average blocks per day, sessions per day) and consistency (variance in daily usage) are stronger predictors of retention than accuracy scores.

2. **Four distinct user profiles** identified through unsupervised learning:
   - **Persistent Challengers** (32.1%): High engagement despite struggling
   - **Efficient Achievers** (25.9%): High performance with moderate effort
   - **Casual Explorers** (36.1%): Minimal commitment across metrics
   - **Deep Thinkers** (5.9%): Deliberate, time-intensive approach

3. **Model Performance**: 
   - Random Forest: AUC = 0.83
   - Neural Network: AUC = 0.85
   - Strong prediction for early churn (0-7 days, AUC = 0.86)

## 📊 Dataset

**Source**: [2018 Duolingo SLAM Dataset](https://doi.org/10.7910/DVN/8SWHNO) (Settles, 2018)

- **Raw data**: 2,243,983 token-level observations
- **Processed**: 5,520 user-level examples
- **Time window**: First 30 days of user activity
- **Train/Test split**: 80/20 with stratified sampling

### Features

The analysis uses 17 engineered features across four categories:

**Correctness Metrics**
- Block average correctness
- Reverse tap, reverse translate, and listen average correctness

**Response Time Metrics**
- Average response times across exercise types
- Response time variance

**Activity Metrics**
- Average blocks per session/day
- Average sessions per day
- User ability score (combined correctness and normalized response time)

**Consistency Measures**
- Session length variance
- Blocks per session/day variance
- Sessions per day variance

### Target Variable

Churn time category:
- Class 1: 0-6 days
- Class 2: 7-13 days
- Class 3: 14-20 days
- Class 4: 21+ days

## 🔬 Methods

### Supervised Learning

1. **Softmax Classifier** (Baseline)
   - Multi-class probabilistic classification
   - Overall AUC: 0.83

2. **Neural Network**
   - 3-layer architecture with ReLU activation
   - Batch normalization and RMSprop optimizer
   - Early stopping to prevent overfitting
   - Overall AUC: 0.85

3. **Random Forest**
   - Hyperparameter optimization via grid search (576 combinations)
   - 10-fold cross-validation
   - Final configuration: 50 estimators, max depth 10
   - Overall AUC: 0.83

### Unsupervised Learning

4. **Principal Component Analysis (PCA)**
   - Reduced 17 features to 4 principal components
   - Captures 60% of total variance
   - Identified four engagement dimensions

5. **Hierarchical Clustering**
   - Agglomerative clustering with Ward's linkage
   - Applied to PCA-transformed space
   - Identified 4 distinct user profiles

### Data Preprocessing

- Log transformation for variance features
- QuantileTransformer for response time variance (handles outliers)
- MinMaxScaler for all numerical features (0-1 normalization)
- SMOTE attempted for class imbalance (modest performance decline observed)

## 📁 Project Structure

```
.
├── data_wrangling.ipynb          # Data preprocessing and feature engineering
├── Neural_Network.ipynb          # Neural network implementation
├── randomforest.ipynb            # Random Forest model
├── xgbboost.ipynb               # XGBoost experiments
├── pca.ipynb                    # PCA and clustering analysis
├── CS129_Final_Project_Paper.pdf # Full research paper
└── README.md                    # This file
```

## 🚀 Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook

### Required Libraries

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
pip install tensorflow  # For Neural Network
pip install xgboost     # For XGBoost experiments
pip install imbalanced-learn  # For SMOTE
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/duolingo-retention-analysis.git
cd duolingo-retention-analysis

# Install dependencies
pip install -r requirements.txt

# Download the dataset from Harvard Dataverse
# https://doi.org/10.7910/DVN/8SWHNO
```

## 💻 Usage

### 1. Data Preprocessing

```bash
jupyter notebook data_wrangling.ipynb
```

This notebook handles:
- Loading raw SLAM data
- Feature engineering
- Data standardization
- Train/test split

### 2. Run Models

**Neural Network:**
```bash
jupyter notebook Neural_Network.ipynb
```

**Random Forest:**
```bash
jupyter notebook randomforest.ipynb
```

**PCA and Clustering:**
```bash
jupyter notebook pca.ipynb
```

### 3. Reproduce Results

Each notebook is self-contained and can be run independently after completing data preprocessing. Results include:
- Model performance metrics (AUC, confusion matrices)
- Feature importance plots
- PCA loadings and variance explained
- User cluster visualizations

## 📈 Results

### Model Performance

| Model | Overall AUC | Class 1 (0-7d) | Class 2 (7-13d) | Class 3 (14-20d) | Class 4 (21+d) |
|-------|-------------|----------------|-----------------|------------------|----------------|
| Neural Network | 0.85 | 0.88 | 0.69 | 0.77 | 0.79 |
| Random Forest | 0.83 | 0.86 | 0.67 | 0.72 | 0.73 |
| Softmax | 0.83 | - | - | - | - |

### Feature Importance (Top 5)

1. **Average blocks per day** (0.168)
2. **Average sessions per day** (0.117)
3. **Blocks per day variance** (0.080)
4. **Sessions per day variance** (0.071)
5. **Blocks per session variance** (0.066)

### PCA Components

Four principal components explain distinct engagement dimensions:

- **PC1**: Engagement Variability & Intensity
- **PC2**: Learning Performance vs. Structure
- **PC3**: Response Time & Learning Efficiency
- **PC4**: Learning Schedule Variability

## 👥 User Profiles

### 1. Persistent Challengers (32.1%)
**Characteristics**: High engagement despite low performance

**Retention Strategy**:
- Provide detailed explanations for incorrect answers
- Implement adaptive difficulty adjustment
- Create "practice mode" without error penalties

### 2. Efficient Achievers (25.9%)
**Characteristics**: High performance with moderate effort

**Retention Strategy**:
- Special achievement badges
- Unlock premium content/advanced lessons
- Competitive elements (leaderboards)

### 3. Casual Explorers (36.1%)
**Characteristics**: Minimal commitment across metrics

**Retention Strategy**:
- Bite-sized daily goals
- Lower initial difficulty
- Immediate positive feedback

### 4. Deep Thinkers (5.9%)
**Characteristics**: Deliberate, time-intensive approach

**Retention Strategy**:
- Remove time pressures
- Reward accuracy over speed
- "Pick up where you left off" features

## 👨‍🔬 Contributors

- **Kaz Fukuhara** - Neural Network implementation, SMOTE analysis
- **Xinman (Yoyo) Liu** - Feature engineering, Random Forest, feature importance
- **Xinyu (Teah) Shi** - PCA, hierarchical clustering, user segmentation

*Stanford University, CS129: Applied Machine Learning*

## 📚 References

Key papers informing this work:

1. Settles, B. (2018). Data for the 2018 Duolingo SLAM. Harvard Dataverse.
2. Chrzan et al. (2023). Analyzing Duolingo User Behavior Data Using Semi-Supervised Learning.
3. Loewen et al. (2019). Mobile-assisted language learning: A Duolingo case study.
4. Mazal, J. (2023). How Duolingo reignited user growth. Lenny's Newsletter.

See `CS129_Final_Project_Paper.pdf` for complete references.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Stanford Graduate School of Education
- Duolingo for making the SLAM dataset publicly available
- CS129 course staff for guidance and feedback

## 📧 Contact

For questions or collaborations, please reach out:
- Kaz Fukuhara: kazf28@stanford.edu
- Xinman Liu: xinman@stanford.edu
- Xinyu Shi: teah2001@stanford.edu

---

**Citation**: If you use this work, please cite:
```
Fukuhara, K., Liu, X., & Shi, X. (2024). Retention in Gamified Learning: 
Analyzing Duolingo User Engagement with Machine Learning. 
Stanford University CS129 Final Project.
```
