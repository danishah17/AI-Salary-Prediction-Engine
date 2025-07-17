# AI Salary Prediction Engine

A sophisticated machine learning system that predicts salaries in the data science and technology sector using ensemble models and advanced feature engineering techniques.

##  Features

- **Advanced Machine Learning**: Ensemble model combining XGBoost and Gradient Boosting Regressors
- **Intelligent Feature Engineering**: Job title clustering, target encoding, and polynomial features
- **Hyperparameter Optimization**: Automated model tuning using RandomizedSearchCV
- **Data Validation**: Comprehensive data cleaning with outlier detection and KNN imputation
- **Interactive Predictions**: Command-line interface for real-time salary predictions
- **Model Persistence**: Save and load trained models for future use
- **Comprehensive Visualizations**: Multiple charts showing salary trends and patterns
- **Cross-Validation**: Robust model evaluation with k-fold cross-validation

##  Model Performance

The system achieves high accuracy with:
- **R² Score**: >0.90 (target performance)
- **Cross-Validation**: K-fold validation for robust performance estimation
- **Confidence Intervals**: 15% prediction intervals for uncertainty quantification

##  Installation

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib scipy
```

### Required Libraries

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: matplotlib, seaborn
- **Utilities**: joblib, scipy

##  Project Structure

```
AI-Salary-Prediction-Engine/
├── AI_Salary_Pridictor.py    # Main prediction engine
├── README.md                 # Project documentation
├── salaries.csv             # Dataset (if available)
└── generated_outputs/       # Model outputs and visualizations
    ├── Figure_1.png         # Salary distribution
    ├── Figure_2.png         # Experience level analysis
    ├── Figure_3.png         # Country-wise salaries
    ├── Figure_4.png         # Company size impact
    ├── Figure_5.png         # Remote work analysis
    ├── Figure_6.png         # Yearly trends
    ├── Figure_7.png         # Top paying jobs
    ├── OUTPUT 1.png         # Model performance
    └── OUTPUT 2.png         # Feature importance
```

##  Quick Start

### 1. Basic Usage

```python
from AI_Salary_Pridictor import OptimizedSalaryPredictor
import pandas as pd

# Initialize predictor
predictor = OptimizedSalaryPredictor()

# Load your data
df = pd.read_csv('salaries.csv')

# Train the model
metrics = predictor.train_model(df)

# Make a prediction
user_input = {
    'experience_level': 'SE',
    'job_title': 'Data Scientist',
    'company_size': 'L',
    'employee_residence': 'US',
    'company_location': 'US',
    'work_year': 2024,
    'remote_ratio': 50,
    'employment_type': 'FT'
}

prediction = predictor.predict_with_confidence(user_input)
print(f"Predicted Salary: ${prediction['predicted_salary']:,.2f}")
```

### 2. Command Line Interface

Run the main script for interactive predictions:

```bash
python AI_Salary_Pridictor.py
```

Follow the prompts to enter job details and get salary predictions.

##  Input Parameters

| Parameter | Description | Valid Values |
|-----------|-------------|--------------|
| `experience_level` | Professional experience level | EN (Entry), MI (Mid), SE (Senior), EX (Executive) |
| `job_title` | Job position title | Any string (e.g., "Data Scientist") |
| `company_size` | Company size category | S (Small), M (Medium), L (Large) |
| `employee_residence` | Employee's country | ISO country codes (e.g., US, CA, GB) |
| `company_location` | Company's country | ISO country codes (e.g., US, CA, GB) |
| `work_year` | Year of employment | 2020-2025 |
| `remote_ratio` | Remote work percentage | 0 (On-site), 50 (Hybrid), 100 (Remote) |
| `employment_type` | Employment contract type | FT (Full-time), PT (Part-time), CT (Contract) |

## 🔧 Advanced Features

### Feature Engineering

The system implements sophisticated feature engineering:

- **Job Title Clustering**: TF-IDF vectorization with K-means clustering
- **Target Encoding**: Mean encoding for categorical variables
- **Polynomial Features**: Interaction terms for numeric variables
- **Geographic Features**: High-paying country indicators
- **Temporal Features**: Year-based trend analysis

### Model Architecture

```python
# Ensemble Model Components
- XGBoost Regressor (60% weight)
- Gradient Boosting Regressor (40% weight)
- Voting Regressor for final predictions
```

### Data Validation

- Salary range filtering (10K - 600K USD)
- Year validation (2020-2025)
- Outlier removal using IQR method
- KNN imputation for missing values
- Duplicate removal

##  Visualizations

The system generates comprehensive visualizations:

1. **Salary Distribution**: Histogram with KDE
2. **Experience Level Analysis**: Average salary by experience
3. **Geographic Analysis**: Top countries by salary
4. **Company Size Impact**: Salary variation by company size
5. **Remote Work Analysis**: Remote vs on-site compensation
6. **Temporal Trends**: Salary evolution over years
7. **Job Title Analysis**: Highest paying positions
8. **Feature Importance**: Model feature rankings

##  Model Evaluation

### Metrics Tracked

- **MAE**: Mean Absolute Error
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of Determination
- **MAPE**: Mean Absolute Percentage Error
- **Cross-Validation**: K-fold R² scores

### Performance Monitoring

```python
# Example output
MODEL EVALUATION RESULTS
==================================================
MAE: 8234.5678
MSE: 123456789.0123
RMSE: 11111.2345
R²: 0.9234
MAPE: 12.34
CV_R²_mean: 0.9123
CV_R²_std: 0.0234
==================================================
```

##  Model Persistence

### Save Trained Model

```python
model_path = predictor.save_model('my_salary_model')
```

### Load Existing Model

```python
predictor = OptimizedSalaryPredictor()
predictor.load_model('my_salary_model_20241201_143022.pkl')
```

##  Sample Data Generation

If you don't have a dataset, the system can generate sample data:

```python
from AI_Salary_Pridictor import create_sample_data

# Generate 5000 sample records
df = create_sample_data(n_samples=5000)
```

##  Configuration

### Hyperparameter Tuning

The system automatically optimizes:

- **XGBoost Parameters**: n_estimators, max_depth, learning_rate, subsample
- **Gradient Boosting Parameters**: n_estimators, max_depth, learning_rate
- **Ensemble Weights**: Optimal combination of base models

### Performance Optimization

- Reduced estimators for faster training
- Parallel processing disabled for stability
- Memory-efficient data processing
- Optimized cross-validation folds

##  Error Handling

The system includes comprehensive error handling:

- Input validation for all parameters
- Model state verification
- File I/O error management
- Prediction error recovery
- Logging for debugging

##  Logging

Detailed logging is implemented throughout:

```python
# Log levels
- INFO: General operation status
- WARNING: Performance alerts
- ERROR: Exception handling
```



##  Future Enhancements

- [ ] Web interface for predictions
- [ ] Real-time data integration
- [ ] Additional ML algorithms
- [ ] Industry-specific models
- [ ] API endpoint development
- [ ] Docker containerization
- [ ] Cloud deployment options


**Note**: This system is designed for educational and research purposes. Always validate predictions with domain expertise and current market data.
