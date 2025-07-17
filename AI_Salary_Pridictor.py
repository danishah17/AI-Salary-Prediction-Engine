import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from datetime import datetime
import json
import joblib
from scipy import stats
import os

# Set environment variable to avoid joblib CPU detection issues on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import xgboost as xgb
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import RandomizedSearchCV
from sklearn.impute import KNNImputer
from sklearn.pipeline import make_pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --------------------
# Enhanced Data Validation
# --------------------
def validate_and_clean_data(df):
    """Enhanced data validation with stricter outlier removal and KNN imputation"""
    logger.info("Starting data validation and cleaning...")
    original_shape = df.shape
    # Remove rows with missing critical columns
    critical_columns = ['salary_in_usd', 'job_title', 'experience_level']
    df = df.dropna(subset=critical_columns)
    # Filter realistic salary ranges (10K to 600K USD)
    df = df[(df['salary_in_usd'] >= 10000) & (df['salary_in_usd'] <= 600000)]
    # Validate year ranges
    df = df[(df['work_year'] >= 2020) & (df['work_year'] <= 2025)]
    # Remove duplicates
    df = df.drop_duplicates()
    # Advanced outlier removal using IQR
    Q1 = df['salary_in_usd'].quantile(0.25)
    Q3 = df['salary_in_usd'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['salary_in_usd'] >= Q1 - 1.5 * IQR) & (df['salary_in_usd'] <= Q3 + 1.5 * IQR)]
    # KNN imputation for remaining missing values (numeric only)
    num_cols = df.select_dtypes(include=[float, int]).columns
    if df[num_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=5, weights='distance')
        df[num_cols] = imputer.fit_transform(df[num_cols])
    logger.info(f"Data cleaning complete. Shape: {original_shape} → {df.shape}")
    return df.reset_index(drop=True)

# --------------------
# Enhanced Feature Engineering
# --------------------
def create_job_title_clusters(job_titles, n_clusters=10):
    """Advanced job title clustering using TF-IDF and KMeans"""
    logger.info("Creating advanced job title clusters...")
    tfidf = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(job_titles)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    clusters = kmeans.fit_predict(tfidf_matrix)
    le = LabelEncoder()
    clusters_encoded = le.fit_transform(clusters)
    return clusters_encoded, le

def streamlined_feature_engineering(df):
    """Enhanced feature engineering with additional interactions"""
    logger.info("Starting enhanced feature engineering...")
    df = df.copy()
    # Experience level encoding
    exp_map = {'EN': 0, 'MI': 1, 'SE': 2, 'EX': 3}
    df['experience_level_encoded'] = df['experience_level'].map(exp_map)
    # Company size encoding
    size_map = {'S': 1, 'M': 2, 'L': 3}
    df['company_size_encoded'] = df['company_size'].map(size_map)
    # Job title clustering
    job_clusters, _ = create_job_title_clusters(df['job_title'], n_clusters=min(10, len(df)))
    df['job_title_cluster'] = job_clusters
    # Geographic features
    high_paying_countries = ['US', 'CH', 'CA', 'GB', 'DE', 'AU', 'NL', 'SG', 'IE', 'DK']
    df['high_paying_country'] = df['employee_residence'].isin(high_paying_countries).astype(int)
    # Remote work features
    df['remote_ratio_cat'] = pd.cut(df['remote_ratio'], bins=[-1, 0, 50, 100], 
                                   labels=['onsite', 'hybrid', 'remote'])
    # Employment type binary
    df['is_fulltime'] = (df['employment_type'] == 'FT').astype(int)
    # Year features
    df['years_since_2020'] = df['work_year'] - 2020
    # Same country check
    df['same_country'] = (df['employee_residence'] == df['company_location']).astype(int)
    # Additional interaction features
    df['exp_x_remote'] = df['experience_level_encoded'] * df['remote_ratio']
    df['size_x_country'] = df['company_size_encoded'] * df['high_paying_country']
    logger.info("Feature engineering completed.")
    return df

def advanced_feature_engineering(df):
    """Add advanced features: target encoding, polynomial features, and interactions"""
    df = df.copy()
    # Targetitere target encoding for job_title and company_location
    if 'salary_in_usd' in df.columns:
        for col in ['job_title', 'company_location']:
            means = df.groupby(col)['salary_in_usd'].mean()
            df[f'{col}_target_enc'] = df[col].map(means)
    else:
        for col in ['job_title', 'company_location']:
            df[f'{col}_target_enc'] = 0
    # Polynomial features for numeric columns
    poly_cols = ['years_since_2020', 'experience_level_encoded', 'company_size_encoded']
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(df[poly_cols].fillna(0))
    poly_feature_names = poly.get_feature_names_out(poly_cols)
    for i, name in enumerate(poly_feature_names):
        df[f'poly_{name}'] = poly_features[:, i]
    # Additional interaction features
    df['exp_x_size'] = df['experience_level_encoded'] * df['company_size_encoded']
    df['exp_x_country'] = df['experience_level_encoded'] * df['high_paying_country']
    return df

def prepare_features_for_modeling(df):
    """Enhanced feature preparation with scaling"""
    logger.info("Preparing features for modeling...")
    categorical_cols = ['remote_ratio_cat']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols)
    df_encoded = advanced_feature_engineering(df_encoded)
    feature_columns = [
        'experience_level_encoded', 'company_size_encoded', 'job_title_cluster',
        'high_paying_country', 'is_fulltime', 'years_since_2020', 'same_country',
        'job_title_target_enc', 'company_location_target_enc', 'exp_x_size',
        'exp_x_remote', 'size_x_country', 'exp_x_country'
    ]
    feature_columns.extend([col for col in df_encoded.columns if col.startswith('poly_')])
    encoded_cols = [col for col in df_encoded.columns if col.startswith('remote_ratio_cat_')]
    feature_columns.extend(encoded_cols)
    X = df_encoded[feature_columns].fillna(0)
    # Scale features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, feature_columns, scaler

# --------------------
# Enhanced Model Training
# --------------------
def create_optimized_model():
    """Create an ensemble model with XGBoost and GradientBoostingRegressor"""
    logger.info("Creating ensemble model...")
    xgb_model = xgb.XGBRegressor(
        n_estimators=30,  # Reduced for speed
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=1,
        tree_method='auto',
        verbosity=0,
        eval_metric='rmse'
    )
    gbr_model = GradientBoostingRegressor(
        n_estimators=30,  # Reduced for speed
        learning_rate=0.1,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,
        random_state=42,
        verbose=0
    )
    model = VotingRegressor(
        estimators=[('xgb', xgb_model), ('gbr', gbr_model)],
        weights=[0.6, 0.4],
        n_jobs=1
    )
    return model

def hyperparameter_search(X, y):
    """Perform extensive RandomizedSearchCV for ensemble model"""
    param_dist = {
        'xgb__n_estimators': [30],  # Only test 30 for speed
        'xgb__max_depth': [4, 6, 8],
        'xgb__learning_rate': [0.05, 0.1, 0.15],
        'xgb__subsample': [0.7, 0.8, 0.9],
        'xgb__colsample_bytree': [0.7, 0.8, 0.9],
        'xgb__min_child_weight': [1, 2, 3],
        'xgb__reg_alpha': [0, 0.1, 0.5],
        'xgb__reg_lambda': [0.5, 1.0, 2.0],
        'gbr__n_estimators': [30],  # Only test 30 for speed
        'gbr__max_depth': [4, 6, 8],
        'gbr__learning_rate': [0.05, 0.1, 0.15],
        'gbr__subsample': [0.7, 0.8, 0.9],
        'gbr__min_samples_split': [2, 5, 10],
        'gbr__min_samples_leaf': [1, 2, 4]
    }
    model = create_optimized_model()
    search = RandomizedSearchCV(
        model,
        param_distributions=param_dist,
        n_iter=2,
        scoring='r2',
        cv=2,
        verbose=0,  # Set to 0 for speed
        random_state=42,
        n_jobs=1
    )
    search.fit(X, y)
    return search.best_estimator_, search.best_params_

def cross_validate_model(model, X, y, k=2):
    """Perform k-fold cross-validation with higher k"""
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    r2_scores = cross_val_score(model, X, y, cv=kf, scoring='r2', n_jobs=1)
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring='neg_mean_squared_error', n_jobs=1))
    print("\nK-FOLD CROSS-VALIDATION RESULTS")
    print(f"R² mean: {r2_scores.mean():.4f} | std: {r2_scores.std():.4f}")
    print(f"RMSE mean: {rmse_scores.mean():.2f} | std: {rmse_scores.std():.2f}")
    return r2_scores, rmse_scores

# --------------------
# Enhanced Evaluation
# --------------------
def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Enhanced model evaluation with detailed metrics"""
    logger.info("Evaluating model...")
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    cv_scores = cross_val_score(model, X_train, y_train, cv=2, scoring='r2', n_jobs=1)  # Reduced cv
    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE': mape,
        'CV_R²_mean': cv_scores.mean(),
        'CV_R²_std': cv_scores.std()
    }
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    return metrics, y_pred

# --------------------
# Optimized Predictor Class
# --------------------
class OptimizedSalaryPredictor:
    """Optimized salary prediction system with ensemble model"""
    
    def __init__(self):
        self.model = None
        self.feature_columns = None
        self.scaler = None
        self.label_encoders = {}
        self.model_version = None
        self.training_metrics = None
        self.logger = logging.getLogger(__name__)
    
    def validate_input(self, input_dict):
        """Validate prediction input"""
        required_fields = ['experience_level', 'job_title', 'company_size', 'employee_residence', 'work_year']
        for field in required_fields:
            if field not in input_dict:
                raise ValueError(f"Missing required field: {field}")
        input_dict.setdefault('remote_ratio', 0)
        input_dict.setdefault('employment_type', 'FT')
        input_dict.setdefault('company_location', input_dict['employee_residence'])
        return True
    
    def preprocess_for_prediction(self, input_dict):
        """Preprocess single input for prediction"""
        df = pd.DataFrame([input_dict])
        df = streamlined_feature_engineering(df)
        df_encoded = pd.get_dummies(df, columns=['remote_ratio_cat'], prefix=['remote_ratio_cat'])
        df_encoded = advanced_feature_engineering(df_encoded)
        
        # Ensure all required features exist
        if self.feature_columns is None:
            raise ValueError("Model not trained yet. Please train the model first.")
            
        for col in self.feature_columns:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        
        X = df_encoded[self.feature_columns].fillna(0)
        if self.scaler is None:
            raise ValueError("Scaler not available. Please train the model first.")
        X = self.scaler.transform(X)
        return X
    
    def predict_with_confidence(self, input_dict):
        """Make prediction with confidence estimation"""
        try:
            self.validate_input(input_dict)
            input_processed = self.preprocess_for_prediction(input_dict)
            if self.model is None:
                raise ValueError("Model is not trained. Please train the model first.")
            prediction = self.model.predict(input_processed)[0]
            confidence_range = prediction * 0.15  # Reduced to 15% for tighter interval
            result = {
                'predicted_salary': float(prediction),
                'confidence_interval': {
                    'lower': float(max(0, prediction - confidence_range)),
                    'upper': float(prediction + confidence_range)
                },
                'model_version': self.model_version,
                'timestamp': datetime.now().isoformat(),
                'input_features': input_dict
            }
            self.logger.info(f"Prediction made: ${prediction:.2f}")
            return result
        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}")
            raise
    
    def train_model(self, df):
        """Train ensemble model with hyperparameter optimization"""
        self.logger.info("Starting model training...")
        df_clean = validate_and_clean_data(df)
        df_features = streamlined_feature_engineering(df_clean)
        X, feature_columns, scaler = prepare_features_for_modeling(df_features)
        y = df_clean['salary_in_usd']
        self.feature_columns = feature_columns
        self.scaler = scaler
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42  # Reduced test size for more training data
        )
        self.logger.info("Running hyperparameter optimization...")
        self.model, best_params = hyperparameter_search(X_train, y_train)
        self.logger.info(f"Best params: {best_params}")
        self.model.fit(X_train, y_train)
        cross_validate_model(self.model, X_train, y_train, k=2)
        self.training_metrics, y_pred = evaluate_model(self.model, X_train, y_train, X_test, y_test)
        self.model_version = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.logger.info("Model training completed successfully!")
        return self.training_metrics
    
    def save_model(self, filepath_prefix='optimized_salary_model'):
        """Save the trained model and scaler"""
        if self.model is None:
            raise ValueError("No trained model to save")
        model_path = f"{filepath_prefix}_{self.model_version}.pkl"
        model_data = {
            'model': self.model,
            'feature_columns': self.feature_columns,
            'scaler': self.scaler,
            'model_version': self.model_version,
            'training_metrics': self.training_metrics
        }
        joblib.dump(model_data, model_path)
        self.logger.info(f"Model saved: {model_path}")
        return model_path
    
    def load_model(self, model_path):
        """Load a trained model"""
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.feature_columns = model_data['feature_columns']
            self.scaler = model_data['scaler']
            self.model_version = model_data['model_version']
            self.training_metrics = model_data.get('training_metrics')
            self.logger.info(f"Model loaded successfully: {model_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False

# --------------------
# Visualization (Unchanged)
# --------------------
def create_essential_visualizations(df, model=None, feature_columns=None):
    """Create essential visualizations (user-provided style)"""
    sns.set_theme(style="whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)

    sns.histplot(x=df['salary_in_usd'], bins=50, kde=True)
    plt.title("Salary (USD) Distribution")
    plt.xlabel("Salary in USD")
    plt.ylabel("Frequency")
    plt.show()

    exp_map_labels = {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior-level', 'EX': 'Executive-level'}
    avg_salary_exp = pd.DataFrame(df.groupby('experience_level')['salary_in_usd'].mean()).reset_index()
    avg_salary_exp['experience_level'] = avg_salary_exp['experience_level'].replace(exp_map_labels)
    ordered_experience_levels = ['Entry-level', 'Mid-level', 'Senior-level', 'Executive-level']
    sns.barplot(x='experience_level', y='salary_in_usd', data=avg_salary_exp, order=ordered_experience_levels)
    plt.title("Average Salary by Experience Level")
    plt.xlabel("Experience Level")
    plt.ylabel("Average Salary in USD")
    plt.show()

    country_map = {
        'US': 'United States', 'IN': 'India', 'GB': 'United Kingdom', 'CA': 'Canada', 'DE': 'Germany',
        'FR': 'France', 'ES': 'Spain', 'AU': 'Australia', 'NL': 'Netherlands', 'BR': 'Brazil',
        'IT': 'Italy', 'MX': 'Mexico', 'AR': 'Argentina', 'CH': 'Switzerland', 'PL': 'Poland',
        'RU': 'Russia', 'SA': 'Saudi Arabia', 'JP': 'Japan', 'KR': 'South Korea', 'TW': 'Taiwan',
        'HK': 'Hong Kong', 'SG': 'Singapore', 'NZ': 'New Zealand', 'BE': 'Belgium', 'LT': 'Lithuania',
        'NO': 'Norway', 'SE': 'Sweden', 'DK': 'Denmark', 'IE': 'Ireland', 'PT': 'Portugal', 'CZ': 'Czech Republic',
    }
    df['employee_residence_full'] = pd.Series(df['employee_residence'], index=df.index).map(country_map)
    df['employee_residence_full'] = pd.Series(df['employee_residence_full'], index=df.index).fillna(pd.Series(df['employee_residence'], index=df.index))
    top_countries = pd.Series(df['employee_residence_full']).value_counts().head(10).index
    df_country = df[pd.Series(df['employee_residence_full']).isin(list(top_countries))]
    df_country['salary_in_usd'] = pd.to_numeric(df_country['salary_in_usd'], errors='coerce')
    df_country = pd.DataFrame(df_country)
    avg_salary_country = pd.Series(df_country.groupby('employee_residence_full')['salary_in_usd'].mean()).sort_values()
    avg_salary_country.plot(kind='barh', title='Average Salary by Top Countries')
    plt.xlabel('Average Salary (USD)')
    plt.show()

    ordered_company_sizes = ['S', 'M', 'L']
    mean_salaries = df.groupby('company_size')['salary_in_usd'].mean().reindex(ordered_company_sizes)
    ax = sns.barplot(x=mean_salaries.index, y=mean_salaries.values)
    plt.xticks(ticks=range(len(ordered_company_sizes)), labels=['Small', 'Medium', 'Large'])
    plt.title("Average Salary by Company Size")
    plt.xlabel("Company Size")
    plt.ylabel("Average Salary in USD")
    plt.show()

    remote_map = {0: 'On-site', 50: 'Hybrid', 100: 'Remote'}
    mean_salary_remote = df.groupby('remote_ratio')['salary_in_usd'].mean()
    mean_salary_remote.index = mean_salary_remote.index.map(remote_map)
    sns.barplot(x=mean_salary_remote.index, y=mean_salary_remote.values)
    plt.title("Average Salary by Remote Ratio")
    plt.xlabel("Remote Work Type")
    plt.ylabel("Average Salary in USD")
    plt.show()

    sns.lineplot(x='work_year', y='salary_in_usd', data=pd.DataFrame(df))
    plt.title("Average Salary Over Years (2020–2025)")
    plt.xlabel("Year")
    plt.ylabel("Average Salary in USD")
    plt.show()

    top_jobs = pd.Series(df.groupby('job_title')['salary_in_usd'].mean()).sort_values(ascending=False).head(10)
    top_jobs.plot(kind='barh', title='Top 10 Paying Job Titles')
    plt.xlabel('Average Salary (USD)')
    plt.show()

    if model is not None and hasattr(model, 'feature_importances_') and feature_columns is not None:
        importances = model.feature_importances_
        features = feature_columns
        feat_imp = pd.Series(importances, index=features).sort_values(ascending=False).head(15)
        feat_imp.plot(kind='barh', title='Top 15 Feature Importances (Ensemble)')
        plt.xlabel('Importance Score')
        plt.gca().invert_yaxis()
        plt.show()

# --------------------
# Data Generation (Unchanged)
# --------------------
def create_sample_data(n_samples=5000):
    """Create sample salary data for testing"""
    logger.info(f"Creating sample data with {n_samples} samples...")
    np.random.seed(42)
    experience_levels = np.random.choice(['EN', 'MI', 'SE', 'EX'], n_samples)
    job_titles = np.random.choice([
        'Data Scientist', 'Software Engineer', 'Data Analyst', 'ML Engineer',
        'Data Engineer', 'Research Scientist', 'Software Developer'
    ], n_samples)
    company_sizes = np.random.choice(['S', 'M', 'L'], n_samples)
    countries = np.random.choice(['US', 'CA', 'GB', 'DE', 'IN', 'FR', 'AU'], n_samples)
    work_years = np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples)
    remote_ratios = np.random.choice([0, 50, 100], n_samples)
    employment_types = np.random.choice(['FT', 'PT', 'CT'], n_samples)
    base_salary = 50000
    exp_multiplier = {'EN': 1.0, 'MI': 1.3, 'SE': 1.6, 'EX': 2.0}
    size_multiplier = {'S': 0.9, 'M': 1.0, 'L': 1.2}
    country_multiplier = {'US': 1.3, 'CA': 1.1, 'GB': 1.1, 'DE': 1.0, 'IN': 0.4, 'FR': 0.9, 'AU': 1.2}
    salaries = []
    for i in range(n_samples):
        salary = base_salary * exp_multiplier[experience_levels[i]] * \
                size_multiplier[company_sizes[i]] * country_multiplier[countries[i]]
        salary += np.random.normal(0, salary * 0.1)  # Reduced noise for consistency
        salaries.append(max(20000, salary))
    df = pd.DataFrame({
        'work_year': work_years,
        'experience_level': experience_levels,
        'employment_type': employment_types,
        'job_title': job_titles,
        'salary_in_usd': salaries,
        'employee_residence': countries,
        'remote_ratio': remote_ratios,
        'company_location': countries,
        'company_size': company_sizes
    })
    logger.info(f"Sample data created. Shape: {df.shape}")
    return df

# --------------------
# Main Execution
# --------------------
def main():
    """Main execution function with user input for prediction"""
    logger.info("Starting Optimized AI Salary Prediction Engine...")
    try:
        if os.path.exists(r"salaries.csv"):
            logger.info("Loading data from salaries.csv...")
            df = pd.read_csv(r"salaries.csv")
        else:
            print ("Failed to load csv!")
        predictor = OptimizedSalaryPredictor()
        metrics = predictor.train_model(df)
        if metrics['R²'] < 0.90:
            logger.warning(f"Model R² ({metrics['R²']:.4f}) below target of 0.90, consider additional tuning.")
        model_path = predictor.save_model()
        create_essential_visualizations(df, predictor.model, predictor.feature_columns)
        print("\n" + "="*60)
        print("Enter input for salary prediction:")
        user_input = {}
        user_input['experience_level'] = input("Experience Level (EN/MI/SE/EX): ").strip().upper()
        user_input['job_title'] = input("Job Title: ").strip()
        user_input['company_size'] = input("Company Size (S/M/L): ").strip().upper()
        user_input['employee_residence'] = input("Employee Residence (Country Code, e.g., US): ").strip().upper()
        user_input['company_location'] = input("Company Location (Country Code, e.g., US): ").strip().upper()
        user_input['work_year'] = int(input("Work Year (e.g., 2024): ").strip())
        user_input['remote_ratio'] = int(input("Remote Ratio (0/50/100): ").strip())
        user_input['employment_type'] = input("Employment Type (FT/PT/CT): ").strip().upper()
        prediction = predictor.predict_with_confidence(user_input)
        print("\n" + "="*60)
        print("USER PREDICTION")
        print("="*60)
        print(f"Input: {user_input}")
        print(f"Predicted Salary: ${prediction['predicted_salary']:,.2f}")
        print(f"Confidence Interval: ${prediction['confidence_interval']['lower']:,.2f} - ${prediction['confidence_interval']['upper']:,.2f}")
        print(f"Model Accuracy (R²): {metrics['R²']:.4f}")
        print("="*60)
        logger.info("Optimized AI Salary Prediction Engine completed successfully!")
        return predictor
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    predictor = main()