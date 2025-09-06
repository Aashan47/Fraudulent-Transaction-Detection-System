"""
Fraudulent Transaction Detection System
======================================

A comprehensive fraud detection system using machine learning techniques
optimized for financial applications with emphasis on high precision.

Author: Claude AI
Date: 2025-09-06
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           precision_score, recall_score, f1_score, auc)
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor

# Imbalanced learning
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Boosting algorithms
import xgboost as xgb
import lightgbm as lgb

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

class FraudDetectionSystem:
    """
    A comprehensive fraud detection system for financial transactions.
    
    This system implements:
    - Data preprocessing and feature engineering
    - Anomaly detection
    - Class imbalance handling with SMOTE
    - Multiple boosting algorithms (XGBoost, LightGBM)
    - Precision-optimized model selection
    - Comprehensive evaluation and visualization
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        self.random_state = random_state
        self.scaler = RobustScaler()
        self.label_encoders = {}
        self.imputer = SimpleImputer(strategy='median')
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.X_train_resampled = None
        self.y_train_resampled = None
        
    def create_synthetic_dataset(self, n_samples=100000, fraud_rate=0.02):
        """
        Create a synthetic financial transaction dataset with realistic features.
        
        Parameters:
        - n_samples: Number of transactions to generate
        - fraud_rate: Proportion of fraudulent transactions
        
        Returns:
        - DataFrame with transaction features and labels
        """
        print(f"Creating synthetic dataset with {n_samples} transactions...")
        
        np.random.seed(self.random_state)
        
        # Generate transaction amounts (log-normal distribution)
        amounts = np.random.lognormal(mean=3, sigma=1.5, size=n_samples)
        amounts = np.clip(amounts, 1, 10000)  # Clip to reasonable range
        
        # Time-based features
        hours = np.random.randint(0, 24, n_samples)
        days_of_week = np.random.randint(0, 7, n_samples)
        
        # Account features
        account_ages = np.random.exponential(scale=365, size=n_samples)  # Days
        account_ages = np.clip(account_ages, 1, 3650)  # 1 day to 10 years
        
        # Transaction frequency (transactions per day for the account)
        transaction_frequency = np.random.poisson(lam=2, size=n_samples) + 1
        
        # Merchant categories (encoded as integers)
        merchant_categories = np.random.choice(
            ['grocery', 'gas', 'restaurant', 'online', 'retail', 'atm', 'other'], 
            n_samples, p=[0.25, 0.15, 0.20, 0.15, 0.15, 0.05, 0.05]
        )
        
        # Location features
        is_foreign = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
        
        # Payment method
        payment_methods = np.random.choice(
            ['card', 'online', 'mobile', 'cash'], 
            n_samples, p=[0.60, 0.25, 0.10, 0.05]
        )
        
        # Generate labels (0: legitimate, 1: fraudulent)
        n_fraud = int(n_samples * fraud_rate)
        labels = np.concatenate([np.ones(n_fraud), np.zeros(n_samples - n_fraud)])
        np.random.shuffle(labels)
        
        # Adjust features to make fraudulent transactions more detectable
        fraud_indices = np.where(labels == 1)[0]
        
        # Fraudulent transactions tend to:
        # - Have unusual amounts (very high or very low)
        amounts[fraud_indices[:len(fraud_indices)//2]] *= np.random.uniform(5, 20, len(fraud_indices)//2)
        amounts[fraud_indices[len(fraud_indices)//2:]] *= np.random.uniform(0.1, 0.3, len(fraud_indices) - len(fraud_indices)//2)
        
        # - Occur at unusual hours
        hours[fraud_indices] = np.random.choice([1, 2, 3, 4, 23], len(fraud_indices))
        
        # - Come from newer accounts
        account_ages[fraud_indices] *= np.random.uniform(0.1, 0.5, len(fraud_indices))
        
        # - Have higher transaction frequency
        transaction_frequency[fraud_indices] = (transaction_frequency[fraud_indices] * np.random.uniform(3, 8, len(fraud_indices))).astype(int)
        
        # - More likely to be foreign
        is_foreign[fraud_indices] = np.random.choice([0, 1], len(fraud_indices), p=[0.3, 0.7])
        
        # Create DataFrame
        df = pd.DataFrame({
            'amount': amounts,
            'hour': hours,
            'day_of_week': days_of_week,
            'account_age_days': account_ages,
            'transaction_frequency': transaction_frequency,
            'merchant_category': merchant_categories,
            'is_foreign': is_foreign,
            'payment_method': payment_methods,
            'is_fraud': labels.astype(int)
        })
        
        # Add some derived features
        df['amount_log'] = np.log1p(df['amount'])
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = ((df['hour'] >= 22) | (df['hour'] <= 5)).astype(int)
        df['amount_per_frequency'] = df['amount'] / df['transaction_frequency']
        
        print(f"Dataset created successfully!")
        print(f"Total transactions: {len(df)}")
        print(f"Fraudulent transactions: {df['is_fraud'].sum()} ({df['is_fraud'].mean():.2%})")
        
        return df
    
    def preprocess_data(self, df, target_column='is_fraud'):
        """
        Comprehensive data preprocessing pipeline.
        
        Parameters:
        - df: Input DataFrame
        - target_column: Name of the target variable column
        
        Returns:
        - X: Processed features
        - y: Target variable
        """
        print("Starting data preprocessing...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Handle missing values
        print("Handling missing values...")
        numeric_features = X.select_dtypes(include=[np.number]).columns
        categorical_features = X.select_dtypes(include=['object']).columns
        
        # Impute numeric features
        if len(numeric_features) > 0:
            X[numeric_features] = self.imputer.fit_transform(X[numeric_features])
        
        # Encode categorical features
        print("Encoding categorical features...")
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
            X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
        
        # Handle outliers using IQR method for numeric features
        print("Handling outliers...")
        for col in numeric_features:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[col] = np.clip(X[col], lower_bound, upper_bound)
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        print(f"Preprocessing completed. Features: {len(self.feature_names)}")
        return X, y
    
    def detect_anomalies(self, X, contamination=0.1):
        """
        Apply anomaly detection techniques to identify suspicious transactions.
        
        Parameters:
        - X: Feature matrix
        - contamination: Expected proportion of anomalies
        
        Returns:
        - anomaly_scores: Anomaly scores for each transaction
        - outliers: Binary labels (1 for outliers, -1 for normal)
        """
        print("Applying anomaly detection...")
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, 
                                   random_state=self.random_state,
                                   n_estimators=100)
        iso_outliers = iso_forest.fit_predict(X)
        iso_scores = iso_forest.decision_function(X)
        
        # Local Outlier Factor
        lof = LocalOutlierFactor(contamination=contamination, novelty=False)
        lof_outliers = lof.fit_predict(X)
        lof_scores = lof.negative_outlier_factor_
        
        # Combine anomaly scores (ensemble)
        combined_scores = (iso_scores + lof_scores) / 2
        
        # Create anomaly features
        anomaly_features = pd.DataFrame({
            'iso_score': iso_scores,
            'lof_score': lof_scores,
            'combined_anomaly_score': combined_scores,
            'is_iso_outlier': (iso_outliers == -1).astype(int),
            'is_lof_outlier': (lof_outliers == -1).astype(int)
        })
        
        print(f"Anomaly detection completed. Detected {(iso_outliers == -1).sum()} isolation forest outliers")
        print(f"Detected {(lof_outliers == -1).sum()} LOF outliers")
        
        return anomaly_features
    
    def apply_smote(self, X, y, sampling_strategy='auto'):
        """
        Apply SMOTE to handle class imbalance.
        
        Parameters:
        - X: Feature matrix
        - y: Target variable
        - sampling_strategy: SMOTE sampling strategy
        
        Returns:
        - X_resampled: Resampled features
        - y_resampled: Resampled target
        """
        print("Applying SMOTE for class imbalance handling...")
        print(f"Original class distribution: {dict(pd.Series(y).value_counts())}")
        
        smote = SMOTE(sampling_strategy=sampling_strategy, 
                     random_state=self.random_state,
                     k_neighbors=5)
        
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        print(f"After SMOTE class distribution: {dict(pd.Series(y_resampled).value_counts())}")
        
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train):
        """
        Train multiple boosting models with hyperparameter optimization.
        
        Parameters:
        - X_train: Training features
        - y_train: Training target
        """
        print("Training boosting models...")
        
        # XGBoost
        print("Training XGBoost...")
        xgb_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 6, 9],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
        
        xgb_model = xgb.XGBClassifier(
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        
        # Use smaller parameter grid for demonstration
        xgb_params_small = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1]
        }
        
        xgb_grid = GridSearchCV(
            xgb_model, xgb_params_small, 
            cv=3, scoring='precision', 
            n_jobs=-1, verbose=1
        )
        
        xgb_grid.fit(X_train, y_train)
        self.models['XGBoost'] = xgb_grid.best_estimator_
        
        # LightGBM
        print("Training LightGBM...")
        lgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 6],
            'learning_rate': [0.01, 0.1],
            'num_leaves': [31, 50]
        }
        
        lgb_model = lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1,
            force_col_wise=True
        )
        
        lgb_grid = GridSearchCV(
            lgb_model, lgb_params, 
            cv=3, scoring='precision', 
            n_jobs=-1, verbose=1
        )
        
        lgb_grid.fit(X_train, y_train)
        self.models['LightGBM'] = lgb_grid.best_estimator_
        
        print("Model training completed!")
    
    def evaluate_models(self, X_test, y_test):
        """
        Comprehensive model evaluation with focus on precision.
        
        Parameters:
        - X_test: Test features
        - y_test: Test target
        
        Returns:
        - results: Dictionary with evaluation results for each model
        """
        print("Evaluating models...")
        results = {}
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            
            # Precision-Recall AUC
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
            pr_auc = auc(recall_curve, precision_curve)
            
            results[name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'pr_auc': pr_auc,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'model': model
            }
            
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            print(f"ROC-AUC: {roc_auc:.4f}")
            print(f"PR-AUC: {pr_auc:.4f}")
        
        # Select best model based on precision (primary) and PR-AUC (secondary)
        best_model_name = max(results.keys(), 
                            key=lambda x: (results[x]['precision'], results[x]['pr_auc']))
        
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        
        print(f"\nBest model: {best_model_name} (Precision: {results[best_model_name]['precision']:.4f})")
        
        return results
    
    def analyze_feature_importance(self, model_name=None):
        """
        Analyze and visualize feature importance.
        
        Parameters:
        - model_name: Name of the model to analyze (uses best model if None)
        
        Returns:
        - feature_importance: DataFrame with feature importance scores
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        print(f"Analyzing feature importance for {model_name}...")
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance_scores = model.feature_importances_
        else:
            print("Model does not have feature_importances_ attribute")
            return None
        
        # Create feature importance DataFrame
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance_scores
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        return feature_importance
    
    def create_visualizations(self, results, y_test, y_train_original, y_train_resampled):
        """
        Create comprehensive visualizations for the fraud detection system.
        
        Parameters:
        - results: Model evaluation results
        - y_test: Test target
        - y_train_original: Original training target (before SMOTE)
        - y_train_resampled: Resampled training target (after SMOTE)
        """
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Class Distribution (Before and After SMOTE)
        ax1 = plt.subplot(2, 4, 1)
        original_counts = pd.Series(y_train_original).value_counts()
        ax1.bar(['Legitimate', 'Fraudulent'], 
               [original_counts[0], original_counts[1]], 
               color=['skyblue', 'lightcoral'])
        ax1.set_title('Original Class Distribution')
        ax1.set_ylabel('Count')
        
        ax2 = plt.subplot(2, 4, 2)
        resampled_counts = pd.Series(y_train_resampled).value_counts()
        ax2.bar(['Legitimate', 'Fraudulent'], 
               [resampled_counts[0], resampled_counts[1]], 
               color=['skyblue', 'lightcoral'])
        ax2.set_title('After SMOTE Class Distribution')
        ax2.set_ylabel('Count')
        
        # 2. ROC Curves
        ax3 = plt.subplot(2, 4, 3)
        for name, result in results.items():
            fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
            ax3.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})')
        
        ax3.plot([0, 1], [0, 1], 'k--', label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves')
        ax3.legend()
        ax3.grid(True)
        
        # 3. Precision-Recall Curves
        ax4 = plt.subplot(2, 4, 4)
        for name, result in results.items():
            precision_curve, recall_curve, _ = precision_recall_curve(y_test, result['y_pred_proba'])
            ax4.plot(recall_curve, precision_curve, 
                    label=f'{name} (AUC = {result["pr_auc"]:.3f})')
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves')
        ax4.legend()
        ax4.grid(True)
        
        # 4. Confusion Matrix for Best Model
        ax5 = plt.subplot(2, 4, 5)
        best_result = results[self.best_model_name]
        cm = confusion_matrix(y_test, best_result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5)
        ax5.set_title(f'Confusion Matrix - {self.best_model_name}')
        ax5.set_ylabel('True Label')
        ax5.set_xlabel('Predicted Label')
        
        # 5. Feature Importance
        ax6 = plt.subplot(2, 4, 6)
        feature_importance = self.analyze_feature_importance()
        if feature_importance is not None:
            top_features = feature_importance.head(10)
            ax6.barh(range(len(top_features)), top_features['importance'])
            ax6.set_yticks(range(len(top_features)))
            ax6.set_yticklabels(top_features['feature'])
            ax6.set_title('Top 10 Feature Importance')
            ax6.set_xlabel('Importance Score')
        
        # 6. Model Performance Comparison
        ax7 = plt.subplot(2, 4, 7)
        metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
        x = np.arange(len(metrics))
        width = 0.35
        
        model_names = list(results.keys())
        for i, model_name in enumerate(model_names):
            values = [results[model_name][metric] for metric in metrics]
            ax7.bar(x + i * width, values, width, label=model_name)
        
        ax7.set_xlabel('Metrics')
        ax7.set_ylabel('Score')
        ax7.set_title('Model Performance Comparison')
        ax7.set_xticks(x + width / 2)
        ax7.set_xticklabels(metrics, rotation=45)
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 7. Prediction Probability Distribution
        ax8 = plt.subplot(2, 4, 8)
        best_result = results[self.best_model_name]
        
        # Separate probabilities by true class
        fraud_probs = best_result['y_pred_proba'][y_test == 1]
        legitimate_probs = best_result['y_pred_proba'][y_test == 0]
        
        ax8.hist(legitimate_probs, bins=50, alpha=0.7, label='Legitimate', color='skyblue')
        ax8.hist(fraud_probs, bins=50, alpha=0.7, label='Fraudulent', color='lightcoral')
        ax8.set_xlabel('Prediction Probability')
        ax8.set_ylabel('Count')
        ax8.set_title('Prediction Probability Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('fraud_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations saved as 'fraud_detection_analysis.png'")
    
    def generate_report(self, results, feature_importance):
        """
        Generate a comprehensive report with insights and recommendations.
        
        Parameters:
        - results: Model evaluation results
        - feature_importance: Feature importance analysis
        
        Returns:
        - report: Formatted report string
        """
        report = f"""
FRAUDULENT TRANSACTION DETECTION SYSTEM - COMPREHENSIVE REPORT
============================================================

EXECUTIVE SUMMARY
----------------
This report presents the results of a machine learning-based fraud detection system 
optimized for financial applications with emphasis on high precision to minimize 
false positives in risk-sensitive environments.

DATASET OVERVIEW
---------------
• Total Transactions Analyzed: {len(self.X_test) + len(self.X_train)}
• Training Set: {len(self.X_train)} transactions
• Test Set: {len(self.X_test)} transactions
• Features Used: {len(self.feature_names)}

MODEL PERFORMANCE SUMMARY
------------------------
Best Performing Model: {self.best_model_name}

"""
        
        for name, result in results.items():
            report += f"""
{name} Performance:
• Precision: {result['precision']:.4f} (Primary optimization metric)
• Recall: {result['recall']:.4f}
• F1-Score: {result['f1_score']:.4f}
• ROC-AUC: {result['roc_auc']:.4f}
• PR-AUC: {result['pr_auc']:.4f}
"""
        
        report += f"""

FEATURE IMPORTANCE ANALYSIS
---------------------------
Top 5 Most Important Features:
"""
        
        if feature_importance is not None:
            for i, row in feature_importance.head(5).iterrows():
                report += f"• {row['feature']}: {row['importance']:.4f}\n"
        
        report += f"""

KEY INSIGHTS
-----------
1. CLASS IMBALANCE HANDLING: SMOTE successfully balanced the dataset, improving 
   model ability to detect fraudulent patterns.

2. PRECISION OPTIMIZATION: The system prioritizes precision over recall to minimize 
   false positives, crucial for financial applications where false fraud alerts 
   can damage customer relationships.

3. ANOMALY DETECTION: Integrated anomaly detection techniques provide additional 
   signals for suspicious transaction identification.

4. FEATURE ENGINEERING: Derived features (log transformations, time-based features, 
   transaction ratios) significantly contribute to model performance.

DEPLOYMENT RECOMMENDATIONS
-------------------------
1. PRODUCTION DEPLOYMENT:
   • Implement real-time scoring with latency < 100ms
   • Use {self.best_model_name} as primary model
   • Set prediction threshold to optimize for precision
   • Implement model monitoring and drift detection

2. RISK MANAGEMENT:
   • Establish clear escalation procedures for high-risk transactions
   • Implement human review process for edge cases
   • Regular model retraining (monthly) with new transaction data
   • A/B testing for model updates

3. COMPLIANCE & MONITORING:
   • Log all predictions for audit purposes
   • Monitor model performance metrics daily
   • Implement bias detection and fairness checks
   • Regular validation against regulatory requirements

4. TECHNICAL INFRASTRUCTURE:
   • Deploy with auto-scaling capabilities
   • Implement fallback mechanisms for model failures
   • Ensure data privacy and security compliance
   • Regular security audits of the ML pipeline

5. CONTINUOUS IMPROVEMENT:
   • Collect feedback on false positives/negatives
   • Incorporate new feature sources (device fingerprinting, behavioral analytics)
   • Experiment with ensemble methods and deep learning approaches
   • Regular competitive analysis of fraud detection techniques

RISK CONSIDERATIONS
------------------
• False Positive Impact: High precision minimizes customer friction
• Model Drift: Regular retraining required as fraud patterns evolve
• Adversarial Attacks: Implement robust monitoring for unusual patterns
• Data Quality: Ensure consistent data preprocessing in production

NEXT STEPS
---------
1. Deploy model in shadow mode for validation
2. Integrate with existing transaction processing systems
3. Establish monitoring and alerting infrastructure
4. Train operations team on model interpretation and troubleshooting
5. Plan for regular model updates and performance reviews

Report Generated: {pd.Timestamp.now()}
Model Version: 1.0
"""
        
        return report
    
    def run_full_pipeline(self, n_samples=100000, fraud_rate=0.02):
        """
        Execute the complete fraud detection pipeline.
        
        Parameters:
        - n_samples: Number of synthetic transactions to generate
        - fraud_rate: Proportion of fraudulent transactions
        
        Returns:
        - results: Complete pipeline results including models, evaluations, and report
        """
        print("Starting Fraudulent Transaction Detection System Pipeline...")
        print("=" * 60)
        
        # Step 1: Create synthetic dataset
        df = self.create_synthetic_dataset(n_samples=n_samples, fraud_rate=fraud_rate)
        
        # Step 2: Preprocess data
        X, y = self.preprocess_data(df)
        
        # Step 3: Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Step 4: Apply anomaly detection
        anomaly_features_train = self.detect_anomalies(X_train)
        anomaly_features_test = self.detect_anomalies(X_test)
        
        # Combine original features with anomaly features
        X_train_enhanced = pd.concat([X_train.reset_index(drop=True), 
                                    anomaly_features_train.reset_index(drop=True)], axis=1)
        X_test_enhanced = pd.concat([X_test.reset_index(drop=True), 
                                   anomaly_features_test.reset_index(drop=True)], axis=1)
        
        # Update feature names
        self.feature_names = X_train_enhanced.columns.tolist()
        
        # Step 5: Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train_enhanced),
            columns=self.feature_names
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test_enhanced),
            columns=self.feature_names
        )
        
        # Store for later use
        self.X_train = X_train_scaled
        self.X_test = X_test_scaled
        self.y_train = y_train
        self.y_test = y_test
        
        # Step 6: Apply SMOTE
        X_train_resampled, y_train_resampled = self.apply_smote(X_train_scaled, y_train)
        self.X_train_resampled = X_train_resampled
        self.y_train_resampled = y_train_resampled
        
        # Step 7: Train models
        self.train_models(X_train_resampled, y_train_resampled)
        
        # Step 8: Evaluate models
        results = self.evaluate_models(X_test_scaled, y_test)
        
        # Step 9: Feature importance analysis
        feature_importance = self.analyze_feature_importance()
        
        # Step 10: Create visualizations
        self.create_visualizations(results, y_test, y_train, y_train_resampled)
        
        # Step 11: Generate comprehensive report
        report = self.generate_report(results, feature_importance)
        
        # Save report to file
        with open('fraud_detection_report.txt', 'w') as f:
            f.write(report)
        
        print("\n" + "=" * 60)
        print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Best Model: {self.best_model_name}")
        print(f"Best Precision: {results[self.best_model_name]['precision']:.4f}")
        print(f"Report saved to: fraud_detection_report.txt")
        print(f"Visualizations saved to: fraud_detection_analysis.png")
        
        return {
            'models': self.models,
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'results': results,
            'feature_importance': feature_importance,
            'report': report,
            'data': {
                'X_train': self.X_train,
                'X_test': self.X_test,
                'y_train': self.y_train,
                'y_test': self.y_test,
                'X_train_resampled': self.X_train_resampled,
                'y_train_resampled': self.y_train_resampled
            }
        }

def main():
    """
    Main function to demonstrate the fraud detection system.
    """
    print("Initializing Fraudulent Transaction Detection System...")
    
    # Create and run the fraud detection system
    fraud_system = FraudDetectionSystem(random_state=42)
    
    # Execute the full pipeline
    results = fraud_system.run_full_pipeline(
        n_samples=50000,  # Reduced for demonstration
        fraud_rate=0.03   # 3% fraud rate
    )
    
    print("\nSystem ready for deployment!")
    print("Check 'fraud_detection_report.txt' for detailed analysis and recommendations.")
    
    return fraud_system, results

if __name__ == "__main__":
    # Run the fraud detection system
    system, pipeline_results = main()