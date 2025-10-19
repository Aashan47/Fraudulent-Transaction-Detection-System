#!/usr/bin/env python3
"""
Kaggle Fraud Detection Model Training
===================================

Simple training script for fraud detection using the Kaggle Credit Card dataset.
Trains a Random Forest model and saves it for use with the Streamlit dashboard.

Usage: python train.py
"""

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, classification_report
import kagglehub
import os
import warnings
warnings.filterwarnings('ignore')

def load_kaggle_dataset():
    """Load the Kaggle credit card fraud dataset."""
    
    print("📊 Loading Kaggle Credit Card Fraud Dataset...")
    
    try:
        # Download the dataset
        dataset_path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        csv_path = os.path.join(dataset_path, "creditcard.csv")
        df = pd.read_csv(csv_path)
        
        print(f"✅ Dataset loaded successfully!")
        print(f"   Total transactions: {len(df):,}")
        print(f"   Fraud cases: {df['Class'].sum():,}")
        print(f"   Fraud rate: {df['Class'].mean():.4%}")
        
        return df
        
    except Exception as e:
        print(f"❌ Failed to load Kaggle dataset: {e}")
        print("💡 Make sure you have kagglehub installed: pip install kagglehub")
        raise

def train_fraud_model(df, sample_size=None):
    """Train fraud detection model."""
    
    print(f"\n🤖 Training Fraud Detection Model")
    print("=" * 50)
    
    # Use subset for faster training or full dataset
    if sample_size and len(df) > sample_size:
        df_sample = df.sample(n=sample_size, random_state=42)
        print(f"⚡ Using {sample_size:,} samples for training (subset for speed)")
    else:
        df_sample = df
        print(f"📊 Using full dataset: {len(df_sample):,} samples")
    
    # Separate features and target
    X = df_sample.drop('Class', axis=1)
    y = df_sample['Class']
    
    print(f"   Features: {X.shape[1]}")
    print(f"   Fraud rate: {y.mean():.4%}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Preprocessing
    print("\n🔧 Preprocessing data...")
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_processed = imputer.fit_transform(X_train)
    X_test_processed = imputer.transform(X_test)
    
    X_train_scaled = scaler.fit_transform(X_train_processed)
    X_test_scaled = scaler.transform(X_test_processed)
    
    print("   ✅ Data preprocessing complete")
    
    # Train model
    print("\n🎯 Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    print("   ✅ Model training complete")
    
    # Evaluate model
    print("\n📊 Evaluating model performance...")
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate metrics
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"   🎯 Precision: {precision:.3f}")
    print(f"   🎯 Recall: {recall:.3f}")
    print(f"   🎯 F1-Score: {f1:.3f}")
    print(f"   🎯 ROC-AUC: {roc_auc:.3f}")
    
    # Detailed classification report
    print(f"\n📈 Detailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
    
    # Create model package for saving
    model_package = {
        'model': model,
        'scaler': scaler,
        'imputer': imputer,
        'feature_names': list(X.columns),
        'performance': {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        },
        'training_info': {
            'training_date': datetime.now().isoformat(),
            'dataset_size': len(df_sample),
            'fraud_rate': y.mean(),
            'test_size': len(X_test)
        }
    }
    
    return model_package

def save_model(model_package, filename="fraud_model.pkl"):
    """Save the trained model."""
    
    print(f"\n💾 Saving model...")
    joblib.dump(model_package, filename)
    
    # Save training summary
    summary_file = "training_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("KAGGLE FRAUD DETECTION MODEL - TRAINING SUMMARY\n")
        f.write("=" * 55 + "\n\n")
        
        f.write(f"Training Date: {model_package['training_info']['training_date']}\n")
        f.write(f"Dataset Size: {model_package['training_info']['dataset_size']:,} transactions\n")
        f.write(f"Fraud Rate: {model_package['training_info']['fraud_rate']:.4%}\n")
        f.write(f"Features: {len(model_package['feature_names'])}\n\n")
        
        f.write("MODEL PERFORMANCE:\n")
        perf = model_package['performance']
        f.write(f"  Precision: {perf['precision']:.3f}\n")
        f.write(f"  Recall: {perf['recall']:.3f}\n")
        f.write(f"  F1-Score: {perf['f1_score']:.3f}\n")
        f.write(f"  ROC-AUC: {perf['roc_auc']:.3f}\n\n")
        
        f.write("NEXT STEPS:\n")
        f.write("1. Run Streamlit dashboard: streamlit run dashboard.py\n")
        f.write("2. Model file ready for deployment: fraud_model.pkl\n")
    
    print(f"   ✅ Model saved to: {filename}")
    print(f"   ✅ Summary saved to: {summary_file}")

def main():
    """Main training function."""
    
    print("🛡️ KAGGLE FRAUD DETECTION MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Load Kaggle dataset
        df = load_kaggle_dataset()
        
        # Train model
        model_package = train_fraud_model(df)
        
        # Save model
        save_model(model_package)
        
        print("\n" + "=" * 60)
        print("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("🚀 Next steps:")
        print("   1. Run dashboard: streamlit run dashboard.py")
        print("   2. Open browser: http://localhost:8501")
        print("   3. Start fraud detection monitoring!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        print("💡 Make sure kagglehub is installed: pip install kagglehub")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)