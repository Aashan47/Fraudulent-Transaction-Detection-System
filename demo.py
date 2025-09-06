#!/usr/bin/env python3
"""
Fraud Detection System Demonstration
===================================

This script demonstrates the key features of the Fraudulent Transaction Detection System.
Run this for a quick demonstration of the system's capabilities.

Usage: python demo.py
"""

from fraud_detection_system import FraudDetectionSystem
import pandas as pd
import numpy as np

def run_demonstration():
    """
    Run a comprehensive demonstration of the fraud detection system.
    """
    print("🔍 FRAUDULENT TRANSACTION DETECTION SYSTEM DEMO")
    print("=" * 60)
    
    # Initialize the system
    print("\n📊 Initializing Fraud Detection System...")
    fraud_system = FraudDetectionSystem(random_state=42)
    
    # Run the complete pipeline with demo parameters
    print("\n🚀 Running Complete Detection Pipeline...")
    print("   • Generating synthetic dataset (30,000 transactions)")
    print("   • Applying preprocessing and anomaly detection")
    print("   • Training XGBoost and LightGBM models")
    print("   • Evaluating with precision-focused metrics")
    
    results = fraud_system.run_full_pipeline(
        n_samples=30000,  # Smaller dataset for demo
        fraud_rate=0.025  # 2.5% fraud rate
    )
    
    # Display key results
    print("\n📈 DEMO RESULTS SUMMARY")
    print("-" * 40)
    
    best_model_name = results['best_model_name']
    best_results = results['results'][best_model_name]
    
    print(f"🥇 Best Model: {best_model_name}")
    print(f"   Precision: {best_results['precision']:.3f}")
    print(f"   Recall: {best_results['recall']:.3f}")
    print(f"   F1-Score: {best_results['f1_score']:.3f}")
    print(f"   ROC-AUC: {best_results['roc_auc']:.3f}")
    print(f"   PR-AUC: {best_results['pr_auc']:.3f}")
    
    # Show feature importance
    print(f"\n🔍 Top 5 Most Important Features:")
    feature_importance = results['feature_importance']
    if feature_importance is not None:
        for i, (_, row) in enumerate(feature_importance.head(5).iterrows()):
            print(f"   {i+1}. {row['feature']}: {row['importance']:.3f}")
    
    # Show model comparison
    print(f"\n⚖️  Model Comparison:")
    for model_name, model_results in results['results'].items():
        print(f"   {model_name}:")
        print(f"     Precision: {model_results['precision']:.3f}")
        print(f"     ROC-AUC: {model_results['roc_auc']:.3f}")
    
    # Demonstrate prediction on new data
    print(f"\n🎯 Sample Predictions on Test Data:")
    X_test = results['data']['X_test']
    y_test = results['data']['y_test']
    best_model = results['best_model']
    
    # Get predictions for first 10 test samples
    sample_predictions = best_model.predict_proba(X_test.head(10))[:, 1]
    sample_labels = y_test.head(10).values
    
    print("   Transaction | True Label | Fraud Probability")
    print("   " + "-" * 45)
    for i in range(10):
        label_text = "FRAUD" if sample_labels[i] == 1 else "LEGIT"
        prob = sample_predictions[i]
        risk_level = "HIGH" if prob > 0.5 else "LOW"
        print(f"   Transaction {i+1:2d} | {label_text:10s} | {prob:.3f} ({risk_level})")
    
    print(f"\n📁 Output Files Generated:")
    print("   • fraud_detection_analysis.png - Comprehensive visualizations")
    print("   • fraud_detection_report.txt - Detailed analysis report")
    
    print(f"\n✅ Demo completed successfully!")
    print("   Check the generated files for detailed analysis.")
    
    return fraud_system, results

def demonstrate_custom_usage():
    """
    Demonstrate custom usage of individual system components.
    """
    print("\n" + "=" * 60)
    print("🛠️  CUSTOM USAGE DEMONSTRATION")
    print("=" * 60)
    
    # Initialize system
    fraud_system = FraudDetectionSystem(random_state=42)
    
    # Step-by-step demonstration
    print("\n1️⃣  Creating Custom Synthetic Dataset...")
    df = fraud_system.create_synthetic_dataset(n_samples=10000, fraud_rate=0.02)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['is_fraud'].mean():.1%}")
    
    print("\n2️⃣  Data Preprocessing...")
    X, y = fraud_system.preprocess_data(df)
    print(f"   Features: {X.shape[1]}")
    print(f"   Feature names: {list(X.columns[:5])}...")
    
    print("\n3️⃣  Anomaly Detection...")
    anomaly_features = fraud_system.detect_anomalies(X, contamination=0.05)
    print(f"   Anomaly features added: {anomaly_features.shape[1]}")
    
    # Combine features
    X_enhanced = pd.concat([X.reset_index(drop=True), 
                           anomaly_features.reset_index(drop=True)], axis=1)
    
    print("\n4️⃣  Class Imbalance Handling...")
    X_resampled, y_resampled = fraud_system.apply_smote(X_enhanced, y)
    print(f"   Original samples: {len(X_enhanced)}")
    print(f"   After SMOTE: {len(X_resampled)}")
    
    print("\n✨ Custom usage demonstration completed!")

if __name__ == "__main__":
    # Run the main demonstration
    system, results = run_demonstration()
    
    # Show custom usage
    demonstrate_custom_usage()
    
    print("\n🎉 All demonstrations completed!")
    print("   The fraud detection system is ready for production use.")
    print("   See README.md for detailed usage instructions.")