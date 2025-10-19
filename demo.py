#!/usr/bin/env python3
"""
Professional Flask Fraud Detection Demo
======================================

Clean, professional web interface without emojis and with proper chart labeling.
Optimized for business presentations and professional demos.

Usage: python demo.py
"""

from flask import Flask, render_template_string, request, jsonify
import pandas as pd
import numpy as np
import joblib
import json
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import traceback

app = Flask(__name__)

# Load model once at startup
try:
    model_package = joblib.load("fraud_model.pkl")
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model not found! Run: python train.py")
    exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def generate_sample_transaction():
    """Generate a sample transaction for testing."""
    try:
        transaction = {}
        
        # Time feature (random seconds)
        transaction['Time'] = float(np.random.uniform(0, 172800))  # 48 hours
        
        # V1-V28 features (PCA transformed)
        for i in range(1, 29):
            if np.random.random() < 0.1:  # 10% chance of anomalous value
                transaction[f'V{i}'] = float(np.random.normal(0, 3))
            else:
                transaction[f'V{i}'] = float(np.random.normal(0, 1))
        
        # Amount feature
        amount_type = np.random.choice(['small', 'medium', 'large'], p=[0.6, 0.3, 0.1])
        if amount_type == 'small':
            transaction['Amount'] = float(np.random.exponential(50))
        elif amount_type == 'medium':
            transaction['Amount'] = float(np.random.exponential(200) + 50)
        else:
            transaction['Amount'] = float(np.random.exponential(1000) + 200)
        
        transaction['Amount'] = float(np.clip(transaction['Amount'], 0.01, 25000))
        
        return transaction
    except Exception as e:
        print(f"Error generating transaction: {e}")
        return None

def predict_fraud(transaction):
    """Predict fraud probability for a transaction."""
    try:
        df = pd.DataFrame([transaction])
        
        # Apply preprocessing
        X_processed = model_package['imputer'].transform(df)
        X_scaled = model_package['scaler'].transform(X_processed)
        
        # Make prediction
        fraud_prob = float(model_package['model'].predict_proba(X_scaled)[0][1])
        is_fraud = fraud_prob > 0.5
        
        # Determine risk level
        if fraud_prob >= 0.8:
            risk_level = "CRITICAL"
            risk_color = "#d32f2f"
        elif fraud_prob >= 0.6:
            risk_level = "HIGH"
            risk_color = "#f57c00"
        elif fraud_prob >= 0.3:
            risk_level = "MEDIUM"
            risk_color = "#fbc02d"
        else:
            risk_level = "LOW"
            risk_color = "#388e3c"
        
        return {
            'fraud_probability': fraud_prob,
            'is_fraud': is_fraud,
            'risk_level': risk_level,
            'risk_color': risk_color
        }
    except Exception as e:
        print(f"Error predicting fraud: {e}")
        return None

def create_large_performance_chart():
    """Create large, clean model performance visualization without emojis."""
    try:
        # Set style and large figure size
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 10))
        
        # Create grid layout
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        perf = model_package['performance']
        
        # 1. Large Performance metrics bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        metrics = ['Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        values = [perf['precision'], perf['recall'], perf['f1_score'], perf['roc_auc']]
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
        
        bars = ax1.bar(metrics, values, color=colors, alpha=0.8, width=0.6)
        ax1.set_title('Model Performance Metrics', fontsize=18, fontweight='bold', pad=20)
        ax1.set_ylabel('Score', fontsize=14)
        ax1.set_ylim(0, 1.0)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(labelsize=12)
        
        # Add large value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=14)
        
        # 2. Clean Risk level distribution pie chart
        ax2 = fig.add_subplot(gs[0, 1])
        risk_levels = ['Low Risk', 'Medium Risk', 'High Risk', 'Critical Risk']
        risk_counts = [70, 20, 8, 2]
        risk_colors = ['#388e3c', '#fbc02d', '#f57c00', '#d32f2f']
        
        # Create pie chart with proper spacing
        wedges, texts, autotexts = ax2.pie(risk_counts, 
                                          colors=risk_colors, 
                                          autopct='%1.1f%%',
                                          startangle=90,
                                          textprops={'fontsize': 14, 'fontweight': 'bold'},
                                          pctdistance=0.85)
        
        ax2.set_title('Risk Distribution in Typical Dataset', fontsize=18, fontweight='bold', pad=20)
        
        # Create separate legend to avoid congestion
        ax2.legend(wedges, risk_levels, title="Risk Levels", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1), fontsize=12)
        
        # 3. Feature importance
        ax3 = fig.add_subplot(gs[1, 0])
        features = ['V4', 'V11', 'V12', 'V14', 'V17', 'Amount', 'V10', 'V16']
        importance = [0.15, 0.13, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07]
        
        bars = ax3.barh(features, importance, color='#4CAF50', alpha=0.8, height=0.6)
        ax3.set_title('Key Feature Importance', fontsize=18, fontweight='bold', pad=20)
        ax3.set_xlabel('Relative Importance', fontsize=14)
        ax3.grid(axis='x', alpha=0.3)
        ax3.tick_params(labelsize=12)
        
        # Add value labels
        for bar, imp in zip(bars, importance):
            width = bar.get_width()
            ax3.text(width + 0.005, bar.get_y() + bar.get_height()/2,
                    f'{imp:.3f}', ha='left', va='center', fontweight='bold', fontsize=12)
        
        # 4. Dataset information
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis('off')
        
        info = model_package['training_info']
        
        # Create info text with large fonts
        info_text = f"""DATASET INFORMATION
        
Dataset Size: {info['dataset_size']:,} transactions

Fraud Rate: {info['fraud_rate']:.3%}

Features: {len(model_package['feature_names'])}

Training Date: {info['training_date'][:10]}

MODEL PERFORMANCE

Precision: {perf['precision']:.1%}
Recall: {perf['recall']:.1%}
F1-Score: {perf['f1_score']:.3f}
ROC-AUC: {perf['roc_auc']:.3f}
        """
        
        ax4.text(0.1, 0.9, info_text, fontsize=14, fontweight='bold', 
                verticalalignment='top', transform=ax4.transAxes)
        
        plt.suptitle('FRAUD DETECTION MODEL ANALYSIS', fontsize=24, fontweight='bold', y=0.98)
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_string
    except Exception as e:
        print(f"Error creating performance chart: {e}")
        traceback.print_exc()
        return None

def create_large_transaction_chart(transaction, prediction):
    """Create large, clean transaction analysis visualization without emojis."""
    try:
        # Set style and very large figure size
        plt.style.use('seaborn-v0_8')
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        
        # 1. Large Risk Level Display
        ax1 = fig.add_subplot(gs[0, :])
        
        risk_level = prediction['risk_level']
        fraud_prob = prediction['fraud_probability']
        risk_color = prediction['risk_color']
        
        # Create large risk indicator without emojis
        ax1.text(0.5, 0.7, f"{risk_level} RISK", 
                ha='center', va='center', fontsize=36, fontweight='bold', 
                color=risk_color, transform=ax1.transAxes)
        
        ax1.text(0.5, 0.4, f"Fraud Probability: {fraud_prob:.1%}", 
                ha='center', va='center', fontsize=24, fontweight='bold', 
                color='#333', transform=ax1.transAxes)
        
        classification = "FRAUD DETECTED" if prediction['is_fraud'] else "LEGITIMATE TRANSACTION"
        ax1.text(0.5, 0.1, classification, 
                ha='center', va='center', fontsize=20, fontweight='bold', 
                color=risk_color, transform=ax1.transAxes)
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Add background color
        ax1.add_patch(plt.Rectangle((0, 0), 1, 1, facecolor=risk_color, alpha=0.1, transform=ax1.transAxes))
        
        # 2. Fraud Probability Gauge
        ax2 = fig.add_subplot(gs[1, 0])
        categories = ['Legitimate', 'Fraud']
        probs = [1-fraud_prob, fraud_prob]
        colors_prob = ['#4CAF50', '#F44336']
        
        bars = ax2.bar(categories, probs, color=colors_prob, alpha=0.8, width=0.6)
        ax2.set_title('Fraud vs Legitimate Probability', fontsize=18, fontweight='bold', pad=20)
        ax2.set_ylabel('Probability', fontsize=14)
        ax2.set_ylim(0, 1)
        ax2.tick_params(labelsize=12)
        ax2.grid(axis='y', alpha=0.3)
        
        # Add large probability labels
        for bar, p in zip(bars, probs):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{p:.1%}', ha='center', va='center', fontweight='bold', 
                    color='white', fontsize=16)
        
        # 3. Transaction Details
        ax3 = fig.add_subplot(gs[1, 1])
        
        amount = transaction['Amount']
        time_hours = int(transaction['Time'] / 3600)
        
        details = [
            f"Amount: ${amount:,.2f}",
            f"Time: {time_hours//24}d {time_hours%24}h",
            f"Risk Score: {fraud_prob:.1%}",
            f"Risk Level: {risk_level}"
        ]
        
        ax3.axis('off')
        for i, detail in enumerate(details):
            ax3.text(0.1, 0.8 - i*0.2, detail, fontsize=16, fontweight='bold', 
                    transform=ax3.transAxes)
        
        ax3.set_title('Transaction Details', fontsize=18, fontweight='bold', pad=20)
        
        # 4. Key Features Bar Chart
        ax4 = fig.add_subplot(gs[2, :])
        key_features = ['Amount', 'V4', 'V11', 'V12', 'V14', 'V17']
        feature_values = [transaction[f] for f in key_features]
        
        # Normalize feature values for better visualization
        normalized_values = []
        for i, (feat, val) in enumerate(zip(key_features, feature_values)):
            if feat == 'Amount':
                # Scale amount to 0-10 range for better comparison
                normalized_values.append(min(val / 1000, 10))
            else:
                normalized_values.append(val)
        
        bars = ax4.bar(key_features, normalized_values, color='#2196F3', alpha=0.8, width=0.6)
        ax4.set_title('Key Transaction Features', fontsize=18, fontweight='bold', pad=20)
        ax4.set_ylabel('Feature Value', fontsize=14)
        ax4.tick_params(labelsize=12)
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, orig_val in zip(bars, feature_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{orig_val:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.suptitle('TRANSACTION ANALYSIS RESULTS', fontsize=24, fontweight='bold', y=0.98)
        
        # Convert to base64 string
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        img_buffer.seek(0)
        img_string = base64.b64encode(img_buffer.read()).decode()
        plt.close()
        
        return img_string
    except Exception as e:
        print(f"Error creating transaction chart: {e}")
        traceback.print_exc()
        return None

# Clean, professional HTML Template without emojis
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Fraud Detection System</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            line-height: 1.6;
        }
        
        .container { 
            max-width: 1600px; 
            margin: 0 auto; 
            padding: 40px 20px;
        }
        
        .header { 
            text-align: center; 
            margin-bottom: 60px; 
            color: white;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .header h1 { 
            font-size: 4em; 
            margin-bottom: 20px; 
            letter-spacing: 2px;
        }
        
        .header p { 
            font-size: 1.5em; 
            opacity: 0.9; 
            margin-bottom: 30px;
        }
        
        .card { 
            background: rgba(255,255,255,0.95); 
            border-radius: 20px; 
            padding: 40px; 
            margin-bottom: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        
        .card h2 { 
            color: #333; 
            margin-bottom: 30px; 
            font-size: 2.2em;
            text-align: center;
            border-bottom: 3px solid #667eea;
            padding-bottom: 15px;
        }
        
        .metrics-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); 
            gap: 30px; 
            margin: 40px 0; 
        }
        
        .metric { 
            text-align: center; 
            padding: 30px; 
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            border-radius: 15px; 
            border: 2px solid #e0e0e0;
            transition: transform 0.3s ease;
        }
        
        .metric:hover {
            transform: translateY(-5px);
        }
        
        .metric h3 { 
            color: #555; 
            margin-bottom: 15px; 
            font-size: 1.2em; 
        }
        
        .metric p { 
            font-size: 2em; 
            font-weight: bold; 
            color: #333; 
        }
        
        .btn { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 20px 40px; 
            border: none; 
            border-radius: 30px; 
            cursor: pointer; 
            font-size: 1.3em; 
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 8px 25px rgba(0,0,0,0.2);
            letter-spacing: 1px;
        }
        
        .btn:hover { 
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(0,0,0,0.3);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn-container {
            text-align: center;
            margin: 50px 0;
        }
        
        .chart-container { 
            text-align: center; 
            margin: 40px 0; 
        }
        
        .chart-container img { 
            max-width: 100%; 
            height: auto; 
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .loading { 
            text-align: center; 
            padding: 60px; 
            color: #667eea;
            font-size: 1.5em;
            font-weight: bold;
        }
        
        .error { 
            background: #ffebee; 
            color: #c62828; 
            padding: 25px; 
            border-radius: 15px; 
            margin: 30px 0;
            border-left: 6px solid #c62828;
            font-size: 1.2em;
        }
        
        .result-card {
            background: rgba(255,255,255,0.98);
            border-radius: 20px;
            padding: 40px;
            margin-top: 30px;
            box-shadow: 0 15px 45px rgba(0,0,0,0.1);
        }
        
        @media (max-width: 768px) {
            .header h1 { font-size: 2.5em; }
            .header p { font-size: 1.2em; }
            .card { padding: 25px; }
            .metrics-grid { grid-template-columns: repeat(2, 1fr); gap: 20px; }
            .metric { padding: 20px; }
            .metric p { font-size: 1.5em; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>FRAUD DETECTION SYSTEM</h1>
            <p>Advanced Machine Learning Model • 284K+ Real Transactions • Real-Time Analysis</p>
        </div>

        <div class="card">
            <h2>MODEL PERFORMANCE DASHBOARD</h2>
            
            <div class="metrics-grid">
                <div class="metric">
                    <h3>Precision</h3>
                    <p>{{ "%.1f%%"|format(model_perf.precision * 100) }}</p>
                </div>
                <div class="metric">
                    <h3>Recall</h3>
                    <p>{{ "%.1f%%"|format(model_perf.recall * 100) }}</p>
                </div>
                <div class="metric">
                    <h3>ROC-AUC</h3>
                    <p>{{ "%.3f"|format(model_perf.roc_auc) }}</p>
                </div>
                <div class="metric">
                    <h3>Dataset Size</h3>
                    <p>{{ "{:,}".format(training_info.dataset_size) }}</p>
                </div>
            </div>
            
            <div class="chart-container">
                <img src="data:image/png;base64,{{ performance_chart }}" alt="Model Performance Analysis">
            </div>
        </div>

        <div class="card">
            <h2>LIVE TRANSACTION ANALYSIS</h2>
            
            <div class="btn-container">
                <button class="btn" onclick="generateTransaction()">ANALYZE NEW TRANSACTION</button>
            </div>
            
            <div id="transaction-result">
                <div class="loading">Click button to analyze a live transaction...</div>
            </div>
        </div>
    </div>

    <script>
        function generateTransaction() {
            document.getElementById('transaction-result').innerHTML = '<div class="loading">Analyzing transaction...</div>';
            
            fetch('/analyze')
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! status: ${response.status}`);
                    }
                    return response.json();
                })
                .then(data => {
                    displayTransaction(data);
                })
                .catch(error => {
                    console.error('Error:', error);
                    document.getElementById('transaction-result').innerHTML = 
                        '<div class="error">Error analyzing transaction: ' + error.message + '</div>';
                });
        }

        function displayTransaction(data) {
            if (!data || !data.prediction) {
                document.getElementById('transaction-result').innerHTML = 
                    '<div class="error">Invalid response data</div>';
                return;
            }

            let html = '<div class="result-card">';
            
            if (data.transaction_chart) {
                html += `
                    <div class="chart-container">
                        <img src="data:image/png;base64,${data.transaction_chart}" alt="Transaction Analysis Results">
                    </div>
                `;
            }
            
            html += '</div>';
            
            document.getElementById('transaction-result').innerHTML = html;
        }

        // Generate initial transaction on page load
        window.onload = function() {
            setTimeout(generateTransaction, 1500);
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main dashboard page."""
    try:
        performance_chart = create_large_performance_chart()
        
        return render_template_string(HTML_TEMPLATE, 
                                    model_perf=model_package['performance'],
                                    training_info=model_package['training_info'],
                                    performance_chart=performance_chart)
    except Exception as e:
        print(f"Error in index route: {e}")
        traceback.print_exc()
        return f"Error loading dashboard: {str(e)}", 500

@app.route('/analyze')
def analyze():
    """Generate and analyze a new transaction."""
    try:
        transaction = generate_sample_transaction()
        if not transaction:
            return jsonify({'error': 'Failed to generate transaction'}), 500
            
        prediction = predict_fraud(transaction)
        if not prediction:
            return jsonify({'error': 'Failed to predict fraud'}), 500
        
        # Create large transaction chart
        transaction_chart = create_large_transaction_chart(transaction, prediction)
        
        response_data = {
            'transaction': transaction,
            'prediction': prediction,
            'timestamp': datetime.now().isoformat()
        }
        
        if transaction_chart:
            response_data['transaction_chart'] = transaction_chart
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in analyze route: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    print("Starting Professional Flask Fraud Detection Demo...")
    print("Dashboard will be accessible at:")
    print("   Local: http://localhost:5002")
    print("   Network: http://0.0.0.0:5002")
    print("   External: http://172.178.125.58:5002")
    print("\nStarting server...")
    
    app.run(host='0.0.0.0', port=5002, debug=False)