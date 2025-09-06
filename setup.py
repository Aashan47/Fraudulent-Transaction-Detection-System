#!/usr/bin/env python3
"""
Setup script for Fraudulent Transaction Detection System
=======================================================

This script sets up the environment and runs a basic test.

Usage: python setup.py
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages from requirements.txt"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements.txt"
        ])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def test_installation():
    """Test if the fraud detection system works"""
    print("\n🧪 Testing fraud detection system...")
    try:
        from fraud_detection_system import FraudDetectionSystem
        
        # Quick test
        system = FraudDetectionSystem()
        df = system.create_synthetic_dataset(n_samples=500, fraud_rate=0.03)
        
        print(f"✅ Test dataset created: {len(df)} transactions")
        print(f"✅ Fraud rate: {df['is_fraud'].mean():.1%}")
        print("✅ System is ready to use!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing system: {e}")
        return False

def main():
    """Main setup function"""
    print("🔍 FRAUDULENT TRANSACTION DETECTION SYSTEM SETUP")
    print("=" * 60)
    
    # Check if requirements.txt exists
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return
    
    # Install packages
    if not install_requirements():
        print("❌ Setup failed during package installation")
        return
    
    # Test installation
    if not test_installation():
        print("❌ Setup failed during system testing")
        return
    
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Run 'python demo.py' for a quick demonstration")
    print("2. Run 'python fraud_detection_system.py' for the full pipeline")
    print("3. Check 'README.md' for detailed usage instructions")
    print("\nFiles you'll get after running:")
    print("• fraud_detection_analysis.png - Comprehensive visualizations")
    print("• fraud_detection_report.txt - Detailed analysis and recommendations")

if __name__ == "__main__":
    main()