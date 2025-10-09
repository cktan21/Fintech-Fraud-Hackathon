"""
Training Module for Fraud Detection System
Trains models on startup using the full dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import time

from fraud_engine import (
    LegitimateTransactionProfiler,
    DataBalancer,
    FraudDetectionEngine,
    FraudTypeClassifier,
    HierarchicalFraudDetector
)


def apply_feature_engineering(df):
    """
    Apply feature engineering to raw transaction data
    Based on Cell 4 from the notebook
    """
    df_features = df.copy()
    
    RISKY_COUNTRIES = ['Nigeria', 'Mexico', 'India']
    
    # 1. Time-based features
    df_features['Time'] = pd.to_datetime(df_features['Time'], format='%H:%M:%S', errors='coerce')
    df_features['Date'] = pd.to_datetime(df_features['Date'], errors='coerce')
    df_features['Hour'] = df_features['Time'].dt.hour
    df_features['Day_of_week'] = df_features['Date'].dt.dayofweek
    df_features['Month'] = df_features['Date'].dt.month
    
    # 2. Amount-based features
    df_features['Amount_log'] = np.log1p(df_features['Amount'])
    df_features['Amount_zscore'] = (df_features['Amount'] - df_features['Amount'].mean()) / df_features['Amount'].std()
    
    # 3. Cross-border and currency features
    df_features['Is_cross_border'] = (df_features['Sender_bank_location'] != df_features['Receiver_bank_location']).astype(int)
    df_features['Is_currency_different'] = (df_features['Payment_currency'] != df_features['Received_currency']).astype(int)
    df_features['Is_high_risk_country'] = df_features['Receiver_bank_location'].isin(RISKY_COUNTRIES).astype(int)
    
    # 4. Payment type risk scoring
    payment_risk_scores = {
        'Cash Deposit': 0.3,
        'Cash Withdrawal': 0.8,
        'Cross-border': 0.7,
        'Credit card': 0.2,
        'Debit card': 0.2,
        'ACH': 0.1,
        'Cheque': 0.4
    }
    df_features['Payment_risk_score'] = df_features['Payment_type'].map(payment_risk_scores).fillna(0.5)
    
    # 5. Amount risk categories
    df_features['Amount_category'] = pd.cut(
        df_features['Amount'], 
        bins=[0, 1000, 5000, 10000, 50000, float('inf')], 
        labels=['Low', 'Medium', 'High', 'Very_High', 'Extreme']
    )
    
    return df_features


def train_fraud_detection_system(data_path, test_size=0.2, random_state=42):
    """
    Train the complete hierarchical fraud detection system
    
    Args:
        data_path: Path to the CSV dataset
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        HierarchicalFraudDetector: Trained fraud detection system
    """
    print("\n" + "="*70)
    print("TRAINING HIERARCHICAL FRAUD DETECTION SYSTEM")
    print("="*70)
    
    start_time = time.time()
    
    # ========================================
    # STEP 1: Load and prepare data
    # ========================================
    print("\n📊 STEP 1: Loading data...")
    df = pd.read_csv(data_path)
    print(f"✓ Loaded {len(df):,} transactions")
    
    # Apply feature engineering
    print("\n📊 STEP 2: Feature engineering...")
    df_features = apply_feature_engineering(df)
    print(f"✓ Created {len(df_features.columns)} features")
    
    # Extract labels
    y_binary = df_features['Is_laundering']
    y_types = df_features['Laundering_type']
    
    # Split data
    print("\n📊 STEP 3: Splitting data...")
    df_train, df_test, y_train_binary, y_test_binary, y_train_types, y_test_types = \
        train_test_split(
            df_features, y_binary, y_types,
            test_size=test_size,
            random_state=random_state,
            stratify=y_binary
        )
    print(f"✓ Training: {len(df_train):,} | Test: {len(df_test):,}")
    
    # ========================================
    # STEP 4: Train Legitimate Transaction Profiler
    # ========================================
    print("\n📊 STEP 4: Training Legitimate Transaction Profiler...")
    profiler = LegitimateTransactionProfiler()
    profiler.analyze_legitimate_patterns(df_train, learn_weights=True)
    
    # Calculate legitimacy scores
    legitimacy_scores = profiler.predict_legitimate(df_train)
    df_train['Legitimacy_score'] = legitimacy_scores
    print("✓ Legitimacy profiler trained")
    
    # ========================================
    # STEP 5: Balance data for binary classification
    # ========================================
    print("\n📊 STEP 5: Balancing data for binary classification...")
    
    fraud_mask_train = (y_train_binary == 1)
    normal_mask_train = (y_train_binary == 0)
    
    fraud_indices = df_train[fraud_mask_train].index
    normal_indices = df_train[normal_mask_train].index
    
    # Undersample normal transactions
    target_normal_samples = 500000
    if len(normal_indices) > target_normal_samples:
        normal_types = y_train_types[normal_mask_train]
        label_counts = normal_types.value_counts()
        proportions = label_counts / len(normal_types)
        
        sampled_normal_indices = []
        for label, proportion in proportions.items():
            label_indices = normal_types[normal_types == label].index
            n_label_samples = int(target_normal_samples * proportion)
            n_label_samples = min(n_label_samples, len(label_indices))
            
            if n_label_samples > 0:
                sampled = np.random.choice(label_indices, n_label_samples, replace=False)
                sampled_normal_indices.extend(sampled)
        
        normal_indices = sampled_normal_indices
    
    # Combine and shuffle
    balanced_indices = list(fraud_indices) + list(normal_indices)
    np.random.shuffle(balanced_indices)
    
    df_train_balanced = df_train.loc[balanced_indices]
    y_train_balanced = y_train_binary.loc[balanced_indices]
    
    print(f"✓ Balanced: Fraud={len(fraud_indices):,}, Normal={len(normal_indices):,}")
    
    # Compute class weights
    balancer = DataBalancer(random_state=random_state)
    balancer.class_weights = balancer._compute_class_weights(y_train_balanced)
    binary_sample_weights = balancer.get_sample_weights(y_train_balanced)
    
    # ========================================
    # STEP 6: Train Binary Fraud Classifier
    # ========================================
    print("\n📊 STEP 6: Training Binary Fraud Classifier...")
    
    binary_classifier = FraudDetectionEngine()
    
    # Learn flagged accounts
    binary_classifier.learn_flagged_accounts(df_train, fraud_column='Is_laundering', min_occurrences=5)
    print(f"✓ Flagged {len(binary_classifier.flagged_sender_accounts)} senders, "
          f"{len(binary_classifier.flagged_receiver_accounts)} receivers")
    
    # Prepare features and train
    X_train_balanced = binary_classifier.prepare_features(df_train_balanced)
    binary_classifier.train_models(
        X_train_balanced,
        y_train_balanced,
        n_jobs=-1,
        sample_weight=binary_sample_weights
    )
    print("✓ Binary classifier trained")
    
    # ========================================
    # STEP 7: Prepare fraud type data
    # ========================================
    print("\n📊 STEP 7: Preparing fraud type data...")
    
    fraud_mask_train = (y_train_binary == 1)
    df_train_fraud = df_train[fraud_mask_train]
    y_train_fraud_types = y_train_types[fraud_mask_train]
    
    # Remove "Normal" types
    actual_fraud_mask = ~y_train_fraud_types.str.startswith('Normal')
    df_train_fraud = df_train_fraud[actual_fraud_mask]
    y_train_fraud_types = y_train_fraud_types[actual_fraud_mask]
    
    print(f"✓ {len(df_train_fraud):,} fraud transactions for type classification")
    
    # ========================================
    # STEP 8: Train Fraud Type Classifier
    # ========================================
    print("\n📊 STEP 8: Training Fraud Type Classifier...")
    
    fraud_type_classifier = FraudTypeClassifier(
        random_state=random_state,
        flagged_sender_accounts=binary_classifier.flagged_sender_accounts,
        flagged_receiver_accounts=binary_classifier.flagged_receiver_accounts
    )
    
    # Prepare features
    X_train_fraud = fraud_type_classifier.prepare_features(df_train_fraud)
    
    # Apply SMOTE
    fraud_type_counts = y_train_fraud_types.value_counts()
    min_class_size = fraud_type_counts.min()
    k_neighbors = min(5, min_class_size - 1)
    
    if k_neighbors >= 1:
        try:
            smote = SMOTE(
                sampling_strategy='not majority',
                k_neighbors=k_neighbors,
                random_state=random_state
            )
            X_train_fraud_balanced, y_train_fraud_balanced = smote.fit_resample(X_train_fraud, y_train_fraud_types)
            print(f"✓ SMOTE applied: {len(X_train_fraud):,} → {len(X_train_fraud_balanced):,} samples")
        except Exception as e:
            print(f"⚠ SMOTE failed: {e}, using original data")
            X_train_fraud_balanced = X_train_fraud
            y_train_fraud_balanced = y_train_fraud_types
    else:
        X_train_fraud_balanced = X_train_fraud
        y_train_fraud_balanced = y_train_fraud_types
    
    # Compute fraud type weights
    balancer.fraud_type_weights = balancer._compute_class_weights(y_train_fraud_balanced)
    fraud_type_weights = balancer.get_sample_weights(y_train_fraud_balanced, balancer.fraud_type_weights)
    
    # Train
    fraud_type_classifier.train(
        X_train_fraud_balanced,
        y_train_fraud_balanced,
        sample_weight=fraud_type_weights,
        n_jobs=-1
    )
    print("✓ Fraud type classifier trained")
    
    # ========================================
    # STEP 9: Create Hierarchical Detector
    # ========================================
    print("\n📊 STEP 9: Creating Hierarchical Detector...")
    hierarchical_detector = HierarchicalFraudDetector(
        legitimacy_profiler=profiler,
        binary_classifier=binary_classifier,
        fraud_type_classifier=fraud_type_classifier
    )
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("✓ TRAINING COMPLETE!")
    print("="*70)
    print(f"Total training time: {elapsed_time:.1f} seconds")
    print(f"Models ready for deployment")
    print("="*70)
    
    return hierarchical_detector

