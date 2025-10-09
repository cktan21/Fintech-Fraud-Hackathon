"""
Core Fraud Detection Classes
Extracted from the Jupyter notebook fraud detection system
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import lightgbm as lgb


# ============================================================
# LEGITIMATE TRANSACTION PROFILER
# ============================================================

class LegitimateTransactionProfiler:
    def __init__(self, weights=None):
        """
        Initialize profiler with optional manual weights
        
        Args:
            weights: Dictionary with keys ['amount', 'timing', 'payment_type', 'currency', 'location']
                    If None, weights will be learned from data automatically
        """
        self.legitimate_patterns = {}
        self.normal_amount_stats = {}
        self.normal_timing_stats = {}
        self.weights = weights
        self.weight_learning_method = 'manual' if weights else 'learned'
        
    def analyze_legitimate_patterns(self, df, learn_weights=True):
        """
        Analyze patterns in legitimate transactions and optionally learn weights
        """
        legitimate_df = df[df['Is_laundering'] == 0]
        
        # 1. Normal amount patterns
        self.normal_amount_stats = {
            'mean': legitimate_df['Amount'].mean(),
            'std': legitimate_df['Amount'].std(),
            'median': legitimate_df['Amount'].median(),
            'q25': legitimate_df['Amount'].quantile(0.25),
            'q75': legitimate_df['Amount'].quantile(0.75)
        }
        
        # 2. Normal timing patterns
        self.normal_timing_stats = {
            'common_hours': legitimate_df['Hour'].mode().tolist(),
            'common_days': legitimate_df['Day_of_week'].mode().tolist(),
            'avg_hour': legitimate_df['Hour'].mean()
        }
        
        # 3. Legitimate payment type patterns
        self.legitimate_patterns['payment_types'] = legitimate_df['Payment_type'].value_counts(normalize=True).to_dict()
        
        # 4. Legitimate currency patterns
        self.legitimate_patterns['currency_combinations'] = legitimate_df.groupby(['Payment_currency', 'Received_currency']).size().to_dict()
        
        # 5. Legitimate location patterns
        self.legitimate_patterns['location_combinations'] = legitimate_df.groupby(['Sender_bank_location', 'Receiver_bank_location']).size().to_dict()
        
        # 6. Learn weights automatically if not provided
        if self.weights is None and learn_weights:
            self._learn_optimal_weights(df)
            self.weight_learning_method = 'learned'
        elif self.weights is None:
            self.weights = {
                'amount': 0.30,
                'timing': 0.25,
                'payment_type': 0.20,
                'currency': 0.15,
                'location': 0.10
            }
            self.weight_learning_method = 'default'
        
        return self.legitimate_patterns
    
    def _learn_optimal_weights(self, df):
        """Learn optimal weights from data using Logistic Regression"""
        X = self._calculate_component_scores_vectorized(df)
        y = (df['Is_laundering'] == 0).astype(int)
        
        lr = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
        lr.fit(X, y)
        
        raw_weights = np.abs(lr.coef_[0])
        normalized_weights = raw_weights / raw_weights.sum()
        
        self.weights = {
            'amount': normalized_weights[0],
            'timing': normalized_weights[1],
            'payment_type': normalized_weights[2],
            'currency': normalized_weights[3],
            'location': normalized_weights[4]
        }
    
    def _calculate_component_scores_vectorized(self, df):
        """Calculate component scores using vectorized operations"""
        n_samples = len(df)
        component_scores = np.zeros((n_samples, 5))
        
        # 1. Amount score
        q25, q75 = self.normal_amount_stats['q25'], self.normal_amount_stats['q75']
        amount = df['Amount'].values
        component_scores[:, 0] = 0.3
        medium_mask = (amount >= q25 * 0.5) & (amount <= q75 * 2)
        component_scores[medium_mask, 0] = 0.7
        perfect_mask = (amount >= q25) & (amount <= q75)
        component_scores[perfect_mask, 0] = 1.0
        
        # 2. Timing score
        hours = df['Hour'].values
        avg_hour = self.normal_timing_stats['avg_hour']
        common_hours = set(self.normal_timing_stats['common_hours'])
        component_scores[:, 1] = 0.4
        close_mask = np.abs(hours - avg_hour) <= 3
        component_scores[close_mask, 1] = 0.7
        exact_mask = df['Hour'].isin(common_hours).values
        component_scores[exact_mask, 1] = 1.0
        
        # 3. Payment type score
        payment_types = df['Payment_type'].values
        payment_map = {k: min(v * 10, 1.0) for k, v in self.legitimate_patterns['payment_types'].items()}
        component_scores[:, 2] = np.array([payment_map.get(pt, 0.2) for pt in payment_types])
        
        # 4. Currency score
        currency_pairs = list(zip(df['Payment_currency'], df['Received_currency']))
        currency_map = {k: min(v / 1000, 1.0) for k, v in self.legitimate_patterns['currency_combinations'].items()}
        component_scores[:, 3] = np.array([currency_map.get(pair, 0.3) for pair in currency_pairs])
        
        # 5. Location score
        location_pairs = list(zip(df['Sender_bank_location'], df['Receiver_bank_location']))
        location_map = {k: min(v / 1000, 1.0) for k, v in self.legitimate_patterns['location_combinations'].items()}
        component_scores[:, 4] = np.array([location_map.get(pair, 0.4) for pair in location_pairs])
        
        return component_scores
    
    def predict_legitimate(self, df):
        """Predict legitimacy scores for all transactions (VECTORIZED)"""
        component_scores = self._calculate_component_scores_vectorized(df)
        
        weights_vector = np.array([
            self.weights['amount'],
            self.weights['timing'],
            self.weights['payment_type'],
            self.weights['currency'],
            self.weights['location']
        ])
        
        final_scores = component_scores @ weights_vector
        return final_scores
    
    def get_weight_info(self):
        """Get information about the weights being used"""
        return {
            'method': self.weight_learning_method,
            'weights': self.weights,
            'description': {
                'manual': 'User-provided weights',
                'learned': 'Automatically learned from data using Logistic Regression',
                'default': 'Default weights (no learning performed)'
            }[self.weight_learning_method]
        }


# ============================================================
# DATA BALANCER
# ============================================================

class DataBalancer:
    """
    Handles class imbalance using hybrid approach:
    - Intelligent undersampling for normal transactions
    - SMOTE oversampling for rare fraud types
    - Class weight computation
    """
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.class_weights = {}
        self.fraud_type_weights = {}
        
    def _stratified_sample(self, labels, n_samples):
        """Stratified sampling to preserve label distribution"""
        label_counts = labels.value_counts()
        proportions = label_counts / len(labels)
        
        sampled_indices = []
        for label, proportion in proportions.items():
            label_indices = labels[labels == label].index
            n_label_samples = int(n_samples * proportion)
            n_label_samples = min(n_label_samples, len(label_indices))
            
            if n_label_samples > 0:
                sampled = np.random.choice(label_indices, n_label_samples, replace=False)
                sampled_indices.extend(sampled)
        
        return sampled_indices
    
    def _compute_class_weights(self, y):
        """Compute class weights for imbalanced data"""
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, weights))
    
    def get_sample_weights(self, y, weight_dict=None):
        """Get sample weights for training"""
        if weight_dict is None:
            weight_dict = self.class_weights
        
        return np.array([weight_dict[label] for label in y])


# ============================================================
# FRAUD DETECTION ENGINE
# ============================================================

class FraudDetectionEngine:
    def __init__(self, flagged_sender_accounts=None, flagged_receiver_accounts=None):
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self._is_fitted = False
        
        self.flagged_sender_accounts = flagged_sender_accounts or set()
        self.flagged_receiver_accounts = flagged_receiver_accounts or set()
    
    def learn_flagged_accounts(self, df, fraud_column='Is_laundering', min_occurrences=5):
        """Automatically learn flagged accounts from training data"""
        fraud_senders = df[df[fraud_column] == 1]['Sender_account'].value_counts()
        fraud_receivers = df[df[fraud_column] == 1]['Receiver_account'].value_counts()
        
        self.flagged_sender_accounts = set(fraud_senders[fraud_senders >= min_occurrences].index)
        self.flagged_receiver_accounts = set(fraud_receivers[fraud_receivers >= min_occurrences].index)
        
        return self.flagged_sender_accounts, self.flagged_receiver_accounts
        
    def prepare_features(self, df):
        """Prepare features for machine learning models"""
        numerical_features = ['Amount', 'Amount_log', 'Amount_zscore', 'Hour', 'Day_of_week', 
                            'Month', 'Is_cross_border', 'Is_currency_different', 'Is_high_risk_country', 
                            'Legitimacy_score']
        
        if 'Payment_risk_score' in df.columns:
            numerical_features.append('Payment_risk_score')
        
        categorical_features = ['Payment_type', 'Payment_currency', 'Received_currency', 
                              'Sender_bank_location', 'Receiver_bank_location', 'Amount_category']
        
        feature_df = df[numerical_features].copy()
        
        # Add flagged account features
        feature_df['Is_flagged_sender'] = df['Sender_account'].isin(self.flagged_sender_accounts).astype(int)
        feature_df['Is_flagged_receiver'] = df['Receiver_account'].isin(self.flagged_receiver_accounts).astype(int)
        
        # Encode categorical variables
        for col in categorical_features:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                feature_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                class_to_code = dict(zip(
                    self.label_encoders[col].classes_, 
                    range(len(self.label_encoders[col].classes_))
                ))
                feature_df[f'{col}_encoded'] = df[col].astype(str).map(class_to_code).fillna(-1).astype(int)
        
        self.feature_columns = feature_df.columns.tolist()
        self._is_fitted = True
        return feature_df
    
    def train_models(self, X, y, n_jobs=-1, sample_weight=None):
        """Train multiple fraud detection models"""
        X_scaled = self.scaler.fit_transform(X)
        
        # 1. LightGBM Classifier
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_jobs': n_jobs
        }
        
        train_data = lgb.Dataset(X_scaled, label=y, weight=sample_weight)
        self.models['lightgbm'] = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # 2. Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000,
            solver='saga',
            n_jobs=n_jobs
        )
        self.models['logistic_regression'].fit(X_scaled, y, sample_weight=sample_weight)
        
        # 3. Isolation Forest
        self.models['isolation_forest'] = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=n_jobs
        )
        self.models['isolation_forest'].fit(X_scaled, sample_weight=sample_weight)
    
    def predict_fraud_probability(self, X):
        """Get fraud probability from ensemble of models"""
        X_scaled = self.scaler.transform(X)
        
        lgb_proba = self.models['lightgbm'].predict(X_scaled)
        lr_proba = self.models['logistic_regression'].predict_proba(X_scaled)[:, 1]
        iso_pred = self.models['isolation_forest'].predict(X_scaled)
        iso_proba = (iso_pred == -1).astype(np.float32)
        
        ensemble_proba = (0.4 * lgb_proba + 0.4 * lr_proba + 0.2 * iso_proba)
        
        return ensemble_proba, {
            'lightgbm': lgb_proba,
            'logistic_regression': lr_proba,
            'isolation_forest': iso_proba
        }


# ============================================================
# FRAUD TYPE CLASSIFIER
# ============================================================

class FraudTypeClassifier:
    """Multi-class classifier for identifying specific fraud types"""
    def __init__(self, random_state=42, flagged_sender_accounts=None, flagged_receiver_accounts=None):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.fraud_types = []
        
        self.flagged_sender_accounts = flagged_sender_accounts or set()
        self.flagged_receiver_accounts = flagged_receiver_accounts or set()
        
    def prepare_features(self, df):
        """Prepare features for fraud type classification"""
        numerical_features = ['Amount', 'Amount_log', 'Amount_zscore', 'Hour', 'Day_of_week', 
                            'Month', 'Is_cross_border', 'Is_currency_different', 'Is_high_risk_country', 
                            'Legitimacy_score']
        
        if 'Payment_risk_score' in df.columns:
            numerical_features.append('Payment_risk_score')
        
        categorical_features = ['Payment_type', 'Payment_currency', 'Received_currency', 
                              'Sender_bank_location', 'Receiver_bank_location', 'Amount_category']
        
        feature_df = df[numerical_features].copy()
        
        # Add flagged account features
        feature_df['Is_flagged_sender'] = df['Sender_account'].isin(self.flagged_sender_accounts).astype(int)
        feature_df['Is_flagged_receiver'] = df['Receiver_account'].isin(self.flagged_receiver_accounts).astype(int)
        
        # Encode categorical variables
        for col in categorical_features:
            if col in df.columns:
                feature_df[f'{col}_encoded'] = df[col].astype('category').cat.codes
        
        self.feature_columns = feature_df.columns.tolist()
        return feature_df
    
    def train(self, X, y_fraud_types, sample_weight=None, n_jobs=-1):
        """Train fraud type classifier"""
        y_encoded = self.label_encoder.fit_transform(y_fraud_types)
        self.fraud_types = self.label_encoder.classes_
        
        X_scaled = self.scaler.fit_transform(X)
        
        # Compute class weights if not provided
        if sample_weight is None:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_encoded), y=y_encoded)
            class_weight_dict = dict(zip(np.unique(y_encoded), class_weights))
            sample_weight = np.array([class_weight_dict[y] for y in y_encoded])
        
        # 1. LightGBM Multi-class
        lgb_params = {
            'objective': 'multiclass',
            'num_class': len(self.fraud_types),
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state,
            'n_jobs': n_jobs
        }
        
        train_data = lgb.Dataset(X_scaled, label=y_encoded, weight=sample_weight)
        self.models['lightgbm'] = lgb.train(
            lgb_params,
            train_data,
            num_boost_round=200,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)]
        )
        
        # 2. Random Forest Multi-class
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            random_state=self.random_state,
            class_weight='balanced',
            n_jobs=n_jobs
        )
        self.models['random_forest'].fit(X_scaled, y_encoded, sample_weight=sample_weight)
    
    def predict_fraud_type(self, X, top_k=3):
        """Predict fraud type with top-K alternatives"""
        X_scaled = self.scaler.transform(X)
        
        lgb_proba = self.models['lightgbm'].predict(X_scaled)
        rf_proba = self.models['random_forest'].predict_proba(X_scaled)
        
        ensemble_proba = (0.6 * lgb_proba + 0.4 * rf_proba)
        
        predictions_encoded = np.argmax(ensemble_proba, axis=1)
        predictions = self.label_encoder.inverse_transform(predictions_encoded)
        probabilities = ensemble_proba[np.arange(len(X)), predictions_encoded]
        
        top_k_indices = np.argsort(-ensemble_proba, axis=1)[:, :top_k]
        top_k_types = []
        top_k_probas = []
        
        for i, indices in enumerate(top_k_indices):
            types = self.label_encoder.inverse_transform(indices)
            probas = ensemble_proba[i, indices]
            top_k_types.append(types.tolist())
            top_k_probas.append(probas.tolist())
        
        return predictions, probabilities, top_k_types, top_k_probas


# ============================================================
# HIERARCHICAL FRAUD DETECTOR
# ============================================================

class HierarchicalFraudDetector:
    """Three-stage hierarchical fraud detection system"""
    
    def __init__(self, legitimacy_profiler, binary_classifier, fraud_type_classifier):
        self.legitimacy_profiler = legitimacy_profiler
        self.binary_classifier = binary_classifier
        self.fraud_type_classifier = fraud_type_classifier
        
    def predict(self, df, legitimacy_threshold=0.7, fraud_threshold=0.5):
        """Predict fraud and fraud type using hierarchical approach"""
        n_transactions = len(df)
        
        # Initialize results
        results = pd.DataFrame(index=df.index)
        results['is_fraud'] = False
        results['fraud_type'] = 'Normal'
        results['fraud_type_confidence'] = 0.0
        results['legitimacy_score'] = 0.0
        results['fraud_probability'] = 0.0
        results['stage'] = ''
        results['top_3_fraud_types'] = ''
        results['top_3_probabilities'] = ''
        
        # STAGE 1: Legitimate Transaction Profiler
        legitimacy_scores = self.legitimacy_profiler.predict_legitimate(df)
        results['legitimacy_score'] = legitimacy_scores
        
        legitimate_mask = legitimacy_scores >= legitimacy_threshold
        n_legitimate = legitimate_mask.sum()
        n_suspicious = n_transactions - n_legitimate
        
        results.loc[legitimate_mask, 'stage'] = 'Stage 1: Legitimate'
        results.loc[legitimate_mask, 'is_fraud'] = False
        results.loc[legitimate_mask, 'fraud_type'] = 'Normal'
        
        if n_suspicious == 0:
            return results
        
        # STAGE 2: Binary Fraud Classifier
        suspicious_df = df[~legitimate_mask]
        suspicious_idx = suspicious_df.index
        
        X_suspicious = self.binary_classifier.prepare_features(suspicious_df)
        fraud_probas, _ = self.binary_classifier.predict_fraud_probability(X_suspicious)
        
        results.loc[suspicious_idx, 'fraud_probability'] = fraud_probas
        
        fraud_mask = fraud_probas >= fraud_threshold
        fraud_idx = suspicious_idx[fraud_mask]
        normal_idx = suspicious_idx[~fraud_mask]
        
        n_fraud = len(fraud_idx)
        
        results.loc[normal_idx, 'stage'] = 'Stage 2: Normal'
        results.loc[normal_idx, 'is_fraud'] = False
        results.loc[normal_idx, 'fraud_type'] = 'Normal'
        
        results.loc[fraud_idx, 'stage'] = 'Stage 2: Fraud Detected'
        results.loc[fraud_idx, 'is_fraud'] = True
        
        if n_fraud == 0:
            return results
        
        # STAGE 3: Fraud Type Classifier
        fraud_df = df.loc[fraud_idx]
        
        X_fraud = self.fraud_type_classifier.prepare_features(fraud_df)
        fraud_types, type_confidence, top_3_types, top_3_probas = \
            self.fraud_type_classifier.predict_fraud_type(X_fraud, top_k=3)
        
        results.loc[fraud_idx, 'fraud_type'] = fraud_types
        results.loc[fraud_idx, 'fraud_type_confidence'] = type_confidence
        results.loc[fraud_idx, 'stage'] = 'Stage 3: Type Identified'
        results.loc[fraud_idx, 'top_3_fraud_types'] = [', '.join(types) for types in top_3_types]
        results.loc[fraud_idx, 'top_3_probabilities'] = [
            ', '.join([f"{p:.3f}" for p in probas]) for probas in top_3_probas
        ]
        
        return results

