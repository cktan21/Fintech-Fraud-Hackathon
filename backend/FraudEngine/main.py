"""
FastAPI Application for Fraud Detection
Trains models on startup and provides real-time fraud detection endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import os

from models import (
    Transaction,
    BatchTransactionRequest,
    FraudDetectionResponse,
    BatchFraudDetectionResponse,
    HealthResponse,
    RootResponse
)
from training import train_fraud_detection_system, apply_feature_engineering

# ============================================================
# FASTAPI APPLICATION SETUP
# ============================================================

app = FastAPI(
    title="Fraud Detection API",
    description="Hierarchical 3-stage fraud detection system for cross-border transactions",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global fraud detector (loaded on startup)
fraud_detector = None
amount_mean = None
amount_std = None


# ============================================================
# STARTUP EVENT - TRAIN MODELS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Train fraud detection models on startup"""
    global fraud_detector, amount_mean, amount_std
    
    print("\n🚀 Starting Fraud Detection API...")
    print("📊 Training models on startup...")
    
    # Determine data path
    data_path = os.path.join(os.path.dirname(__file__), ".", "data", "SAML-D.csv")
    
    if not os.path.exists(data_path):
        print(f"⚠ WARNING: Data file not found at {data_path}")
        print("  Models will not be trained. Please ensure data/SAML-D.csv exists.")
        return
    
    try:
        # Train the system
        fraud_detector = train_fraud_detection_system(data_path)
        
        # Store statistics for feature engineering
        df_sample = pd.read_csv(data_path, nrows=10000)
        amount_mean = df_sample['Amount'].mean()
        amount_std = df_sample['Amount'].std()
        
        print("\n✅ Fraud detection system ready!")
        print("🌐 API is now accepting requests\n")
        
    except Exception as e:
        print(f"\n❌ ERROR during model training: {e}")
        print("  API will start but predictions will fail\n")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def transaction_to_dataframe(transaction: Transaction) -> pd.DataFrame:
    """Convert a single Transaction to DataFrame"""
    return pd.DataFrame([transaction.dict()])


def prepare_transaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering to transaction(s)
    Uses global amount_mean and amount_std for consistency
    """
    df_features = df.copy()
    
    RISKY_COUNTRIES = ['Nigeria', 'Mexico', 'India']
    
    # Time-based features
    df_features['Time'] = pd.to_datetime(df_features['Time'], format='%H:%M:%S', errors='coerce')
    df_features['Date'] = pd.to_datetime(df_features['Date'], errors='coerce')
    df_features['Hour'] = df_features['Time'].dt.hour
    df_features['Day_of_week'] = df_features['Date'].dt.dayofweek
    df_features['Month'] = df_features['Date'].dt.month
    
    # Amount-based features
    df_features['Amount_log'] = np.log1p(df_features['Amount'])
    df_features['Amount_zscore'] = (df_features['Amount'] - amount_mean) / amount_std
    
    # Risk features
    df_features['Is_cross_border'] = (df_features['Sender_bank_location'] != df_features['Receiver_bank_location']).astype(int)
    df_features['Is_currency_different'] = (df_features['Payment_currency'] != df_features['Received_currency']).astype(int)
    df_features['Is_high_risk_country'] = df_features['Receiver_bank_location'].isin(RISKY_COUNTRIES).astype(int)
    
    # Payment risk scoring
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
    
    # Amount categories
    df_features['Amount_category'] = pd.cut(
        df_features['Amount'], 
        bins=[0, 1000, 5000, 10000, 50000, float('inf')], 
        labels=['Low', 'Medium', 'High', 'Very_High', 'Extreme']
    )
    
    # Calculate legitimacy score
    df_features['Legitimacy_score'] = fraud_detector.legitimacy_profiler.predict_legitimate(df_features)
    
    return df_features


def identify_risk_factors(row: pd.Series, result: pd.Series) -> list:
    """Identify risk factors for a transaction"""
    risk_factors = []
    
    if row['Is_cross_border'] == 1:
        risk_factors.append('Cross-border transaction')
    
    if row['Is_currency_different'] == 1:
        risk_factors.append('Currency conversion involved')
    
    if row['Is_high_risk_country'] == 1:
        risk_factors.append('High-risk destination')
    
    if row['Amount'] > 10000:
        risk_factors.append('High transaction amount (>$10,000)')
    
    if row['Payment_risk_score'] > 0.6:
        risk_factors.append('High-risk payment type')
    
    if result['legitimacy_score'] < 0.4:
        risk_factors.append('Very low legitimacy score')
    
    if result['fraud_probability'] > 0.8:
        risk_factors.append('High fraud probability')
    
    # Check flagged accounts
    if row['Sender_account'] in fraud_detector.binary_classifier.flagged_sender_accounts:
        risk_factors.append('Flagged sender account')
    
    if row['Receiver_account'] in fraud_detector.binary_classifier.flagged_receiver_accounts:
        risk_factors.append('Flagged receiver account')
    
    return risk_factors


def make_recommendation(fraud_probability: float) -> str:
    """Generate recommendation based on fraud probability"""
    if fraud_probability > 0.8:
        return "BLOCK"
    elif fraud_probability > 0.5:
        return "REVIEW"
    else:
        return "APPROVE"


def process_prediction_result(df_original: pd.DataFrame, results: pd.DataFrame) -> list:
    """Process hierarchical detector results into API response format"""
    responses = []
    
    for idx in range(len(results)):
        result = results.iloc[idx]
        original = df_original.iloc[idx]
        
        # Parse top 3 fraud types
        if result['is_fraud'] and result['top_3_fraud_types']:
            top_3_types = result['top_3_fraud_types'].split(', ')
            top_3_probas = [float(p) for p in result['top_3_probabilities'].split(', ')]
        else:
            top_3_types = []
            top_3_probas = []
        
        # Identify risk factors
        risk_factors = identify_risk_factors(original, result)
        
        # Make recommendation
        recommendation = make_recommendation(result['fraud_probability'])
        
        response = FraudDetectionResponse(
            is_fraud=bool(result['is_fraud']),
            fraud_type=result['fraud_type'],
            confidence=float(result['fraud_type_confidence']),
            legitimacy_score=float(result['legitimacy_score']),
            fraud_probability=float(result['fraud_probability']),
            top_3_fraud_types=top_3_types,
            top_3_probabilities=top_3_probas,
            risk_factors=risk_factors,
            recommendation=recommendation,
            stage=result['stage']
        )
        
        responses.append(response)
    
    return responses


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/", response_model=RootResponse)
async def root():
    """API information and available endpoints"""
    return RootResponse(
        service="Fraud Detection API",
        version="1.0.0",
        description="Hierarchical 3-stage fraud detection system for cross-border transactions",
        endpoints={
            "/": "API information",
            "/docs": "Interactive API documentation (Swagger UI)",
            "/redoc": "Alternative API documentation (ReDoc)",
            "/health": "Health check endpoint",
            "/predict": "Single transaction fraud detection (POST)",
            "/predict_batch": "Batch transaction fraud detection (POST)"
        }
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    if fraud_detector is None:
        return HealthResponse(
            status="unhealthy",
            model_loaded=False,
            message="Models not loaded. Check startup logs for errors."
        )
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        message="Fraud detection system is ready"
    )


@app.post("/predict", response_model=FraudDetectionResponse)
async def predict_fraud(transaction: Transaction):
    """
    Predict fraud for a single transaction
    
    This endpoint uses a 3-stage hierarchical detection system:
    1. Legitimacy profiler (rule-based filtering)
    2. Binary fraud classifier (ML-based fraud detection)
    3. Fraud type classifier (17 fraud type identification)
    """
    if fraud_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection system not ready. Models are still training or failed to load."
        )
    
    try:
        # Convert to DataFrame
        df = transaction_to_dataframe(transaction)
        
        # Apply feature engineering
        df_features = prepare_transaction_features(df)
        
        # Run hierarchical detection
        results = fraud_detector.predict(df_features)
        
        # Process results
        responses = process_prediction_result(df_features, results)
        
        return responses[0]
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/predict_batch", response_model=BatchFraudDetectionResponse)
async def predict_fraud_batch(request: BatchTransactionRequest):
    """
    Predict fraud for multiple transactions in batch
    
    More efficient than calling /predict multiple times.
    Returns results for each transaction plus summary statistics.
    """
    if fraud_detector is None:
        raise HTTPException(
            status_code=503,
            detail="Fraud detection system not ready. Models are still training or failed to load."
        )
    
    if not request.transactions:
        raise HTTPException(
            status_code=400,
            detail="No transactions provided"
        )
    
    try:
        # Convert to DataFrame
        df = pd.DataFrame([t.dict() for t in request.transactions])
        
        # Apply feature engineering
        df_features = prepare_transaction_features(df)
        
        # Run hierarchical detection
        results = fraud_detector.predict(df_features)
        
        # Process results
        responses = process_prediction_result(df_features, results)
        
        # Calculate summary statistics
        total = len(responses)
        fraud_count = sum(1 for r in responses if r.is_fraud)
        fraud_percentage = (fraud_count / total * 100) if total > 0 else 0
        
        recommendations = {}
        for r in responses:
            recommendations[r.recommendation] = recommendations.get(r.recommendation, 0) + 1
        
        summary = {
            "total_transactions": total,
            "fraud_detected": fraud_count,
            "fraud_percentage": round(fraud_percentage, 2),
            "recommendations": recommendations
        }
        
        return BatchFraudDetectionResponse(
            results=responses,
            summary=summary
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction failed: {str(e)}"
        )


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=5600,
        reload=False  # Disable reload since we train on startup
    )

