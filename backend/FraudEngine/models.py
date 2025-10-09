"""
Pydantic Models for FastAPI Request/Response
"""

from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


# ============================================================
# REQUEST MODELS
# ============================================================

class Transaction(BaseModel):
    """Single transaction for fraud detection"""
    Time: str = Field(..., description="Transaction time in HH:MM:SS format", example="14:30:25")
    Date: str = Field(..., description="Transaction date in YYYY-MM-DD format", example="2022-10-07")
    Sender_account: int = Field(..., description="Sender account number", example=8724731955)
    Receiver_account: int = Field(..., description="Receiver account number", example=2769355426)
    Amount: float = Field(..., gt=0, description="Transaction amount", example=1459.15)
    Payment_currency: str = Field(..., description="Currency used for payment", example="UK pounds")
    Received_currency: str = Field(..., description="Currency received", example="UK pounds")
    Sender_bank_location: str = Field(..., description="Sender's bank location", example="UK")
    Receiver_bank_location: str = Field(..., description="Receiver's bank location", example="UK")
    Payment_type: str = Field(..., description="Type of payment", example="Cash Deposit")
    
    class Config:
        json_schema_extra = {
            "example": {
                "Time": "14:30:25",
                "Date": "2022-10-07",
                "Sender_account": 8724731955,
                "Receiver_account": 2769355426,
                "Amount": 1459.15,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UAE",
                "Payment_type": "Cross-border"
            }
        }


class BatchTransactionRequest(BaseModel):
    """Batch of transactions for fraud detection"""
    transactions: List[Transaction] = Field(..., description="List of transactions to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "Time": "14:30:25",
                        "Date": "2022-10-07",
                        "Sender_account": 8724731955,
                        "Receiver_account": 2769355426,
                        "Amount": 1459.15,
                        "Payment_currency": "UK pounds",
                        "Received_currency": "UK pounds",
                        "Sender_bank_location": "UK",
                        "Receiver_bank_location": "UAE",
                        "Payment_type": "Cross-border"
                    },
                    {
                        "Time": "10:15:30",
                        "Date": "2022-10-07",
                        "Sender_account": 1234567890,
                        "Receiver_account": 9876543210,
                        "Amount": 25000.00,
                        "Payment_currency": "US Dollar",
                        "Received_currency": "Euro",
                        "Sender_bank_location": "USA",
                        "Receiver_bank_location": "Nigeria",
                        "Payment_type": "Cash Withdrawal"
                    }
                ]
            }
        }


# ============================================================
# RESPONSE MODELS
# ============================================================

class FraudDetectionResponse(BaseModel):
    """Detailed fraud detection response for a single transaction"""
    is_fraud: bool = Field(..., description="Whether the transaction is fraudulent")
    fraud_type: str = Field(..., description="Type of fraud detected (or 'Normal' if not fraud)")
    confidence: float = Field(..., ge=0, le=1, description="Confidence score for fraud type prediction")
    legitimacy_score: float = Field(..., ge=0, le=1, description="Legitimacy score from Stage 1")
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability from Stage 2")
    top_3_fraud_types: List[str] = Field(..., description="Top 3 most likely fraud types")
    top_3_probabilities: List[float] = Field(..., description="Probabilities for top 3 fraud types")
    risk_factors: List[str] = Field(..., description="Identified risk factors")
    recommendation: str = Field(..., description="Action recommendation: APPROVE, REVIEW, or BLOCK")
    stage: str = Field(..., description="Detection stage where decision was made")
    
    class Config:
        json_schema_extra = {
            "example": {
                "is_fraud": True,
                "fraud_type": "Smurfing",
                "confidence": 0.85,
                "legitimacy_score": 0.35,
                "fraud_probability": 0.92,
                "top_3_fraud_types": ["Smurfing", "Structuring", "Cash_Withdrawal"],
                "top_3_probabilities": [0.85, 0.10, 0.05],
                "risk_factors": [
                    "Cross-border transaction",
                    "High fraud probability",
                    "Flagged sender account"
                ],
                "recommendation": "BLOCK",
                "stage": "Stage 3: Type Identified"
            }
        }


class BatchFraudDetectionResponse(BaseModel):
    """Response for batch fraud detection"""
    results: List[FraudDetectionResponse] = Field(..., description="Detection results for each transaction")
    summary: dict = Field(..., description="Summary statistics for the batch")
    
    class Config:
        json_schema_extra = {
            "example": {
                "results": [
                    {
                        "is_fraud": False,
                        "fraud_type": "Normal",
                        "confidence": 0.0,
                        "legitimacy_score": 0.95,
                        "fraud_probability": 0.0,
                        "top_3_fraud_types": [],
                        "top_3_probabilities": [],
                        "risk_factors": [],
                        "recommendation": "APPROVE",
                        "stage": "Stage 1: Legitimate"
                    },
                    {
                        "is_fraud": True,
                        "fraud_type": "Cash_Withdrawal",
                        "confidence": 0.78,
                        "legitimacy_score": 0.25,
                        "fraud_probability": 0.89,
                        "top_3_fraud_types": ["Cash_Withdrawal", "Structuring", "Smurfing"],
                        "top_3_probabilities": [0.78, 0.15, 0.07],
                        "risk_factors": [
                            "High-risk country",
                            "High amount (>$10,000)",
                            "High fraud probability"
                        ],
                        "recommendation": "BLOCK",
                        "stage": "Stage 3: Type Identified"
                    }
                ],
                "summary": {
                    "total_transactions": 2,
                    "fraud_detected": 1,
                    "fraud_percentage": 50.0,
                    "recommendations": {
                        "APPROVE": 1,
                        "REVIEW": 0,
                        "BLOCK": 1
                    }
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether models are loaded")
    message: str = Field(..., description="Additional information")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "message": "Fraud detection system is ready"
            }
        }


class RootResponse(BaseModel):
    """Root endpoint response"""
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="API version")
    description: str = Field(..., description="Service description")
    endpoints: dict = Field(..., description="Available endpoints")
    
    class Config:
        json_schema_extra = {
            "example": {
                "service": "Fraud Detection API",
                "version": "1.0.0",
                "description": "Hierarchical 3-stage fraud detection system for cross-border transactions",
                "endpoints": {
                    "/": "API information",
                    "/docs": "Interactive API documentation (Swagger UI)",
                    "/redoc": "Alternative API documentation (ReDoc)",
                    "/health": "Health check endpoint",
                    "/predict": "Single transaction fraud detection",
                    "/predict_batch": "Batch transaction fraud detection"
                }
            }
        }

