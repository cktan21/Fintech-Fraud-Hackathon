# Fraud Detection API

A production-ready FastAPI service that provides real-time fraud detection for cross-border financial transactions using a 3-stage hierarchical detection system.

## Features

- **3-Stage Hierarchical Detection**:
  1. Legitimate Transaction Profiler (rule-based, fast filtering)
  2. Binary Fraud Classifier (ML-based fraud detection)
  3. Fraud Type Classifier (identifies 17 specific fraud types)

- **Detailed Analysis**: Returns fraud probability, confidence scores, risk factors, and recommendations
- **Batch Processing**: Efficient processing of multiple transactions
- **Auto-Training**: Models train automatically on startup from the dataset
- **RESTful API**: Well-documented endpoints with automatic Swagger UI

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure data file exists**:
Place the `SAML-D.csv` dataset in `../../data/SAML-D.csv` (relative to this directory)

## Running the API

Start the server:
```bash
cd backend/FraudEngine
python main.py
```

Or with uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 5600
```

The API will:
- Train models on startup (takes ~30-60 seconds for 9.5M transactions)
- Start accepting requests once training is complete
- Be available at `http://localhost:5600`

## API Endpoints

### Root - `GET /`
Returns API information and available endpoints.

### Health Check - `GET /health`
Check if the service and models are ready.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "message": "Fraud detection system is ready"
}
```

### Single Prediction - `POST /predict`
Analyze a single transaction for fraud.

**Request Body**:
```json
{
  "Time": "14:30:25",
  "Date": "2022-10-07",
  "Sender_account": 8724731955,
  "Receiver_account": 2769355426,
  "Amount": 15000.00,
  "Payment_currency": "UK pounds",
  "Received_currency": "Dirham",
  "Sender_bank_location": "UK",
  "Receiver_bank_location": "UAE",
  "Payment_type": "Cross-border"
}
```

**Response**:
```json
{
  "is_fraud": true,
  "fraud_type": "Smurfing",
  "confidence": 0.85,
  "legitimacy_score": 0.35,
  "fraud_probability": 0.92,
  "top_3_fraud_types": ["Smurfing", "Structuring", "Cash_Withdrawal"],
  "top_3_probabilities": [0.85, 0.10, 0.05],
  "risk_factors": [
    "Cross-border transaction",
    "High fraud probability",
    "Currency conversion involved"
  ],
  "recommendation": "BLOCK",
  "stage": "Stage 3: Type Identified"
}
```

### Batch Prediction - `POST /predict_batch`
Analyze multiple transactions in one request (more efficient).

**Request Body**:
```json
{
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
      "Receiver_bank_location": "UK",
      "Payment_type": "Cash Deposit"
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
```

**Response**:
```json
{
  "results": [
    {
      "is_fraud": false,
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
      "is_fraud": true,
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
      "BLOCK": 1
    }
  }
}
```

## Interactive Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:5600/docs
- **ReDoc**: http://localhost:5600/redoc

## Fraud Types Detected

The system can identify 17 specific fraud types:
- Smurfing
- Structuring
- Cash_Withdrawal
- Deposit-Send
- Layered_Fan_In
- Layered_Fan_Out
- Stacked Bipartite
- Bipartite
- Cycle
- Fan_In
- Fan_Out
- Behavioural_Change_1
- Behavioural_Change_2
- Gather-Scatter
- Scatter-Gather
- Single_large
- Over-Invoicing

## Risk Factors Identified

The system analyzes and reports various risk factors:
- Cross-border transactions
- Currency conversion
- High-risk destination countries (Nigeria, Mexico, India)
- High transaction amounts (>$10,000)
- High-risk payment types (Cash Withdrawal, Cross-border)
- Low legitimacy scores
- High fraud probabilities
- Flagged sender/receiver accounts (recurring fraud patterns)

## Recommendations

Based on fraud probability, the system provides action recommendations:
- **APPROVE**: Low risk (fraud_probability ≤ 0.5)
- **REVIEW**: Medium risk (0.5 < fraud_probability ≤ 0.8)
- **BLOCK**: High risk (fraud_probability > 0.8)

## Architecture

### Training Pipeline (on startup)
1. Load CSV dataset (9.5M transactions)
2. Apply feature engineering (time, amount, risk features)
3. Train Legitimate Transaction Profiler (learns patterns from normal transactions)
4. Balance data (undersample normal, keep all fraud)
5. Train Binary Fraud Classifier (LightGBM + Logistic Regression + Isolation Forest)
6. Apply SMOTE to rare fraud types
7. Train Fraud Type Classifier (LightGBM + Random Forest)

### Prediction Pipeline
1. **Stage 1**: Legitimate Transaction Profiler filters ~94% of normal transactions
2. **Stage 2**: Binary Fraud Classifier analyzes suspicious transactions
3. **Stage 3**: Fraud Type Classifier identifies specific fraud type for detected fraud

## Performance

- **Training Time**: 30-60 seconds (9.5M transactions)
- **Prediction Latency**: <100ms per transaction
- **Throughput**: ~1.9M transactions/second (legitimacy profiler)
- **Memory**: ~500MB after training

## Example Usage with cURL

**Single prediction**:
```bash
curl -X POST "http://localhost:5600/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Time": "14:30:25",
    "Date": "2022-10-07",
    "Sender_account": 8724731955,
    "Receiver_account": 2769355426,
    "Amount": 15000.00,
    "Payment_currency": "UK pounds",
    "Received_currency": "Dirham",
    "Sender_bank_location": "UK",
    "Receiver_bank_location": "UAE",
    "Payment_type": "Cross-border"
  }'
```

**Health check**:
```bash
curl http://localhost:5600/health
```

## Development

To run in development mode with auto-reload (note: models will retrain on each reload):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 5600
```

## Troubleshooting

**Models not loading**:
- Check that `data/SAML-D.csv` exists in the correct location
- Check startup logs for errors
- Visit `/health` endpoint to check status

**Slow startup**:
- Normal for large datasets (9.5M transactions)
- Training takes 30-60 seconds
- Models are cached in memory after training

**Prediction errors**:
- Ensure all required fields are provided
- Check field types match the schema
- Visit `/docs` for field descriptions and examples

