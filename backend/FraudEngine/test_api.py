"""
Test script for Fraud Detection API
Run this after starting the API server to verify it works correctly
"""

import requests
import json

BASE_URL = "http://localhost:8000"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*70)
    print("Testing Health Check Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def test_single_prediction():
    """Test single transaction prediction"""
    print("\n" + "="*70)
    print("Testing Single Transaction Prediction")
    print("="*70)
    
    # Normal transaction
    normal_transaction = {
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
    }
    
    print("\n📝 Request (Normal Transaction):")
    print(json.dumps(normal_transaction, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=normal_transaction)
    print(f"\nStatus Code: {response.status_code}")
    print(f"\n📊 Response:")
    print(json.dumps(response.json(), indent=2))
    
    # Suspicious transaction
    suspicious_transaction = {
        "Time": "02:30:00",
        "Date": "2022-10-07",
        "Sender_account": 1234567890,
        "Receiver_account": 9876543210,
        "Amount": 25000.00,
        "Payment_currency": "US Dollar",
        "Received_currency": "Naira",
        "Sender_bank_location": "USA",
        "Receiver_bank_location": "Nigeria",
        "Payment_type": "Cash Withdrawal"
    }
    
    print("\n" + "-"*70)
    print("\n📝 Request (Suspicious Transaction):")
    print(json.dumps(suspicious_transaction, indent=2))
    
    response = requests.post(f"{BASE_URL}/predict", json=suspicious_transaction)
    print(f"\nStatus Code: {response.status_code}")
    print(f"\n📊 Response:")
    print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200


def test_batch_prediction():
    """Test batch transaction prediction"""
    print("\n" + "="*70)
    print("Testing Batch Transaction Prediction")
    print("="*70)
    
    batch_request = {
        "transactions": [
            {
                "Time": "10:00:00",
                "Date": "2022-10-07",
                "Sender_account": 1111111111,
                "Receiver_account": 2222222222,
                "Amount": 500.00,
                "Payment_currency": "UK pounds",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "UK",
                "Receiver_bank_location": "UK",
                "Payment_type": "Debit card"
            },
            {
                "Time": "14:30:25",
                "Date": "2022-10-07",
                "Sender_account": 3333333333,
                "Receiver_account": 4444444444,
                "Amount": 8500.00,
                "Payment_currency": "Euro",
                "Received_currency": "UK pounds",
                "Sender_bank_location": "Germany",
                "Receiver_bank_location": "UK",
                "Payment_type": "Cross-border"
            },
            {
                "Time": "03:00:00",
                "Date": "2022-10-07",
                "Sender_account": 5555555555,
                "Receiver_account": 6666666666,
                "Amount": 45000.00,
                "Payment_currency": "US Dollar",
                "Received_currency": "Peso",
                "Sender_bank_location": "USA",
                "Receiver_bank_location": "Mexico",
                "Payment_type": "Cash Withdrawal"
            }
        ]
    }
    
    print("\n📝 Request (3 transactions):")
    print(f"  Transaction 1: £500 UK → UK (Debit card)")
    print(f"  Transaction 2: €8,500 Germany → UK (Cross-border)")
    print(f"  Transaction 3: $45,000 USA → Mexico (Cash Withdrawal)")
    
    response = requests.post(f"{BASE_URL}/predict_batch", json=batch_request)
    print(f"\nStatus Code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print(f"\n📊 Summary:")
        print(json.dumps(result['summary'], indent=2))
        
        print(f"\n📊 Individual Results:")
        for i, res in enumerate(result['results'], 1):
            print(f"\n  Transaction {i}:")
            print(f"    Is Fraud: {res['is_fraud']}")
            print(f"    Fraud Type: {res['fraud_type']}")
            print(f"    Confidence: {res['confidence']:.3f}")
            print(f"    Recommendation: {res['recommendation']}")
            print(f"    Risk Factors: {', '.join(res['risk_factors']) if res['risk_factors'] else 'None'}")
    else:
        print(f"\n❌ Error:")
        print(json.dumps(response.json(), indent=2))
    
    return response.status_code == 200


def test_root():
    """Test root endpoint"""
    print("\n" + "="*70)
    print("Testing Root Endpoint")
    print("="*70)
    
    response = requests.get(f"{BASE_URL}/")
    print(f"Status Code: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    
    return response.status_code == 200


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 FRAUD DETECTION API TEST SUITE")
    print("="*70)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the API server is running!")
    
    try:
        # Run tests
        test_results = {
            "Root Endpoint": test_root(),
            "Health Check": test_health(),
            "Single Prediction": test_single_prediction(),
            "Batch Prediction": test_batch_prediction()
        }
        
        # Print summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        for test_name, passed in test_results.items():
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"{test_name:25s}: {status}")
        
        all_passed = all(test_results.values())
        print("\n" + "="*70)
        if all_passed:
            print("🎉 ALL TESTS PASSED!")
        else:
            print("⚠️  SOME TESTS FAILED")
        print("="*70 + "\n")
        
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Could not connect to API server")
        print(f"   Make sure the server is running at {BASE_URL}")
        print("   Start it with: python main.py\n")
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")


if __name__ == "__main__":
    main()

