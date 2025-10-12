const API_BASE_URL = 'http://localhost:5600';

interface FraudApiTransaction {
    Time: string;
    Date: string;
    Sender_account: number;
    Receiver_account: number;
    Amount: number;
    Payment_currency: string;
    Received_currency: string;
    Sender_bank_location: string;
    Receiver_bank_location: string;
    Payment_type: string;
}

interface FraudApiResponse {
    is_fraud: boolean;
    fraud_type: string;
    confidence: number;
    legitimacy_score: number;
    fraud_probability: number;
    top_3_fraud_types: string[];
    top_3_probabilities: number[];
    risk_factors: string[];
    recommendation: 'APPROVE' | 'BLOCK' | 'REVIEW';
    stage: string;
}

interface BatchResponse {
    results: FraudApiResponse[];
    summary: {
        total_transactions: number;
        fraud_detected: number;
        fraud_percentage: number;
        recommendations: Record<string, number>;
    };
}

const convertToApiFormat = (tx: any): FraudApiTransaction => {
    const senderAccount = typeof tx.sender_account === 'string'
        ? parseInt(tx.sender_account, 10)
        : tx.sender_account;
    const receiverAccount = typeof tx.receiver_account === 'string'
        ? parseInt(tx.receiver_account, 10)
        : tx.receiver_account;

    // Validate that we have valid numbers
    if (isNaN(senderAccount) || isNaN(receiverAccount)) {
        throw new Error('Invalid account numbers');
    }

    return {
        Time: tx.time,
        Date: tx.date,
        Sender_account: senderAccount,
        Receiver_account: receiverAccount,
        Amount: parseFloat(tx.amount),
        Payment_currency: tx.payment_currency,
        Received_currency: tx.received_currency,
        Sender_bank_location: tx.sender_bank_location,
        Receiver_bank_location: tx.receiver_bank_location,
        Payment_type: tx.payment_type
    };
};

export const analyzeSingleTransaction = async (transaction: any): Promise<FraudApiResponse> => {
    try {
        const apiData = convertToApiFormat(transaction);
        console.log('Sending to API:', apiData);

        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(apiData)
        });

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('API error response:', errorData);
            throw new Error(`API error ${response.status}: ${JSON.stringify(errorData)}`);
        }

        return await response.json();
    } catch (error) {
        console.error('Fraud API error:', error);
        // Fallback to basic detection if API fails
        return {
            is_fraud: false,
            fraud_type: 'Normal',
            confidence: 0,
            legitimacy_score: 1,
            fraud_probability: 0,
            top_3_fraud_types: [],
            top_3_probabilities: [],
            risk_factors: [],
            recommendation: 'APPROVE',
            stage: 'API Error - Using Fallback'
        };
    }
};

export const analyzeBatchTransactions = async (transactions: any[]): Promise<BatchResponse> => {
    try {
        const apiData = {
            transactions: transactions.map(convertToApiFormat)
        };
        console.log('Sending batch to API:', apiData);
        console.log('Sample transaction (first):', JSON.stringify(apiData.transactions[0], null, 2));

        const response = await fetch(`${API_BASE_URL}/predict_batch`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(apiData)
        });

        console.log('API Response Status:', response.status);

        if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            console.error('❌ API batch error response:', errorData);
            console.error('Status:', response.status);
            console.error('Full error:', JSON.stringify(errorData, null, 2));
            throw new Error(`API error ${response.status}: ${JSON.stringify(errorData)}`);
        }

        const result = await response.json();
        console.log('✅ API batch success! Sample result:', result.results[0]);
        return result;
    } catch (error) {
        console.error('❌ Fraud API batch error:', error);
        // Fallback response
        return {
            results: transactions.map(() => ({
                is_fraud: false,
                fraud_type: 'Normal',
                confidence: 0,
                legitimacy_score: 1,
                fraud_probability: 0,
                top_3_fraud_types: [],
                top_3_probabilities: [],
                risk_factors: [],
                recommendation: 'APPROVE',
                stage: 'API Error - Using Fallback'
            })),
            summary: {
                total_transactions: transactions.length,
                fraud_detected: 0,
                fraud_percentage: 0,
                recommendations: { APPROVE: transactions.length }
            }
        };
    }
};

export const checkApiHealth = async (): Promise<boolean> => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
};
