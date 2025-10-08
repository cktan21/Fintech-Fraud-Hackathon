import { Transaction } from '@/types/transaction';

export const useFraudDetection = () => {
  const analyzeFraud = (transaction: Omit<Transaction, 'id' | 'is_fraudulent' | 'fraud_score' | 'status' | 'created_at'>): { is_fraudulent: boolean; fraud_score: number } => {
    let score = 0;
    
    // High amount transactions are riskier
    if (transaction.amount > 10000) score += 30;
    else if (transaction.amount > 5000) score += 15;
    
    // Cross-border transactions
    if (transaction.sender_bank_location !== transaction.receiver_bank_location) {
      score += 20;
    }
    
    // Currency mismatch
    if (transaction.payment_currency !== transaction.received_currency) {
      score += 15;
    }
    
    // Specific payment types
    if (transaction.payment_type === 'Wire Transfer') score += 10;
    if (transaction.payment_type === 'Cash') score += 25;
    
    // Round numbers are suspicious
    if (transaction.amount % 1000 === 0) score += 10;
    
    // Add some randomness to simulate ML model
    score += Math.random() * 20;
    
    const fraud_score = Math.min(Math.round(score), 100);
    const is_fraudulent = fraud_score > 60;
    
    return { is_fraudulent, fraud_score };
  };

  return { analyzeFraud };
};
