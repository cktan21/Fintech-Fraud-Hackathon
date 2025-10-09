import { analyzeSingleTransaction, analyzeBatchTransactions } from '@/services/fraudApi';

export const useFraudDetection = () => {
  const analyzeFraud = async (transaction: any) => {
    const result = await analyzeSingleTransaction(transaction);
    return {
      is_fraudulent: result.is_fraud,
      fraud_score: Math.round(result.fraud_probability * 100),
      fraud_type: result.fraud_type,
      confidence: result.confidence,
      legitimacy_score: result.legitimacy_score,
      fraud_probability: result.fraud_probability,
      top_3_fraud_types: result.top_3_fraud_types,
      top_3_probabilities: result.top_3_probabilities,
      risk_factors: result.risk_factors,
      recommendation: result.recommendation,
      stage: result.stage
    };
  };

  const analyzeBatch = async (transactions: any[]) => {
    const result = await analyzeBatchTransactions(transactions);
    return result.results.map((r, i) => ({
      ...transactions[i],
      is_fraudulent: r.is_fraud,
      fraud_score: Math.round(r.fraud_probability * 100),
      fraud_type: r.fraud_type,
      confidence: r.confidence,
      legitimacy_score: r.legitimacy_score,
      fraud_probability: r.fraud_probability,
      top_3_fraud_types: r.top_3_fraud_types,
      top_3_probabilities: r.top_3_probabilities,
      risk_factors: r.risk_factors,
      recommendation: r.recommendation,
      stage: r.stage
    }));
  };

  return { analyzeFraud, analyzeBatch };
};
