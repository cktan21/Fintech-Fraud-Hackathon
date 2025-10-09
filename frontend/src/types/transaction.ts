export interface Transaction {
  id?: number;
  time: string;
  date: string;
  sender_account: string;
  receiver_account: string;
  amount: number;
  payment_currency: string;
  received_currency: string;
  sender_bank_location: string;
  receiver_bank_location: string;
  payment_type: string;
  is_fraudulent: boolean;
  fraud_score: number;
  status: 'pending' | 'approved' | 'blocked';
  created_at: string;
  // API response fields
  fraud_type?: string;
  confidence?: number;
  legitimacy_score?: number;
  fraud_probability?: number;
  top_3_fraud_types?: string[];
  top_3_probabilities?: number[];
  risk_factors?: string[];
  recommendation?: 'APPROVE' | 'BLOCK' | 'REVIEW';
  stage?: string;
}

export interface TransactionStats {
  total: number;
  fraudulent: number;
  blocked: number;
  approved: number;
  totalAmount: number;
  avgAmount: number;
  fraudRate: number;
}
