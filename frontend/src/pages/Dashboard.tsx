import { useEffect, useState } from 'react';
import { useDatabase } from '@/hooks/useDatabase';
import { useFraudDetection } from '@/hooks/useFraudDetection';
import { Transaction } from '@/types/transaction';
import { TransactionList } from '@/components/TransactionList';
import { TransactionUpload } from '@/components/TransactionUpload';
import { ManualTransactionDialog } from '@/components/ManualTransactionDialog';
import { StatsTab } from '@/components/StatsTab';
import { ReportGenerator } from '@/components/ReportGenerator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { useToast } from '@/hooks/use-toast';
import { Shield } from 'lucide-react';

const Dashboard = () => {
  const { isReady, addTransaction, getTransactions, updateTransactionStatus } = useDatabase();
  const { analyzeFraud, analyzeBatch } = useFraudDetection();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [filteredTransactions, setFilteredTransactions] = useState<Transaction[]>([]);
  const [paymentTypeFilter, setPaymentTypeFilter] = useState<string>('all');
  const [statusFilter, setStatusFilter] = useState<string>('all');
  const { toast } = useToast();

  useEffect(() => {
    if (isReady) {
      refreshTransactions();
    }
  }, [isReady]);

  useEffect(() => {
    let filtered = transactions;
    
    if (paymentTypeFilter !== 'all') {
      filtered = filtered.filter(t => t.payment_type === paymentTypeFilter);
    }
    
    if (statusFilter !== 'all') {
      filtered = filtered.filter(t => t.status === statusFilter);
    }
    
    setFilteredTransactions(filtered);
  }, [transactions, paymentTypeFilter, statusFilter]);

  const refreshTransactions = () => {
    const txs = getTransactions();
    setTransactions(txs);
  };

  const handleAddTransaction = async (txData: any) => {
    const analysisResult = await analyzeFraud(txData);
    
    const status = analysisResult.recommendation === 'APPROVE' ? 'approved' : 'pending';
    
    const transaction: Omit<Transaction, 'id'> = {
      ...txData,
      is_fraudulent: analysisResult.is_fraudulent,
      fraud_score: analysisResult.fraud_score,
      fraud_type: analysisResult.fraud_type,
      confidence: analysisResult.confidence,
      legitimacy_score: analysisResult.legitimacy_score,
      fraud_probability: analysisResult.fraud_probability,
      top_3_fraud_types: analysisResult.top_3_fraud_types,
      top_3_probabilities: analysisResult.top_3_probabilities,
      risk_factors: analysisResult.risk_factors,
      recommendation: analysisResult.recommendation,
      stage: analysisResult.stage,
      status,
      created_at: new Date().toISOString()
    };

    addTransaction(transaction);
    refreshTransactions();

    if (status === 'approved') {
      toast({
        title: '✓ Transaction Approved',
        description: `Transaction automatically approved (${analysisResult.fraud_score}% fraud score)`
      });
    } else {
      toast({
        title: '⚠️ Review Required',
        description: `Transaction flagged for review. Recommendation: ${analysisResult.recommendation}`,
        variant: 'destructive'
      });
    }
  };

  const handleBulkUpload = async (txs: any[]) => {
    toast({
      title: 'Processing Batch',
      description: `Analyzing ${txs.length} transactions...`
    });

    const analysisResults = await analyzeBatch(txs);
    
    analysisResults.forEach((result: any) => {
      const status = result.recommendation === 'APPROVE' ? 'approved' : 'pending';
      
      const transaction: Omit<Transaction, 'id'> = {
        ...result,
        status,
        created_at: new Date().toISOString()
      };
      
      addTransaction(transaction);
    });
    
    refreshTransactions();

    const approvedCount = analysisResults.filter((r: any) => r.recommendation === 'APPROVE').length;
    const reviewCount = analysisResults.length - approvedCount;

    toast({
      title: 'Batch Processing Complete',
      description: `${approvedCount} approved, ${reviewCount} pending review`
    });
  };

  const handleBlockTransaction = (id: number) => {
    updateTransactionStatus(id, 'blocked');
    refreshTransactions();
    toast({
      title: 'Transaction Blocked',
      description: 'The transaction has been blocked successfully'
    });
  };

  const handleApproveTransaction = (id: number) => {
    updateTransactionStatus(id, 'approved');
    refreshTransactions();
    toast({
      title: 'Transaction Approved',
      description: 'The transaction has been approved'
    });
  };

  if (!isReady) {
    return (
      <div className="flex min-h-screen items-center justify-center">
        <div className="text-center">
          <Shield className="mx-auto h-12 w-12 animate-pulse text-primary" />
          <p className="mt-4 text-lg">Initializing Fraud Detection System...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b">
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center gap-3">
            <Shield className="h-8 w-8 text-primary" />
            <div>
              <h1 className="text-3xl font-bold">Fraud Detection Dashboard</h1>
              <p className="text-muted-foreground">Real-time transaction monitoring and fraud analysis</p>
            </div>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        <Tabs defaultValue="transactions" className="space-y-6">
          <TabsList>
            <TabsTrigger value="transactions">Transactions</TabsTrigger>
            <TabsTrigger value="stats">Statistics</TabsTrigger>
            <TabsTrigger value="reports">Reports</TabsTrigger>
          </TabsList>

          <TabsContent value="transactions" className="space-y-6">
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between flex-wrap gap-4">
                  <CardTitle>Transaction Management</CardTitle>
                  <div className="flex gap-2">
                    <TransactionUpload onUpload={handleBulkUpload} />
                    <ManualTransactionDialog onAdd={handleAddTransaction} />
                  </div>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="flex gap-4 flex-wrap">
                  <div className="flex-1 min-w-[200px]">
                    <Select value={paymentTypeFilter} onValueChange={setPaymentTypeFilter}>
                      <SelectTrigger>
                        <SelectValue placeholder="Filter by Payment Type" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Payment Types</SelectItem>
                        <SelectItem value="Cash Deposit">Cash Deposit</SelectItem>
                        <SelectItem value="Cash Withdrawal">Cash Withdrawal</SelectItem>
                        <SelectItem value="Wire Transfer">Wire Transfer</SelectItem>
                        <SelectItem value="Cross-border">Cross-border</SelectItem>
                        <SelectItem value="Credit Card">Credit Card</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex-1 min-w-[200px]">
                    <Select value={statusFilter} onValueChange={setStatusFilter}>
                      <SelectTrigger>
                        <SelectValue placeholder="Filter by Status" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="all">All Statuses</SelectItem>
                        <SelectItem value="approved">Approved</SelectItem>
                        <SelectItem value="pending">Pending</SelectItem>
                        <SelectItem value="blocked">Blocked</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
                <TransactionList
                  transactions={filteredTransactions}
                  onBlock={handleBlockTransaction}
                  onApprove={handleApproveTransaction}
                />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="stats">
            <StatsTab transactions={transactions} />
          </TabsContent>

          <TabsContent value="reports">
            <ReportGenerator transactions={transactions} />
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
};

export default Dashboard;
