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
import { useToast } from '@/hooks/use-toast';
import { Shield } from 'lucide-react';

const Dashboard = () => {
  const { isReady, addTransaction, getTransactions, updateTransactionStatus } = useDatabase();
  const { analyzeFraud } = useFraudDetection();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const { toast } = useToast();

  useEffect(() => {
    if (isReady) {
      refreshTransactions();
    }
  }, [isReady]);

  const refreshTransactions = () => {
    const txs = getTransactions();
    setTransactions(txs);
  };

  const handleAddTransaction = (txData: any) => {
    const { is_fraudulent, fraud_score } = analyzeFraud(txData);
    
    const transaction: Omit<Transaction, 'id'> = {
      ...txData,
      is_fraudulent,
      fraud_score,
      status: 'pending',
      created_at: new Date().toISOString()
    };

    addTransaction(transaction);
    refreshTransactions();

    if (is_fraudulent) {
      toast({
        title: '⚠️ Fraudulent Transaction Detected',
        description: `Transaction flagged with ${fraud_score}% fraud score. Review required.`,
        variant: 'destructive'
      });
    } else {
      toast({
        title: 'Transaction Added',
        description: `Transaction added with ${fraud_score}% fraud score`
      });
    }
  };

  const handleBulkUpload = (txs: any[]) => {
    txs.forEach(tx => handleAddTransaction(tx));
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
                <div className="flex items-center justify-between">
                  <CardTitle>Transaction Management</CardTitle>
                  <div className="flex gap-2">
                    <TransactionUpload onUpload={handleBulkUpload} />
                    <ManualTransactionDialog onAdd={handleAddTransaction} />
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <TransactionList
                  transactions={transactions}
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
