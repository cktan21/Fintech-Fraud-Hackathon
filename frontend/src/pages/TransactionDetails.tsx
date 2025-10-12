import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { useDatabase } from '@/hooks/useDatabase';
import { Transaction } from '@/types/transaction';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { useToast } from '@/hooks/use-toast';
import { ArrowLeft, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

const TransactionDetails = () => {
    const { id } = useParams();
    const navigate = useNavigate();
    const { isReady, getTransactions, updateTransactionStatus } = useDatabase();
    const [transaction, setTransaction] = useState<Transaction | null>(null);
    const { toast } = useToast();

    useEffect(() => {
        if (!isReady) return;

        const txs = getTransactions();
        const found = txs.find(t => t.id === Number(id));
        setTransaction(found || null);
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [id, isReady]);

    const handleApprove = () => {
        if (transaction?.id) {
            updateTransactionStatus(transaction.id, 'approved');
            toast({
                title: 'Transaction Approved',
                description: 'The transaction has been approved'
            });
            navigate('/');
        }
    };

    const handleBlock = () => {
        if (transaction?.id) {
            updateTransactionStatus(transaction.id, 'blocked');
            toast({
                title: 'Transaction Blocked',
                description: 'The transaction has been blocked'
            });
            navigate('/');
        }
    };

    if (!isReady) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <Card>
                    <CardContent className="pt-6">
                        <p className="text-muted-foreground">Loading transaction...</p>
                    </CardContent>
                </Card>
            </div>
        );
    }

    if (!transaction) {
        return (
            <div className="min-h-screen bg-background flex items-center justify-center">
                <Card>
                    <CardContent className="pt-6">
                        <p className="text-muted-foreground">Transaction not found</p>
                    </CardContent>
                </Card>
            </div>
        );
    }

    const getFraudBadge = () => {
        if (transaction.fraud_probability && transaction.fraud_probability > 0.8) {
            return <Badge variant="destructive">High Risk</Badge>;
        } else if (transaction.fraud_probability && transaction.fraud_probability > 0.5) {
            return <Badge className="bg-yellow-500">Medium Risk</Badge>;
        }
        return <Badge className="bg-green-500">Low Risk</Badge>;
    };

    return (
        <div className="min-h-screen bg-background">
            <header className="border-b">
                <div className="container mx-auto px-4 py-4">
                    <Button variant="ghost" onClick={() => navigate('/')}>
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Back to Dashboard
                    </Button>
                </div>
            </header>

            <main className="container mx-auto px-4 py-8">
                <div className="max-w-4xl mx-auto space-y-6">
                    <div className="flex items-center justify-between">
                        <h1 className="text-3xl font-bold">Transaction Details</h1>
                        {getFraudBadge()}
                    </div>

                    <Card>
                        <CardHeader>
                            <CardTitle>Transaction Information</CardTitle>
                        </CardHeader>
                        <CardContent className="space-y-4">
                            <div className="grid grid-cols-2 gap-4">
                                <div>
                                    <p className="text-sm text-muted-foreground">Date & Time</p>
                                    <p className="font-medium">{transaction.date} {transaction.time}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Amount</p>
                                    <p className="font-medium text-lg">{transaction.payment_currency} {transaction.amount.toLocaleString()}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Sender Account</p>
                                    <p className="font-medium">{transaction.sender_account}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Receiver Account</p>
                                    <p className="font-medium">{transaction.receiver_account}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Sender Location</p>
                                    <p className="font-medium">{transaction.sender_bank_location}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Receiver Location</p>
                                    <p className="font-medium">{transaction.receiver_bank_location}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Payment Type</p>
                                    <p className="font-medium">{transaction.payment_type}</p>
                                </div>
                                <div>
                                    <p className="text-sm text-muted-foreground">Status</p>
                                    <Badge variant={transaction.status === 'approved' ? 'default' : transaction.status === 'blocked' ? 'destructive' : 'outline'}>
                                        {transaction.status.toUpperCase()}
                                    </Badge>
                                </div>
                            </div>
                        </CardContent>
                    </Card>

                    {transaction.fraud_type && (
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-destructive" />
                                    Fraud Analysis
                                </CardTitle>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="grid grid-cols-2 gap-4">
                                    <div>
                                        <p className="text-sm text-muted-foreground">Fraud Type</p>
                                        <p className="font-medium">{transaction.fraud_type}</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground">Confidence</p>
                                        <p className="font-medium">{((transaction.confidence || 0) * 100).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground">Legitimacy Score</p>
                                        <p className="font-medium">{((transaction.legitimacy_score || 0) * 100).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground">Fraud Probability</p>
                                        <p className="font-medium">{((transaction.fraud_probability || 0) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>

                                {transaction.top_3_fraud_types && transaction.top_3_fraud_types.length > 0 && (
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-2">Top 3 Fraud Types</p>
                                        <div className="space-y-2">
                                            {transaction.top_3_fraud_types.map((type, i) => (
                                                <div key={i} className="flex justify-between items-center">
                                                    <span className="font-medium">{type}</span>
                                                    <span className="text-sm text-muted-foreground">
                                                        {((transaction.top_3_probabilities?.[i] || 0) * 100).toFixed(1)}%
                                                    </span>
                                                </div>
                                            ))}
                                        </div>
                                    </div>
                                )}

                                {transaction.risk_factors && transaction.risk_factors.length > 0 && (
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-2">Risk Factors</p>
                                        <ul className="space-y-1">
                                            {transaction.risk_factors.map((factor, i) => (
                                                <li key={i} className="flex items-start gap-2">
                                                    <span className="text-destructive mt-1">•</span>
                                                    <span>{factor}</span>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                )}

                                {transaction.recommendation && (
                                    <div>
                                        <p className="text-sm text-muted-foreground mb-2">Recommendation</p>
                                        <Badge
                                            variant={
                                                transaction.recommendation === 'APPROVE' ? 'default' :
                                                    transaction.recommendation === 'BLOCK' ? 'destructive' :
                                                        'outline'
                                            }
                                            className="text-base px-4 py-1"
                                        >
                                            {transaction.recommendation}
                                        </Badge>
                                    </div>
                                )}
                            </CardContent>
                        </Card>
                    )}

                    {transaction.status === 'pending' && (
                        <Card>
                            <CardContent className="pt-6">
                                <div className="flex gap-4 justify-end">
                                    <Button
                                        variant="outline"
                                        size="lg"
                                        onClick={handleApprove}
                                        className="gap-2"
                                    >
                                        <CheckCircle className="h-4 w-4" />
                                        Approve Transaction
                                    </Button>
                                    <Button
                                        variant="destructive"
                                        size="lg"
                                        onClick={handleBlock}
                                        className="gap-2"
                                    >
                                        <XCircle className="h-4 w-4" />
                                        Block Transaction
                                    </Button>
                                </div>
                            </CardContent>
                        </Card>
                    )}
                </div>
            </main>
        </div>
    );
};

export default TransactionDetails;
