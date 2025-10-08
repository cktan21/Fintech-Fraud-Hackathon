import { Transaction, TransactionStats } from '@/types/transaction';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { AlertTriangle, CheckCircle, DollarSign, TrendingUp, XCircle } from 'lucide-react';

interface StatsTabProps {
  transactions: Transaction[];
}

export const StatsTab = ({ transactions }: StatsTabProps) => {
  const stats: TransactionStats = {
    total: transactions.length,
    fraudulent: transactions.filter(t => t.is_fraudulent).length,
    blocked: transactions.filter(t => t.status === 'blocked').length,
    approved: transactions.filter(t => t.status === 'approved').length,
    totalAmount: transactions.reduce((sum, t) => sum + t.amount, 0),
    avgAmount: transactions.length > 0 ? transactions.reduce((sum, t) => sum + t.amount, 0) / transactions.length : 0,
    fraudRate: transactions.length > 0 ? (transactions.filter(t => t.is_fraudulent).length / transactions.length) * 100 : 0
  };

  const topLocations = transactions.reduce((acc, t) => {
    acc[t.sender_bank_location] = (acc[t.sender_bank_location] || 0) + 1;
    return acc;
  }, {} as Record<string, number>);

  const topLocationsList = Object.entries(topLocations)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Transactions</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.total}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Fraudulent</CardTitle>
            <AlertTriangle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.fraudulent}</div>
            <p className="text-xs text-muted-foreground">
              {stats.fraudRate.toFixed(1)}% fraud rate
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Blocked</CardTitle>
            <XCircle className="h-4 w-4 text-destructive" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.blocked}</div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Approved</CardTitle>
            <CheckCircle className="h-4 w-4 text-primary" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.approved}</div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Financial Overview</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Total Amount</span>
              <span className="text-lg font-bold">${stats.totalAmount.toLocaleString()}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-muted-foreground">Average Transaction</span>
              <span className="text-lg font-bold">${stats.avgAmount.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Top Sender Locations</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            {topLocationsList.length > 0 ? (
              topLocationsList.map(([location, count]) => (
                <div key={location} className="flex items-center justify-between">
                  <span className="text-sm">{location}</span>
                  <span className="text-sm font-semibold">{count}</span>
                </div>
              ))
            ) : (
              <p className="text-sm text-muted-foreground">No data available</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
};
