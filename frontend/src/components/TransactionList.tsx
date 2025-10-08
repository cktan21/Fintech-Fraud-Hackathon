import { Transaction } from '@/types/transaction';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { AlertTriangle, CheckCircle, XCircle } from 'lucide-react';

interface TransactionListProps {
  transactions: Transaction[];
  onBlock: (id: number) => void;
  onApprove: (id: number) => void;
}

export const TransactionList = ({ transactions, onBlock, onApprove }: TransactionListProps) => {
  return (
    <div className="rounded-md border">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Date/Time</TableHead>
            <TableHead>Sender</TableHead>
            <TableHead>Receiver</TableHead>
            <TableHead>Amount</TableHead>
            <TableHead>Type</TableHead>
            <TableHead>Fraud Risk</TableHead>
            <TableHead>Status</TableHead>
            <TableHead>Actions</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {transactions.length === 0 ? (
            <TableRow>
              <TableCell colSpan={8} className="text-center text-muted-foreground">
                No transactions yet
              </TableCell>
            </TableRow>
          ) : (
            transactions.map((tx) => (
              <TableRow key={tx.id}>
                <TableCell className="whitespace-nowrap">
                  <div className="text-sm">{tx.date}</div>
                  <div className="text-xs text-muted-foreground">{tx.time}</div>
                </TableCell>
                <TableCell>
                  <div className="text-sm">{tx.sender_account}</div>
                  <div className="text-xs text-muted-foreground">{tx.sender_bank_location}</div>
                </TableCell>
                <TableCell>
                  <div className="text-sm">{tx.receiver_account}</div>
                  <div className="text-xs text-muted-foreground">{tx.receiver_bank_location}</div>
                </TableCell>
                <TableCell className="whitespace-nowrap">
                  {tx.amount.toLocaleString()} {tx.payment_currency}
                </TableCell>
                <TableCell>{tx.payment_type}</TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    {tx.is_fraudulent && <AlertTriangle className="h-4 w-4 text-destructive" />}
                    <Badge variant={tx.is_fraudulent ? 'destructive' : 'secondary'}>
                      {tx.fraud_score}%
                    </Badge>
                  </div>
                </TableCell>
                <TableCell>
                  <Badge
                    variant={
                      tx.status === 'approved'
                        ? 'default'
                        : tx.status === 'blocked'
                        ? 'destructive'
                        : 'secondary'
                    }
                  >
                    {tx.status}
                  </Badge>
                </TableCell>
                <TableCell>
                  {tx.status === 'pending' && (
                    <div className="flex gap-2">
                      <Button size="sm" variant="outline" onClick={() => onApprove(tx.id!)}>
                        <CheckCircle className="h-4 w-4" />
                      </Button>
                      <Button size="sm" variant="destructive" onClick={() => onBlock(tx.id!)}>
                        <XCircle className="h-4 w-4" />
                      </Button>
                    </div>
                  )}
                </TableCell>
              </TableRow>
            ))
          )}
        </TableBody>
      </Table>
    </div>
  );
};
