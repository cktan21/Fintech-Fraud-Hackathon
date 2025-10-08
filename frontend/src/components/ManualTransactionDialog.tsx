import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { PlusCircle } from 'lucide-react';

interface ManualTransactionDialogProps {
  onAdd: (transaction: any) => void;
}

export const ManualTransactionDialog = ({ onAdd }: ManualTransactionDialogProps) => {
  const [open, setOpen] = useState(false);
  const [formData, setFormData] = useState({
    time: new Date().toLocaleTimeString(),
    date: new Date().toISOString().split('T')[0],
    sender_account: '',
    receiver_account: '',
    amount: '',
    payment_currency: 'USD',
    received_currency: 'USD',
    sender_bank_location: '',
    receiver_bank_location: '',
    payment_type: 'Bank Transfer'
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onAdd({
      ...formData,
      amount: parseFloat(formData.amount)
    });
    setOpen(false);
    setFormData({
      time: new Date().toLocaleTimeString(),
      date: new Date().toISOString().split('T')[0],
      sender_account: '',
      receiver_account: '',
      amount: '',
      payment_currency: 'USD',
      received_currency: 'USD',
      sender_bank_location: '',
      receiver_bank_location: '',
      payment_type: 'Bank Transfer'
    });
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>
          <PlusCircle className="mr-2 h-4 w-4" />
          Add Transaction
        </Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Add Manual Transaction</DialogTitle>
        </DialogHeader>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="date">Date</Label>
              <Input
                id="date"
                type="date"
                value={formData.date}
                onChange={(e) => setFormData({ ...formData, date: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="time">Time</Label>
              <Input
                id="time"
                type="time"
                value={formData.time}
                onChange={(e) => setFormData({ ...formData, time: e.target.value })}
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="sender_account">Sender Account</Label>
              <Input
                id="sender_account"
                value={formData.sender_account}
                onChange={(e) => setFormData({ ...formData, sender_account: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="sender_bank_location">Sender Bank Location</Label>
              <Input
                id="sender_bank_location"
                value={formData.sender_bank_location}
                onChange={(e) => setFormData({ ...formData, sender_bank_location: e.target.value })}
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <Label htmlFor="receiver_account">Receiver Account</Label>
              <Input
                id="receiver_account"
                value={formData.receiver_account}
                onChange={(e) => setFormData({ ...formData, receiver_account: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="receiver_bank_location">Receiver Bank Location</Label>
              <Input
                id="receiver_bank_location"
                value={formData.receiver_bank_location}
                onChange={(e) => setFormData({ ...formData, receiver_bank_location: e.target.value })}
                required
              />
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div className="space-y-2">
              <Label htmlFor="amount">Amount</Label>
              <Input
                id="amount"
                type="number"
                step="0.01"
                value={formData.amount}
                onChange={(e) => setFormData({ ...formData, amount: e.target.value })}
                required
              />
            </div>
            <div className="space-y-2">
              <Label htmlFor="payment_currency">Payment Currency</Label>
              <Select
                value={formData.payment_currency}
                onValueChange={(value) => setFormData({ ...formData, payment_currency: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="USD">USD</SelectItem>
                  <SelectItem value="EUR">EUR</SelectItem>
                  <SelectItem value="GBP">GBP</SelectItem>
                  <SelectItem value="JPY">JPY</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="received_currency">Received Currency</Label>
              <Select
                value={formData.received_currency}
                onValueChange={(value) => setFormData({ ...formData, received_currency: value })}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="USD">USD</SelectItem>
                  <SelectItem value="EUR">EUR</SelectItem>
                  <SelectItem value="GBP">GBP</SelectItem>
                  <SelectItem value="JPY">JPY</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="space-y-2">
            <Label htmlFor="payment_type">Payment Type</Label>
            <Select
              value={formData.payment_type}
              onValueChange={(value) => setFormData({ ...formData, payment_type: value })}
            >
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="Bank Transfer">Bank Transfer</SelectItem>
                <SelectItem value="Wire Transfer">Wire Transfer</SelectItem>
                <SelectItem value="Credit Card">Credit Card</SelectItem>
                <SelectItem value="Cash">Cash</SelectItem>
                <SelectItem value="Check">Check</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <Button type="submit" className="w-full">Add Transaction</Button>
        </form>
      </DialogContent>
    </Dialog>
  );
};
