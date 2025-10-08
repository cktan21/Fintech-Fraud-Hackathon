import { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface TransactionUploadProps {
  onUpload: (transactions: any[]) => void;
}

export const TransactionUpload = ({ onUpload }: TransactionUploadProps) => {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (e) => {
      const text = e.target?.result as string;
      const lines = text.split('\n').filter(line => line.trim());
      
      if (lines.length < 2) {
        toast({
          title: 'Invalid CSV',
          description: 'CSV file is empty or invalid',
          variant: 'destructive'
        });
        return;
      }

      const headers = lines[0].split(',').map(h => h.trim());
      const transactions = [];

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',').map(v => v.trim());
        const transaction: any = {};
        
        headers.forEach((header, index) => {
          const key = header.toLowerCase().replace(/ /g, '_');
          transaction[key] = values[index];
        });

        transaction.amount = parseFloat(transaction.amount);
        transactions.push(transaction);
      }

      onUpload(transactions);
      toast({
        title: 'Success',
        description: `Uploaded ${transactions.length} transactions`
      });
    };

    reader.readAsText(file);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  return (
    <div>
      <input
        ref={fileInputRef}
        type="file"
        accept=".csv"
        onChange={handleFileUpload}
        className="hidden"
      />
      <Button onClick={() => fileInputRef.current?.click()}>
        <Upload className="mr-2 h-4 w-4" />
        Upload CSV
      </Button>
    </div>
  );
};
