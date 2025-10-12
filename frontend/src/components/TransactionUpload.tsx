import { useRef } from 'react';
import { Button } from '@/components/ui/button';
import { Upload } from 'lucide-react';
import { useToast } from '@/hooks/use-toast';

interface ParsedTransaction {
    time?: string;
    date?: string;
    sender_account?: number;
    receiver_account?: number;
    amount?: number;
    payment_currency?: string;
    received_currency?: string;
    sender_bank_location?: string;
    receiver_bank_location?: string;
    payment_type?: string;
    [key: string]: string | number | undefined;
}

interface TransactionUploadProps {
    onUpload: (transactions: ParsedTransaction[]) => void;
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
            const transactions: ParsedTransaction[] = [];

            for (let i = 1; i < lines.length; i++) {
                const values = lines[i].split(',').map(v => v.trim());
                const transaction: ParsedTransaction = {};

                headers.forEach((header, index) => {
                    let key = header.toLowerCase().replace(/ /g, '_');
                    // Handle common CSV header variations
                    if (key === 'sender_ac' || key === 'sender_account') key = 'sender_account';
                    if (key === 'receiver_ac' || key === 'receiver_account') key = 'receiver_account';
                    if (key === 'payment_currency' || key === 'payment_c') key = 'payment_currency';
                    if (key === 'received_currency' || key === 'received_c') key = 'received_currency';
                    if (key === 'sender_bank_location' || key === 'sender_ba') key = 'sender_bank_location';
                    if (key === 'receiver_bank_location' || key === 'receiver_ba') key = 'receiver_bank_location';
                    if (key === 'payment_type' || key === 'payment_t') key = 'payment_type';

                    transaction[key] = values[index];
                });

                // Parse numeric fields properly (handles scientific notation like 1.15E+08)
                if (typeof transaction.sender_account === 'string') {
                    transaction.sender_account = Math.round(parseFloat(transaction.sender_account));
                }
                if (typeof transaction.receiver_account === 'string') {
                    transaction.receiver_account = Math.round(parseFloat(transaction.receiver_account));
                }
                if (typeof transaction.amount === 'string') {
                    transaction.amount = parseFloat(transaction.amount);
                }

                // Convert date format from DD/MM/YYYY or MM/DD/YYYY to YYYY-MM-DD
                if (transaction.date && typeof transaction.date === 'string' && transaction.date.includes('/')) {
                    const parts = transaction.date.split('/');
                    if (parts.length === 3) {
                        // Try to parse as DD/MM/YYYY or MM/DD/YYYY
                        const [part1, part2, part3] = parts;
                        // Assume MM/DD/YYYY if first part <= 12, otherwise DD/MM/YYYY
                        if (parseInt(part1) <= 12) {
                            // MM/DD/YYYY format
                            transaction.date = `${part3}-${part1.padStart(2, '0')}-${part2.padStart(2, '0')}`;
                        } else {
                            // DD/MM/YYYY format
                            transaction.date = `${part3}-${part2.padStart(2, '0')}-${part1.padStart(2, '0')}`;
                        }
                    }
                }

                // Validate required fields exist
                if (transaction.time && transaction.date &&
                    typeof transaction.sender_account === 'number' &&
                    typeof transaction.receiver_account === 'number' &&
                    typeof transaction.amount === 'number' &&
                    transaction.payment_type) {
                    transactions.push(transaction);
                }
            }

            if (transactions.length === 0) {
                toast({
                    title: 'No Valid Transactions',
                    description: 'Could not parse any valid transactions from the CSV',
                    variant: 'destructive'
                });
                return;
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
