import { useEffect, useState } from 'react';
import initSqlJs, { Database } from 'sql.js';
import { Transaction } from '@/types/transaction';

export const useDatabase = () => {
    const [db, setDb] = useState<Database | null>(null);
    const [isReady, setIsReady] = useState(false);

    useEffect(() => {
        const initDb = async () => {
            const SQL = await initSqlJs({
                locateFile: (file) => `https://sql.js.org/dist/${file}`
            });

            // Try to load existing database from localStorage
            const savedDb = localStorage.getItem('fraudDetectionDb');
            let database: Database;

            if (savedDb) {
                // Load existing database
                const uint8Array = new Uint8Array(JSON.parse(savedDb));
                database = new SQL.Database(uint8Array);
            } else {
                // Create new database
                database = new SQL.Database();

                database.run(`
          CREATE TABLE IF NOT EXISTS transactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT,
            date TEXT,
            sender_account TEXT,
            receiver_account TEXT,
            amount REAL,
            payment_currency TEXT,
            received_currency TEXT,
            sender_bank_location TEXT,
            receiver_bank_location TEXT,
            payment_type TEXT,
            is_fraudulent INTEGER,
            fraud_score REAL,
            status TEXT,
            created_at TEXT,
            fraud_type TEXT,
            confidence REAL,
            legitimacy_score REAL,
            fraud_probability REAL,
            top_3_fraud_types TEXT,
            top_3_probabilities TEXT,
            risk_factors TEXT,
            recommendation TEXT,
            stage TEXT
          )
        `);
            }

            setDb(database);
            setIsReady(true);
        };

        initDb();
    }, []);

    const saveDb = () => {
        if (!db) return;
        const data = db.export();
        const array = Array.from(data);
        localStorage.setItem('fraudDetectionDb', JSON.stringify(array));
    };

    const addTransaction = (transaction: Omit<Transaction, 'id'>) => {
        if (!db) return;

        db.run(`
      INSERT INTO transactions (
        time, date, sender_account, receiver_account, amount,
        payment_currency, received_currency, sender_bank_location,
        receiver_bank_location, payment_type, is_fraudulent,
        fraud_score, status, created_at, fraud_type, confidence,
        legitimacy_score, fraud_probability, top_3_fraud_types,
        top_3_probabilities, risk_factors, recommendation, stage
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `, [
            transaction.time,
            transaction.date,
            transaction.sender_account,
            transaction.receiver_account,
            transaction.amount,
            transaction.payment_currency,
            transaction.received_currency,
            transaction.sender_bank_location,
            transaction.receiver_bank_location,
            transaction.payment_type,
            transaction.is_fraudulent ? 1 : 0,
            transaction.fraud_score,
            transaction.status,
            transaction.created_at,
            transaction.fraud_type || null,
            transaction.confidence || null,
            transaction.legitimacy_score || null,
            transaction.fraud_probability || null,
            transaction.top_3_fraud_types ? JSON.stringify(transaction.top_3_fraud_types) : null,
            transaction.top_3_probabilities ? JSON.stringify(transaction.top_3_probabilities) : null,
            transaction.risk_factors ? JSON.stringify(transaction.risk_factors) : null,
            transaction.recommendation || null,
            transaction.stage || null
        ]);
        saveDb();
    };

    const getTransactions = (): Transaction[] => {
        if (!db) return [];

        const result = db.exec('SELECT * FROM transactions ORDER BY id DESC');
        if (result.length === 0) return [];

        return result[0].values.map((row) => ({
            id: row[0] as number,
            time: row[1] as string,
            date: row[2] as string,
            sender_account: row[3] as string,
            receiver_account: row[4] as string,
            amount: row[5] as number,
            payment_currency: row[6] as string,
            received_currency: row[7] as string,
            sender_bank_location: row[8] as string,
            receiver_bank_location: row[9] as string,
            payment_type: row[10] as string,
            is_fraudulent: row[11] === 1,
            fraud_score: row[12] as number,
            status: row[13] as 'pending' | 'approved' | 'blocked',
            created_at: row[14] as string,
            fraud_type: row[15] as string | undefined,
            confidence: row[16] as number | undefined,
            legitimacy_score: row[17] as number | undefined,
            fraud_probability: row[18] as number | undefined,
            top_3_fraud_types: row[19] ? JSON.parse(row[19] as string) : undefined,
            top_3_probabilities: row[20] ? JSON.parse(row[20] as string) : undefined,
            risk_factors: row[21] ? JSON.parse(row[21] as string) : undefined,
            recommendation: row[22] as 'APPROVE' | 'BLOCK' | 'REVIEW' | undefined,
            stage: row[23] as string | undefined
        }));
    };

    const updateTransactionStatus = (id: number, status: 'approved' | 'blocked') => {
        if (!db) return;
        db.run('UPDATE transactions SET status = ? WHERE id = ?', [status, id]);
        saveDb();
    };

    return {
        isReady,
        addTransaction,
        getTransactions,
        updateTransactionStatus
    };
};
