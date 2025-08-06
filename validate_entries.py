import csv
import pandas as pd

# Read the CSV file
df = pd.read_csv('final_consolidated_output.csv')

print('=== CREDIT/DEBIT MISCLASSIFICATION ANALYSIS ===\n')

# Find entries that are likely misclassified
misclassified = []

# 1. Check for Purchase transactions in Credit column (should be Debit)
print('1. PURCHASE TRANSACTIONS IN CREDIT COLUMN (Should be Debit):')
print('-' * 60)

credit_entries = df[df['Credit'].notna() & (df['Credit'] != '')]
for idx, row in credit_entries.iterrows():
    particulars = str(row['Particulars']).lower()
    vch_type = str(row['Vch Type']).lower()
    
    # Purchase transactions should be in Debit, not Credit
    if ('(as per details)' in particulars and 'purchases' in particulars) or 'purchase' in vch_type:
        # Exclude bank payments (these are correctly in Credit)
        if 'hdfc bank' not in particulars and 'payment' not in vch_type:
            print(f'❌ {row["File Name"]} | {row["Date"]} | Vch {row["Vch No."]}')
            print(f'   Amount: {row["Credit"]} (in Credit, should be Debit)')
            print(f'   Type: {row["Vch Type"]}')
            print()
            misclassified.append({
                'file': row["File Name"],
                'date': row["Date"],
                'vch_no': row["Vch No."],
                'amount': row["Credit"],
                'issue': 'Purchase in Credit (should be Debit)'
            })

print(f'\n2. PAYMENT TRANSACTIONS IN DEBIT COLUMN (Should be Credit):')
print('-' * 60)

# 2. Check for Payment transactions in Debit column (should be Credit)
debit_entries = df[df['Debit'].notna() & (df['Debit'] != '')]
for idx, row in debit_entries.iterrows():
    particulars = str(row['Particulars']).lower()
    vch_type = str(row['Vch Type']).lower()
    
    # Bank payments should be in Credit, not Debit
    if 'hdfc bank' in particulars and 'payment' in particulars:
        print(f'❌ {row["File Name"]} | {row["Date"]} | Vch {row["Vch No."]}')
        print(f'   Amount: {row["Debit"]} (in Debit, should be Credit)')
        print(f'   Particulars: {particulars[:60]}...')
        print()
        misclassified.append({
            'file': row["File Name"],
            'date': row["Date"],
            'vch_no': row["Vch No."],
            'amount': row["Debit"],
            'issue': 'Bank Payment in Debit (should be Credit)'
        })

print(f'\n3. SUMMARY:')
print('-' * 30)
print(f'Total entries analyzed: {len(df)}')
print(f'Potential misclassifications found: {len(misclassified)}')
print(f'Accuracy rate: {((len(df) - len(misclassified)) / len(df) * 100):.1f}%')

if misclassified:
    print(f'\nMISCLASSIFIED ENTRIES SUMMARY:')
    for item in misclassified:
        print(f'- {item["file"]} | {item["date"]} | Vch {item["vch_no"]} | {item["issue"]}')
