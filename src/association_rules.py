import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from .config import REPORT_DIR


def make_transaction_items(row):
    items = []
    if row.get("telecommuting", 0) == 1:
        items.append("Remote Job")
    if row.get("has_company_logo", 0) == 0:
        items.append("Missing Logo")
    if row.get("has_questions", 0) == 0:
        items.append("No Screening Questions")
    if row.get("profile_missing", 0) == 1:
        items.append("Missing Company Profile")
    if row.get("salary_missing", 0) == 1:
        items.append("Salary Not Mentioned")
    if row.get("suspicious_keyword_count", 0) >= 1:
        items.append("Suspicious Keywords Present")
    if row.get("fee_keyword_count", 0) >= 1:
        items.append("Payment Or Deposit Request")
    if row.get("urgency_keyword_count", 0) >= 1:
        items.append("Urgency Or Guaranteed Selection")
    if row.get("contact_risk_keyword_count", 0) >= 1:
        items.append("Risky Contact Channel")
    if row.get("sensitive_info_keyword_count", 0) >= 1:
        items.append("Sensitive Info Requested")
    if row.get("fraudulent", 0) == 1:
        items.append("Fraudulent")
    else:
        items.append("Genuine")
    return items


def generate_simple_association_rules(data, min_support=0.005, min_confidence=0.3):
    """Generate association rules using mlxtend Apriori algorithm."""
    transactions = [make_transaction_items(row) for _, row in data.iterrows()]

    te = TransactionEncoder()
    te_ary = te.fit_transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    # Generate frequent itemsets
    frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)

    if frequent_itemsets.empty:
        print("No frequent itemsets found with current min_support. Try lowering min_support.")
        rules_df = pd.DataFrame()
    else:
        # Generate association rules
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

        if rules.empty:
            print("No association rules found with current parameters.")
            rules_df = pd.DataFrame()
        else:
            # Filter rules related to fraudulent jobs
            fraud_rules = rules[rules['consequents'].apply(lambda x: 'Fraudulent' in x)]

            if not fraud_rules.empty:
                rules_df = fraud_rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].copy()
                rules_df['antecedents'] = rules_df['antecedents'].apply(lambda x: ', '.join(list(x)))
                rules_df['consequents'] = rules_df['consequents'].apply(lambda x: ', '.join(list(x)))
                rules_df = rules_df.sort_values(by=['lift', 'confidence'], ascending=False)
            else:
                print("No rules with 'Fraudulent' as consequent found.")
                rules_df = pd.DataFrame()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    rules_df.to_csv(REPORT_DIR / "association_rules.csv", index=False)
    return rules_df
