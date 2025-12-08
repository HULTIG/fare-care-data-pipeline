import pandas as pd
import os
import sys

def get_config_for_dataset(name):
    # Minimal config mapping based on common usage or inspection
    if name == 'compas':
        return {'target': 'two_year_recid', 'sensitive': ['race', 'sex']}
    elif name == 'adult':
        return {'target': 'income', 'sensitive': ['race', 'sex']}
    elif name == 'german':
        return {'target': 'risk', 'sensitive': ['sex', 'age']} # 'credit_risk' maybe?
    elif name == 'nij':
        return {'target': 'Recidivism_Arrest_Year1', 'sensitive': ['Race', 'Gender']} 
    return {'target': 'target', 'sensitive': []}

def analyze_dataset(name, path):
    print(f"\n--- Analyzing {name} ---")
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    
    try:
        if name == 'adult':
            # Adult often has no header. Providing standard names.
            cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                    "hours-per-week", "native-country", "income"]
            df = pd.read_csv(path, names=cols, skipinitialspace=True)
        elif name == 'german':
            # German is often space separated
            cols = ["status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", 
                    "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property", 
                    "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", 
                    "telephone", "foreign_worker", "risk"]
            df = pd.read_csv(path, sep=r'\s+', names=cols)
            # Map personal_status_sex to sex if needed (complex encoding in german dataset), 
            # or just report age as sensitive.
        else:
            df = pd.read_csv(path)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    config = get_config_for_dataset(name)
    target = config['target']
    sensitive_cols = config['sensitive']

    print(f"Rows (N): {len(df)}")
    print(f"Columns: {len(df.columns)}")
    
    # Missing data
    missing_pct = df.isnull().mean().mean() * 100
    print(f"Global Missingness: {missing_pct:.2f}%")

    # Target balance
    if target in df.columns:
        print(f"Target Variable: {target}")
        print(df[target].value_counts(normalize=True))
    else:
        print(f"Target '{target}' not found in columns: {list(df.columns[:5])}...")

    # Sensitive Attributes
    for col in sensitive_cols:
        if col in df.columns:
            print(f"Sensitive Attribute: {col}")
            print(df[col].value_counts(normalize=True))
        else:
            print(f"Sensitive '{col}' not found.")

def main():
    base_dir = "data/raw"
    datasets = {
        "compas": f"{base_dir}/compas/compas.csv",
        "adult": f"{base_dir}/adult/adult.csv",
        "german": f"{base_dir}/german/german.csv",
        "nij": f"{base_dir}/nij/nij.csv"
    }

    for name, path in datasets.items():
        analyze_dataset(name, path)

if __name__ == "__main__":
    main()
