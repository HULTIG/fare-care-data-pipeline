import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Set style
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

def load_dataset(name, path):
    try:
        if name == 'adult':
            cols = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status", 
                    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss", 
                    "hours-per-week", "native-country", "income"]
            df = pd.read_csv(path, names=cols, skipinitialspace=True)
            return df, 'income', ['sex', 'race']
        elif name == 'german':
            cols = ["status", "duration", "credit_history", "purpose", "amount", "savings", "employment_duration", 
                    "installment_rate", "personal_status_sex", "other_debtors", "residence_since", "property", 
                    "age", "other_installment_plans", "housing", "number_credits", "job", "people_liable", 
                    "telephone", "foreign_worker", "risk"]
            df = pd.read_csv(path, sep=r'\s+', names=cols)
            return df, 'risk', ['age', 'personal_status_sex']
        elif name == 'compas':
            df = pd.read_csv(path)
            return df, 'two_year_recid', ['race', 'sex']
        elif name == 'nij':
            df = pd.read_csv(path)
            return df, 'Recidivism_Arrest_Year1', ['Race', 'Gender']
        else:
            return None, None, None
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None, None

def plot_distributions(name, df, target, sensitive_cols, output_dir):
    print(f"Plotting for {name}...")
    
    # 1. Target Distribution
    if target in df.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df, x=target, palette='viridis')
        plt.title(f'{name}: Target Distribution', fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{name}_target_dist.png'))
        plt.close()

    # 2. Sensitive Attribute Distributions
    for col in sensitive_cols:
        if col in df.columns:
            plt.figure(figsize=(10, 6))
            if len(df[col].unique()) > 10:
                 # If many categories (e.g. age), use distplot or histplot
                 if pd.api.types.is_numeric_dtype(df[col]):
                     sns.histplot(data=df, x=col, kde=True, color='purple')
                 else:
                     # Top 10 categories
                     top_10 = df[col].value_counts().nlargest(10).index
                     sns.countplot(data=df[df[col].isin(top_10)], x=col, order=top_10, palette='magma')
            else:
                sns.countplot(data=df, x=col, palette='magma')
            
            plt.title(f'{name}: {col} Distribution', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{name}_{col}_dist.png'))
            plt.close()

def main():
    base_dir = "data/raw"
    output_dir = "results/figures/eda"
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        "compas": f"{base_dir}/compas/compas.csv",
        "adult": f"{base_dir}/adult/adult.csv",
        "german": f"{base_dir}/german/german.csv",
        "nij": f"{base_dir}/nij/nij.csv"
    }

    for name, path in datasets.items():
        if os.path.exists(path):
            df, target, sensitive = load_dataset(name, path)
            if df is not None:
                plot_distributions(name, df, target, sensitive, output_dir)
        else:
            print(f"Dataset {name} not found at {path}")

if __name__ == "__main__":
    main()
