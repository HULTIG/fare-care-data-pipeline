import pandas as pd
import scipy.stats as stats
import sys

def analyze_exp1(filepath):
    print(f"\nAnalyzing {filepath} (Ablation)...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Pivot to see scores for each config per dataset
    pivot = df.pivot(index='dataset', columns='config', values='faircarescore')
    print("Scores by Config:")
    print(pivot)
    
    # Compare Baseline vs Review (assuming 'configc' or similar is the full model? 
    # Need to check semantic of configs. 
    # Based on README: configa, configb, configc. 
    # Paper says: Baseline, Config A, Config B, Full FAIR-CARE.
    # Let's assume 'configc' might be the full one or I'll look for another one.
    # Actually, let's look at the columns in df.
    pass

def analyze_exp2(filepath):
    print(f"\nAnalyzing {filepath} (Benchmarking)...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return
    
    # Just stats
    print(df.describe())

def analyze_exp3(filepath):
    print(f"\nAnalyzing {filepath} (Regulations)...")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    print(df.groupby('dataset')[['faircarescore', 'privacy_risk']].mean())

def statistical_test(df, col1, col2):
    # Paired t-test
    stat, p = stats.ttest_rel(df[col1], df[col2])
    return stat, p

if __name__ == "__main__":
    base_path = "results/"
    analyze_exp1(base_path + "exp1.csv")
    analyze_exp2(base_path + "exp2.csv")
    analyze_exp3(base_path + "exp3.csv")
    
    # Run t-test Baseline vs Config A (example)
    try:
        df1 = pd.read_csv(base_path + "exp1.csv")
        # Filter for Baseline and ConfigA
        baseline = df1[df1['config'] == 'baseline']['faircarescore'].values
        configa = df1[df1['config'] == 'configa']['faircarescore'].values
        # Ensure they are paired by dataset order
        # (Assuming rows are ordered or pivot ensures it. Let's pivot)
        pivot = df1.pivot(index='dataset', columns='config', values='faircarescore')
        stat, p = statistical_test(pivot, 'baseline', 'configa')
        print(f"\nT-Test Baseline vs ConfigA: stat={stat:.3f}, p={p:.3f}")
        
        stat_b, p_b = statistical_test(pivot, 'baseline', 'configb')
        print(f"T-Test Baseline vs ConfigB: stat={stat_b:.3f}, p={p_b:.3f}")

        stat_c, p_c = statistical_test(pivot, 'baseline', 'configc')
        print(f"T-Test Baseline vs ConfigC: stat={stat_c:.3f}, p={p_c:.3f}")
        
    except Exception as e:
        print(f"Error in t-test: {e}")
