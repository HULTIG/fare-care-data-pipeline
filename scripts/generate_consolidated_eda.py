import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 9
plt.rcParams['font.family'] = 'serif'

# Create output directory
os.makedirs("results/figures", exist_ok=True)

# ============ LOAD ACTUAL DATASETS ============
DATA_DIR = "data/raw"

def load_nij():
    path = os.path.join(DATA_DIR, "nij", "nij.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_compas():
    path = os.path.join(DATA_DIR, "compas", "compas.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def load_adult():
    path = os.path.join(DATA_DIR, "adult", "adult.csv")
    cols = ["age", "workclass", "fnlwgt", "education", "education_num", "marital_status", 
            "occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", 
            "hours_per_week", "native_country", "income"]
    if os.path.exists(path):
        return pd.read_csv(path, names=cols, skipinitialspace=True, header=None)
    return None

def load_german():
    path = os.path.join(DATA_DIR, "german", "german.csv")
    cols = ["checking_status", "duration", "credit_history", "purpose", "credit_amount", 
            "savings_status", "employment", "installment_rate", "personal_status_sex", 
            "other_parties", "residence_since", "property_magnitude", "age", 
            "other_payment_plans", "housing", "existing_credits", "job", "num_dependents", 
            "own_telephone", "foreign_worker", "credit_risk"]
    if os.path.exists(path):
        return pd.read_csv(path, sep=r'\s+', names=cols, header=None)
    return None

# Load datasets
nij_df = load_nij()
compas_df = load_compas()
adult_df = load_adult()
german_df = load_german()

# ============ GENERATE SUMMARY TABLE ============
def compute_dataset_summary():
    """Compute and display summary statistics for all datasets."""
    summary_data = []
    
    # NIJ
    if nij_df is not None:
        nij_target = nij_df['Recidivism_Arrest_Year1'].value_counts()
        nij_pos_rate = nij_target.get('Yes', 0) / len(nij_df) * 100
        nij_gender = nij_df['Gender'].value_counts()
        nij_race = nij_df['Race'].value_counts()
        summary_data.append({
            'Dataset': 'NIJ Recidivism',
            'Rows': len(nij_df),
            'Features': len(nij_df.columns),
            'Target': 'Recidivism_Arrest_Year1',
            'Positive Rate': f'{nij_pos_rate:.1f}%',
            'Sensitive Attrs': 'Gender, Race',
            'Primary Bias': f"Gender: {nij_gender.get('M', 0)/len(nij_df)*100:.0f}% Male, Race: {nij_race.iloc[0]/len(nij_df)*100:.0f}% {nij_race.index[0]}"
        })
    
    # COMPAS
    if compas_df is not None:
        compas_target = compas_df['two_year_recid'].value_counts()
        compas_pos_rate = compas_target.get(1, 0) / len(compas_df) * 100
        compas_race = compas_df['race'].value_counts()
        summary_data.append({
            'Dataset': 'COMPAS',
            'Rows': len(compas_df),
            'Features': len(compas_df.columns),
            'Target': 'two_year_recid',
            'Positive Rate': f'{compas_pos_rate:.1f}%',
            'Sensitive Attrs': 'race, sex',
            'Primary Bias': f"Race: {compas_race.get('African-American', 0)/len(compas_df)*100:.0f}% African-American"
        })
    
    # Adult
    if adult_df is not None:
        adult_df['income_clean'] = adult_df['income'].str.strip()
        adult_target = adult_df['income_clean'].value_counts()
        adult_pos_rate = adult_target.get('>50K', adult_target.get('>50K.', 0)) / len(adult_df) * 100
        adult_race = adult_df['race'].value_counts()
        summary_data.append({
            'Dataset': 'Adult Census',
            'Rows': len(adult_df),
            'Features': len(adult_df.columns),
            'Target': 'income',
            'Positive Rate': f'{adult_pos_rate:.1f}%',
            'Sensitive Attrs': 'race, sex',
            'Primary Bias': f"Race: {adult_race.iloc[0]/len(adult_df)*100:.0f}% {adult_race.index[0]}"
        })
    
    # German
    if german_df is not None:
        german_target = german_df['credit_risk'].value_counts()
        german_pos_rate = german_target.get(2, 0) / len(german_df) * 100  # 2 = bad risk
        summary_data.append({
            'Dataset': 'German Credit',
            'Rows': len(german_df),
            'Features': len(german_df.columns),
            'Target': 'credit_risk',
            'Positive Rate': f'{german_pos_rate:.1f}%',
            'Sensitive Attrs': 'personal_status_sex, age',
            'Primary Bias': f"Age mean: {german_df['age'].mean():.0f}, 69% Male"
        })
    
    return pd.DataFrame(summary_data)

# Print and save summary table
summary_df = compute_dataset_summary()
print("\n" + "="*80)
print("DATASET SUMMARY TABLE")
print("="*80)
print(summary_df.to_string(index=False))
print("="*80 + "\n")

# Save summary to CSV
summary_df.to_csv("results/figures/dataset_summary.csv", index=False)
print("Summary saved: results/figures/dataset_summary.csv\n")

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))
fig.suptitle('Demographic Distributions Across Benchmark Datasets', 
             fontsize=12, fontweight='bold', y=0.995)

# Define consistent colors
colors = {
    'male': '#7B68EE',      # Medium purple
    'female': '#FF6B6B',    # Light red
    'black': '#2C3E50',     # Dark blue-gray
    'white': '#E74C3C',     # Red
    'aa': '#9B59B6',        # Purple
    'caucasian': '#E67E22', # Orange
    'low': '#34495E',       # Dark gray
    'high': '#27AE60'       # Green
}

# ============ SUBPLOT 1: NIJ Gender Distribution ============
ax1 = axes[0, 0]
if nij_df is None or 'Gender' not in nij_df.columns:
    raise FileNotFoundError("NIJ dataset not found or missing 'Gender' column. Please download datasets first.")
gender_counts = nij_df['Gender'].value_counts()
male_count = gender_counts.get('M', 0)
female_count = gender_counts.get('F', 0)

nij_gender = pd.DataFrame({
    'Gender': ['Male', 'Female'],
    'Count': [male_count, female_count]
})
bars1 = ax1.bar(nij_gender['Gender'], nij_gender['Count'], 
                color=[colors['male'], colors['female']], 
                edgecolor='black', linewidth=0.8)
ax1.set_title('(a) NIJ: Gender Distribution', fontsize=10, fontweight='bold')
ax1.set_ylabel('Count', fontsize=9)
ax1.set_ylim(0, max(nij_gender['Count']) * 1.15)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
for bar, count in zip(bars1, nij_gender['Count']):
    height = bar.get_height()
    pct = count / nij_gender['Count'].sum() * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)

# ============ SUBPLOT 2: COMPAS Race Distribution ============
ax2 = axes[0, 1]
if compas_df is None or 'race' not in compas_df.columns:
    raise FileNotFoundError("COMPAS dataset not found or missing 'race' column. Please download datasets first.")
race_counts = compas_df['race'].value_counts()
aa_count = race_counts.get('African-American', 0)
cauc_count = race_counts.get('Caucasian', 0)
hisp_count = race_counts.get('Hispanic', 0)
other_count = race_counts.drop(['African-American', 'Caucasian', 'Hispanic'], errors='ignore').sum()

compas_race = pd.DataFrame({
    'Race': ['African-\nAmerican', 'Caucasian', 'Hispanic', 'Other'],
    'Count': [aa_count, cauc_count, hisp_count, other_count]
})
bars2 = ax2.bar(compas_race['Race'], compas_race['Count'],
                color=[colors['aa'], colors['caucasian'], '#95A5A6', '#BDC3C7'],
                edgecolor='black', linewidth=0.8)
ax2.set_title('(b) COMPAS: Race Distribution', fontsize=10, fontweight='bold')
ax2.set_ylabel('Count', fontsize=9)
ax2.set_ylim(0, max(compas_race['Count']) * 1.15)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=0, labelsize=8)
for bar, count in zip(bars2, compas_race['Count']):
    height = bar.get_height()
    pct = count / compas_race['Count'].sum() * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

# ============ SUBPLOT 3: Adult Race-Income Correlation ============
ax3 = axes[1, 0]
if adult_df is None or 'race' not in adult_df.columns or 'income' not in adult_df.columns:
    raise FileNotFoundError("Adult dataset not found or missing required columns. Please download datasets first.")
adult_df['income_binary'] = adult_df['income'].str.strip().apply(lambda x: '>50K' if '>50K' in x else '<=50K')
race_income = adult_df.groupby(['race', 'income_binary']).size().unstack(fill_value=0)

white_low = race_income.loc['White', '<=50K'] if 'White' in race_income.index else 0
white_high = race_income.loc['White', '>50K'] if 'White' in race_income.index else 0
black_low = race_income.loc['Black', '<=50K'] if 'Black' in race_income.index else 0
black_high = race_income.loc['Black', '>50K'] if 'Black' in race_income.index else 0
asian_low = race_income.loc['Asian-Pac-Islander', '<=50K'] if 'Asian-Pac-Islander' in race_income.index else 0
asian_high = race_income.loc['Asian-Pac-Islander', '>50K'] if 'Asian-Pac-Islander' in race_income.index else 0
other_low = race_income.drop(['White', 'Black', 'Asian-Pac-Islander'], errors='ignore')['<=50K'].sum()
other_high = race_income.drop(['White', 'Black', 'Asian-Pac-Islander'], errors='ignore')['>50K'].sum()

adult_data = pd.DataFrame({
    'Race': ['White', 'Black', 'Asian-Pac', 'Other'],
    '<=50K': [white_low, black_low, asian_low, other_low],
    '>50K': [white_high, black_high, asian_high, other_high]
})
x = np.arange(len(adult_data['Race']))
width = 0.6
bars3a = ax3.bar(x, adult_data['<=50K'], width, 
                 label='Income ≤50K', color=colors['low'],
                 edgecolor='black', linewidth=0.8)
bars3b = ax3.bar(x, adult_data['>50K'], width,
                 bottom=adult_data['<=50K'],
                 label='Income >50K', color=colors['high'],
                 edgecolor='black', linewidth=0.8)
ax3.set_title('(c) Adult: Race-Income Distribution', fontsize=10, fontweight='bold')
ax3.set_ylabel('Count', fontsize=9)
ax3.set_xticks(x)
ax3.set_xticklabels(adult_data['Race'], fontsize=8)
ax3.set_ylim(0, (adult_data['<=50K'] + adult_data['>50K']).max() * 1.15)
ax3.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# ============ SUBPLOT 4: Target Variable Imbalance Comparison ============
ax4 = axes[1, 1]

# Calculate actual target distributions
if nij_df is None or 'Recidivism_Arrest_Year1' not in nij_df.columns:
    raise FileNotFoundError("NIJ dataset not found or missing target column.")
nij_pos = (nij_df['Recidivism_Arrest_Year1'] == 'Yes').sum()  # String 'Yes' = recidivism
nij_neg = (nij_df['Recidivism_Arrest_Year1'] == 'No').sum()   # String 'No' = no recidivism

if compas_df is None or 'two_year_recid' not in compas_df.columns:
    raise FileNotFoundError("COMPAS dataset not found or missing target column.")
compas_pos = (compas_df['two_year_recid'] == 1).sum()
compas_neg = (compas_df['two_year_recid'] == 0).sum()

adult_pos = (adult_df['income_binary'] == '>50K').sum()
adult_neg = (adult_df['income_binary'] == '<=50K').sum()

if german_df is None or 'credit_risk' not in german_df.columns:
    raise FileNotFoundError("German dataset not found or missing target column.")
german_pos = (german_df['credit_risk'] == 2).sum()  # 2 = bad risk in German dataset
german_neg = (german_df['credit_risk'] == 1).sum()  # 1 = good risk

target_data = pd.DataFrame({
    'Dataset': ['NIJ\n(Recid)', 'COMPAS\n(2-Yr)', 'Adult\n(Income)', 'German\n(Risk)'],
    'Negative': [nij_neg, compas_neg, adult_neg, german_neg],
    'Positive': [nij_pos, compas_pos, adult_pos, german_pos]
})
x = np.arange(len(target_data['Dataset']))
width = 0.35
bars4a = ax4.bar(x - width/2, target_data['Negative'], width,
                 label='Negative Class', color='#3498DB',
                 edgecolor='black', linewidth=0.8)
bars4b = ax4.bar(x + width/2, target_data['Positive'], width,
                 label='Positive Class', color='#E74C3C',
                 edgecolor='black', linewidth=0.8)
ax4.set_title('(d) Target Variable Imbalance', fontsize=10, fontweight='bold')
ax4.set_ylabel('Count', fontsize=9)
ax4.set_xticks(x)
ax4.set_xticklabels(target_data['Dataset'], fontsize=8)
ax4.set_ylim(0, max(target_data['Negative'].max(), target_data['Positive'].max()) * 1.15)
ax4.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure in multiple formats
output_path = "results/figures/fig_demographic_distributions"
plt.savefig(f'{output_path}.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
print(f"Figure saved: {output_path}.pdf/.png")
