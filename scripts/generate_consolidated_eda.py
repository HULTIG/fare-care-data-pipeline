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
nij_gender = pd.DataFrame({
    'Gender': ['Male', 'Female'],
    'Count': [22500, 3190]  # Approximate from your chart
})
bars1 = ax1.bar(nij_gender['Gender'], nij_gender['Count'], 
                color=[colors['male'], colors['female']], 
                edgecolor='black', linewidth=0.8)
ax1.set_title('(a) NIJ: Gender Distribution', fontsize=10, fontweight='bold')
ax1.set_ylabel('Count', fontsize=9)
ax1.set_ylim(0, 25000)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
# Add percentage labels on bars
for bar, count in zip(bars1, nij_gender['Count']):
    height = bar.get_height()
    pct = count / nij_gender['Count'].sum() * 100
    ax1.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=8)

# ============ SUBPLOT 2: COMPAS Race Distribution ============
ax2 = axes[0, 1]
compas_race = pd.DataFrame({
    'Race': ['African-\nAmerican', 'Caucasian', 'Hispanic', 'Other'],
    'Count': [3696, 2454, 637, 427]  # Approximate from your chart
})
bars2 = ax2.bar(compas_race['Race'], compas_race['Count'],
                color=[colors['aa'], colors['caucasian'], '#95A5A6', '#BDC3C7'],
                edgecolor='black', linewidth=0.8)
ax2.set_title('(b) COMPAS: Race Distribution', fontsize=10, fontweight='bold')
ax2.set_ylabel('Count', fontsize=9)
ax2.set_ylim(0, 4000)
ax2.grid(axis='y', alpha=0.3, linestyle='--')
ax2.tick_params(axis='x', rotation=0, labelsize=8)
# Add percentage labels
for bar, count in zip(bars2, compas_race['Count']):
    height = bar.get_height()
    pct = count / compas_race['Count'].sum() * 100
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{pct:.0f}%', ha='center', va='bottom', fontsize=7)

# ============ SUBPLOT 3: Adult Race-Income Correlation ============
ax3 = axes[1, 0]
# Stacked bar showing race distribution split by income
adult_data = pd.DataFrame({
    'Race': ['White', 'Black', 'Asian-Pac', 'Other'],
    '<=50K': [20000, 2400, 700, 200],
    '>50K': [7500, 400, 300, 100]
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
ax3.set_ylim(0, 28000)
ax3.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, linestyle='--')

# ============ SUBPLOT 4: Target Variable Imbalance Comparison ============
ax4 = axes[1, 1]
target_data = pd.DataFrame({
    'Dataset': ['NIJ\n(Recid)', 'COMPAS\n(2-Yr)', 'Adult\n(Income)', 'German\n(Risk)'],
    'Negative': [17800, 3900, 24600, 700],
    'Positive': [7890, 3314, 7961, 300]
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
ax4.set_ylim(0, 26000)
ax4.legend(loc='upper right', fontsize=7, framealpha=0.9)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save figure in multiple formats
output_path = "results/figures/fig_demographic_distributions"
plt.savefig(f'{output_path}.pdf', bbox_inches='tight', dpi=300)
plt.savefig(f'{output_path}.png', bbox_inches='tight', dpi=300)
print(f"✅ Figure saved: {output_path}.pdf/.png")
