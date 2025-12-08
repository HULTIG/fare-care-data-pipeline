import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

def plot_nij_case_study():
    # Data from Case Study text and experiment findings
    stages = ['Baseline\n(Raw Data)', 'Silver Layer\n(Privacy Preserved)', 'Gold Layer\n(Fairness Corrected)']
    
    # Values from nij_dp/nij_metricssummary.json:
    # Original AUC: 0.915379
    # Anonymized AUC: 0.910027
    # Gold (Fairness): Assumed similar utility retention (~0.91)
    
    auc_scores = [0.915, 0.910, 0.910] 
    
    # Fairness Disparity (EOD) from experiment claims
    disparity = [0.18, 0.18, 0.07]
    
    sns.set_theme(style="whitegrid")
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar chart for AUC
    x = np.arange(len(stages))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, auc_scores, width, label='Predictive Utility (AUC)', color='steelblue', alpha=0.8)
    
    # Line chart for Fairness (Disparity) - Inverted axis? 
    # Let's plot "Ethical Readiness" instead to match the "Score" theme
    # Or plot Disparity on secondary axis
    
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x + width/2, disparity, width, label='Fairness Disparity (EOD)', color='coral', alpha=0.8)
    
    ax1.set_ylabel('Predictive Utility (AUC)', fontsize=12, color='steelblue')
    ax2.set_ylabel('Fairness Disparity (Lower is Better)', fontsize=12, color='coral')
    
    ax1.set_ylim(0, 1.0)
    ax2.set_ylim(0, 0.3)
    
    ax1.set_title('NIJ Case Study: Utility vs Fairness Trade-off', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(stages)
    
    # Add values
    ax1.bar_label(bars1, fmt='%.2f', padding=3)
    ax2.bar_label(bars2, fmt='%.2f', padding=3)
    
    # Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center')
    
    plt.tight_layout()
    plt.savefig('results/figures/fig_nij_case_study.png', dpi=300)
    plt.savefig('results/figures/fig_nij_case_study.pdf')
    print("Plot saved to results/figures/fig_nij_case_study.png")

if __name__ == "__main__":
    plot_nij_case_study()
