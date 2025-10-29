import matplotlib.pyplot as plt
import numpy as np

def create_learning_curve():
    plt.figure(figsize=(8, 6))
    
    # Simulated data - replace with your actual results
    data_percentage = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    
    # Simulated performance metrics
    efficient_rlhf = np.array([35, 42, 48, 51, 53, 54, 55, 55, 56, 56])
    dpo_random = np.array([30, 35, 40, 43, 45, 47, 48, 49, 49, 50])
    dpo_full = np.array([50] * 10)  # Constant reference
    
    plt.plot(data_percentage, efficient_rlhf, 'o-', linewidth=2, 
             label='EfficientRLHF (Ours)', color='#2E8B57')
    plt.plot(data_percentage, dpo_random, 's--', linewidth=2, 
             label='DPO (Random Sampling)', color='#FF6347')
    plt.axhline(y=50, color='#1E90FF', linestyle=':', linewidth=2, 
                label='DPO (Full Data)')
    
    plt.xlabel('Human Data Used (%)', fontsize=12)
    plt.ylabel('AI Win Rate (%)', fontsize=12)
    plt.title('Sample Efficiency Comparison', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 100)
    plt.ylim(25, 60)
    
    plt.tight_layout()
    plt.savefig('figures/learning_curve.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/learning_curve.png', bbox_inches='tight', dpi=300)
    plt.close()

create_learning_curve()
