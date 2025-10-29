import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

def create_framework_diagram():
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 13)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colors for different phases
    colors = {'phase1': '#E3F2FD', 'phase2': '#E8F5E8', 'phase3': '#FFEBEE'}
    
    # ===== PHASE 1: Synthetic Warm-Start =====
    phase1_y = 10.5
    phase1_box = FancyBboxPatch((1, phase1_y), 8, 1.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['phase1'], edgecolor='black', linewidth=2)
    ax.add_patch(phase1_box)
    
    # Phase label ON TOP of the box
    ax.text(5, phase1_y + 1.8, 'PHASE 1: Synthetic Warm-Start', ha='center', va='bottom', 
            fontsize=14, fontweight='bold', color='#1565C0',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#1565C0', linewidth=2))
    
    # Phase 1 components with proper spacing
    phase1_components = [
        (2.5, phase1_y + 0.5, 'Input:\nPrompts', '#FFF9C4', '#FFA000'),
        (5.0, phase1_y + 0.5, 'Synthetic\nLabeler\n(GPT-4)', '#FFEBEE', '#C62828'),
        (7.5, phase1_y + 0.5, 'Initial Reward\nModel Training', '#E3F2FD', '#1565C0')
    ]
    
    for x, y, text, bg_color, edge_color in phase1_components:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor=edge_color, linewidth=2))
    
    # Arrows between Phase 1 components
    ax.annotate('', xy=(3.8, phase1_y + 0.5), xytext=(3.2, phase1_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', linewidth=2))
    ax.annotate('', xy=(6.3, phase1_y + 0.5), xytext=(5.7, phase1_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', linewidth=2))
    
    # Output of Phase 1
    ax.text(9.2, phase1_y + 0.5, '$RM_{init}$', ha='center', va='center', 
            fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle="circle,pad=0.5", facecolor='#FFF59D', edgecolor='#F57C00', linewidth=2))
    
    # ===== CONNECTION ARROW to Phase 2 =====
    ax.annotate('', xy=(5, 9.9), xytext=(5, 10.2),
                arrowprops=dict(arrowstyle='->', color='#1565C0', linewidth=2, shrinkA=5, shrinkB=5))
    
    # ===== PHASE 2: Active Learning =====
    phase2_y = 6.8
    phase2_box = FancyBboxPatch((1, phase2_y), 8, 2, boxstyle="round,pad=0.1", 
                               facecolor=colors['phase2'], edgecolor='black', linewidth=2)
    ax.add_patch(phase2_box)
    
    # Phase label ON TOP of the box
    ax.text(5, phase2_y + 2.8, 'PHASE 2: Uncertainty-Aware Active Learning', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#2E7D32',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2E7D32', linewidth=2))
    
    # Phase 2 components in circular layout with proper spacing
    center_x, center_y = 5, phase2_y + 0.8
    radius = 1.3
    
    phase2_components = [
        (center_x - radius, center_y, 'Uncertainty\nCalculation', '#FFE0B2', '#EF6C00'),
        (center_x, center_y + radius, 'Human\nLabeling', '#FFEBEE', '#C62828'),
        (center_x + radius, center_y, 'RM\nFine-tuning', '#E3F2FD', '#1565C0'),
        (center_x, center_y - radius, 'Data\nCollection', '#E8F5E9', '#2E7D32')
    ]
    
    for x, y, text, bg_color, edge_color in phase2_components:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor=edge_color, linewidth=2))
    
    # Circular arrows connecting Phase 2 components
    angles = [0, np.pi/2, np.pi, 3*np.pi/2]
    for i in range(4):
        start_angle = angles[i]
        end_angle = angles[(i + 1) % 4]
        
        # Calculate positions for arrows (slightly inside the circle)
        arrow_radius = radius - 0.2
        start_x = center_x + arrow_radius * np.cos(start_angle)
        start_y = center_y + arrow_radius * np.sin(start_angle)
        end_x = center_x + arrow_radius * np.cos(end_angle)
        end_y = center_y + arrow_radius * np.sin(end_angle)
        
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color='#2E7D32', linewidth=2,
                                 connectionstyle="arc3,rad=0.3"))
    
    # Iterative process label
    ax.text(center_x, center_y, 'Iterative\nProcess', ha='center', va='center', fontsize=10, 
            fontweight='bold', style='italic', color='#2E7D32',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='#2E7D32'))
    
    # ===== CONNECTION ARROW to Phase 3 =====
    ax.annotate('', xy=(5, 6.5), xytext=(5, 6.8),
                arrowprops=dict(arrowstyle='->', color='#2E7D32', linewidth=2, shrinkA=5, shrinkB=5))
    
    # ===== PHASE 3: Fine-grained Feedback + DPO =====
    phase3_y = 3.0
    phase3_box = FancyBboxPatch((1, phase3_y), 8, 1.8, boxstyle="round,pad=0.1", 
                               facecolor=colors['phase3'], edgecolor='black', linewidth=2)
    ax.add_patch(phase3_box)
    
    # Phase label ON TOP of the box
    ax.text(5, phase3_y + 1.8, 'PHASE 3: Fine-grained Feedback + DPO Optimization', ha='center', va='bottom',
            fontsize=14, fontweight='bold', color='#C62828',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#C62828', linewidth=2))
    
    # Phase 3 components with proper spacing
    phase3_components = [
        (3.0, phase3_y + 0.5, 'Attribute-based\nFeedback Modeling', '#F3E5F5', '#7B1FA2'),
        (7.0, phase3_y + 0.5, 'DPO Policy\nOptimization', '#E3F2FD', '#1565C0')
    ]
    
    for x, y, text, bg_color, edge_color in phase3_components:
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=bg_color, edgecolor=edge_color, linewidth=2))
    
    # Arrow between Phase 3 components
    ax.annotate('', xy=(6.2, phase3_y + 0.5), xytext=(4.3, phase3_y + 0.5),
                arrowprops=dict(arrowstyle='->', color='#C62828', linewidth=2))
    
    # ===== CONNECTION ARROW to Final Output =====
    ax.annotate('', xy=(5, 2.5), xytext=(5, 2.8),
                arrowprops=dict(arrowstyle='->', color='#C62828', linewidth=2, shrinkA=5, shrinkB=5))
    
    # ===== FINAL OUTPUT =====
    final_box = FancyBboxPatch((4, 1.2), 2, 0.8, boxstyle="round,pad=0.3",
                             facecolor='#FFF9C4', edgecolor='#FF8F00', linewidth=3)
    ax.add_patch(final_box)
    ax.text(5, 1.6, 'Aligned Policy', ha='center', va='center', 
            fontsize=12, fontweight='bold', color='#E65100')
    ax.text(5, 1.2, '$\\pi_{\\theta}$', ha='center', va='center',
            fontsize=14, fontweight='bold', color='#E65100')
    
    # ===== EXTERNAL CONNECTIONS =====
    # Synthetic data input
    ax.annotate('Synthetic Data', xy=(2.5, phase1_y + 0.5), xytext=(0.5, phase1_y + 1.5),
                arrowprops=dict(arrowstyle='->', color='#1565C0', linewidth=1.5, alpha=0.7),
                fontsize=8, fontweight='bold', color='#1565C0', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    # Human feedback input
    ax.annotate('Human Feedback', xy=(5, phase2_y), xytext=(8.5, phase2_y + 2.3),
                arrowprops=dict(arrowstyle='->', color='#C62828', linewidth=1.5, alpha=0.7),
                fontsize=8, fontweight='bold', color='#C62828', ha='center',
                bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('figures/framework_clean.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('figures/framework_clean.png', bbox_inches='tight', dpi=300)
    plt.show()
    plt.close()

print("Generating clean framework diagram...")
create_framework_diagram()
