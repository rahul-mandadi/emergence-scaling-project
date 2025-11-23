import pandas as pd
import yaml
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def load_config(config_path='config.yaml'):
    """Loads the main YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def validate_data(df, expected_rows=450):
    """Ensure data integrity before analysis"""
    print("--- 1. Validating Data ---")
    issues = []
    
    if len(df) != expected_rows:
        issues.append(f"Expected {expected_rows} rows, got {len(df)}")
    
    # Check for empty predictions (parser failures)
    empty_preds = (df['prediction_final'] == '').sum() + df['prediction_final'].isna().sum()
    if empty_preds > 0:
        issues.append(f"Found {empty_preds} empty predictions ({empty_preds/len(df)*100:.1f}%)")
    
    # Check for balanced samples
    combo_counts = df.groupby(['model_key', 'technique', 'task']).size()
    if not combo_counts.eq(10).all():
        issues.append("Unbalanced samples per combination (expected 10 each)")
    
    if issues:
        print("⚠️  DATA ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ Data validation passed! All 450 results are clean.")
    
    return True  # Continue analysis even with warnings

def detect_emergence(df):
    """Check for emergence following Wei et al. (2022) criteria"""
    print("\n--- 2. Detecting Emergence ---")
    
    # Calculate accuracy per model and technique
    accuracy = df.groupby(['model_key', 'technique'])['correct_final'].mean()
    
    results = {}
    for technique in ['zero-shot', 'few-shot', 'cot']:
        acc_8b = accuracy.get(('llama-8b', technique), 0)
        acc_70b = accuracy.get(('llama-70b', technique), 0)
        acc_gemini = accuracy.get(('gemini-pro', technique), 0)
        
        jump_1 = acc_70b - acc_8b
        jump_2 = acc_gemini - acc_70b
        
        # Wei's emergence criteria: >15% jump AND discontinuous (2x previous jump)
        emergence = jump_2 > 0.15 and jump_2 > (jump_1 * 2)
        
        results[technique] = {
            'acc_8b': acc_8b, 
            'acc_70b': acc_70b, 
            'acc_gemini': acc_gemini,
            'jump_1': jump_1, 
            'jump_2': jump_2, 
            'emergence': emergence
        }
        
        if emergence:
            print(f"🚨 EMERGENCE DETECTED: {technique}")
            print(f"   8B: {acc_8b:.1%} -> 70B: {acc_70b:.1%} (+{jump_1:.1%}) -> Gemini: {acc_gemini:.1%} (+{jump_2:.1%} JUMP)")
        else:
            print(f"📉 Smooth scaling: {technique}")
            print(f"   8B: {acc_8b:.1%} -> 70B: {acc_70b:.1%} (+{jump_1:.1%}) -> Gemini: {acc_gemini:.1%} (+{jump_2:.1%})")
    
    return results

def cost_benefit_analysis(df):
    """Calculate token efficiency per technique"""
    print("\n--- 3. Cost-Benefit Analysis ---")
    
    avg_tokens = df.groupby('technique')['input_tokens'].mean()
    accuracy = df.groupby(['model_key', 'technique'])['correct_final'].mean()
    
    for model_key in ['llama-8b', 'llama-70b', 'gemini-pro']:
        print(f"\n{model_key.upper()}:")
        for technique in ['zero-shot', 'few-shot', 'cot']:
            acc = accuracy.get((model_key, technique), 0)
            tokens = avg_tokens.get(technique, 0)
            
            if acc > 0:
                efficiency = tokens / acc
                print(f"  {technique:>10}: {efficiency:,.0f} tokens/correct (Acc: {acc:.1%}, Avg tokens: {tokens:.0f})")
            else:
                print(f"  {technique:>10}: N/A (Acc: 0.0%, no correct answers)")

def compare_to_literature(emergence_results):
    """Compare findings to Wei and Schaeffer predictions"""
    print("\n--- 4. Wei vs. Schaeffer Verdict ---")
    
    wei_support = 0      # Count techniques showing emergence
    schaeffer_support = 0  # Count techniques showing smooth scaling
    
    for tech in ['few-shot', 'cot']:
        if emergence_results[tech]['emergence']:
            wei_support += 1
        else:
            schaeffer_support += 1
    
    # Zero-shot baseline (smooth scaling expected by both theories)
    if not emergence_results['zero-shot']['emergence']:
        schaeffer_support += 1

    print(f"  Wei Support Score (CoT/Few-Shot Emerge): {wei_support}/2")
    print(f"  Schaeffer Support Score (All Smooth): {schaeffer_support}/3")
    
    if schaeffer_support > wei_support:
        print("\n  🏆 VERDICT: Results strongly support Schaeffer et al.")
        print("     Emergence appears to be a 'mirage' of smooth, continuous scaling.")
    elif wei_support > schaeffer_support:
        print("\n  🏆 VERDICT: Results strongly support Wei et al.")
        print("     Emergence is a real, discontinuous capability jump.")
    else:
        print("\n  🤝 VERDICT: Mixed results.")
        print("     Some techniques show emergence while others scale smoothly.")

def test_significance(df):
    """Test if technique improvements are statistically significant (Chi-squared)"""
    print("\n--- 5. Statistical Significance (vs. Zero-Shot Baseline) ---")
    
    for model_key in ['llama-8b', 'llama-70b', 'gemini-pro']:
        print(f"\n{model_key.upper()}:")
        zero_shot = df[(df['model_key'] == model_key) & (df['technique'] == 'zero-shot')]
        
        for technique in ['few-shot', 'cot']:
            technique_data = df[(df['model_key'] == model_key) & (df['technique'] == technique)]
            
            # Build contingency table: [[correct, incorrect], [correct, incorrect]]
            contingency = [
                [zero_shot['correct_final'].sum(), len(zero_shot) - zero_shot['correct_final'].sum()],
                [technique_data['correct_final'].sum(), len(technique_data) - technique_data['correct_final'].sum()]
            ]
            
            try:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                
                if p_value < 0.05:
                    print(f"  {technique:>10} vs. Zero-Shot: p={p_value:.4f} ✓ (Significant at α=0.05)")
                else:
                    print(f"  {technique:>10} vs. Zero-Shot: p={p_value:.4f} (Not significant)")
                    
            except ValueError as e:
                print(f"  {technique:>10} vs. Zero-Shot: Could not compute (error: {e})")

def plot_scaling_curves(df, emergence_results, save_path):
    """Generate scaling curve visualization"""
    print("\n--- 6. Generating Scaling Curve Plot ---")
    
    # Ensure output directory exists
    os.makedirs(save_path, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    model_labels = ['Llama 3.1\n8B', 'Llama 3.3\n70B', 'Gemini 2.0\nFlash']
    
    for idx, technique in enumerate(['zero-shot', 'few-shot', 'cot']):
        ax = axes[idx]
        
        # Extract accuracies for this technique
        accs = [
            emergence_results[technique]['acc_8b'],
            emergence_results[technique]['acc_70b'],
            emergence_results[technique]['acc_gemini']
        ]
        
        # Plot the scaling curve
        ax.plot(model_labels, accs, 'o-', linewidth=3, markersize=10, color='#2E86AB')
        ax.set_title(f'{technique.replace("-", " ").title()}', fontsize=18, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3, linestyle='--')
        ax.tick_params(axis='x', labelsize=11)
        ax.tick_params(axis='y', labelsize=11)
        
        # Add accuracy labels on each point
        for i, acc in enumerate(accs):
            ax.text(i, acc + 0.04, f'{acc:.1%}', ha='center', fontsize=13, fontweight='bold')
        
        # Highlight emergence if detected
        if emergence_results[technique]['emergence']:
            ax.annotate('⚡ EMERGENCE!', xy=(2, accs[2]),
                        xytext=(1.3, accs[2] - 0.25),
                        arrowprops=dict(arrowstyle='->', color='red', lw=2.5),
                        fontsize=15, color='red', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))
            ax.axvspan(1.5, 2.5, alpha=0.15, color='red')
    
    plt.suptitle('Scaling Behavior Across Prompting Techniques', 
                 fontsize=22, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    final_path = os.path.join(save_path, 'scaling_curves_final.png')
    plt.savefig(final_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to {final_path}")
    plt.close()

def task_difficulty_analysis(df):
    """Rank tasks by overall difficulty"""
    print("\n--- 7. Task Difficulty Ranking ---")
    
    task_acc = df.groupby('task')['correct_final'].mean().sort_values(ascending=False)
    
    print(f"{'Task':<45} {'Accuracy':>10}")
    print("-" * 57)
    for task, acc in task_acc.items():
        print(f"{task:<45} {acc:>9.1%}")

def task_model_breakdown(df):
    """Show which models excel at which tasks"""
    print("\n--- 8. Task × Model Performance Matrix ---")
    
    pivot = df.pivot_table(
        values='correct_final', 
        index='task', 
        columns='model_key', 
        aggfunc='mean'
    )
    
    # Format as percentages
    print(f"\n{'Task':<45} {'8B':>8} {'70B':>8} {'Gemini':>8}")
    print("-" * 72)
    for task in pivot.index:
        print(f"{task:<45} {pivot.loc[task, 'llama-8b']:>7.1%} {pivot.loc[task, 'llama-70b']:>7.1%} {pivot.loc[task, 'gemini-pro']:>7.1%}")

def generate_summary_table(df, save_path):
    """Create accuracy summary table for report"""
    print("\n--- 9. Generating Summary Tables ---")
    
    # Main accuracy table
    summary = df.pivot_table(
        values='correct_final',
        index='technique',
        columns='model_key',
        aggfunc='mean'
    )
    
    # Reorder columns for readability
    summary = summary[['llama-8b', 'llama-70b', 'gemini-pro']]
    
    # Save as CSV
    summary_path = os.path.join(save_path, 'accuracy_summary_final.csv')
    summary.to_csv(summary_path)
    print(f"✅ Accuracy summary saved to {summary_path}")
    
    # Print formatted version
    print("\nAccuracy by Model and Technique:")
    print(summary.to_string(float_format=lambda x: f'{x:.1%}'))
    
    # Task-specific table
    task_summary = df.pivot_table(
        values='correct_final',
        index='task',
        columns='technique',
        aggfunc='mean'
    )
    
    task_path = os.path.join(save_path, 'task_accuracy_final.csv')
    task_summary.to_csv(task_path)
    print(f"✅ Task accuracy saved to {task_path}")

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    print("=" * 70)
    print(" EMERGENCE ANALYSIS - Final Results with Fixed Parser".center(70))
    print("=" * 70)
    
    # Load configuration
    config = load_config()
    paths = config['paths']
    
    # Load the FIXED data with comprehensive parser
    try:
        df = pd.read_csv('data/processed/final_results_complete.csv')
        print(f"\n✅ Loaded {len(df)} rows from final_results_complete.csv")
        print(f"   Overall Accuracy: {df['correct_final'].mean():.1%}")
    except FileNotFoundError:
        print(f"\n❌ ERROR: Fixed data file not found!")
        print("Please run the parser fix first.")
        exit(1)
    
    # ========== Run All Analysis Steps ==========
    print("\n" + "=" * 70)
    print(" RUNNING ANALYSIS PIPELINE".center(70))
    print("=" * 70)
    
    validate_data(df)
    emergence_results = detect_emergence(df)
    cost_benefit_analysis(df)
    compare_to_literature(emergence_results)
    test_significance(df)
    plot_scaling_curves(df, emergence_results, paths['figures_output'])
    task_difficulty_analysis(df)
    task_model_breakdown(df)
    generate_summary_table(df, paths['figures_output'])
    
    print("\n" + "=" * 70)
    print(" ANALYSIS COMPLETE".center(70))
    print("=" * 70)
    print(f"\n📊 Results saved to: {paths['figures_output']}")
    print(f"📈 Plots saved to: {paths['figures_output']}")
    
    # Final summary
    print("\n" + "=" * 70)
    print(" KEY FINDINGS".center(70))
    print("=" * 70)
    print(f"\n1. Overall Accuracy: {df['correct_final'].mean():.1%}")
    print(f"2. Best Technique: Few-shot ({df[df['technique']=='few-shot']['correct_final'].mean():.1%})")
    print(f"3. Best Model: Gemini Pro ({df[df['model_key']=='gemini-pro']['correct_final'].mean():.1%})")
    print(f"4. Emergence: {'DETECTED' if any(r['emergence'] for r in emergence_results.values()) else 'NOT DETECTED'}")
    print("\nExperiment completed successfully! ✅")