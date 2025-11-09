# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import json
import os
from scipy import stats

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Set font size
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
})


def load_experiment_data():
    """Load all experiment data from CSV files"""
    data_files = glob.glob("experiment_results/*.csv")
    
    # Filter out summary files if any
    data_files = [f for f in data_files if "summary" not in f]
    
    if not data_files:
        print("No data files found in experiment_results/ directory")
        return None, None
    
    print(f"Found {len(data_files)} data files")
    
    # Load detailed data
    all_data = {}
    for file in data_files:
        try:
            filename = os.path.basename(file)
            
            # Extract model type from filename - only normal and depressed_mild
            if filename.startswith('normal'):
                model_key = 'normal'
            elif filename.startswith('depressed_mild'):
                model_key = 'depressed'
            else:
                continue  # Skip other files
            
            if model_key not in all_data:
                all_data[model_key] = []
            
            df = pd.read_csv(file)
            df['Model'] = model_key
            df['Run'] = len(all_data[model_key])  # Add run identifier
            all_data[model_key].append(df)
            print(f"Loaded {filename} as {model_key} run {len(all_data[model_key])}")
            
        except Exception as e:
            print(f"Error loading {file}: {e}")
    
    # Try to load summary data if available
    summary_files = glob.glob("experiment_results/summary_*.json")
    summary_data = None
    if summary_files:
        latest_summary = max(summary_files, key=os.path.getctime)
        try:
            with open(latest_summary, 'r') as f:
                summary_data = json.load(f)
            print(f"Loaded summary data from {latest_summary}")
        except Exception as e:
            print(f"Error loading summary data: {e}")
    
    return all_data, summary_data


def create_model_comparison_plots(all_data):
    """Create comparison plots between normal and depressed models"""
    if not all_data:
        print("No data to plot")
        return
    
    if 'normal' not in all_data or 'depressed' not in all_data:
        print("Need both normal and depressed models for comparison")
        return
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Comparison: Normal vs Depressed Models', fontsize=14, fontweight='bold')
    
    # Colors for models
    colors = {'normal': 'blue', 'depressed': 'red'}
    
    # 1. Learning curves - Duration over episodes
    ax = axes[0, 0]
    
    for model_key in ['normal', 'depressed']:
        runs = all_data[model_key]
        if not runs:
            continue
            
        # Find the maximum episode length across all runs
        max_episodes = min(len(run) for run in runs)  # Use minimum for alignment
        
        # Initialize arrays for aggregation
        all_durations = np.zeros((len(runs), max_episodes))
        
        # Collect data from all runs
        for i, df in enumerate(runs):
            episodes_to_use = min(len(df), max_episodes)
            all_durations[i, :episodes_to_use] = df['Duration'].values[:episodes_to_use]
        
        # Calculate mean and std across runs for each episode
        mean_duration = np.mean(all_durations, axis=0)
        std_duration = np.std(all_durations, axis=0)
        
        episodes = range(1, len(mean_duration) + 1)
        ax.plot(episodes, mean_duration, label=model_key, 
                color=colors[model_key], linewidth=2)
        ax.fill_between(episodes, 
                       mean_duration - std_duration, 
                       mean_duration + std_duration, 
                       alpha=0.2, color=colors[model_key])
    
    ax.axhline(y=500, color='green', linestyle='--', alpha=0.7, label='Success Threshold')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Duration')
    ax.set_title('Learning Curves')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Final performance comparison
    ax = axes[0, 1]
    final_data = []
    for model_key in ['normal', 'depressed']:
        for df in all_data[model_key]:
            if len(df) >= 50:
                final_perf = df['Duration'].iloc[-50:].mean()
            else:
                final_perf = df['Duration'].iloc[-10:].mean() if len(df) >= 10 else df['Duration'].mean()
            final_data.append({'Model': model_key, 'Performance': final_perf})
    
    if final_data:
        final_df = pd.DataFrame(final_data)
        # Create boxplot
        model_data = [final_df[final_df['Model'] == 'normal']['Performance'],
                     final_df[final_df['Model'] == 'depressed']['Performance']]
        
        box_plot = ax.boxplot(model_data, labels=['Normal', 'Depressed'],
                             patch_artist=True)
        
        # Color the boxes
        colors_list = [colors['normal'], colors['depressed']]
        for patch, color in zip(box_plot['boxes'], colors_list):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title('Final Performance Comparison')
        ax.set_ylabel('Average Duration (last episodes)')
        ax.grid(True, alpha=0.3)
    
    # 3. Success rate over time
    ax = axes[1, 0]
    for model_key in ['normal', 'depressed']:
        runs = all_data[model_key]
        if not runs:
            continue
            
        max_episodes = min(len(run) for run in runs)
        all_success = np.zeros((len(runs), max_episodes))
        
        for i, df in enumerate(runs):
            episodes_to_use = min(len(df), max_episodes)
            all_success[i, :episodes_to_use] = df['Success_Rate'].values[:episodes_to_use]
        
        mean_success = np.mean(all_success, axis=0)
        
        episodes = range(1, len(mean_success) + 1)
        ax.plot(episodes, mean_success, label=model_key, 
                color=colors[model_key], linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Success Rate')
    ax.set_title('Success Rate Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Confidence evolution
    ax = axes[1, 1]
    for model_key in ['normal', 'depressed']:
        runs = all_data[model_key]
        if not runs:
            continue
            
        max_episodes = min(len(run) for run in runs)
        all_confidence = np.zeros((len(runs), max_episodes))
        
        for i, df in enumerate(runs):
            episodes_to_use = min(len(df), max_episodes)
            all_confidence[i, :episodes_to_use] = df['Confidence'].values[:episodes_to_use]
        
        mean_confidence = np.mean(all_confidence, axis=0)
        
        episodes = range(1, len(mean_confidence) + 1)
        ax.plot(episodes, mean_confidence, label=model_key, 
                color=colors[model_key], linewidth=2)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Confidence')
    ax.set_title('Confidence Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()


def create_combined_learning_curve(all_data):
    """Create a combined learning curve plot with both models"""
    if not all_data or 'normal' not in all_data or 'depressed' not in all_data:
        return
    
    plt.figure(figsize=(10, 6))
    
    colors = {'normal': 'blue', 'depressed': 'red'}
    
    for model_key in ['normal', 'depressed']:
        runs = all_data[model_key]
        
        # Find the maximum episode length across all runs
        max_episodes = min(len(run) for run in runs)
        
        # Initialize arrays for aggregation
        all_durations = np.zeros((len(runs), max_episodes))
        
        # Collect data from all runs
        for i, df in enumerate(runs):
            episodes_to_use = min(len(df), max_episodes)
            all_durations[i, :episodes_to_use] = df['Duration'].values[:episodes_to_use]
        
        # Calculate mean and std across runs for each episode
        mean_duration = np.mean(all_durations, axis=0)
        std_duration = np.std(all_durations, axis=0)
        
        episodes = range(1, len(mean_duration) + 1)
        plt.plot(episodes, mean_duration, label=model_key, 
                color=colors[model_key], linewidth=2)
        plt.fill_between(episodes, 
                        mean_duration - std_duration, 
                        mean_duration + std_duration, 
                        alpha=0.2, color=colors[model_key])
    
    plt.axhline(y=500, color='green', linestyle='--', alpha=0.7, 
                label='Success Threshold (500)')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.title('Learning Curves: Normal vs Depressed Models')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('combined_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.show()


def analyze_model_performance(all_data):
    """Analyze performance differences between normal and depressed models"""
    print("\n" + "="*60)
    print("MODEL PERFORMANCE ANALYSIS: Normal vs Depressed")
    print("="*60)
    
    if 'normal' not in all_data or 'depressed' not in all_data:
        print("Need both normal and depressed models for comparison")
        return
    
    # Extract final performances
    normal_final = []
    depressed_final = []
    
    for df in all_data['normal']:
        if len(df) >= 50:
            normal_final.append(df['Duration'].iloc[-50:].mean())
        elif len(df) >= 10:
            normal_final.append(df['Duration'].iloc[-10:].mean())
        else:
            normal_final.append(df['Duration'].mean())
    
    for df in all_data['depressed']:
        if len(df) >= 50:
            depressed_final.append(df['Duration'].iloc[-50:].mean())
        elif len(df) >= 10:
            depressed_final.append(df['Duration'].iloc[-10:].mean())
        else:
            depressed_final.append(df['Duration'].mean())
    
    if not normal_final or not depressed_final:
        print("Not enough data for statistical analysis")
        return
    
    # Statistical test
    t_stat, p_value = stats.ttest_ind(normal_final, depressed_final)
    
    print(f"Normal Model Final Performance: {np.mean(normal_final):.1f} ± {np.std(normal_final):.1f} (n={len(normal_final)})")
    print(f"Depressed Model Final Performance: {np.mean(depressed_final):.1f} ± {np.std(depressed_final):.1f} (n={len(depressed_final)})")
    print(f"T-test: t = {t_stat:.3f}, p = {p_value:.4f}")
    
    if p_value < 0.05:
        print("*** Significant difference found (p < 0.05) ***")
        if np.mean(normal_final) > np.mean(depressed_final):
            print("Normal model performs significantly better than depressed model")
        else:
            print("Depressed model performs significantly better than normal model")
    else:
        print("No significant difference found (p >= 0.05)")
    
    # Success rate analysis
    normal_success = [df['Success_Rate'].iloc[-1] for df in all_data['normal'] if len(df) > 0]
    depressed_success = [df['Success_Rate'].iloc[-1] for df in all_data['depressed'] if len(df) > 0]
    
    print(f"\nSuccess Rate Analysis:")
    print(f"Normal Model: {np.mean(normal_success):.3f} ± {np.std(normal_success):.3f}")
    print(f"Depressed Model: {np.mean(depressed_success):.3f} ± {np.std(depressed_success):.3f}")
    
    # Learning speed analysis (episodes to reach certain performance)
    print(f"\nLearning Speed Analysis:")
    threshold = 400  # Define a performance threshold
    
    for model_key in ['normal', 'depressed']:
        episodes_to_threshold = []
        for df in all_data[model_key]:
            # Find first episode where performance exceeds threshold
            above_threshold = df[df['Duration'] >= threshold]
            if not above_threshold.empty:
                first_episode = above_threshold.index[0] + 1  # +1 because index starts at 0
                episodes_to_threshold.append(first_episode)
        
        if episodes_to_threshold:
            print(f"{model_key} - Episodes to reach {threshold}: {np.mean(episodes_to_threshold):.1f} ± {np.std(episodes_to_threshold):.1f}")
        else:
            print(f"{model_key} - Never reached performance threshold of {threshold}")


def print_data_summary(all_data):
    """Print summary of loaded data"""
    print("\n" + "="*60)
    print("DATA LOADING SUMMARY")
    print("="*60)
    
    if not all_data:
        print("No data loaded")
        return
    
    for model_key in ['normal', 'depressed']:
        if model_key in all_data:
            runs = all_data[model_key]
            print(f"\n{model_key}:")
            print(f"  Number of runs: {len(runs)}")
            if runs:
                episode_counts = [len(df) for df in runs]
                print(f"  Episodes per run: {episode_counts}")
                print(f"  Average episodes: {np.mean(episode_counts):.1f}")
                
                # Calculate average final performance
                final_perfs = []
                for df in runs:
                    if len(df) >= 50:
                        final_perfs.append(df['Duration'].iloc[-50:].mean())
                    elif len(df) >= 10:
                        final_perfs.append(df['Duration'].iloc[-10:].mean())
                    else:
                        final_perfs.append(df['Duration'].mean())
                
                if final_perfs:
                    print(f"  Average final performance: {np.mean(final_perfs):.1f} ± {np.std(final_perfs):.1f}")


if __name__ == "__main__":
    print("Loading experiment data...")
    all_data, summary_data = load_experiment_data()
    
    if all_data is None:
        print("No data found. Please check the experiment_results/ directory.")
        exit()
    
    print("\nModels found:", list(all_data.keys()))
    
    # Print data loading summary
    print_data_summary(all_data)
    
    print("\nCreating model comparison plots...")
    create_model_comparison_plots(all_data)
    
    print("Creating combined learning curve...")
    create_combined_learning_curve(all_data)
    
    print("Analyzing model performance differences...")
    analyze_model_performance(all_data)
    
    print("\nAnalysis complete!")
    print("Plots saved to ./ directory")