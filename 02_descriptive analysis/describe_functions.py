import matplotlib.pyplot as plt  # Changed this line
import numpy as np
import pandas as pd
import seaborn as sns



def recode_response_variable(df, response_col='response', concept_col='concept', 
                           reverse_concepts=None, verbose=True):
    """
    Recode response variable by reverse-coding negatively poled concepts
    
    Args:
        df: DataFrame with response and concept columns
        response_col: Name of response column to recode (default: 'response')
        concept_col: Name of concept column (default: 'concept')
        reverse_concepts: List of concepts to reverse-code. If None, uses default list
        verbose: Whether to print detailed statistics
    
    Returns:
        DataFrame with new 'response_recoded' column
    """
    
    # Make a copy to avoid modifying original
    df_recoded = df.copy()
    
    # Default reverse concepts if not provided
    if reverse_concepts is None:
        reverse_concepts = ['AH', 'PH', 'ENV', 'CON', 'PIT']
    
    if verbose:
        print("=== RESPONSE VARIABLE RECODING ===")
        print(f"Original df shape: {df_recoded.shape}")
        print(f"Response column: {response_col}")
        print(f"Concept column: {concept_col}")
        print(f"Concepts to be reverse-poled: {reverse_concepts}")
    
    # Check if required columns exist
    if response_col not in df_recoded.columns:
        print(f" ERROR: Column '{response_col}' not found in DataFrame")
        return df_recoded
    
    if concept_col not in df_recoded.columns:
        print(f" ERROR: Column '{concept_col}' not found in DataFrame")
        return df_recoded
    
    # Convert response to numeric
    df_recoded[response_col] = pd.to_numeric(df_recoded[response_col], errors='coerce')
    
    # Create the recoded response column (start with original values)
    df_recoded['response_recoded'] = df_recoded[response_col].copy()
    
    if verbose:
        print(f"\nConcept distribution:")
        print(df_recoded[concept_col].value_counts())
        
        # Before recoding stats
        before_mean = df_recoded[response_col].mean()
        before_std = df_recoded[response_col].std()
        print(f"\nBefore recoding - Mean: {before_mean:.3f}, Std: {before_std:.3f}")
    
    # Create mask for rows that need reverse coding
    reverse_mask = df_recoded[concept_col].isin(reverse_concepts)
    n_reverse = reverse_mask.sum()
    n_total = df_recoded[response_col].notna().sum()
    
    if verbose:
        print(f"Rows to reverse: {n_reverse:,} out of {n_total:,} ({n_reverse/n_total*100:.1f}%)")
    
    # Apply reverse coding: multiply by -1
    df_recoded.loc[reverse_mask, 'response_recoded'] = df_recoded.loc[reverse_mask, response_col] * -1
    
    if verbose:
        # After recoding stats
        after_mean = df_recoded['response_recoded'].mean()
        after_std = df_recoded['response_recoded'].std()
        print(f"After recoding  - Mean: {after_mean:.3f}, Std: {after_std:.3f}")
        
        # Show means by concept after recoding
        print(f"\nMean {response_col} by concept after recoding:")
        concept_means = df_recoded.groupby(concept_col)['response_recoded'].mean().round(3)
        for concept in concept_means.index:
            marker = " (REVERSED)" if concept in reverse_concepts else ""
            print(f"  {concept}: {concept_means[concept]:.3f}{marker}")
        
        # Verification
        print(f"\n=== VERIFICATION ===")
        print("Comparing original vs recoded responses by concept group:")
        
        # Group concepts
        normal_concepts = [c for c in df_recoded[concept_col].unique() 
                          if c not in reverse_concepts and pd.notna(c)]
        
        if len(reverse_concepts) > 0:
            reverse_data_orig = df_recoded[df_recoded[concept_col].isin(reverse_concepts)][response_col]
            reverse_data_recoded = df_recoded[df_recoded[concept_col].isin(reverse_concepts)]['response_recoded']
            
            if len(reverse_data_orig) > 0:
                print(f"\nReverse-poled concepts:")
                print(f"  Original mean: {reverse_data_orig.mean():.3f}")
                print(f"  Recoded mean:  {reverse_data_recoded.mean():.3f}")
                print(f"  N = {len(reverse_data_orig)}")
        
        if len(normal_concepts) > 0:
            normal_data_orig = df_recoded[df_recoded[concept_col].isin(normal_concepts)][response_col]
            normal_data_recoded = df_recoded[df_recoded[concept_col].isin(normal_concepts)]['response_recoded']
            
            if len(normal_data_orig) > 0:
                print(f"\nNormal concepts:")
                print(f"  Original mean: {normal_data_orig.mean():.3f}")
                print(f"  Recoded mean:  {normal_data_recoded.mean():.3f}")
                print(f"  N = {len(normal_data_orig)}")
        
        print(f"\n=== RECODING COMPLETE ===")
        print("The following concepts have been reverse-poled (multiplied by -1):")
        for concept in reverse_concepts:
            concept_count = (df_recoded[concept_col] == concept).sum()
            print(f"  {concept}: {concept_count:,} observations")
        
        print(f"\nNew column 'response_recoded' created")
        print("All response variables now coded in same direction")
    
    return df_recoded


def recode_all_models(model_dict, save_files=True, output_dir=""):
    """
    Apply response recoding to multiple model dataframes
    
    Args:
        model_dict: Dictionary with model names as keys and DataFrames as values
        save_files: Whether to save recoded files
        output_dir: Directory to save files (empty string for current directory)
    
    Returns:
        Dictionary with recoded DataFrames
    """
    
    print("RECODING RESPONSES FOR ALL MODELS")
    print("=" * 60)
    
    recoded_models = {}
    
    for model_name, df in model_dict.items():
        if df is not None:
            print(f"\nProcessing {model_name.upper()}...")
            
            try:
                recoded_df = recode_response_variable(df, verbose=True)
                recoded_models[model_name] = recoded_df
                
                if save_files:
                    filename = f"{output_dir}{model_name}_recoded.csv"
                    recoded_df.to_csv(filename, index=False)
                    print(f"Saved as: {filename}")
                
            except Exception as e:
                print(f" Error processing {model_name}: {e}")
                recoded_models[model_name] = df  # Return original if recoding fails
        else:
            print(f"Skipping {model_name} (no data)")
            recoded_models[model_name] = None
    
    print("\nALL MODELS RECODED!")
    return recoded_models





def plot_response_distribution_by_item(df, model_name, figsize=(15, 10), save_plot=True):
    """
    Visualize response distribution by item_id, ordered by concepts
    
    Args:
        df: DataFrame with columns 'item_id', 'concept', 'response'
        model_name: String name for the model (for titles and saving)
        figsize: Tuple for figure size
        save_plot: Boolean to save the plot as PNG
    
    Returns:
        matplotlib figure object
    """
    
    # Remove rows with missing responses or concepts
    df_clean = df.dropna(subset=['response', 'concept', 'item_id']).copy()
    
    # Get unique concepts and create color palette
    concepts = sorted(df_clean['concept'].unique())
    colors = sns.color_palette("Set2", len(concepts))
    concept_colors = dict(zip(concepts, colors))
    
    # Sort items by concept, then by item_id within concept
    df_clean['concept_order'] = df_clean['concept'].map({c: i for i, c in enumerate(concepts)})
    df_clean = df_clean.sort_values(['concept_order', 'item_id'])
    
    # Get ordered item list
    ordered_items = df_clean.groupby(['concept', 'item_id']).size().reset_index()
    ordered_items = ordered_items.sort_values(['concept', 'item_id'])
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[3, 1])
    
    # Top plot: Box plots by item
    item_positions = []
    item_labels = []
    concept_positions = []
    concept_labels = []
    
    current_pos = 0
    for concept in concepts:
        concept_items = ordered_items[ordered_items['concept'] == concept]['item_id'].tolist()
        concept_start = current_pos
        
        for item in concept_items:
            item_data = df_clean[df_clean['item_id'] == item]['response']
            
            # Create box plot
            bp = ax1.boxplot(item_data, positions=[current_pos], widths=0.6, 
                           patch_artist=True, 
                           boxprops=dict(facecolor=concept_colors[concept], alpha=0.7),
                           medianprops=dict(color='black', linewidth=2))
            
            item_positions.append(current_pos)
            item_labels.append(f"{item}")
            current_pos += 1
        
        # Mark concept boundaries
        concept_positions.append((concept_start + current_pos - 1) / 2)
        concept_labels.append(concept)
        
        # Add vertical line between concepts (except after last)
        if concept != concepts[-1]:
            ax1.axvline(x=current_pos - 0.5, color='gray', linestyle='--', alpha=0.5)
    
    # Customize top plot
    ax1.set_xticks(item_positions)
    ax1.set_xticklabels(item_labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Response Value', fontsize=12)
    ax1.set_title(f'{model_name} - Response Distribution by Item (Grouped by Concept)', 
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add concept labels on secondary x-axis
    ax1_concept = ax1.twiny()
    ax1_concept.set_xlim(ax1.get_xlim())
    ax1_concept.set_xticks(concept_positions)
    ax1_concept.set_xticklabels(concept_labels, fontsize=12, fontweight='bold')
    
    # Bottom plot: Sample sizes by item
    sample_sizes = []
    for item in ordered_items['item_id']:
        count = len(df_clean[df_clean['item_id'] == item])
        sample_sizes.append(count)
    
    bars = ax2.bar(item_positions, sample_sizes, color=[concept_colors[concept] 
                   for concept in ordered_items['concept']], alpha=0.7)
    
    ax2.set_xticks(item_positions)
    ax2.set_xticklabels(item_labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Sample Size', fontsize=12)
    ax2.set_title('Sample Sizes by Item', fontsize=12)
    ax2.grid(True, alpha=0.3)
    
    # Add concept boundaries to bottom plot
    for concept_idx in range(len(concepts) - 1):
        boundary_pos = item_positions[sum(len(ordered_items[ordered_items['concept'] == c]) 
                                         for c in concepts[:concept_idx + 1])] - 0.5
        ax2.axvline(x=boundary_pos, color='gray', linestyle='--', alpha=0.5)
    
    # Create legend
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=concept_colors[concept], 
                                   alpha=0.7, label=concept) for concept in concepts]
    ax1.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
    
    # Add summary statistics
    total_items = len(ordered_items)
    total_responses = len(df_clean)
    avg_responses_per_item = total_responses / total_items if total_items > 0 else 0
    
    fig.suptitle(f'{model_name} Response Analysis\n'
                f'Items: {total_items} | Responses: {total_responses:,} | '
                f'Avg per Item: {avg_responses_per_item:.1f}', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save plot if requested
    if save_plot:
        filename = f'{model_name.lower()}_response_distribution_by_item.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    # Print summary statistics
    print(f"\n📊 {model_name} Response Distribution Summary:")
    print(f"   Total unique items: {total_items}")
    print(f"   Total responses: {total_responses:,}")
    print(f"   Average responses per item: {avg_responses_per_item:.1f}")
    print(f"   Response range: {df_clean['response'].min():.2f} to {df_clean['response'].max():.2f}")
    
    concept_summary = df_clean.groupby('concept').agg({
        'item_id': 'nunique',
        'response': ['count', 'mean', 'std']
    }).round(2)
    print(f"\n   By concept:")
    for concept in concepts:
        items = concept_summary.loc[concept, ('item_id', 'nunique')]
        responses = concept_summary.loc[concept, ('response', 'count')]
        mean_resp = concept_summary.loc[concept, ('response', 'mean')]
        std_resp = concept_summary.loc[concept, ('response', 'std')]
        print(f"     {concept}: {items} items, {responses} responses, "
              f"mean={mean_resp:.2f}, std={std_resp:.2f}")
    
    return fig




def plot_means_by_demographics(df, model_name, figsize=(16, 12), save_plot=True):
    """
    Create subplots showing mean responses by demographic groups for each concept
    
    Args:
        df: DataFrame with columns 'concept', 'response', demographic columns
        model_name: String name for the model (for titles and saving)
        figsize: Tuple for figure size
        save_plot: Boolean to save the plot as PNG
    
    Returns:
        matplotlib figure object
    """
    
    # Clean data
    df_clean = df.dropna(subset=['response', 'concept']).copy()
    
    # Get unique concepts
    concepts = sorted(df_clean['concept'].unique())
    
    # Demographic columns to analyze
    demo_cols = ['race', 'gender', 'religion', 'gender_alignment']
    available_demos = [col for col in demo_cols if col in df_clean.columns]
    
    # Create subplot grid: 4 demographic categories × number of concepts
    n_demos = len(available_demos)
    n_concepts = len(concepts)
    
    fig, axes = plt.subplots(n_demos, n_concepts, figsize=figsize, 
                            sharex=False, sharey=True)
    
    # Handle case where there's only one concept or one demographic
    if n_concepts == 1:
        axes = axes.reshape(-1, 1)
    if n_demos == 1:
        axes = axes.reshape(1, -1)
    
    # Color palette for concepts
    concept_colors = dict(zip(concepts, sns.color_palette("Set2", n_concepts)))
    
    # Process each demographic and concept combination
    for demo_idx, demo_col in enumerate(available_demos):
        for concept_idx, concept in enumerate(concepts):
            ax = axes[demo_idx, concept_idx]
            
            # Filter data for this concept
            concept_data = df_clean[df_clean['concept'] == concept].copy()
            
            # Remove rows where demographic is missing or 'not_mentioned'
            concept_data = concept_data[
                (concept_data[demo_col].notna()) & 
                (concept_data[demo_col] != 'not_mentioned')
            ]
            
            if len(concept_data) > 0:
                # Calculate means and standard errors by demographic group
                demo_stats = concept_data.groupby(demo_col)['response'].agg([
                    'mean', 'std', 'count', 'sem'
                ]).reset_index()
                
                # Create bar plot
                bars = ax.bar(demo_stats[demo_col], demo_stats['mean'], 
                             color=concept_colors[concept], alpha=0.7,
                             edgecolor='black', linewidth=0.5)
                
                # Add error bars
                ax.errorbar(demo_stats[demo_col], demo_stats['mean'], 
                           yerr=demo_stats['sem'], fmt='none', color='black', 
                           capsize=3, capthick=1)
                
                # Add sample sizes on bars
                for i, (bar, count) in enumerate(zip(bars, demo_stats['count'])):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + demo_stats['sem'].iloc[i],
                           f'n={int(count)}', ha='center', va='bottom', fontsize=8)
                
                # Customize axis
                ax.set_title(f'{concept}', fontsize=12, fontweight='bold')
                ax.tick_params(axis='x', rotation=45, labelsize=9)
                ax.grid(True, alpha=0.3)
                
                # Set y-axis label only for leftmost column
                if concept_idx == 0:
                    ax.set_ylabel(f'{demo_col.replace("_", " ").title()}\nMean Response', 
                                 fontsize=11, fontweight='bold')
                
                # Add overall statistics text
                overall_mean = concept_data['response'].mean()
                ax.axhline(y=overall_mean, color='red', linestyle='--', alpha=0.5, linewidth=1)
                
            else:
                # No data for this combination
                ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12, color='gray')
                ax.set_title(f'{concept}', fontsize=12, fontweight='bold')
                if concept_idx == 0:
                    ax.set_ylabel(f'{demo_col.replace("_", " ").title()}\nMean Response', 
                                 fontsize=11, fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'{model_name} - Mean Responses by Demographics Across Concepts', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    # Save plot if requested
    if save_plot:
        filename = f'{model_name.lower()}_means_by_demographics.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    # Print summary statistics
    print(f"\n📊 {model_name} Demographics Summary:")
    for concept in concepts:
        concept_data = df_clean[df_clean['concept'] == concept]
        print(f"\n   {concept}:")
        
        for demo_col in available_demos:
            if demo_col in concept_data.columns:
                demo_data = concept_data[
                    (concept_data[demo_col].notna()) & 
                    (concept_data[demo_col] != 'not_mentioned')
                ]
                
                if len(demo_data) > 0:
                    demo_means = demo_data.groupby(demo_col)['response'].agg(['mean', 'count'])
                    print(f"     {demo_col}:")
                    for group, stats in demo_means.iterrows():
                        print(f"       {group}: {stats['mean']:.2f} (n={int(stats['count'])})")
    
    return fig


def plot_demographic_comparison_single(df, model_name, figsize=(12, 8), save_plot=True):
    """
    Create a single comprehensive plot comparing all demographics across concepts
    
    Args:
        df: DataFrame with columns 'concept', 'response', demographic columns
        model_name: String name for the model
        figsize: Tuple for figure size
        save_plot: Boolean to save the plot as PNG
    
    Returns:
        matplotlib figure object
    """
    
    # Clean data
    df_clean = df.dropna(subset=['response', 'concept']).copy()
    
    # Get unique concepts
    concepts = sorted(df_clean['concept'].unique())
    
    # Demographic columns to analyze
    demo_cols = ['race', 'gender', 'religion', 'gender_alignment']
    available_demos = [col for col in demo_cols if col in df_clean.columns]
    
    # Create figure with subplots for each concept
    fig, axes = plt.subplots(1, len(concepts), figsize=figsize, sharey=True)
    
    if len(concepts) == 1:
        axes = [axes]
    
    # Color palette for demographics
    demo_colors = dict(zip(available_demos, sns.color_palette("Set1", len(available_demos))))
    
    for concept_idx, concept in enumerate(concepts):
        ax = axes[concept_idx]
        concept_data = df_clean[df_clean['concept'] == concept].copy()
        
        # Collect all demographic means for this concept
        x_pos = 0
        x_labels = []
        x_positions = []
        
        for demo_col in available_demos:
            # Remove not_mentioned values
            demo_data = concept_data[
                (concept_data[demo_col].notna()) & 
                (concept_data[demo_col] != 'not_mentioned')
            ]
            
            if len(demo_data) > 0:
                demo_means = demo_data.groupby(demo_col)['response'].agg(['mean', 'sem', 'count'])
                
                for group, stats in demo_means.iterrows():
                    ax.bar(x_pos, stats['mean'], color=demo_colors[demo_col], 
                          alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.errorbar(x_pos, stats['mean'], yerr=stats['sem'], 
                               fmt='none', color='black', capsize=3)
                    
                    # Add sample size
                    ax.text(x_pos, stats['mean'] + stats['sem'], f'n={int(stats["count"])}',
                           ha='center', va='bottom', fontsize=8)
                    
                    x_labels.append(f"{demo_col[:4]}:\n{group}")
                    x_positions.append(x_pos)
                    x_pos += 1
                
                # Add separator between demographic categories
                if demo_col != available_demos[-1]:
                    ax.axvline(x=x_pos - 0.5, color='gray', linestyle=':', alpha=0.5)
        
        # Customize axis
        ax.set_title(f'{concept}', fontsize=14, fontweight='bold')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Set y-axis label only for leftmost plot
        if concept_idx == 0:
            ax.set_ylabel('Mean Response', fontsize=12, fontweight='bold')
    
    # Add overall title
    fig.suptitle(f'{model_name} - Demographics Comparison Across Concepts', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    # Save plot if requested
    if save_plot:
        filename = f'{model_name.lower()}_demographics_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Plot saved as: {filename}")
    
    return fig



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def compute_global_yaxis(dfs, col="response", pad=0.10, step=5, min_ymax=5):
    """
    Compute shared y-axis max and yticks for multiple datasets.
    """
    global_max = 0
    for df in dfs:
        data = pd.to_numeric(df[col], errors="coerce").dropna().round().astype(int)
        data = data[(data >= -5) & (data <= 5)]
        if len(data) == 0:
            continue
        global_max = max(global_max, int(data.value_counts().max()))

    ymax = max(min_ymax, int(np.ceil(global_max * (1 + pad))))

    # round ymax up to next tick step
    ymax = int(np.ceil(ymax / step) * step)
    yticks = np.arange(0, ymax + 1, step)

    return ymax, yticks

def plot_response_histogram(
    df,
    model_name,
    save=True,
    ymax=None,
    yticks=None,
):
    # Theme / typography
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1.6)

    # ensure integer discrete responses in [-5, 5]
    data = pd.to_numeric(df["response"], errors="coerce").dropna().round().astype(int)
    data = data[(data >= -5) & (data <= 5)]

    palette = {"Llama": "#1f77b4", "Mistral": "#ff7f0e", "Qwen": "#2ca02c"}
    color = palette.get(model_name, "#4c72b0")

    vals = list(range(-5, 6))
    counts = data.value_counts().reindex(vals, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        vals,
        counts.values,
        color=color,
        edgecolor=color,
        linewidth=0,
        alpha=0.9,
    )

    # annotate counts
    for bar, cnt in zip(bars, counts.values):
        if cnt > 0:
            ax.annotate(
                str(int(cnt)),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(vals)
    ax.set_xlabel("Response")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{model_name} — Distribution of responses (counts)",
        fontsize=16,
        pad=10,
    )

    # force identical y-axis across plots
    if ymax is not None:
        ax.set_ylim(0, ymax)

    if yticks is not None:
        ax.set_yticks(yticks)

    # grid follows yticks exactly
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="gray", alpha=0.6)
    ax.xaxis.grid(False)

    sns.despine(trim=True)
    plt.tight_layout()

    if save:
        fname_base = f"hist_response_{model_name.lower()}"
        fig.savefig(fname_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fname_base + ".pdf", dpi=300, bbox_inches="tight")

    plt.show()
    return fig


def plot_response_histogram_recoded(
    df,
    model_name,
    save=True,
    ymax=None,
    yticks=None,
):
    # Theme / typography
    sns.set_theme(style="white")
    sns.set_context("paper", font_scale=1.6)

    # ensure integer discrete responses in [-5, 5]
    data = pd.to_numeric(df["response_recoded"], errors="coerce").dropna().round().astype(int)
    data = data[(data >= -5) & (data <= 5)]

    palette = {"Llama": "#1f77b4", "Mistral": "#ff7f0e", "Qwen": "#2ca02c"}
    color = palette.get(model_name, "#4c72b0")

    vals = list(range(-5, 6))
    counts = data.value_counts().reindex(vals, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(
        vals,
        counts.values,
        color=color,
        edgecolor=color,
        linewidth=0,
        alpha=0.9,
    )

    # annotate counts
    for bar, cnt in zip(bars, counts.values):
        if cnt > 0:
            ax.annotate(
                str(int(cnt)),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )

    ax.set_xticks(vals)
    ax.set_xlabel("Response (recoded)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"{model_name} — Distribution of responses (counts)",
        fontsize=16,
        pad=10,
    )

    # force identical y-axis across plots
    if ymax is not None:
        ax.set_ylim(0, ymax)

    if yticks is not None:
        ax.set_yticks(yticks)

    # grid follows yticks exactly
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, color="gray", alpha=0.6)
    ax.xaxis.grid(False)

    sns.despine(trim=True)
    plt.tight_layout()

    if save:
        fname_base = f"hist_response_recoded_{model_name.lower()}"
        fig.savefig(fname_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(fname_base + ".pdf", dpi=300, bbox_inches="tight")

    plt.show()
    return fig


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_response_histograms_fixed_scale(
    dfs,
    model_names,
    col="response_recoded",
    x_min=-5,
    x_max=5,
    y_max=575_000,
    y_step=50_000,
    figsize=(10, 6),
    font_scale=1.6,
    palette=None,
    save=False,
    out_dir=".",
    fname_prefix="hist_response",
    close=True,                 # keep default: don't display in notebooks
):
    """
    Plot one histogram (count bar chart) per model with fixed x/y scales.
    Works for both 'response' and 'response_recoded' via col=...
    """

    if len(dfs) != len(model_names):
        raise ValueError("dfs and model_names must have the same length.")

    # Ensure output directory exists if saving
    out_path = Path(out_dir)
    if save:
        out_path.mkdir(parents=True, exist_ok=True)

    # Bigger fonts everywhere
    plt.rcParams.update({
        "font.size": 10 * font_scale,
        "axes.titlesize": 11 * font_scale,
        "axes.labelsize": 10 * font_scale,
        "xtick.labelsize": 9 * font_scale,
        "ytick.labelsize": 9 * font_scale,
    })

    # Default palette if none given
    if palette is None:
        palette = {"Llama": "#1f77b4", "Mistral": "#ff7f0e", "Qwen": "#2ca02c"}

    vals = np.arange(x_min, x_max + 1)
    yticks = np.arange(0, y_max + 1, y_step)

    figs = []

    for df, model_name in zip(dfs, model_names):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in dataframe for model '{model_name}'.")

        s = pd.to_numeric(df[col], errors="coerce")
        s = s.dropna()
        s = s[(s >= x_min) & (s <= x_max)]
        s = s.round().astype(int)

        counts = s.value_counts().reindex(vals, fill_value=0)

        color = palette.get(model_name, None)

        fig, ax = plt.subplots(figsize=figsize)
        bars = ax.bar(
            vals,
            counts.values,
            color=color,
            edgecolor=color if color else None,
            linewidth=0,
            alpha=0.9
        )

        # ---- ADD COUNTS ON TOP OF BARS ----
        for bar, cnt in zip(bars, counts.values):
            if cnt > 0:
                ax.annotate(
                    f"{cnt:,}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),                 # vertical offset in points
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8 * font_scale,
                    fontweight="bold"
                )
        # ----------------------------------

        ax.set_xlim(x_min - 0.5, x_max + 0.5)
        ax.set_ylim(0, y_max)
        ax.set_xticks(vals)
        ax.set_yticks(yticks)

        ax.set_xlabel("Response")
        ax.set_ylabel("Count")
        ax.set_title(f"{model_name} — Distribution of {col}")

        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{int(x):,}"))
        ax.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        ax.grid(axis="x", visible=False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()

        if save:
            safe_model = str(model_name).lower().replace(" ", "_")
            safe_col = str(col).lower().replace(" ", "_")
            base = out_path / f"{fname_prefix}_{safe_col}_{safe_model}"
            fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")
            fig.savefig(str(base) + ".pdf", dpi=300, bbox_inches="tight")

        figs.append(fig)
        if close:
            plt.close(fig)

    return figs, y_max, yticks






def collapse_vignette_responses(
    df,
    id_cols=("item_id", "vignette_id"),
    response_cols=("response", "response_recoded"),
    metadata_strategy="first",
    verbose=True
):
    """
    Collapse raw LLM responses to mean values per (item_id, vignette_id)
    while preserving metadata.

    Args:
        df: pandas DataFrame
        id_cols: columns identifying a unique vignette-item combination
        response_cols: numeric columns to average
        metadata_strategy: "first" or "check"
            - "first": take first value in group (efficient, default)
            - "check": verify all values identical, raise warning if not
        verbose: print diagnostics

    Returns:
        DataFrame collapsed to one row per (item_id, vignette_id)
    """

    df = df.copy()

    # -------------------------
    # Ensure required columns exist
    # -------------------------
    for col in id_cols + response_cols:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in DataFrame")

    # -------------------------
    # Identify metadata columns
    # -------------------------
    meta_cols = [c for c in df.columns
                 if c not in id_cols and c not in response_cols]

    if verbose:
        print(f"Collapsing {len(df):,} rows → 1 per {id_cols}")
        print(f"Response columns averaged: {list(response_cols)}")
        print(f"Metadata columns preserved: {len(meta_cols)}")

    # -------------------------
    # Aggregation dictionary
    # -------------------------
    agg_dict = {}

    # average numeric response columns
    for c in response_cols:
        agg_dict[c] = "mean"

    # metadata columns
    if metadata_strategy == "first":
        for c in meta_cols:
            agg_dict[c] = "first"

    elif metadata_strategy == "check":
        # This strict option ensures metadata consistency
        for c in meta_cols:
            agg_dict[c] = lambda x, col=c: (
                x.iloc[0] if x.nunique() == 1
                else (_warn_inconsistent(col, x))
            )
    else:
        raise ValueError("metadata_strategy must be 'first' or 'check'")

    # -------------------------
    # Perform aggregation
    # -------------------------
    collapsed = (
        df.groupby(list(id_cols), as_index=False)
          .agg(agg_dict)
    )

    if verbose:
        print(f"Collapsed dataset size: {len(collapsed):,} rows")

    return collapsed


def _warn_inconsistent(col, series):
    print(f"WARNING: Column '{col}' has inconsistent values in group!")
    return series.iloc[0]




def compute_scm_scores(df, response_col='response', concept_col='concept', 
                       groupby_col='vignette_id', keep_cols=None, use_absolute=False):
    """
    Compute SCM warmth and competence scores from BIAS map emotion/behavior items.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with individual responses
    response_col : str, default='response'
        Column name containing response values (-5 to +5)
    concept_col : str, default='concept'
        Column name containing BIAS map categories ('ADM', 'PIT', 'ENV', 'CON', 
        'active_facilitation', 'passive_facilitation', 'active_harm', 'passive_harm')
    groupby_col : str, default='vignette_id'
        Column name to group by for aggregation (e.g., vignette ID or stratum ID)
    keep_cols : list of str, optional
        Additional columns to keep in aggregated output (e.g., ['gender', 'race', 'religion'])
        These will be aggregated using 'first' (assumes they're constant within groups)
    
    Returns
    -------
    pd.DataFrame
        Aggregated dataframe with warmth_score and competence_score for each group
    
    Notes
    -----
    - Emotion items (ADM, PIT, ENV, CON) contribute to both warmth and competence
    - Behavioral items contribute to only one dimension:
        * active_facilitation, active_harm → warmth only
        * passive_facilitation, passive_harm → competence only
    - Items with polarity=0 for a dimension are excluded from that dimension's mean
    - Response values of 0 are meaningful (neutral) and included in calculations
    
    Examples
    --------
    >>> df_scm = compute_scm_scores(
    ...     df, 
    ...     groupby_col='vignette_id',
    ...     keep_cols=['gender', 'race', 'religion', 'transness']
    ... )
    """
    
    # Define polarity mappings
    warmth_polarity = {
        # Emotion items (measure both dimensions)
        "ADM": +1,   # Admiration: high warmth
        "PIT": +1,   # Pity: high warmth
        "ENV": -1,   # Envy: low warmth
        "CON": -1,   # Contempt: low warmth
        # Behavioral items (measure one dimension)
        "AF": +1,    # High warmth
        "AH": -1,            # Low warmth
        "PF": 0,    # Doesn't measure warmth
        "PH": 0,            # Doesn't measure warmth
    }
    
    competence_polarity = {
        # Emotion items (measure both dimensions)
        "ADM": +1,   # Admiration: high competence
        "PIT": -1,   # Pity: low competence
        "ENV": +1,   # Envy: high competence
        "CON": -1,   # Contempt: low competence
        # Behavioral items (measure one dimension)
        "AF": 0,     # Doesn't measure competence
        "AH": 0,             # Doesn't measure competence
        "PF": +1,   # High competence
        "PH": -1,           # Low competence
    }
    
     # Create working copy
    df_work = df.copy()
    
    # Map polarities
    df_work['warmth_polarity'] = df_work[concept_col].map(warmth_polarity)
    df_work['competence_polarity'] = df_work[concept_col].map(competence_polarity)
    
    # Check for unmapped concepts
    if df_work['warmth_polarity'].isna().any():
        unmapped = df_work[df_work['warmth_polarity'].isna()][concept_col].unique()
        raise ValueError(f"Unmapped concepts found: {unmapped}")
    
    # Use absolute values if requested
    if use_absolute:
        response_values = df_work[response_col].abs()
    else:
        response_values = df_work[response_col]
    
    # Compute dimension scores for each response
    df_work['warmth_item'] = df_work['warmth_polarity'] * response_values
    df_work['competence_item'] = df_work['competence_polarity'] * response_values
    
    # Create flags for which items measure which dimensions
    df_work['measures_warmth'] = df_work['warmth_polarity'] != 0
    df_work['measures_competence'] = df_work['competence_polarity'] != 0
    
    # Prepare aggregation dictionary
    agg_dict = {
        'warmth_item': lambda x: x[df_work.loc[x.index, 'measures_warmth']].mean(),
        'competence_item': lambda x: x[df_work.loc[x.index, 'measures_competence']].mean(),
        'measures_warmth': 'sum',
        'measures_competence': 'sum',
    }
    
    # Add additional columns to keep
    if keep_cols:
        for col in keep_cols:
            if col not in df_work.columns:
                raise ValueError(f"Column '{col}' not found in dataframe")
            agg_dict[col] = 'first'
    
    # Aggregate by group
    df_result = df_work.groupby(groupby_col).agg(agg_dict).reset_index()
    
    # Rename columns for clarity
    df_result = df_result.rename(columns={
        'warmth_item': 'warmth_score',
        'competence_item': 'competence_score',
        'measures_warmth': 'n_warmth_items',
        'measures_competence': 'n_competence_items'
    })
    
    return df_result
