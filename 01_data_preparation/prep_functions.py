import json
import pandas as pd
from pathlib import Path
import os
import numpy as np
import re

def combine_llm_results(file_pattern, output_name):
    """Combine JSONL files by item_id and vignette_id"""
    
    print(f"Combining files matching: {file_pattern}")
    combined_data = {}
    
    for file_path in Path('.').glob(file_pattern):
        print(f"   Processing {file_path}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    
                    # Create unique key from item_id and vignette_id
                    key = (entry['item_id'], entry['vignette_id'])
                    
                    if key not in combined_data:
                        combined_data[key] = {
                            'item_id': entry['item_id'],
                            'vignette_id': entry['vignette_id'],
                            'responses': []
                        }
                    
                    # Add responses from this file
                    combined_data[key]['responses'].extend(entry['responses'])
    
    # Convert to DataFrame with one row per response
    result_list = []
    for key, data in combined_data.items():
        for i, response in enumerate(data['responses']):
            result_list.append({
                'item_id': data['item_id'],
                'vignette_id': data['vignette_id'],
                'response_index': i + 1,
                'response': response
            })
    
    df = pd.DataFrame(result_list)
    df.to_csv(f'{output_name}.csv', index=False, encoding='utf-8')
    
    print(f"Combined {len(combined_data)} unique items with {len(result_list)} total responses")
    print(f"Saved as {output_name}.csv")
    
    return df


def load_all_llm_data(data_dir="C:\\Users\\ThinkPad\\Desktop\\llm_biases_fse\\00_data\\raw"):
    """
    Load and combine all LLM results from the specified directory
    Returns dictionary with DataFrames for each model
    """
    
    print("Loading and combining all LLM data")
    print("=" * 60)
    
    # Change to data directory
    original_dir = os.getcwd()
    os.chdir(data_dir)
    print(f"Working in directory: {data_dir}")
    
    results = {}
    
    # Process Llama results
    print("\n Processing Llama results")
    try:
        llama_df = combine_llm_results('llm_results_chatformat_llama*.jsonl', 'combined_llama_results')
        results['llama'] = llama_df
        print(f" Llama: {len(llama_df):,} responses loaded")
    except Exception as e:
        print(f"Error processing Llama: {e}")
        results['llama'] = None
    
    # Process Qwen results
    print("\n Processing Qwen results")
    try:
        qwen_df = combine_llm_results('llm_results_chatformat_qwen_independent*.jsonl', 'combined_qwen_results')
        results['qwen'] = qwen_df
        print(f" Qwen: {len(qwen_df):,} responses loaded")
    except Exception as e:
        print(f"Error processing Qwen: {e}")
        results['qwen'] = None
    
    # Process Mistral results
    print("\n Processing Mistral results")
    try:
        mistral_df = combine_llm_results('llm_results_chatformat_mistral*.jsonl', 'combined_mistral_results')
        results['mistral'] = mistral_df
        print(f" Mistral: {len(mistral_df):,} responses loaded")
    except Exception as e:
        print(f"Error processing Mistral: {e}")
        results['mistral'] = None
    
    # Return to original directory
    os.chdir(original_dir)
    
    print("\n COMPLETE!")
    print(f"Loaded {sum(1 for v in results.values() if v is not None)} out of 3 models")
    
    return results



def balance_dataset(df, item_col=None, vignette_col=None, response_col="response_index"):
    """
    Balance dataset by trimming combinations with >100 responses to exactly 100.
    Report combinations with <100 responses but keep original data.
    
    Args:
        df: Input DataFrame
        item_col: Column name for item ID (auto-detected if None)
        vignette_col: Column name for vignette ID (auto-detected if None) 
        response_col: Column name for response index
        
    Returns:
        tuple: (processed_df, report_dict)
    """
    # Auto-detect column names if not provided
    if item_col is None:
        possible_item_cols = ['item_id', 'item', 'item_name']
        item_col = next((col for col in possible_item_cols if col in df.columns), None)
    
    if vignette_col is None:
        possible_vignette_cols = ['vignette_id', 'vignette', 'vignette_number']
        vignette_col = next((col for col in possible_vignette_cols if col in df.columns), None)
    
    if response_col not in df.columns:
        response_col = 'response_index'
    
    print(f"Using columns: item_col='{item_col}', vignette_col='{vignette_col}', response_col='{response_col}'")
    
    # Check if we found the required columns
    if item_col is None or item_col not in df.columns:
        print(f"ERROR: Could not find item column. Available columns: {df.columns.tolist()}")
        return df, {"error": "item_col not found"}
    
    if vignette_col is None or vignette_col not in df.columns:
        print(f"ERROR: Could not find vignette column. Available columns: {df.columns.tolist()}")
        return df, {"error": "vignette_col not found"}
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Analyze group sizes
    group_sizes = df.groupby([item_col, vignette_col]).size()
    print(f"Found {len(group_sizes)} unique combinations of {item_col} x {vignette_col}")
    print(f"Group sizes: min={group_sizes.min()}, max={group_sizes.max()}, mean={group_sizes.mean():.1f}")
    
    # Initialize report
    report = {
        "gekürzt": 0, 
        "unverändert": 0, 
        "less_than_100": 0,
        "gekürzt_gruppen": [], 
        "unverändert_gruppen": [], 
        "less_than_100_gruppen": []
    }
    
    # Process each group manually
    result_dfs = []
    
    for (item_id, vignette_id), group in df.groupby([item_col, vignette_col]):
        n = len(group)
        key = (item_id, vignette_id)
        
        if n > 100:
            # Trim to exactly 100 (take first 100 rows)
            report["gekürzt"] += 1
            report["gekürzt_gruppen"].append(key)
            group_processed = group.sample(n=100, replace=False, random_state=42).copy()
            
        elif n < 100:
            # Keep original data, but report it
            report["less_than_100"] += 1
            report["less_than_100_gruppen"].append((key, n))  # Include count
            group_processed = group.copy()
            
        else:
            # Exactly 100 - perfect
            report["unverändert"] += 1
            report["unverändert_gruppen"].append(key)
            group_processed = group.copy()
        
        # Reset index and assign sequential response_index
        group_processed = group_processed.reset_index(drop=True)
        group_processed[response_col] = range(1, len(group_processed) + 1)
        result_dfs.append(group_processed)
    
    # Concatenate all processed groups
    df_balanced = pd.concat(result_dfs, ignore_index=True)
    
    # Report results
    print(f"\n PROCESSING SUMMARY:")
    print(f"   Trimmed (>100):     {report['gekürzt']:,} combinations")
    print(f"   Unchanged (=100):   {report['unverändert']:,} combinations") 
    print(f"   Less than 100:     {report['less_than_100']:,} combinations")
    
    if report['less_than_100'] > 0:
        print(f"\n  COMBINATIONS WITH <100 RESPONSES:")
        for (item_id, vignette_id), count in report['less_than_100_gruppen']:
            print(f"   {item_id} × {vignette_id}: {count} responses")
        
        # Create DataFrame for easy analysis
        less_than_100_df = pd.DataFrame([
            {"item_id": item_id, "vignette_id": vignette_id, "n_responses": count}
            for (item_id, vignette_id), count in report['less_than_100_gruppen']
        ])
        report['less_than_100_df'] = less_than_100_df
        
        print(f"\n DISTRIBUTION OF <100 GROUPS:")
        count_dist = less_than_100_df['n_responses'].value_counts().sort_index()
        for resp_count, freq in count_dist.items():
            print(f"   {resp_count} responses: {freq} combinations")
    
    return df_balanced, report


def merge_with_metadata_all(llm_results_dict, metadata_file="C:\\Users\\ThinkPad\\Desktop\\llm_biases_fse\\00_data\\complete_vignette_llm.xlsx"):
    """
    Load metadata and merge with all LLM DataFrames
    
    Args:
        llm_results_dict: Dictionary from load_all_llm_data()
        metadata_file: Path to metadata file
    
    Returns:
        Dictionary with merged DataFrames
    """
    
    print("\n Loading metadata and merging with LLM data")
    print("=" * 60)
    
    # Load vignette metadata
    print("Loading vignette metadata...")
    try:
        vignette_df = pd.read_excel(metadata_file)
        print(f"Loaded {len(vignette_df)} vignettes with metadata")
        print(f"Columns: {list(vignette_df.columns)}")
    except FileNotFoundError:
        print(f"ERROR: {metadata_file} not found!")
        return {}
    except Exception as e:
        print(f"Error loading metadata: {e}")
        return {}
    
    merged_results = {}
    
    # Helper function for merging
    def merge_single_model(df, model_name):
        """Merge single model DataFrame with metadata"""
        
        print(f"\n Merging {model_name} with vignette metadata...")
        
        # Extract vignette number from "vignette_X" format
        def extract_vignette_number(vignette_id):
            if isinstance(vignette_id, str) and vignette_id.startswith('vignette_'):
                return int(vignette_id.replace('vignette_', ''))
            return vignette_id
        
        # Apply mapping and ensure data types match
        df['vignette_number'] = df['vignette_id'].apply(extract_vignette_number)
        vignette_df['vignette_id'] = vignette_df['vignette_id'].astype(int)
        
        # Merge datasets - now including n_dim
        merged_df = pd.merge(
            df, 
            vignette_df[['vignette_id', 'race', 'gender', 'gender_alignment', 'religion', 'n_dim']], 
            left_on='vignette_number',
            right_on='vignette_id', 
            how='left',
            suffixes=('', '_excel')
        )
        
        # Clean up temporary columns
        merged_df = merged_df.drop(['vignette_number', 'vignette_id_excel'], axis=1)
        
        # Analyze merge results
        total_entries = len(merged_df)
        entries_with_demographics = merged_df['race'].notna().sum()
        entries_with_nas = merged_df['race'].isna().sum()
        
        print(f" {model_name} merge completed!")
        print(f"   Total entries: {total_entries:,}")
        print(f"   With demographic data: {entries_with_demographics:,} ({entries_with_demographics/total_entries*100:.1f}%)")
        print(f"   With NAs (by design): {entries_with_nas:,} ({entries_with_nas/total_entries*100:.1f}%)")
        
        # Show n_dim distribution
        n_dim_counts = merged_df['n_dim'].value_counts().sort_index()
        print(f"   n_dim distribution: {dict(n_dim_counts)}")
        
        return merged_df
    
    # Merge each model
    for model_name, df in llm_results_dict.items():
        if df is not None:
            try:
                merged_df = merge_single_model(df, model_name)
                merged_results[model_name] = merged_df
                
                # Save merged dataset
                output_file = f'combined_{model_name}_results_with_metadata.csv'
                merged_df.to_csv(output_file, index=False, encoding='utf-8')
                print(f" Saved as {output_file}")
                
            except Exception as e:
                print(f"Error merging {model_name}: {e}")
                merged_results[model_name] = None
        else:
            print(f"Skipping {model_name} (no data loaded)")
            merged_results[model_name] = None
    
    print("\n STEP 2 COMPLETE!")
    print("=" * 60)
    
    # Summary
    successful_merges = sum(1 for v in merged_results.values() if v is not None)
    print(f"Successfully merged {successful_merges} out of {len(llm_results_dict)} models")
    
    print("\n📁 Files created:")
    for model_name, df in merged_results.items():
        if df is not None:
            print(f"   • combined_{model_name}_results.csv")
            print(f"   • combined_{model_name}_results_with_metadata.csv")
    
    print("\n Final Statistics:")
    for model_name, df in merged_results.items():
        if df is not None:
            print(f"   {model_name.capitalize()}: {len(df):,} total responses")
    
    return merged_results


def main():
    """Main function for two-step process"""
    
    print(" LLM Data Preparation - Two Step Process")
    print("=" * 60)
    
    # Step 1: Load all LLM data
    llm_data = load_all_llm_data()
    
    # Step 2: Merge with metadata
    if any(df is not None for df in llm_data.values()):
        merged_data = merge_with_metadata_all(llm_data)
        
        print("\n ALL PROCESSING COMPLETE!")
        return merged_data
    else:
        print("\n No LLM data could be loaded. Check file paths and patterns.")
        return {}


if __name__ == "__main__":
    main()


## Concept mapping
def add_concept_mapping(df):
    """Add concept mapping based on item_id"""
    # Create `item_id` and `vignette_id` from numeric columns when missing
    if 'item_id' not in df.columns:
        if 'item' in df.columns:
            df['item_id'] = df['item'].astype(str).apply(lambda x: f"item_{x}")
        else:
            df['item_id'] = np.nan

    if 'vignette_id' not in df.columns:
        if 'vignr' in df.columns:
            df['vignette_id'] = df['vignr'].astype(str)
        elif 'vignette' in df.columns:
            df['vignette_id'] = df['vignette'].astype(str)
        else:
            df['vignette_id'] = np.nan

    # initialize concept column as object dtype so we can assign strings safely
    df['concept'] = pd.Series(index=df.index, dtype=object)

    # If the dataset uses a simple 1-4 coding for `item`, map directly to broad concepts
    # 1 -> Admiration (ADM), 2 -> Envy (ENV), 3 -> Pity (PIT), 4 -> Contempt (CON)
    try:
        if 'item' in df.columns:
            item_nums = pd.to_numeric(df['item'], errors='coerce').dropna().unique().astype(int)
            if set(item_nums).issubset({1, 2, 3, 4}):
                map_simple = {1: 'ADM', 2: 'ENV', 3: 'PIT', 4: 'CON'}
                df['concept'] = pd.to_numeric(df['item'], errors='coerce').map(lambda x: map_simple.get(int(x)) if pd.notna(x) else None)
                print('Applied simple item(1-4) -> concept mapping')
                return df
    except Exception:
        # Fall back to the more detailed mapping below
        pass
    
    # Active
    df.loc[df['item_id'].isin(["item_1", "item_13", "item_29", "item_49", "item_51"]), 'concept'] = "AF"
    df.loc[df['item_id'].isin(["item_2", "item_14", "item_30", "item_50"]), 'concept'] = "AF"
    df.loc[df['item_id'].isin(["item_5", "item_17", "item_33", "item_55", "item_57"]), 'concept'] = "AH"
    df.loc[df['item_id'].isin(["item_6", "item_18", "item_34", "item_56"]), 'concept'] = "AH"

    # Passive
    df.loc[df['item_id'].isin(["item_3", "item_15", "item_31", "item_52", "item_54"]), 'concept'] = "PF"
    df.loc[df['item_id'].isin(["item_4", "item_16", "item_32", "item_53"]), 'concept'] = "PF"
    df.loc[df['item_id'].isin(["item_7", "item_19", "item_35", "item_58"]), 'concept'] = "PH"
    df.loc[df['item_id'].isin(["item_8", "item_20", "item_36", "item_59", "item_60"]), 'concept'] = "PH"

    # Emotions
    df.loc[df['item_id'].isin(["item_9", "item_21", "item_25", "item_37", "item_38", "item_39", "item_61", "item_62", "item_63"]), 'concept'] = "ADM"
    df.loc[df['item_id'].isin(["item_11", "item_23", "item_27", "item_43", "item_44", "item_45", "item_67", "item_68", "item_69"]), 'concept'] = "PIT"
    df.loc[df['item_id'].isin(["item_10", "item_22", "item_26", "item_40", "item_41", "item_42", "item_64", "item_65", "item_66"]), 'concept'] = "ENV"
    df.loc[df['item_id'].isin(["item_12", "item_24", "item_28", "item_46", "item_47", "item_48", "item_70", "item_71", "item_72"]), 'concept'] = "CON"
    
    print(f"Concept distribution:")
    print(df['concept'].value_counts())
    
    return df

def add_concept_to_all(merged_results):
    """Add concept mapping to all merged DataFrames"""
    print("\n Adding concept mapping to all merged DataFrames")
    print("=" * 60)
    
    concept_results = {}
    
    for model_name, df in merged_results.items():
        if df is not None:
            try:
                print(f"\n Adding concepts to {model_name}...")
                df_with_concept = add_concept_mapping(df)
                concept_results[model_name] = df_with_concept
                
                # Save updated dataset
                output_file = f'combined_{model_name}_results_with_metadata_and_concept.csv'
                df_with_concept.to_csv(output_file, index=False, encoding='utf-8')
                print(f" Saved as {output_file}")
                
            except Exception as e:
                print(f"Error adding concepts to {model_name}: {e}")
                concept_results[model_name] = None
        else:
            print(f"Skipping {model_name} (no data loaded)")
            concept_results[model_name] = None
    
    print("\n COMPLETE!")
    print("=" * 60)
    
    # Summary
    successful_concepts = sum(1 for v in concept_results.values() if v is not None)
    print(f"Successfully added concepts to {successful_concepts} out of {len(merged_results)} models")
    
    return concept_results

def recode_demographic_columns(df, df_name=""):
    """
    Efficiently recode demographic columns for a dataframe
    """
    print(f"\n Recoding demographic columns for {df_name}...")
    
    # Explizite, theoretisch sinnvolle Referenzkategorien
    if 'race' in df.columns:
        # Ensure 'not_mentioned' is present and used as reference category
        race_cats = ['not_mentioned', 'White', 'Black', 'Asian']
        df['race_cat'] = pd.Categorical(df['race'].fillna('not_mentioned'), categories=race_cats)
        print(f"   Race categories (not_mentioned=reference): {df['race_cat'].cat.categories.tolist()}")

    if 'religion' in df.columns:
        religion_cats = ['not_mentioned', 'christian', 'muslim', 'jewish']
        df['religion_cat'] = pd.Categorical(df['religion'].fillna('not_mentioned'), categories=religion_cats)
        print(f"   Religion categories (not_mentioned=reference): {df['religion_cat'].cat.categories.tolist()}")

    if 'gender' in df.columns:
        gender_cats = ['not_mentioned', 'man', 'woman', 'nonbinary']
        df['gender_cat'] = pd.Categorical(df['gender'].fillna('not_mentioned'), categories=gender_cats)
        print(f"   Gender categories (not_mentioned=reference): {df['gender_cat'].cat.categories.tolist()}")

    if 'gender_alignment' in df.columns:
        ga_cats = ['not_mentioned', 'cis', 'trans']
        df['gender_alignment_cat'] = pd.Categorical(df['gender_alignment'].fillna('not_mentioned'), categories=ga_cats)
        print(f"   Gender alignment categories (not_mentioned=reference): {df['gender_alignment_cat'].cat.categories.tolist()}")

    # Create clean columns with 'not_mentioned' for NAs
    df['race_clean'] = df['race'].fillna('not_mentioned')
    df['gender_clean'] = df['gender'].fillna('not_mentioned')
    df['religion_clean'] = df['religion'].fillna('not_mentioned')
    df['gender_alignment_clean'] = df['gender_alignment'].fillna('not_mentioned')

    # Identify scale score columns
    potential_scales = ['AF', 'AH', 'PF', 'PH', 'ADM', 'PIT', 'ENV', 'CON']
    scale_columns = [col for col in df.columns if col in potential_scales]
    
    if scale_columns:
        print(f"   Scale columns found: {scale_columns}")
    else:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        exclude_cols = ['vignette_id', 'vignette_id_x', 'vignette_id_y', 'n_dim', 'response_index']
        scale_columns = [col for col in numeric_cols if col not in exclude_cols]
        print(f"   Alternative scale columns: {scale_columns}")
    
    return df


def clean_response_column(df, df_name=""):
    """
    Clean the response column while preserving ALL information.
    
    Adds:
        - response_clean: numeric or NaN
        - response_status: categorical label describing what happened
        - response_text_original: original string
    
    Categories in response_status:
        - valid_numeric
        - extracted_numeric
        - refusal
        - no_number_found
        - original_nan
        - unparseable
    """

    print(f"\n Cleaning response column for {df_name}...")

    df = df.copy()
    df['response_text_original'] = df['response'].astype(str)

    # Initialize status
    df['response_status'] = "unprocessed"

    # Identify originally missing
    df.loc[df['response'].isna(), 'response_status'] = "original_nan"

    # Convert to string for consistent processing
    text = df['response_text_original']

    # ---------- 1. Detect pure numeric responses ----------
    numeric_response = pd.to_numeric(text, errors='coerce')
    numeric_mask = numeric_response.notna()

    df.loc[numeric_mask, 'response_clean'] = numeric_response[numeric_mask]
    df.loc[numeric_mask, 'response_status'] = "valid_numeric"

    # ---------- 2. Detect refusals ----------
    refusal_patterns = [
        r"cannot answer", r"can't answer", r"i cannot", r"i can't",
        r"as an ai", r"as a language model",
        r"unable to", r"not able to",
        r"i do not have", r"i don't have",
        r"i cannot provide", r"i can't provide",
        r"refuse", r"refusal",
        r"i apologize", r"sorry", r"cannot comply",
    ]

    refusal_mask = text.str.lower().str.contains("|".join(refusal_patterns), regex=True)
    df.loc[refusal_mask, 'response_clean'] = np.nan
    df.loc[refusal_mask, 'response_status'] = "refusal"

    # ---------- 3. Extract numbers from non-numeric text ----------
    # Only process rows not yet classified as numeric or refusal
    remaining_mask = (df['response_status'] == "unprocessed")

    extract_pattern = r'-?\d+\.?\d*'
    extracted = text[remaining_mask].str.extract(f'({extract_pattern})', expand=False)
    extracted_numeric = pd.to_numeric(extracted, errors='coerce')

    extracted_mask = extracted_numeric.notna()

    # Fill extracted values
    df.loc[remaining_mask & extracted_mask, 'response_clean'] = extracted_numeric[extracted_mask]
    df.loc[remaining_mask & extracted_mask, 'response_status'] = "extracted_numeric"

    # ---------- 4. Mark remaining as no-number-found ----------
    leftover_mask = (df['response_status'] == "unprocessed")
    df.loc[leftover_mask, 'response_clean'] = np.nan
    df.loc[leftover_mask, 'response_status'] = "no_number_found"

    # ---------- Report ----------
    counts = df['response_status'].value_counts()
    print("\n Response Cleaning Summary:")
    for status, count in counts.items():
        print(f"  {status:20s}: {count:,} ({count/len(df)*100:.2f}%)")

    numeric_mask_final = df['response_clean'].notna()
    if numeric_mask_final.sum() > 0:
        print("\n Numeric summary:")
        print(f"  Range: {df['response_clean'].min():.2f} to {df['response_clean'].max():.2f}")
        print(f"  Mean:  {df['response_clean'].mean():.2f}")

    return df



def clean_responses_all_models(model_dict):
    """
    Clean response columns for all models in the dictionary
    
    Args:
        model_dict: Dictionary with model DataFrames
    
    Returns:
        Dictionary with cleaned DataFrames
    """
    print("\n CLEANING RESPONSE COLUMNS FOR ALL MODELS")
    print("=" * 60)
    
    cleaned_results = {}
    
    for model_name, df in model_dict.items():
        if df is not None:
            try:
                cleaned_df = clean_response_column(df, model_name.capitalize())
                cleaned_results[model_name] = cleaned_df
                
            except Exception as e:
                print(f" Error cleaning {model_name}: {e}")
                cleaned_results[model_name] = df  # Return original if cleaning fails
        else:
            print(f"  Skipping {model_name} (no data loaded)")
            cleaned_results[model_name] = None
    
    print("\n RESPONSE CLEANING COMPLETE!")
    print("=" * 60)
    
    return cleaned_results





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


