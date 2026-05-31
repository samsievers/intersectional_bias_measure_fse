import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
import statsmodels.stats.multicomp as multi
import statsmodels.stats.anova as anova
import pingouin as pg
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor  
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split    
import os
import itertools
from matplotlib.patches import Patch
from scipy import stats as st  # Fix missing import
from scipy import stats 
from adjustText import adjust_text

def run_per_concept_analysis(df, out_fname, plot_dir,
                             vignette_cols=None,
                             response_var="response_recoded",
                             concept_col="concept",
                             continuous_vars=None,
                             min_obs_per_concept=30,
                             min_obs_per_interaction=10,
                             min_groups_per_interaction=2):
    """
    Führt separate Mixed-Effects-Analysen für jedes Concept durch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Der Input-Dataframe mit allen benötigten Variablen
    out_fname : str
        Pfad und Name der Output-Excel-Datei (z.B. "per_concept_results.xlsx")
    plot_dir : str
        Hauptverzeichnis für die Plots (Subdirectories werden für jedes Concept erstellt)
    vignette_cols : list, optional
        Liste der Vignetten-Variablen (z.B. ["race", "gender", "religion", "transness"])
        Default: ["race", "gender", "religion", "transness"]
    response_var : str, default="response_recoded"
        Name der abhängigen Variable
    concept_col : str, default="concept"
        Name der Spalte mit den Concepts
    continuous_vars : list, optional
        Liste der kontinuierlichen Variablen (z.B. ["n_dim"])
        Default: ["n_dim"]
    min_obs_per_concept : int, default=30
        Minimale Anzahl Beobachtungen pro Concept
    min_obs_per_interaction : int, default=10
        Minimale Anzahl Beobachtungen für Interaktionsanalyse
    min_groups_per_interaction : int, default=2
        Minimale Anzahl Gruppen für Interaktionsanalyse
        
    Returns:
    --------
    dict : Dictionary mit allen Ergebnissen
        - 'fixed_effects': DataFrame mit allen Fixed Effects
        - 'random_effects': DataFrame mit allen Random Effects
        - 're_summary': Summary-Statistiken der Random Effects
    """
    
    # Defaults setzen
    if vignette_cols is None:
        vignette_cols = ['race', 'gender', 'religion', 'transness']
    if continuous_vars is None:
        continuous_vars = ['n_dim']
    
    # Dataframe kopieren um Original nicht zu verändern
    df = df.copy()
    
    # ------------------------
    # Daten vorbereiten
    # ------------------------
    print(" RUNNING SEPARATE MODELS FOR EACH CONCEPT")
    print("="*60)
    
    # Recode gender value nonbinary person to nonbinary
    if 'gender' in df.columns:
        df['gender'] = df['gender'].replace({'nonbinary person': 'nonbinary'})
    
    # Rename column gender_alignment to transness if exists
    if 'gender_alignment' in df.columns and 'transness' not in df.columns:
        df = df.rename(columns={'gender_alignment': 'transness'})
    
    # "not_mentioned" als Referenzlevel setzen
    for col in vignette_cols:
        if col in df.columns:
            levels = df[col].dropna().unique().tolist()
            levels = ["not_mentioned"] + [x for x in levels if x != "not_mentioned"]
            df[col] = pd.Categorical(df[col].fillna("not_mentioned"), categories=levels, ordered=False)
    
    concepts = df[concept_col].unique()
    print(f"Total dataset size: {len(df):,} observations")
    print(f"Number of concepts: {len(concepts)}")
    print(f"Concepts: {concepts}")
    
    # Plot-Verzeichnis erstellen
    os.makedirs(plot_dir, exist_ok=True)
    
    # ------------------------
    # Interaktionen erstellen (mit EXCLUDE für not_mentioned)
    # ------------------------
    interaction_cols = []
    for a, b in itertools.combinations(vignette_cols, 2):
        if a in df.columns and b in df.columns:
            col_name = f"{a}_{b}"
            df[col_name] = np.where(
                (df[a].astype(str) != "not_mentioned") & (df[b].astype(str) != "not_mentioned"),
                df[a].astype(str) + "_" + df[b].astype(str),
                "EXCLUDE"
            )
            interaction_cols.append(col_name)
    
    print(f"Created {len(interaction_cols)} 2-way interaction columns")
    
    interaction_cols_3way = []
    for a, b, c in itertools.combinations(vignette_cols, 3):
        if a in df.columns and b in df.columns and c in df.columns:
            col_name = f"{a}_{b}_{c}"
            df[col_name] = np.where(
                (df[a].astype(str) != "not_mentioned") & 
                (df[b].astype(str) != "not_mentioned") & 
                (df[c].astype(str) != "not_mentioned"),
                df[a].astype(str) + "_" + df[b].astype(str) + "_" + df[c].astype(str),
                "EXCLUDE"
            )
            interaction_cols_3way.append(col_name)
    
    print(f"Created {len(interaction_cols_3way)} 3-way interaction columns")
    
    # ------------------------
    # ExcelWriter vorbereiten
    # ------------------------
    all_fe_results = []
    all_re_results = []
    
    with pd.ExcelWriter(out_fname, engine="xlsxwriter") as writer:
        
        # ------------------------
        # Loop über alle Concepts
        # ------------------------
        for concept in concepts:
            print(f"\n{'='*60}")
            print(f"CONCEPT: {concept}")
            print(f"{'='*60}")
            
            # Filter Daten für dieses Concept
            df_concept = df[df[concept_col] == concept].copy()
            n_obs = len(df_concept)
            
            print(f"Observations for {concept}: {n_obs}")
            
            if n_obs < min_obs_per_concept:
                print(f" Zu wenige Beobachtungen für {concept}, überspringe...")
                continue
            
            # Create concept-specific plot directory
            concept_plot_dir = os.path.join(plot_dir, str(concept).replace('/', '_'))
            os.makedirs(concept_plot_dir, exist_ok=True)
            
            # ------------------------
            # Model 1: Fixed Effects (OLS)
            # ------------------------
            print(f"\n Model 1: Fixed Effects (OLS) for {concept}")
            
            # Formula aufbauen
            fixed_parts = [f"C({col})" for col in vignette_cols if col in df_concept.columns]
            fixed_parts += [col for col in continuous_vars if col in df_concept.columns]
            
            if not fixed_parts:
                print(f" No valid variables for {concept}, skipping...")
                continue
                
            formula_fe = f"{response_var} ~ " + " + ".join(fixed_parts)
            
            try:
                md_fe = smf.ols(formula_fe, df_concept)
                mdf_fe = md_fe.fit()
                
                fe_out = pd.DataFrame({
                    "concept": concept,
                    "coef": mdf_fe.params,
                    "se": mdf_fe.bse,
                    "pvalue": mdf_fe.pvalues,
                    "ci_lower": mdf_fe.conf_int()[0],
                    "ci_upper": mdf_fe.conf_int()[1]
                })
                fe_out['variable'] = fe_out.index
                all_fe_results.append(fe_out)
                
                print(f" Fixed Effects fitted | R² = {mdf_fe.rsquared:.4f}")
                
                # Plot Fixed Effects für dieses Concept
                fe_plot = fe_out[fe_out['variable'] != 'Intercept']
                
                if len(fe_plot) > 0:
                    fig, ax = plt.subplots(figsize=(10, max(6, 0.3*len(fe_plot))))
                    y_pos = range(len(fe_plot))
                    colors_sig = ['red' if p < 0.05 else 'gray' for p in fe_plot['pvalue']]
                    
                    for idx, (_, row) in enumerate(fe_plot.iterrows()):
                        color = colors_sig[idx]
                        xerr_lower = row['coef'] - row['ci_lower']
                        xerr_upper = row['ci_upper'] - row['coef']
                        
                        ax.errorbar(row['coef'], idx,
                                   xerr=[[xerr_lower], [xerr_upper]],
                                   fmt='o', capsize=3, color=color, ecolor=color, 
                                   markersize=5, alpha=0.7)
                    
                    ax.set_yticks(y_pos)
                    ax.set_yticklabels(fe_plot['variable'], fontsize=9)
                    ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                    ax.set_xlabel('Coefficient', fontsize=11)
                    ax.set_title(f"Fixed Effects: {concept}\n(n={n_obs}, R²={mdf_fe.rsquared:.3f})", 
                                fontsize=12, fontweight='bold')
                    ax.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{concept_plot_dir}/Fixed_Effects.png", dpi=300, bbox_inches='tight')
                    plt.close()  # Changed from plt.show() to plt.close()
                
            except Exception as e:
                print(f" Fixed Effects for {concept} failed: {e}")
            
            # ------------------------
            # Model 2: Random Effects for Interactions
            # ------------------------
            print(f"\n🔀 Model 2: Random Effects for Interactions - {concept}")
            
            re_concept_combined = pd.DataFrame()
            
            # Formula für Mixed Model
            cont_formula = " + ".join([col for col in continuous_vars if col in df_concept.columns])
            formula_mixed = f"{response_var} ~ {cont_formula}" if cont_formula else f"{response_var} ~ 1"
            
            for interaction in interaction_cols + interaction_cols_3way:
                try:
                    if interaction not in df_concept.columns:
                        continue
                        
                    vars_in_inter = interaction.split('_')
                    
                    # Gültige Beobachtungen
                    valid_mask = df_concept[interaction] != "EXCLUDE"
                    valid_obs = valid_mask.sum()
                    valid_groups = df_concept.loc[valid_mask, interaction].unique()
                    n_valid_groups = len(valid_groups)
                    
                    if n_valid_groups < min_groups_per_interaction or valid_obs < min_obs_per_interaction:
                        continue
                    
                    # Fit mixed model mit Interaktion als Gruppierung
                    md_inter = smf.mixedlm(formula_mixed, df_concept, groups=df_concept[interaction])
                    mdf_inter = md_inter.fit(reml=False, method='powell', maxiter=200)
                    
                    if not mdf_inter.converged:
                        continue
                    
                    # Random Effects extrahieren
                    valid_random_effects = {
                        g: mdf_inter.random_effects[g]['Group'] 
                        for g in mdf_inter.random_effects 
                        if g != "EXCLUDE"
                    }
                    
                    for group, re_val in valid_random_effects.items():
                        re_concept_combined = pd.concat([
                            re_concept_combined,
                            pd.DataFrame({
                                'random_intercept': [re_val],
                                'interaction_type': [len(vars_in_inter)],
                                'interaction': [interaction],
                                'group': [group],
                                'concept': [concept]
                            })
                        ], axis=0)
                    
                    print(f" {interaction}: {n_valid_groups} groups, {valid_obs} obs")
                    
                except Exception as e:
                    print(f" {interaction}: {str(e)[:60]}")
            
            # Speichere Random Effects für dieses Concept
            if not re_concept_combined.empty:
                sheet_name = f"{str(concept)[:28]}_RE"
                re_concept_combined.to_excel(writer, sheet_name=sheet_name, index=False)
                all_re_results.append(re_concept_combined)
                
                # Plot Random Effects
                fig, ax = plt.subplots(figsize=(12, max(8, 0.25*len(re_concept_combined))))
                re_sorted = re_concept_combined.sort_values("random_intercept")
                colors = ['#3498db' if x == 2 else '#e74c3c' for x in re_sorted["interaction_type"]]
                
                ax.barh(range(len(re_sorted)), re_sorted["random_intercept"], 
                       color=colors, alpha=0.7)
                ax.set_yticks(range(len(re_sorted)))
                labels = [f"{row['interaction']}: {row['group']}" 
                         for _, row in re_sorted.iterrows()]
                ax.set_yticklabels(labels, fontsize=7)
                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                ax.set_xlabel('Random Intercept', fontsize=11)
                ax.set_title(f"Random Effects for Interactions: {concept}\n(n={len(re_sorted)} effects)", 
                            fontsize=12, fontweight='bold')
                
                legend_elements = [
                    Patch(facecolor='#3498db', alpha=0.7, label='2-way interactions'),
                    Patch(facecolor='#e74c3c', alpha=0.7, label='3-way interactions')
                ]
                ax.legend(handles=legend_elements, loc='best')
                ax.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{concept_plot_dir}/Interaction_Random_Effects.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()  # Changed from plt.show() to plt.close()
                
                print(f" Random Effects gespeichert: {len(re_concept_combined)} effects")
        
        # ------------------------
        # Zusammenfassung über alle Concepts
        # ------------------------
        print(f"\n{'='*60}")
        print("SUMMARY ACROSS ALL CONCEPTS")
        print(f"{'='*60}")
        
        if all_fe_results:
            df_all_fe = pd.concat(all_fe_results, ignore_index=True)
            df_all_fe.to_excel(writer, sheet_name="All_Concepts_FE", index=False)
            print(f" Combined Fixed Effects: {len(df_all_fe)} estimates")
            
            # Vergleichsplot: Koeffizienten über Concepts
            main_vars = [f'C({col})' for col in vignette_cols]
            
            for var_pattern in main_vars:
                var_data = df_all_fe[df_all_fe['variable'].str.contains(var_pattern, na=False)]
                
                if len(var_data) > 0:
                    unique_vars = var_data['variable'].unique()
                    
                    for unique_var in unique_vars[:5]:  # Max 5 pro Variable
                        var_subset = var_data[var_data['variable'] == unique_var]
                        
                        if len(var_subset) > 1:
                            fig, ax = plt.subplots(figsize=(10, max(6, 0.4*len(var_subset))))
                            
                            var_sorted = var_subset.sort_values('coef')
                            y_pos = range(len(var_sorted))
                            colors_sig = ['red' if p < 0.05 else 'gray' 
                                         for p in var_sorted['pvalue']]
                            
                            for idx, (_, row) in enumerate(var_sorted.iterrows()):
                                color = colors_sig[idx]
                                xerr_lower = row['coef'] - row['ci_lower']
                                xerr_upper = row['ci_upper'] - row['coef']
                                
                                ax.errorbar(row['coef'], idx,
                                           xerr=[[xerr_lower], [xerr_upper]],
                                           fmt='o', capsize=3, color=color, 
                                           ecolor=color, markersize=6, alpha=0.7)
                            
                            ax.set_yticks(y_pos)
                            ax.set_yticklabels(var_sorted['concept'], fontsize=9)
                            ax.axvline(0, color='k', linestyle='--', alpha=0.3)
                            ax.set_xlabel('Coefficient', fontsize=11)
                            ax.set_title(f"Comparison Across Concepts: {unique_var}", 
                                        fontsize=12, fontweight='bold')
                            ax.grid(axis='x', alpha=0.3)
                            
                            plt.tight_layout()
                            safe_name = unique_var.replace('C(', '').replace(')', '').replace('[', '').replace(']', '').replace('T.', '')
                            plt.savefig(f"{plot_dir}/Comparison_{safe_name}.png", 
                                       dpi=300, bbox_inches='tight')
                            plt.close()  # Changed from plt.show() to plt.close()
        
        if all_re_results:
            df_all_re = pd.concat(all_re_results, ignore_index=True)
            df_all_re.to_excel(writer, sheet_name="All_Concepts_RE", index=False)
            print(f" Combined Random Effects: {len(df_all_re)} estimates")
            
            # Summary Statistiken
            re_summary = df_all_re.groupby(['interaction', 'group']).agg({
                'random_intercept': ['mean', 'std', 'count']
            }).reset_index()
            re_summary.columns = ['interaction', 'group', 'mean_re', 'std_re', 'n_concepts']
            re_summary = re_summary.sort_values('mean_re', key=abs, ascending=False)
            re_summary.to_excel(writer, sheet_name="RE_Summary", index=False)
            
            print(f" Random Effects Summary created")
    
    print(f"\n Per-concept analysis completed!")
    print(f" Results saved in '{out_fname}'")
    print(f"📁 Plots saved in '{plot_dir}/' directory")
    print(f"📁 Individual concept plots in subdirectories")
    
    # Rückgabe der Ergebnisse
    results = {}
    if all_fe_results:
        results['fixed_effects'] = pd.concat(all_fe_results, ignore_index=True)
    if all_re_results:
        results['random_effects'] = pd.concat(all_re_results, ignore_index=True)
        if 'df_all_re' in locals():  # Check if df_all_re exists
            results['re_summary'] = re_summary
    
    return results


def run_mixed_effects_analysis(df, out_fname, plot_dir, 
                                interaction_cols=None, 
                                interaction_cols_3way=None,
                                response_var="response_recoded",
                                fixed_effects=["race", "gender", "religion", "transness"],
                                continuous_vars=["n_dim"],
                                concept_col="concept",
                                vignette_col="vignette_id",
                                compute_full_maihda=True,
                                compute_vignette_maihda=True,
                                min_obs_per_stratum=5,
                                min_obs_per_interaction=20,
                                min_groups_per_interaction=2,
                                save_plots=True,
                                verbose=True):
    """
    Führt eine umfassende Mixed-Effects-Analyse durch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Der Input-Dataframe mit allen benötigten Variablen
    out_fname : str
        Pfad und Name der Output-Excel-Datei (z.B. "results.xlsx")
    plot_dir : str
        Verzeichnis für die Plots (wird erstellt falls nicht vorhanden)
    interaction_cols : list, optional
        Liste der 2-way Interaktionsspalten (z.B. ["race_gender", "race_religion"])
    interaction_cols_3way : list, optional
        Liste der 3-way Interaktionsspalten
    response_var : str, default="response_recoded"
        Name der abhängigen Variable
    fixed_effects : list, default=["race", "gender", "religion", "transness"]
        Liste der kategorialen Fixed Effects
    continuous_vars : list, default=["n_dim"]
        Liste der kontinuierlichen Variablen
    concept_col : str, default="concept"
        Name der Spalte für Concept Random Effects
    vignette_col : str, default="vignette_id"
        Name der Spalte für Vignette Random Effects (corrected MAIHDA)
    compute_full_maihda : bool, default=True
        Ob das klassische MAIHDA mit intersectionalen Strata berechnet werden soll
    compute_vignette_maihda : bool, default=True
        Ob das korrigierte MAIHDA mit vignette_id als Gruppierung berechnet werden soll
    min_obs_per_stratum : int, default=5
        Minimale Anzahl Beobachtungen pro Stratum für MAIHDA
    min_obs_per_interaction : int, default=20
        Minimale Anzahl Beobachtungen für Interaktionsanalyse
    min_groups_per_interaction : int, default=2
        Minimale Anzahl Gruppen für Interaktionsanalyse
    save_plots : bool, default=True
        Ob Plots gespeichert werden sollen
    verbose : bool, default=True
        Ob detaillierte Ausgaben angezeigt werden sollen
        
    Returns:
    --------
    dict : Dictionary mit allen gefitteten Modellen und Ergebnissen
    """
    
    # Input validation
    if response_var not in df.columns:
        raise ValueError(f"Response variable '{response_var}' not found in dataframe")
    
    # Defaults für Interaktionen
    if interaction_cols is None:
        interaction_cols = []
    if interaction_cols_3way is None:
        interaction_cols_3way = []
    
    # Plot-Verzeichnis erstellen
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Ergebnis-Dictionary
    results = {}
    
    # Copy dataframe to avoid modifying original
    df_work = df.copy()
    
    if verbose:
        print("\n RUNNING COMPREHENSIVE MIXED-EFFECTS ANALYSIS")
        print("="*60)
        print(f" Dataset: {len(df_work):,} observations")
        print(f" Response variable: {response_var}")
        print(f" Available columns: {list(df_work.columns)}")
    
    # Formula aufbauen - only include variables that exist in dataframe
    fixed_formula_parts = []
    missing_vars = []
    
    for var in fixed_effects:
        if var in df_work.columns:
            fixed_formula_parts.append(f"C({var})")
        else:
            missing_vars.append(var)
    
    for var in continuous_vars:
        if var in df_work.columns:
            fixed_formula_parts.append(var)
        else:
            missing_vars.append(var)
    
    if missing_vars and verbose:
        print(f" Missing variables (will be excluded): {missing_vars}")
    
    if not fixed_formula_parts:
        print(" No valid variables found in dataframe")
        return results
    
    fixed_formula = " + ".join(fixed_formula_parts)
    if verbose:
        print(f" Fixed effects formula: {fixed_formula}")
    
    # ------------------------
    # ExcelWriter vorbereiten
    # ------------------------
    with pd.ExcelWriter(out_fname, engine="xlsxwriter") as writer:

        # ------------------------
        # Model 1: Fixed Effects Only (OLS baseline)
        # ------------------------
        if verbose:
            print("\n Model 1: Fixed Effects Only (OLS baseline)")
        
        if concept_col in df_work.columns:
            formula_fe = f"{response_var} ~ {fixed_formula} + C({concept_col})"
        else:
            formula_fe = f"{response_var} ~ {fixed_formula}"
        
        try:
            md_fe = smf.ols(formula_fe, df_work)
            mdf_fe = md_fe.fit()
            results['fixed_effects_model'] = mdf_fe
            
            fe_out = pd.DataFrame({
                "coef": mdf_fe.params,
                "se": mdf_fe.bse,
                "pvalue": mdf_fe.pvalues,
                "ci_lower": mdf_fe.conf_int()[0],
                "ci_upper": mdf_fe.conf_int()[1]
            })
            fe_out.to_excel(writer, sheet_name="Fixed_Effects_Only")
            
            if verbose:
                print(f" Fixed Effects model fitted successfully")
                print(f"   R-squared: {mdf_fe.rsquared:.4f}")
                print(f"   Adj. R-squared: {mdf_fe.rsquared_adj:.4f}")
            
            # Plot Fixed Effects
            if save_plots:
                plt.figure(figsize=(10, max(6, 0.3*len(fe_out))))
                y_pos = range(len(fe_out))
                colors = ['red' if p < 0.05 else 'gray' for p in fe_out['pvalue']]
                plt.errorbar(fe_out["coef"], y_pos,
                             xerr=[fe_out["coef"]-fe_out["ci_lower"], fe_out["ci_upper"]-fe_out["coef"]],
                             fmt='o', capsize=3, color='black', ecolor=colors)
                plt.yticks(y_pos, fe_out.index)
                plt.axvline(0, color='k', linestyle='--')
                plt.xlabel('Coefficient')
                plt.title("Fixed Effects Model - Whole Dataset")
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/Fixed_Effects_Model.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            if verbose:
                print(f" Fixed Effects model failed: {e}")

        # ------------------------
        # Model 2: Mixed Model with Concepts as Random Effects
        # ------------------------
        if concept_col in df_work.columns:
            if verbose:
                print("\n🔀 Model 2: Mixed Model with Concepts as Random Effects")
            formula_mixed = f"{response_var} ~ {fixed_formula}"
            
            try:
                md_mixed = smf.mixedlm(formula_mixed, df_work, groups=df_work[concept_col])
                mdf_mixed = md_mixed.fit(reml=False, method='powell', maxiter=3000, disp=False)
                results['mixed_model_concepts'] = mdf_mixed
                
                # Calculate ICC for concepts
                tau_sq_concepts = mdf_mixed.cov_re.iloc[0, 0]
                sigma_sq_concepts = mdf_mixed.scale
                icc_concepts = tau_sq_concepts / (tau_sq_concepts + sigma_sq_concepts)
                results['concept_icc'] = icc_concepts
                
                # Fixed effects
                fe_mixed = pd.DataFrame({
                    "coef": mdf_mixed.fe_params,
                    "se": mdf_mixed.bse,
                    "pvalue": mdf_mixed.pvalues,
                    "ci_lower": mdf_mixed.conf_int()[0],
                    "ci_upper": mdf_mixed.conf_int()[1]
                })
                fe_mixed.to_excel(writer, sheet_name="Mixed_Fixed_Effects")
                
                # Random effects
                re_mixed = pd.DataFrame({
                    "random_intercept": [mdf_mixed.random_effects[c]['Group'] for c in mdf_mixed.random_effects.keys()]
                }, index=list(mdf_mixed.random_effects.keys()))
                re_mixed.index.name = concept_col
                re_mixed.to_excel(writer, sheet_name="Mixed_Random_Effects")
                
                if verbose:
                    print(f" Mixed model with concept random effects fitted successfully")
                    print(f"   Log-likelihood: {mdf_mixed.llf:.2f}")
                    print(f"   AIC: {mdf_mixed.aic:.2f}")
                    print(f"   BIC: {mdf_mixed.bic:.2f}")
                    print(f"   ICC (concepts): {icc_concepts:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f" Mixed model with concepts failed: {e}")

        # ------------------------
        # Model 3: CORRECTED MAIHDA - Vignette-Level Model
        # ------------------------
        if compute_vignette_maihda and vignette_col in df_work.columns:
            if verbose:
                print("\n🌟 Model 3: CORRECTED MAIHDA - Vignette-Level Model")
                print("="*60)
            
            # Check vignette statistics
            vignette_stats = df_work.groupby(vignette_col).agg({
                response_var: ['count', 'mean', 'std']
            }).round(4)
            
            vignette_stats.columns = ['n_obs', 'mean_response', 'std_response']
            vignette_stats = vignette_stats.reset_index()
            
            if verbose:
                print(f" Vignette Statistics:")
                print(f"   Total vignettes: {len(vignette_stats)}")
                print(f"   Obs per vignette: min={vignette_stats['n_obs'].min()}, "
                      f"mean={vignette_stats['n_obs'].mean():.1f}, max={vignette_stats['n_obs'].max()}")
                print(f"   Mean response range: {vignette_stats['mean_response'].min():.3f} to "
                      f"{vignette_stats['mean_response'].max():.3f}")
            
            try:
                # Fit corrected MAIHDA model
                formula_vignette_maihda = f"{response_var} ~ {fixed_formula}"
                md_vignette_maihda = smf.mixedlm(formula_vignette_maihda, df_work, groups=df_work[vignette_col])
                mdf_vignette_maihda = md_vignette_maihda.fit(reml=False, method='powell', maxiter=3000, disp=False)
                results['vignette_maihda_model'] = mdf_vignette_maihda
                
                # Calculate variance components
                tau_sq_vignette = mdf_vignette_maihda.cov_re.iloc[0, 0]
                sigma_sq_vignette = mdf_vignette_maihda.scale
                icc_vignette = tau_sq_vignette / (tau_sq_vignette + sigma_sq_vignette)
                
                results['vignette_maihda_icc'] = icc_vignette
                results['vignette_maihda_tau_sq'] = tau_sq_vignette
                results['vignette_maihda_sigma_sq'] = sigma_sq_vignette
                
                if verbose:
                    print(f"\n CORRECTED MAIHDA RESULTS:")
                    print(f"   ═══════════════════════════════════════")
                    print(f"   τ² (Between-vignette variance): {tau_sq_vignette:.4f}")
                    print(f"   σ² (Within-vignette variance):  {sigma_sq_vignette:.4f}")
                    print(f"   ICC (Intraclass Correlation):  {icc_vignette:.4f}")
                    print(f"   ═══════════════════════════════════════")
                    print(f"   Interpretation: {icc_vignette*100:.2f}% of variance is")
                    print(f"                   at the vignette level")
                
                # Extract random effects (BLUPs for each vignette)
                re_vignettes = pd.DataFrame({
                    vignette_col: list(mdf_vignette_maihda.random_effects.keys()),
                    "random_intercept": [mdf_vignette_maihda.random_effects[v]['Group'] 
                                        for v in mdf_vignette_maihda.random_effects.keys()]
                })
                
                # Add observed means for comparison
                re_vignettes = re_vignettes.merge(
                    df_work.groupby(vignette_col)[response_var].mean().reset_index(),
                    on=vignette_col
                )
                re_vignettes.columns = [vignette_col, 'random_intercept', 'observed_mean']
                re_vignettes = re_vignettes.sort_values('random_intercept', ascending=False)
                
                # Save to Excel
                vignette_maihda_fe = pd.DataFrame({
                    "coef": mdf_vignette_maihda.fe_params,
                    "se": mdf_vignette_maihda.bse,
                    "pvalue": mdf_vignette_maihda.pvalues,
                    "ci_lower": mdf_vignette_maihda.conf_int()[0],
                    "ci_upper": mdf_vignette_maihda.conf_int()[1]
                })
                vignette_maihda_fe.to_excel(writer, sheet_name="Vignette_MAIHDA_FE")
                
                re_vignettes.to_excel(writer, sheet_name="Vignette_MAIHDA_RE", index=False)
                results['vignette_maihda_random_effects'] = re_vignettes
                
                # MAIHDA Metrics Summary
                vignette_maihda_metrics = pd.DataFrame({
                    'Metric': ['Number of Vignettes', 'Number of Observations',
                              'Between-Vignette Variance (τ²)', 
                              'Within-Vignette Variance (σ²)',
                              'ICC (Intraclass Correlation)',
                              '% Variance at Vignette Level'],
                    'Value': [len(vignette_stats), len(df_work), 
                             tau_sq_vignette, sigma_sq_vignette,
                             icc_vignette, icc_vignette*100]
                })
                vignette_maihda_metrics.to_excel(writer, sheet_name="Vignette_MAIHDA_Metrics", index=False)
                
                if verbose:
                    print(f"\n Top 10 vignettes by random effect:")
                    print(re_vignettes.head(10)[[vignette_col, 'random_intercept', 'observed_mean']].to_string(index=False))
                
                # Correlation between random effects and observed means
                correlation = re_vignettes['random_intercept'].corr(re_vignettes['observed_mean'])
                if verbose:
                    print(f"\n📈 Correlation between random effects and observed means: {correlation:.4f}")
                
                # Create comparison plot
                if save_plots:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot 1: Random effects vs observed means
                    ax1.scatter(re_vignettes['observed_mean'], re_vignettes['random_intercept'], alpha=0.6)
                    ax1.set_xlabel('Observed Vignette Mean')
                    ax1.set_ylabel('Random Effect (BLUP)')
                    ax1.set_title(f'Random Effects vs Observed Means\n(r = {correlation:.3f})')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add diagonal reference line
                    min_val = min(re_vignettes[['observed_mean', 'random_intercept']].min())
                    max_val = max(re_vignettes[['observed_mean', 'random_intercept']].max())
                    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                    
                    # Plot 2: Distribution of random effects
                    ax2.hist(re_vignettes['random_intercept'], bins=20, alpha=0.7, edgecolor='black')
                    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Random Intercept')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title(f'Distribution of Vignette Random Effects\n(ICC = {icc_vignette:.3f})')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/Corrected_MAIHDA_Vignette_Analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
            except Exception as e:
                if verbose:
                    print(f" Corrected MAIHDA model failed: {e}")
                    import traceback
                    traceback.print_exc()

        # ------------------------
        # Model 4: KLASSISCHES MAIHDA - Full Intersectional Strata
        # ------------------------
        if compute_full_maihda:
            if verbose:
                print("\n🌟 Model 4: CLASSIC MAIHDA - Full Intersectional Strata Model")
                print("="*60)
            
            available_fe = [fe for fe in fixed_effects if fe in df_work.columns]
            
            if len(available_fe) >= 2:
                # Create proper intersectional strata using the existing function
                df_copy = df_work.copy()
                stratum_ids = create_proper_stratum_id(df_copy, available_fe, exclude_val="not_mentioned")
                df_copy['stratum_all'] = pd.Categorical(stratum_ids)
                
                # Count dimensions properly
                def count_dimensions(stratum_name):
                    if pd.isna(stratum_name) or stratum_name == "baseline":
                        return 0
                    if isinstance(stratum_name, str):
                        return len(stratum_name.split("|"))
                    return 0
                
                df_copy['n_dimensions'] = df_copy['stratum_all'].apply(count_dimensions)
                
                # Filter to only TRUE INTERSECTIONS (2+ dimensions)
                df_maihda = df_copy[df_copy['n_dimensions'] >= 2].copy()
                
                if len(df_maihda) == 0:
                    if verbose:
                        print(" No true intersectional observations found")
                else:
                    # Get stratum counts
                    stratum_counts = df_maihda['stratum_all'].value_counts()
                    n_strata = len(stratum_counts)
                    
                    # Check variance in means BEFORE filtering
                    stratum_means = df_maihda.groupby('stratum_all')[response_var].mean()
                    mean_variance = stratum_means.var()
                    
                    if verbose:
                        print(f" MAIHDA Statistics BEFORE filtering:")
                        print(f"   Total intersectional strata: {n_strata}")
                        print(f"   Intersectional observations: {len(df_maihda)}")
                        print(f"   Variance in stratum means: {mean_variance:.6f}")
                        print(f"   Range: {stratum_means.min():.3f} to {stratum_means.max():.3f}")
                        
                        if mean_variance < 0.001:
                            print("    WARNING: Very low variance in stratum means!")
                            print("   This suggests the response variable has little variation")
                            print("   across intersectional groups.")
                    
                    # Filter by minimum observations per stratum
                    valid_strata = stratum_counts[stratum_counts >= min_obs_per_stratum].index
                    df_maihda_final = df_maihda[df_maihda['stratum_all'].isin(valid_strata)].copy()
                    
                    n_strata_used = len(valid_strata)
                    n_obs_used = len(df_maihda_final)
                    
                    if verbose:
                        print(f"\n   After min {min_obs_per_stratum} obs filter:")
                        print(f"   Strata used: {n_strata_used}")
                        print(f"   Observations used: {n_obs_used}")
                    
                    if n_strata_used < 2:
                        if verbose:
                            print(" Not enough strata for MAIHDA model after filtering")
                    else:
                        try:
                            # 🔥 KEY FIX: Use proper formula without fixed effects
                            # The stratum random effects REPLACE the fixed effects in classic MAIHDA
                            formula_maihda = f"{response_var} ~ 1"  # Only intercept!
                            
                            # Add concept as fixed effect if available and requested
                            if concept_col in df_maihda_final.columns:
                                formula_maihda = f"{response_var} ~ C({concept_col})"
                            

                            # Fit MAIHDA model
                            md_maihda = smf.mixedlm(formula_maihda, df_maihda_final, 
                                                   groups=df_maihda_final['stratum_all'])
                            mdf_maihda = md_maihda.fit(reml=False, method='powell', 
                                                      maxiter=3000, disp=False)
                            
                            # Calculate variance components
                            tau_sq_maihda = mdf_maihda.cov_re.iloc[0, 0]
                            sigma_sq_maihda = mdf_maihda.scale
                            icc_maihda = tau_sq_maihda / (tau_sq_maihda + sigma_sq_maihda)
                            

                            results['maihda_full_model'] = mdf_maihda
                            results['maihda_icc'] = icc_maihda
                            results['maihda_tau_sq'] = tau_sq_maihda
                            results['maihda_sigma_sq'] = sigma_sq_maihda
                            
                            if verbose:
                                print(f"\n MAIHDA RESULTS:")
                                print(f"   ═══════════════════════════════════════")
                                print(f"   τ² (Between-stratum variance): {tau_sq_maihda:.6f}")
                                print(f"   σ² (Within-stratum variance):  {sigma_sq_maihda:.6f}")
                                print(f"   ICC (Intraclass Correlation):  {icc_maihda:.6f}")
                                print(f"   ═══════════════════════════════════════")
                                
                                # Compare with observed variance
                                expected_between_var = mean_variance * len(df_maihda_final) / n_strata_used
                                print(f"   Expected between-group variance: {expected_between_var:.6f}")
                                print(f"   Ratio (observed/expected): {tau_sq_maihda/expected_between_var:.3f}")
                            

                            # Save results...
                            fe_maihda = pd.DataFrame({
                                "coef": mdf_maihda.fe_params,
                                "se": mdf_maihda.bse,
                                "pvalue": mdf_maihda.pvalues,
                                "ci_lower": mdf_maihda.conf_int()[0],
                                "ci_upper": mdf_maihda.conf_int()[1]
                            })
                            fe_maihda.to_excel(writer, sheet_name="MAIHDA_Fixed_Effects")
                            
                            # Random effects
                            re_maihda = pd.DataFrame({
                                "random_intercept": [mdf_maihda.random_effects[s]['Group'] 
                                                    for s in mdf_maihda.random_effects.keys()],
                                "stratum": list(mdf_maihda.random_effects.keys())
                            })
                            re_maihda = re_maihda.sort_values('random_intercept')
                            re_maihda.to_excel(writer, sheet_name="MAIHDA_Random_Effects", index=False)
                            results['maihda_random_effects'] = re_maihda
                            

                            # MAIHDA metrics
                            maihda_metrics = pd.DataFrame({
                                'Metric': ['Number of Strata', 'Number of Observations',
                                          'Between-Stratum Variance (τ²)', 
                                          'Within-Stratum Variance (σ²)',
                                          'ICC (Intraclass Correlation)',
                                          'Observed Stratum Mean Variance',
                                          '% Variance at Stratum Level'],
                                'Value': [n_strata_used, n_obs_used, 
                                         tau_sq_maihda, sigma_sq_maihda,
                                         icc_maihda, mean_variance, icc_maihda*100]
                            })
                            maihda_metrics.to_excel(writer, sheet_name="MAIHDA_Metrics", index=False)
                            
                            # Plots for MAIHDA
                            if save_plots:
                                # Plot 1: Top and Bottom Random Effects
                                n_top = min(20, len(re_maihda) // 2)
                                top_high = re_maihda.nlargest(n_top, 'random_intercept')
                                top_low = re_maihda.nsmallest(n_top, 'random_intercept')
                                top_combined = pd.concat([top_high, top_low]).sort_values('random_intercept')
                                
                                fig, ax = plt.subplots(figsize=(12, max(8, 0.35*len(top_combined))))
                                y_pos = range(len(top_combined))
                                colors = ['darkgreen' if x > 0 else 'darkred' 
                                         for x in top_combined['random_intercept']]
                                
                                ax.barh(y_pos, top_combined['random_intercept'], 
                                       color=colors, alpha=0.7)
                                ax.set_yticks(y_pos)
                                ax.set_yticklabels(top_combined['stratum'], fontsize=8)
                                ax.axvline(0, color='k', linestyle='--', linewidth=1)
                                ax.set_xlabel('Random Intercept (BLUP)', fontsize=11)
                                ax.set_title(f'MAIHDA: Top {n_top} Highest/Lowest Intersectional Effects\n' +
                                           f'(ICC = {icc_maihda:.3f}, {n_strata_used} strata)', 
                                           fontsize=13, fontweight='bold')
                                ax.grid(axis='x', alpha=0.3)
                                
                                plt.tight_layout()
                                plt.savefig(f"{plot_dir}/MAIHDA_Top_Strata.png", 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                                
                                # Plot 2: Distribution of Random Effects
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # Histogram
                                ax1.hist(re_maihda['random_intercept'], bins=30, 
                                        color='steelblue', alpha=0.7, edgecolor='black')
                                ax1.axvline(0, color='red', linestyle='--', linewidth=2, 
                                           label='Grand Mean')
                                ax1.set_xlabel('Random Intercept', fontsize=11)
                                ax1.set_ylabel('Frequency', fontsize=11)
                                ax1.set_title('Distribution of Stratum Random Effects', 
                                             fontsize=12, fontweight='bold')
                                ax1.legend()
                                ax1.grid(alpha=0.3)
                                
                                # QQ-plot (check normality)
                                from scipy import stats
                                stats.probplot(re_maihda['random_intercept'], dist="norm", plot=ax2)
                                ax2.set_title('Q-Q Plot (Normality Check)', 
                                             fontsize=12, fontweight='bold')
                                ax2.grid(alpha=0.3)
                                
                                plt.tight_layout()
                                plt.savefig(f"{plot_dir}/MAIHDA_RE_Distribution.png", 
                                           dpi=300, bbox_inches='tight')
                                plt.close()
                            

                        except Exception as e:
                            if verbose:
                                print(f" MAIHDA model failed: {e}")
                                import traceback
                                traceback.print_exc()
            else:
                if verbose:
                    print(" Need at least 2 fixed effects variables for MAIHDA")

        # ------------------------
        # Model 5: Mixed Model with Interaction Random Effects
        # ------------------------
        if verbose:
            print("\n Model 5: Mixed Model with Specific Interaction Random Effects")
        
        re_interactions_combined = pd.DataFrame()
        formula_mixed = f"{response_var} ~ {fixed_formula}"
        
        for interaction in interaction_cols + interaction_cols_3way:
            try:
                if interaction not in df_work.columns:
                    continue
                    
                vars_in_inter = interaction.split('_')
                
                # Step 1: Anzahl gültiger Beobachtungen (alles außer EXCLUDE)
                valid_mask = df_work[interaction] != "EXCLUDE"
                valid_obs = valid_mask.sum()
                valid_groups = df_work.loc[valid_mask, interaction].unique()
                n_valid_groups = len(valid_groups)
                
                if n_valid_groups < min_groups_per_interaction or valid_obs < min_obs_per_interaction:
                    if verbose:
                        print(f" Zu wenige gültige Gruppen/Beobachtungen für {interaction}: {n_valid_groups} groups, {valid_obs} obs")
                    continue
                
                # Step 2: Fit mixed model
                md_inter = smf.mixedlm(formula_mixed, df_work, groups=df_work[interaction])
                mdf_inter = md_inter.fit(reml=False, method="powell", maxiter=2000, disp=False)

                # Step 3: Random Effects extrahieren (nur gültige Gruppen)
                valid_random_effects = {
                    g: mdf_inter.random_effects[g]['Group'] 
                    for g in mdf_inter.random_effects 
                    if g != "EXCLUDE"
                }
                
                if len(valid_random_effects) == 0:
                    if verbose:
                        print(f" Keine gültigen Random Effects für {interaction}")
                    continue

                # Step 4: Ergebnisse speichern
                re_df = pd.DataFrame({
                    "random_intercept": list(valid_random_effects.values()),
                    "interaction_type": [len(vars_in_inter)] * len(valid_random_effects)
                }, index=list(valid_random_effects.keys()))
                re_df.index.name = interaction
                re_interactions_combined = pd.concat([re_interactions_combined, re_df], axis=0)

                if verbose:
                    print(f" Random Intercepts für {interaction} erfolgreich ({valid_obs} valid obs, {n_valid_groups} valid groups)")

            except Exception as e:
                if verbose:
                    print(f" Random Intercepts für {interaction} failed: {e}")

        # ------------------------
        # Save and Plot Interaction Random Effects
        # ------------------------
        if not re_interactions_combined.empty:
            re_interactions_combined.to_excel(writer, sheet_name="Interaction_Random_Effects")
            results['interaction_random_effects'] = re_interactions_combined
            
            # Plot
            if save_plots:
                plt.figure(figsize=(12, max(8, 0.2*len(re_interactions_combined))))
                re_sorted = re_interactions_combined.sort_values("random_intercept")
                colors = ['#3498db' if x == 2 else '#e74c3c' for x in re_sorted["interaction_type"]]
                
                plt.barh(range(len(re_sorted)), re_sorted["random_intercept"], color=colors, alpha=0.7)
                plt.yticks(range(len(re_sorted)), re_sorted.index, fontsize=8)
                plt.axvline(0, color='k', linestyle='--')
                plt.xlabel('Random Intercept')
                plt.title("Specific Interaction Random Effects (Blue=2-way, Red=3-way)")
                legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='2-way interactions'),
                                  Patch(facecolor='#e74c3c', alpha=0.7, label='3-way interactions')]
                plt.legend(handles=legend_elements)
                plt.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/Interaction_Random_Effects.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            if verbose:
                print(f" Interaction Random Effects gespeichert: {len(re_interactions_combined)} effects")
        
        else:
            if verbose:
                print(f" Keine Interaction Random Effects zu speichern")

    if verbose:
        print(f"\n Whole dataset analysis completed! Results saved in '{out_fname}'")
        if save_plots:
            print(f"📁 Plots saved in '{plot_dir}/' directory")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF MODELS:")
        print("="*60)
        if 'fixed_effects_model' in results:
            print(f" Model 1: Fixed Effects Only (OLS)")
        if 'mixed_model_concepts' in results:
            print(f" Model 2: Concepts as Random Effects (ICC: {results.get('concept_icc', 0):.3f})")
        if 'vignette_maihda_model' in results:
            print(f" Model 3: CORRECTED MAIHDA - Vignette-Level (ICC: {results.get('vignette_maihda_icc', 0):.3f})")
            print(f"   → {results.get('vignette_maihda_icc', 0)*100:.1f}% of variance at vignette level")
        if 'maihda_full_model' in results:
            print(f" Model 4: CLASSIC MAIHDA - Full Strata (ICC: {results.get('maihda_icc', 0):.3f})")
            print(f"   → {results.get('maihda_icc', 0)*100:.1f}% of variance at intersectional stratum level")
        if 'interaction_random_effects' in results:
            print(f" Model 5: Mixed Model with Specific Interaction Random Effects ({len(results['interaction_random_effects'])} effects)")
        print("="*60)
    
    return results

def run_mixed_effects_analysis(df, out_fname, plot_dir, 
                                interaction_cols=None, 
                                interaction_cols_3way=None,
                                response_var="response_recoded",
                                fixed_effects=["race", "gender", "religion", "transness"],
                                continuous_vars=["n_dim"],
                                concept_col="concept",
                                vignette_col="vignette_id",
                                compute_full_maihda=True,
                                compute_vignette_maihda=True,
                                min_obs_per_stratum=5,
                                min_obs_per_interaction=20,
                                min_groups_per_interaction=2,
                                save_plots=True,
                                verbose=True):
    """
    Führt eine umfassende Mixed-Effects-Analyse durch.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Der Input-Dataframe mit allen benötigten Variablen
    out_fname : str
        Pfad und Name der Output-Excel-Datei (z.B. "results.xlsx")
    plot_dir : str
        Verzeichnis für die Plots (wird erstellt falls nicht vorhanden)
    interaction_cols : list, optional
        Liste der 2-way Interaktionsspalten (z.B. ["race_gender", "race_religion"])
    interaction_cols_3way : list, optional
        Liste der 3-way Interaktionsspalten
    response_var : str, default="response_recoded"
        Name der abhängigen Variable
    fixed_effects : list, default=["race", "gender", "religion", "transness"]
        Liste der kategorialen Fixed Effects
    continuous_vars : list, default=["n_dim"]
        Liste der kontinuierlichen Variablen
    concept_col : str, default="concept"
        Name der Spalte für Concept Random Effects
    vignette_col : str, default="vignette_id"
        Name der Spalte für Vignette Random Effects (corrected MAIHDA)
    compute_full_maihda : bool, default=True
        Ob das klassische MAIHDA mit intersectionalen Strata berechnet werden soll
    compute_vignette_maihda : bool, default=True
        Ob das korrigierte MAIHDA mit vignette_id als Gruppierung berechnet werden soll
    min_obs_per_stratum : int, default=5
        Minimale Anzahl Beobachtungen pro Stratum für MAIHDA
    min_obs_per_interaction : int, default=20
        Minimale Anzahl Beobachtungen für Interaktionsanalyse
    min_groups_per_interaction : int, default=2
        Minimale Anzahl Gruppen für Interaktionsanalyse
    save_plots : bool, default=True
        Ob Plots gespeichert werden sollen
    verbose : bool, default=True
        Ob detaillierte Ausgaben angezeigt werden sollen
        
    Returns:
    --------
    dict : Dictionary mit allen gefitteten Modellen und Ergebnissen
    """
    
    # Input validation
    if response_var not in df.columns:
        raise ValueError(f"Response variable '{response_var}' not found in dataframe")
    
    # Defaults für Interaktionen
    if interaction_cols is None:
        interaction_cols = []
    if interaction_cols_3way is None:
        interaction_cols_3way = []
    
    # Plot-Verzeichnis erstellen
    if save_plots:
        os.makedirs(plot_dir, exist_ok=True)
    
    # Ergebnis-Dictionary
    results = {}
    
    # Copy dataframe to avoid modifying original
    df_work = df.copy()
    
    if verbose:
        print("\n RUNNING COMPREHENSIVE MIXED-EFFECTS ANALYSIS")
        print("="*60)
        print(f" Dataset: {len(df_work):,} observations")
        print(f" Response variable: {response_var}")
        print(f" Available columns: {list(df_work.columns)}")
    
    # Formula aufbauen - only include variables that exist in dataframe
    fixed_formula_parts = []
    missing_vars = []
    
    for var in fixed_effects:
        if var in df_work.columns:
            fixed_formula_parts.append(f"C({var})")
        else:
            missing_vars.append(var)
    
    for var in continuous_vars:
        if var in df_work.columns:
            fixed_formula_parts.append(var)
        else:
            missing_vars.append(var)
    
    if missing_vars and verbose:
        print(f" Missing variables (will be excluded): {missing_vars}")
    
    if not fixed_formula_parts:
        print(" No valid variables found in dataframe")
        return results
    
    fixed_formula = " + ".join(fixed_formula_parts)
    if verbose:
        print(f" Fixed effects formula: {fixed_formula}")
    
    # ------------------------
    # ExcelWriter vorbereiten
    # ------------------------
    with pd.ExcelWriter(out_fname, engine="xlsxwriter") as writer:

        # ------------------------
        # Model 1: Fixed Effects Only (OLS baseline)
        # ------------------------
        if verbose:
            print("\n Model 1: Fixed Effects Only (OLS baseline)")
        
        if concept_col in df_work.columns:
            formula_fe = f"{response_var} ~ {fixed_formula} + C({concept_col})"
        else:
            formula_fe = f"{response_var} ~ {fixed_formula}"
        
        try:
            md_fe = smf.ols(formula_fe, df_work)
            mdf_fe = md_fe.fit()
            results['fixed_effects_model'] = mdf_fe
            
            fe_out = pd.DataFrame({
                "coef": mdf_fe.params,
                "se": mdf_fe.bse,
                "pvalue": mdf_fe.pvalues,
                "ci_lower": mdf_fe.conf_int()[0],
                "ci_upper": mdf_fe.conf_int()[1]
            })
            fe_out.to_excel(writer, sheet_name="Fixed_Effects_Only")
            
            if verbose:
                print(f" Fixed Effects model fitted successfully")
                print(f"   R-squared: {mdf_fe.rsquared:.4f}")
                print(f"   Adj. R-squared: {mdf_fe.rsquared_adj:.4f}")
            
            # Plot Fixed Effects
            if save_plots:
                plt.figure(figsize=(10, max(6, 0.3*len(fe_out))))
                y_pos = range(len(fe_out))
                colors = ['red' if p < 0.05 else 'gray' for p in fe_out['pvalue']]
                plt.errorbar(fe_out["coef"], y_pos,
                             xerr=[fe_out["coef"]-fe_out["ci_lower"], fe_out["ci_upper"]-fe_out["coef"]],
                             fmt='o', capsize=3, color='black', ecolor=colors)
                plt.yticks(y_pos, fe_out.index)
                plt.axvline(0, color='k', linestyle='--')
                plt.xlabel('Coefficient')
                plt.title("Fixed Effects Model - Whole Dataset")
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/Fixed_Effects_Model.png", dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            if verbose:
                print(f" Fixed Effects model failed: {e}")

        # ------------------------
        # Model 2: Mixed Model with Concepts as Random Effects
        # ------------------------
        if concept_col in df_work.columns:
            if verbose:
                print("\n🔀 Model 2: Mixed Model with Concepts as Random Effects")
            formula_mixed = f"{response_var} ~ {fixed_formula}"
            
            try:
                md_mixed = smf.mixedlm(formula_mixed, df_work, groups=df_work[concept_col])
                mdf_mixed = md_mixed.fit(reml=False, method='powell', maxiter=3000, disp=False)
                results['mixed_model_concepts'] = mdf_mixed
                
                # Calculate ICC for concepts
                tau_sq_concepts = mdf_mixed.cov_re.iloc[0, 0]
                sigma_sq_concepts = mdf_mixed.scale
                icc_concepts = tau_sq_concepts / (tau_sq_concepts + sigma_sq_concepts)
                results['concept_icc'] = icc_concepts
                
                # Fixed effects
                fe_mixed = pd.DataFrame({
                    "coef": mdf_mixed.fe_params,
                    "se": mdf_mixed.bse,
                    "pvalue": mdf_mixed.pvalues,
                    "ci_lower": mdf_mixed.conf_int()[0],
                    "ci_upper": mdf_mixed.conf_int()[1]
                })
                fe_mixed.to_excel(writer, sheet_name="Mixed_Fixed_Effects")
                
                # Random effects
                re_mixed = pd.DataFrame({
                    "random_intercept": [mdf_mixed.random_effects[c]['Group'] for c in mdf_mixed.random_effects.keys()]
                }, index=list(mdf_mixed.random_effects.keys()))
                re_mixed.index.name = concept_col
                re_mixed.to_excel(writer, sheet_name="Mixed_Random_Effects")
                
                if verbose:
                    print(f" Mixed model with concept random effects fitted successfully")
                    print(f"   Log-likelihood: {mdf_mixed.llf:.2f}")
                    print(f"   AIC: {mdf_mixed.aic:.2f}")
                    print(f"   BIC: {mdf_mixed.bic:.2f}")
                    print(f"   ICC (concepts): {icc_concepts:.4f}")
                
            except Exception as e:
                if verbose:
                    print(f" Mixed model with concepts failed: {e}")

        # ------------------------
        # Model 3A: MAIHDA - Vignette-Level Model - The simple intersectional model 
        # ------------------------
        if compute_vignette_maihda and vignette_col in df_work.columns:
            if verbose:
                print("\n🌟 Model 3: MAIHDA - The simple intersectional model")
                print("="*60)
            
            # Check vignette statistics
            vignette_stats = df_work.groupby(vignette_col).agg({
                response_var: ['count', 'mean', 'std']
            }).round(4)
            
            vignette_stats.columns = ['n_obs', 'mean_response', 'std_response']
            vignette_stats = vignette_stats.reset_index()
            
            if verbose:
                print(f" Vignette Statistics:")
                print(f"   Total vignettes: {len(vignette_stats)}")
                print(f"   Obs per vignette: min={vignette_stats['n_obs'].min()}, "
                      f"mean={vignette_stats['n_obs'].mean():.1f}, max={vignette_stats['n_obs'].max()}")
                print(f"   Mean response range: {vignette_stats['mean_response'].min():.3f} to "
                      f"{vignette_stats['mean_response'].max():.3f}")
            
            try:
                # Fit simple MAIHDA model without fixed effects

                md_simple_maihda = smf.mixedlm("response_recoded ~ 1", df_work, groups=df_work[vignette_col])
                mdf_simple_maihda = md_simple_maihda.fit(reml=False, method='powell', maxiter=3000, disp=False)
                results['vignette_maihda_model'] = mdf_simple_maihda
                results['predictions'] = mdf_simple_maihda.predict()
                # Calculate variance components
                tau_sq_vignette = mdf_simple_maihda.cov_re.iloc[0, 0]
                sigma_sq_vignette = mdf_simple_maihda.scale
                icc_vignette = tau_sq_vignette / (tau_sq_vignette + sigma_sq_vignette)
                
                results['vignette_maihda_icc'] = icc_vignette
                results['vignette_maihda_tau_sq'] = tau_sq_vignette
                results['vignette_maihda_sigma_sq'] = sigma_sq_vignette
                
                if verbose:
                    print(f"\n SIMPLE MAIHDA RESULTS:")
                    print(f"   ═══════════════════════════════════════")
                    print(f"   τ² (Between-vignette variance): {tau_sq_vignette:.4f}")
                    print(f"   σ² (Within-vignette variance):  {sigma_sq_vignette:.4f}")
                    print(f"   ICC (Intraclass Correlation):  {icc_vignette:.4f}")
                    print(f"   ═══════════════════════════════════════")
                    print(f"   Interpretation: {icc_vignette*100:.2f}% of variance is")
                    print(f"                   at the vignette level")
                
                # Extract random effects (BLUPs for each vignette)
                re_vignettes = pd.DataFrame({
                    vignette_col: list(mdf_simple_maihda.random_effects.keys()),
                    "random_intercept": [mdf_simple_maihda.random_effects[v]['Group'] 
                                        for v in mdf_simple_maihda.random_effects.keys()]
                })
                
                # Add observed means for comparison
                re_vignettes = re_vignettes.merge(
                    df_work.groupby(vignette_col)[response_var].mean().reset_index(),
                    on=vignette_col
                )
                re_vignettes.columns = [vignette_col, 'random_intercept', 'observed_mean']
                re_vignettes = re_vignettes.sort_values('random_intercept', ascending=False)
                
                # Save to Excel
                vignette_maihda_fe = pd.DataFrame({
                    "coef": mdf_simple_maihda.fe_params,
                    "se": mdf_simple_maihda.bse,
                    "pvalue": mdf_simple_maihda.pvalues,
                    "ci_lower": mdf_simple_maihda.conf_int()[0],
                    "ci_upper": mdf_simple_maihda.conf_int()[1]
                })
                vignette_maihda_fe.to_excel(writer, sheet_name="Vignette_MAIHDA_FE")
                
                re_vignettes.to_excel(writer, sheet_name="Vignette_MAIHDA_RE", index=False)
                results['vignette_maihda_random_effects'] = re_vignettes
                
                # MAIHDA Metrics Summary
                vignette_maihda_metrics = pd.DataFrame({
                    'Metric': ['Number of Vignettes', 'Number of Observations',
                              'Between-Vignette Variance (τ²)', 
                              'Within-Vignette Variance (σ²)',
                              'ICC (Intraclass Correlation)',
                              '% Variance at Vignette Level'],
                    'Value': [len(vignette_stats), len(df_work), 
                             tau_sq_vignette, sigma_sq_vignette,
                             icc_vignette, icc_vignette*100]
                })
                vignette_maihda_metrics.to_excel(writer, sheet_name="Vignette_MAIHDA_Metrics", index=False)
                
                if verbose:
                    print(f"\n Top 10 vignettes by random effect:")
                    print(re_vignettes.head(10)[[vignette_col, 'random_intercept', 'observed_mean']].to_string(index=False))
                
                # Correlation between random effects and observed means
                correlation = re_vignettes['random_intercept'].corr(re_vignettes['observed_mean'])
                if verbose:
                    print(f"\n📈 Correlation between random effects and observed means: {correlation:.4f}")
                
                # Create comparison plot
                if save_plots:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                    
                    # Plot 1: Random effects vs observed means
                    ax1.scatter(re_vignettes['observed_mean'], re_vignettes['random_intercept'], alpha=0.6)
                    ax1.set_xlabel('Observed Vignette Mean')
                    ax1.set_ylabel('Random Effect (BLUP)')
                    ax1.set_title(f'Random Effects vs Observed Means\n(r = {correlation:.3f})')
                    ax1.grid(True, alpha=0.3)
                    
                    # Add diagonal reference line
                    min_val = min(re_vignettes[['observed_mean', 'random_intercept']].min())
                    max_val = max(re_vignettes[['observed_mean', 'random_intercept']].max())
                    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
                    
                    # Plot 2: Distribution of random effects
                    ax2.hist(re_vignettes['random_intercept'], bins=20, alpha=0.7, edgecolor='black')
                    ax2.axvline(0, color='red', linestyle='--', linewidth=2)
                    ax2.set_xlabel('Random Intercept')
                    ax2.set_ylabel('Frequency')
                    ax2.set_title(f'Distribution of Vignette Random Effects\n(ICC = {icc_vignette:.3f})')
                    ax2.grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    plt.savefig(f'{plot_dir}/Corrected_MAIHDA_Vignette_Analysis.png', dpi=300, bbox_inches='tight')
                    plt.close()
                
            except Exception as e:
                if verbose:
                    print(f" Corrected MAIHDA model failed: {e}")
                    import traceback
                    traceback.print_exc()

        # ------------------------
        # Model 3B: Mixed Model with Covariates
        # ------------------------
        if compute_full_maihda:
            import numpy as np  # Ensure numpy is imported for this section
            if verbose:
                print("\n🌟 Model 3B: Mixed Model with Covariates")
                print("=" * 60)

            try:
                # Ensure all covariates are available in the dataset
                covariates = ["sex", "race", "education", "income", "age"]
                # Also check for alternative column names that might be in your dataset
                alternative_names = {
                    "sex": ["gender"],
                    "race": ["ethnicity"],
                    "education": ["edu", "education_level"],
                    "income": ["income_level"],
                    "age": ["age_group"]
                }
                
                available_covs = []
                for cov in covariates:
                    if cov in df_work.columns:
                        available_covs.append(cov)
                    else:
                        # Check alternative names
                        for alt in alternative_names.get(cov, []):
                            if alt in df_work.columns:
                                available_covs.append(alt)
                                break

                if len(available_covs) == 0:
                    raise ValueError("No valid covariates found in dataset.")

                # Create stratum variable if it doesn't exist
                if 'stratum' not in df_work.columns:
                    # Create stratum from available fixed effects
                    available_fe = [fe for fe in fixed_effects if fe in df_work.columns]
                    if len(available_fe) >= 2:
                        stratum_ids = create_proper_stratum_id(df_work, available_fe, exclude_val="not_mentioned")
                        df_work['stratum'] = pd.Categorical(stratum_ids)
                    else:
                        # Fallback: use concept or vignette as stratum
                        if concept_col in df_work.columns:
                            df_work['stratum'] = df_work[concept_col]
                        elif vignette_col in df_work.columns:
                            df_work['stratum'] = df_work[vignette_col]
                        else:
                            raise ValueError("Cannot create stratum variable - insufficient grouping variables")

                # Model formula as in R
                cov_formula = f"{response_var} ~ " + " + ".join([f"C({c})" if df_work[c].dtype == 'object' else c for c in available_covs])

                # Fit Mixed Model (random intercept for 'stratum')
                md_cov = smf.mixedlm(cov_formula, df_work, groups=df_work["stratum"])
                mdf_cov = md_cov.fit(reml=False, method='powell', maxiter=2000, disp=False)

                if verbose:
                    print(mdf_cov.summary())

                # --- Calculate variance components and ICC ---
                tau_sq = mdf_cov.cov_re.iloc[0, 0]  # Variance of random intercept
                sigma_sq = mdf_cov.scale             # Residual variance
                icc = tau_sq / (tau_sq + sigma_sq)

                results["model3b_covariates"] = mdf_cov
                results["model3b_icc"] = icc
                results["model3b_tau_sq"] = tau_sq
                results["model3b_sigma_sq"] = sigma_sq

                if verbose:
                    print(f"\n Model 3B Results:")
                    print(f"   τ² (Between-stratum variance): {tau_sq:.6f}")
                    print(f"   σ² (Within-stratum variance):  {sigma_sq:.6f}")
                    print(f"   ICC (Intraclass Correlation):  {icc:.6f}")

                # --- Predictions (mean outcome) ---
                try:
                    df_work["m3b_pred"] = mdf_cov.predict(df_work)

                    # Confidence intervals for predictions (simplified approach)
                    # Use residual standard error as approximation for prediction intervals
                    import numpy as np
                    residual_se = np.sqrt(mdf_cov.scale)  # This is sigma from the model
                    
                    # Simple prediction intervals using residual standard error
                    df_work["m3b_lower"] = df_work["m3b_pred"] - 1.96 * residual_se
                    df_work["m3b_upper"] = df_work["m3b_pred"] + 1.96 * residual_se
                    
                except Exception as e:
                    print(f"Warning: Could not calculate prediction intervals: {e}")
                    df_work["m3b_pred"] = mdf_cov.predict(df_work)
                    df_work["m3b_lower"] = np.nan
                    df_work["m3b_upper"] = np.nan

                # ID variable as in R
                df_work["id"] = range(1, len(df_work) + 1)

                # --- Random Effects (Random Intercepts) & SEs (analogous to REsim) ---
                try:
                    re = mdf_cov.random_effects
                    re_df = pd.DataFrame({
                        "stratum": list(re.keys()),
                        "random_intercept": [v[0] if isinstance(v, np.ndarray) and len(v) > 0 else 
                                           v["Group"] if isinstance(v, dict) and "Group" in v else 
                                           list(v.values())[0] if isinstance(v, dict) else 
                                           0.0 for v in re.values()]
                    })

                    # Estimation of uncertainty (standard errors of random effects)
                    # Approximation: sqrt(tau_sq) as SE per intercept (R: REsim provides similar)
                    re_df["se_random_intercept"] = np.sqrt(tau_sq)
                    results["model3b_random_effects"] = re_df

                    if verbose:
                        print(f"\nRandom Effects (first 5):\n{re_df.head()}")
                        
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not extract random effects: {e}")
                    results["model3b_random_effects"] = pd.DataFrame()

                # Save to Excel
                fe_cov = pd.DataFrame({
                    "coef": mdf_cov.fe_params,
                    "se": mdf_cov.bse,
                    "pvalue": mdf_cov.pvalues,
                    "ci_lower": mdf_cov.conf_int()[0],
                    "ci_upper": mdf_cov.conf_int()[1]
                })
                fe_cov.to_excel(writer, sheet_name="Model3B_Fixed_Effects")

                re_df.to_excel(writer, sheet_name="Model3B_Random_Effects", index=False)

                model3b_metrics = pd.DataFrame({
                    'Metric': ['Between-Stratum Variance (τ²)', 
                              'Within-Stratum Variance (σ²)',
                              'ICC (Intraclass Correlation)',
                              'Number of Covariates Used',
                              'Number of Strata'],
                    'Value': [tau_sq, sigma_sq, icc, len(available_covs), df_work['stratum'].nunique()]
                })
                model3b_metrics.to_excel(writer, sheet_name="Model3B_Metrics", index=False)

                # Optional plotting for covariate model
                if save_plots:
                    # Plot random effects distribution
                    plt.figure(figsize=(12, 6))
                    plt.hist(re_df['random_intercept'], bins=20, alpha=0.7, edgecolor='black')
                    plt.axvline(0, color='red', linestyle='--', linewidth=2)
                    plt.xlabel('Random Intercept')
                    plt.ylabel('Frequency')
                    plt.title(f'Distribution of Stratum Random Effects (Model 3B)\n(ICC = {icc:.3f})')
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(f"{plot_dir}/Model3B_Random_Effects.png", dpi=300, bbox_inches='tight')
                    plt.close()

            except Exception as e:
                if verbose:
                    print(f" Model 3B failed: {e}")
                    import traceback
                    traceback.print_exc()
        else:
            if verbose:
                print(" Need at least 2 fixed effects variables for Model 3B")

        # ------------------------
        # Model 5: Mixed Model with Interaction Random Effects
        # ------------------------
        if verbose:
            print("\n Model 5: Mixed Model with Specific Interaction Random Effects")
        
        re_interactions_combined = pd.DataFrame()
        formula_mixed = f"{response_var} ~ {fixed_formula}"
        
        for interaction in interaction_cols + interaction_cols_3way:
            try:
                if interaction not in df_work.columns:
                    continue
                    
                vars_in_inter = interaction.split('_')
                
                # Step 1: Anzahl gültiger Beobachtungen (alles außer EXCLUDE)
                valid_mask = df_work[interaction] != "EXCLUDE"
                valid_obs = valid_mask.sum()
                valid_groups = df_work.loc[valid_mask, interaction].unique()
                n_valid_groups = len(valid_groups)
                
                if n_valid_groups < min_groups_per_interaction or valid_obs < min_obs_per_interaction:
                    if verbose:
                        print(f" Zu wenige gültige Gruppen/Beobachtungen für {interaction}: {n_valid_groups} groups, {valid_obs} obs")
                    continue
                
                # Step 2: Fit mixed model
                md_inter = smf.mixedlm(formula_mixed, df_work, groups=df_work[interaction])
                mdf_inter = md_inter.fit(reml=False, method="powell", maxiter=2000, disp=False)

                # Step 3: Random Effects extrahieren (nur gültige Gruppen)
                valid_random_effects = {
                    g: mdf_inter.random_effects[g]['Group'] 
                    for g in mdf_inter.random_effects 
                    if g != "EXCLUDE"
                }
                
                if len(valid_random_effects) == 0:
                    if verbose:
                        print(f" Keine gültigen Random Effects für {interaction}")
                    continue

                # Step 4: Ergebnisse speichern
                re_df = pd.DataFrame({
                    "random_intercept": list(valid_random_effects.values()),
                    "interaction_type": [len(vars_in_inter)] * len(valid_random_effects)
                }, index=list(valid_random_effects.keys()))
                re_df.index.name = interaction
                re_interactions_combined = pd.concat([re_interactions_combined, re_df], axis=0)

                if verbose:
                    print(f" Random Intercepts für {interaction} erfolgreich ({valid_obs} valid obs, {n_valid_groups} valid groups)")

            except Exception as e:
                if verbose:
                    print(f" Random Intercepts für {interaction} failed: {e}")

        # ------------------------
        # Save and Plot Interaction Random Effects
        # ------------------------
        if not re_interactions_combined.empty:
            re_interactions_combined.to_excel(writer, sheet_name="Interaction_Random_Effects")
            results['interaction_random_effects'] = re_interactions_combined
            
            # Plot
            if save_plots:
                plt.figure(figsize=(12, max(8, 0.2*len(re_interactions_combined))))
                re_sorted = re_interactions_combined.sort_values("random_intercept")
                colors = ['#3498db' if x == 2 else '#e74c3c' for x in re_sorted["interaction_type"]]
                
                plt.barh(range(len(re_sorted)), re_sorted["random_intercept"], color=colors, alpha=0.7)
                plt.yticks(range(len(re_sorted)), re_sorted.index, fontsize=8)
                plt.axvline(0, color='k', linestyle='--')
                plt.xlabel('Random Intercept')
                plt.title("Specific Interaction Random Effects (Blue=2-way, Red=3-way)")
                legend_elements = [Patch(facecolor='#3498db', alpha=0.7, label='2-way interactions'),
                                  Patch(facecolor='#e74c3c', alpha=0.7, label='3-way interactions')]
                plt.legend(handles=legend_elements)
                plt.grid(axis='x', alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{plot_dir}/Interaction_Random_Effects.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            if verbose:
                print(f" Interaction Random Effects gespeichert: {len(re_interactions_combined)} effects")
        
        else:
            if verbose:
                print(f" Keine Interaction Random Effects zu speichern")

    if verbose:
        print(f"\n Whole dataset analysis completed! Results saved in '{out_fname}'")
        if save_plots:
            print(f"📁 Plots saved in '{plot_dir}/' directory")
        
        # Print summary
        print("\n" + "="*60)
        print("SUMMARY OF MODELS:")
        print("="*60)
        if 'fixed_effects_model' in results:
            print(f" Model 1: Fixed Effects Only (OLS)")
        if 'mixed_model_concepts' in results:
            print(f" Model 2: Concepts as Random Effects (ICC: {results.get('concept_icc', 0):.3f})")
        if 'vignette_maihda_model' in results:
            print(f" Model 3: CORRECTED MAIHDA - Vignette-Level (ICC: {results.get('vignette_maihda_icc', 0):.3f})")
            print(f"   → {results.get('vignette_maihda_icc', 0)*100:.1f}% of variance at vignette level")
        if 'maihda_full_model' in results:
            print(f" Model 4: CLASSIC MAIHDA - Full Strata (ICC: {results.get('maihda_icc', 0):.3f})")
            print(f"   → {results.get('maihda_icc', 0)*100:.1f}% of variance at intersectional stratum level")
        if 'interaction_random_effects' in results:
            print(f" Model 5: Mixed Model with Specific Interaction Random Effects ({len(results['interaction_random_effects'])} effects)")
        print("="*60)
    
    return results


def create_proper_stratum_id(df, vignette_cols, exclude_val="not_mentioned"):
    """
    Create proper stratum IDs that only combine truly present dimensions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    vignette_cols : list
        List of vignette column names
    exclude_val : str
        Value to exclude (typically "not_mentioned")
    
    Returns:
    --------
    list : List of stratum IDs
    """
    
    stratum_ids = []
    
    for _, row in df.iterrows():
        present_dims = []
        
        for col in vignette_cols:
            if col in row and pd.notna(row[col]) and row[col] != exclude_val:
                present_dims.append(f"{col}:{row[col]}")
        
        if len(present_dims) == 0:
            stratum_ids.append("baseline")
        else:
            stratum_ids.append("|".join(present_dims))
    
    return stratum_ids





def compute_excess_risk_residual(df, plot_dir, response_var="response_recoded", 
                               fixed_effects=["race", "gender", "religion", "transness"],
                               concept_col="concept", stratum_col="stratum_all", 
                               save_excel_path=None, min_dimensions=2):
    """
    Compute excess risk using residual-based approach with proper parameter handling
    """
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_1samp
    
    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n COMPUTING EXCESS RISK ANALYSIS")
    print("="*60)
    print(f" Dataset: {len(df):,} observations")
    print(f" Strata: {df[stratum_col].nunique()} unique")
    print(f" Concepts: {df[concept_col].nunique()}")
    
    # Build formula for main effects model
    formula_parts = []
    for var in fixed_effects:
        if var in df.columns:
            formula_parts.append(f"C({var})")
    
    if not formula_parts:
        raise ValueError(f"None of the fixed effects {fixed_effects} found in dataframe columns")
    
    formula = f"{response_var} ~ " + " + ".join(formula_parts)
    print(f" Model formula: {formula}")
    
    # Fit main effects model
    try:
        print(" Fitting main effects model...")
        model_main = smf.ols(formula, data=df).fit()
        print(f" Model fitted successfully (R² = {model_main.rsquared:.4f})")
    except Exception as e:
        print(f" Model fitting failed: {e}")
        return None
    
    # Calculate residuals
    df_copy = df.copy()
    df_copy['residuals'] = model_main.resid
    df_copy['fitted'] = model_main.fittedvalues
    
    print(f" Residuals calculated (mean = {df_copy['residuals'].mean():.6f})")
    
    # Group by stratum and calculate excess risk metrics
    print(" Calculating stratum-level excess risk...")
    
    stratum_results = []
    
    for stratum in df_copy[stratum_col].unique():
        if stratum == 'baseline' or stratum == 'not_mentioned':  # Skip baseline
            continue
            
        stratum_data = df_copy[df_copy[stratum_col] == stratum]
        n_obs = len(stratum_data)
        
        if n_obs < 5:  # Skip strata with too few observations
            continue
        
        # Count dimensions (non-baseline characteristics)
        if isinstance(stratum, str):
            dimensions = stratum.split('|')
            n_dimensions = len([d for d in dimensions if d and 'not_mentioned' not in d])
        else:
            n_dimensions = 0
        
        if n_dimensions < min_dimensions:
            continue
        
        # Calculate excess risk metrics
        mean_residual = stratum_data['residuals'].mean()
        std_residual = stratum_data['residuals'].std()
        se_residual = std_residual / np.sqrt(n_obs)
        
        # One-sample t-test against 0
        if std_residual > 0:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(stratum_data['residuals'], 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # Confidence interval
        from scipy.stats import t as t_dist
        alpha = 0.05
        df_freedom = n_obs - 1
        t_critical = t_dist.ppf(1 - alpha/2, df_freedom) if df_freedom > 0 else np.nan
        
        ci_lower = mean_residual - t_critical * se_residual if not np.isnan(t_critical) else np.nan
        ci_upper = mean_residual + t_critical * se_residual if not np.isnan(t_critical) else np.nan
        
        # Calculate observed mean (for context)
        observed_mean = stratum_data[response_var].mean()
        
        stratum_results.append({
            'stratum': stratum,
            'n_observations': n_obs,
            'n_dimensions': n_dimensions,
            'excess_risk': mean_residual,
            'std_residual': std_residual,
            'se_residual': se_residual,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'observed_mean': observed_mean,
            'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    # Convert to DataFrame
    excess_df = pd.DataFrame(stratum_results)
    
    if len(excess_df) == 0:
        print(" No valid strata found for analysis")
        return excess_df
    
    print(f" Analyzed {len(excess_df)} strata")
    print(f" Significant excess risk: {excess_df['is_significant'].sum()} strata")
    
    # Sort by excess risk magnitude
    excess_df = excess_df.sort_values('excess_risk', key=abs, ascending=False).reset_index(drop=True)
    
    # Create visualizations
    print(" Creating visualizations...")
    
    # 1. Excess risk plot
    plt.figure(figsize=(14, 8))
    
    # Color by significance
    colors = ['red' if sig else 'lightblue' for sig in excess_df['is_significant']]
    
    plt.barh(range(len(excess_df)), excess_df['excess_risk'], color=colors, alpha=0.7)
    plt.yticks(range(len(excess_df)), [s[:50] + '...' if len(s) > 50 else s for s in excess_df['stratum']])
    plt.xlabel('Excess Risk (Mean Residual)')
    plt.title(f'Excess Risk by Stratum (n={len(excess_df)})')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='lightblue', alpha=0.7, label='Not significant')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'excess_risk_by_stratum.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Changed from plt.show() to plt.close()
    
    # 2. Significance plot
    if excess_df['is_significant'].sum() > 0:
        sig_data = excess_df[excess_df['is_significant']].copy()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(sig_data['excess_risk'], -np.log10(sig_data['p_value']), 
                   s=sig_data['n_observations']*5, alpha=0.6, c='red')
        plt.xlabel('Excess Risk')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Significant Excess Risk (n={len(sig_data)})')
        plt.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        
        # Annotate top results
        for i, row in sig_data.head(5).iterrows():
            plt.annotate(row['stratum'][:30], 
                        (row['excess_risk'], -np.log10(row['p_value'])),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'excess_risk_significance.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Changed from plt.show() to plt.close()
    
    # Export to Excel if path provided
    if save_excel_path:
        try:
            with pd.ExcelWriter(save_excel_path, engine="xlsxwriter") as writer:
                excess_df.to_excel(writer, sheet_name="Excess_Risk_AllStrata", index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Strata', 'Significant Strata', 'Mean Excess Risk', 
                              'Max Excess Risk', 'Min Excess Risk', 'Std Excess Risk'],
                    'Value': [len(excess_df), excess_df['is_significant'].sum(), 
                             excess_df['excess_risk'].mean(), excess_df['excess_risk'].max(),
                             excess_df['excess_risk'].min(), excess_df['excess_risk'].std()]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
            print(f" Results saved to: {save_excel_path}")
        except Exception as e:
            print(f" Failed to save Excel file: {e}")
    
    # Print summary
    print(f"\n SUMMARY")
    print(f" Total intersections: {len(df_sorted)}")
    if len(df_sorted) > 0:
        print(f" Top 5 highest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
        
        print(f" Top 5 lowest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.tail(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
    
    return tables


def create_proper_stratum_id(df, vignette_cols, exclude_val="not_mentioned"):
    """
    Create proper stratum IDs that only combine truly present dimensions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    vignette_cols : list
        List of vignette column names
    exclude_val : str
        Value to exclude (typically "not_mentioned")
    
    Returns:
    --------
    list : List of stratum IDs
    """
    
    stratum_ids = []
    
    for _, row in df.iterrows():
        present_dims = []
        
        for col in vignette_cols:
            if col in row and pd.notna(row[col]) and row[col] != exclude_val:
                present_dims.append(f"{col}:{row[col]}")
        
        if len(present_dims) == 0:
            stratum_ids.append("baseline")
        else:
            stratum_ids.append("|".join(present_dims))
    
    return stratum_ids





def compute_excess_risk_residual(df, plot_dir, response_var="response_recoded", 
                               fixed_effects=["race", "gender", "religion", "transness"],
                               concept_col="concept", stratum_col="stratum_all", 
                               save_excel_path=None, min_dimensions=2):
    """
    Compute excess risk using residual-based approach with proper parameter handling
    """
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_1samp
    
    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n COMPUTING EXCESS RISK ANALYSIS")
    print("="*60)
    print(f" Dataset: {len(df):,} observations")
    print(f" Strata: {df[stratum_col].nunique()} unique")
    print(f" Concepts: {df[concept_col].nunique()}")
    
    # Build formula for main effects model
    formula_parts = []
    for var in fixed_effects:
        if var in df.columns:
            formula_parts.append(f"C({var})")
    
    if not formula_parts:
        raise ValueError(f"None of the fixed effects {fixed_effects} found in dataframe columns")
    
    formula = f"{response_var} ~ " + " + ".join(formula_parts)
    print(f" Model formula: {formula}")
    
    # Fit main effects model
    try:
        print(" Fitting main effects model...")
        model_main = smf.ols(formula, data=df).fit()
        print(f" Model fitted successfully (R² = {model_main.rsquared:.4f})")
    except Exception as e:
        print(f" Model fitting failed: {e}")
        return None
    
    # Calculate residuals
    df_copy = df.copy()
    df_copy['residuals'] = model_main.resid
    df_copy['fitted'] = model_main.fittedvalues
    
    print(f" Residuals calculated (mean = {df_copy['residuals'].mean():.6f})")
    
    # Group by stratum and calculate excess risk metrics
    print(" Calculating stratum-level excess risk...")
    
    stratum_results = []
    
    for stratum in df_copy[stratum_col].unique():
        if stratum == 'baseline' or stratum == 'not_mentioned':  # Skip baseline
            continue
            
        stratum_data = df_copy[df_copy[stratum_col] == stratum]
        n_obs = len(stratum_data)
        
        if n_obs < 5:  # Skip strata with too few observations
            continue
        
        # Count dimensions (non-baseline characteristics)
        if isinstance(stratum, str):
            dimensions = stratum.split('|')
            n_dimensions = len([d for d in dimensions if d and 'not_mentioned' not in d])
        else:
            n_dimensions = 0
        
        if n_dimensions < min_dimensions:
            continue
        
        # Calculate excess risk metrics
        mean_residual = stratum_data['residuals'].mean()
        std_residual = stratum_data['residuals'].std()
        se_residual = std_residual / np.sqrt(n_obs)
        
        # One-sample t-test against 0
        if std_residual > 0:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(stratum_data['residuals'], 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # Confidence interval
        from scipy.stats import t as t_dist
        alpha = 0.05
        df_freedom = n_obs - 1
        t_critical = t_dist.ppf(1 - alpha/2, df_freedom) if df_freedom > 0 else np.nan
        
        ci_lower = mean_residual - t_critical * se_residual if not np.isnan(t_critical) else np.nan
        ci_upper = mean_residual + t_critical * se_residual if not np.isnan(t_critical) else np.nan
        
        # Calculate observed mean (for context)
        observed_mean = stratum_data[response_var].mean()
        
        stratum_results.append({
            'stratum': stratum,
            'n_observations': n_obs,
            'n_dimensions': n_dimensions,
            'excess_risk': mean_residual,
            'std_residual': std_residual,
            'se_residual': se_residual,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'observed_mean': observed_mean,
            'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    # Convert to DataFrame
    excess_df = pd.DataFrame(stratum_results)
    
    if len(excess_df) == 0:
        print(" No valid strata found for analysis")
        return excess_df
    
    print(f" Analyzed {len(excess_df)} strata")
    print(f" Significant excess risk: {excess_df['is_significant'].sum()} strata")
    
    # Sort by excess risk magnitude
    excess_df = excess_df.sort_values('excess_risk', key=abs, ascending=False).reset_index(drop=True)
    
    # Create visualizations
    print(" Creating visualizations...")
    
    # 1. Excess risk plot
    plt.figure(figsize=(14, 8))
    
    # Color by significance
    colors = ['red' if sig else 'lightblue' for sig in excess_df['is_significant']]
    
    plt.barh(range(len(excess_df)), excess_df['excess_risk'], color=colors, alpha=0.7)
    plt.yticks(range(len(excess_df)), [s[:50] + '...' if len(s) > 50 else s for s in excess_df['stratum']])
    plt.xlabel('Excess Risk (Mean Residual)')
    plt.title(f'Excess Risk by Stratum (n={len(excess_df)})')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='lightblue', alpha=0.7, label='Not significant')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'excess_risk_by_stratum.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Changed from plt.show() to plt.close()
    
    # 2. Significance plot
    if excess_df['is_significant'].sum() > 0:
        sig_data = excess_df[excess_df['is_significant']].copy()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(sig_data['excess_risk'], -np.log10(sig_data['p_value']), 
                   s=sig_data['n_observations']*5, alpha=0.6, c='red')
        plt.xlabel('Excess Risk')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Significant Excess Risk (n={len(sig_data)})')
        plt.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        
        # Annotate top results
        for i, row in sig_data.head(5).iterrows():
            plt.annotate(row['stratum'][:30], 
                        (row['excess_risk'], -np.log10(row['p_value'])),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'excess_risk_significance.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Changed from plt.show() to plt.close()
    
    # Export to Excel if path provided
    if save_excel_path:
        try:
            with pd.ExcelWriter(save_excel_path, engine="xlsxwriter") as writer:
                excess_df.to_excel(writer, sheet_name="Excess_Risk_AllStrata", index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Strata', 'Significant Strata', 'Mean Excess Risk', 
                              'Max Excess Risk', 'Min Excess Risk', 'Std Excess Risk'],
                    'Value': [len(excess_df), excess_df['is_significant'].sum(), 
                             excess_df['excess_risk'].mean(), excess_df['excess_risk'].max(),
                             excess_df['excess_risk'].min(), excess_df['excess_risk'].std()]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
            print(f" Results saved to: {save_excel_path}")
        except Exception as e:
            print(f" Failed to save Excel file: {e}")
    
    # Print summary
    print(f"\n SUMMARY")
    print(f" Total intersections: {len(df_sorted)}")
    if len(df_sorted) > 0:
        print(f" Top 5 highest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
        
        print(f" Top 5 lowest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.tail(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
    
    return tables


def create_proper_stratum_id(df, vignette_cols, exclude_val="not_mentioned"):
    """
    Create proper stratum IDs that only combine truly present dimensions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    vignette_cols : list
        List of vignette column names
    exclude_val : str
        Value to exclude (typically "not_mentioned")
    
    Returns:
    --------
    list : List of stratum IDs
    """
    
    stratum_ids = []
    
    for _, row in df.iterrows():
        present_dims = []
        
        for col in vignette_cols:
            if col in row and pd.notna(row[col]) and row[col] != exclude_val:
                present_dims.append(f"{col}:{row[col]}")
        
        if len(present_dims) == 0:
            stratum_ids.append("baseline")
        else:
            stratum_ids.append("|".join(present_dims))
    
    return stratum_ids




def compute_excess_risk_residual(df, plot_dir, response_var="response_recoded", 
                               fixed_effects=["race", "gender", "religion", "transness"],
                               concept_col="concept", stratum_col="stratum_all", 
                               save_excel_path=None, min_dimensions=2):
    """
    Compute excess risk using residual-based approach with proper parameter handling
    """
    
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.formula.api as smf
    from scipy.stats import ttest_1samp
    
    # Create output directory
    os.makedirs(plot_dir, exist_ok=True)
    
    print(f"\n COMPUTING EXCESS RISK ANALYSIS")
    print("="*60)
    print(f" Dataset: {len(df):,} observations")
    print(f" Strata: {df[stratum_col].nunique()} unique")
    print(f" Concepts: {df[concept_col].nunique()}")
    
    # Build formula for main effects model
    formula_parts = []
    for var in fixed_effects:
        if var in df.columns:
            formula_parts.append(f"C({var})")
    
    if not formula_parts:
        raise ValueError(f"None of the fixed effects {fixed_effects} found in dataframe columns")
    
    formula = f"{response_var} ~ " + " + ".join(formula_parts)
    print(f" Model formula: {formula}")
    
    # Fit main effects model
    try:
        print(" Fitting main effects model...")
        model_main = smf.ols(formula, data=df).fit()
        print(f" Model fitted successfully (R² = {model_main.rsquared:.4f})")
    except Exception as e:
        print(f" Model fitting failed: {e}")
        return None
    
    # Calculate residuals
    df_copy = df.copy()
    df_copy['residuals'] = model_main.resid
    df_copy['fitted'] = model_main.fittedvalues
    
    print(f" Residuals calculated (mean = {df_copy['residuals'].mean():.6f})")
    
    # Group by stratum and calculate excess risk metrics
    print(" Calculating stratum-level excess risk...")
    
    stratum_results = []
    
    for stratum in df_copy[stratum_col].unique():
        if stratum == 'baseline' or stratum == 'not_mentioned':  # Skip baseline
            continue
            
        stratum_data = df_copy[df_copy[stratum_col] == stratum]
        n_obs = len(stratum_data)
        
        if n_obs < 5:  # Skip strata with too few observations
            continue
        
        # Count dimensions (non-baseline characteristics)
        if isinstance(stratum, str):
            dimensions = stratum.split('|')
            n_dimensions = len([d for d in dimensions if d and 'not_mentioned' not in d])
        else:
            n_dimensions = 0
        
        if n_dimensions < min_dimensions:
            continue
        
        # Calculate excess risk metrics
        mean_residual = stratum_data['residuals'].mean()
        std_residual = stratum_data['residuals'].std()
        se_residual = std_residual / np.sqrt(n_obs)
        
        # One-sample t-test against 0
        if std_residual > 0:
            from scipy.stats import ttest_1samp
            t_stat, p_value = ttest_1samp(stratum_data['residuals'], 0)
        else:
            t_stat, p_value = np.nan, np.nan
        
        # Confidence interval
        from scipy.stats import t as t_dist
        alpha = 0.05
        df_freedom = n_obs - 1
        t_critical = t_dist.ppf(1 - alpha/2, df_freedom) if df_freedom > 0 else np.nan
        
        ci_lower = mean_residual - t_critical * se_residual if not np.isnan(t_critical) else np.nan
        ci_upper = mean_residual + t_critical * se_residual if not np.isnan(t_critical) else np.nan
        
        # Calculate observed mean (for context)
        observed_mean = stratum_data[response_var].mean()
        
        stratum_results.append({
            'stratum': stratum,
            'n_observations': n_obs,
            'n_dimensions': n_dimensions,
            'excess_risk': mean_residual,
            'std_residual': std_residual,
            'se_residual': se_residual,
            't_statistic': t_stat,
            'p_value': p_value,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'observed_mean': observed_mean,
            'is_significant': p_value < 0.05 if not np.isnan(p_value) else False
        })
    
    # Convert to DataFrame
    excess_df = pd.DataFrame(stratum_results)
    
    if len(excess_df) == 0:
        print(" No valid strata found for analysis")
        return excess_df
    
    print(f" Analyzed {len(excess_df)} strata")
    print(f" Significant excess risk: {excess_df['is_significant'].sum()} strata")
    
    # Sort by excess risk magnitude
    excess_df = excess_df.sort_values('excess_risk', key=abs, ascending=False).reset_index(drop=True)
    
    # Create visualizations
    print(" Creating visualizations...")
    
    # 1. Excess risk plot
    plt.figure(figsize=(14, 8))
    
    # Color by significance
    colors = ['red' if sig else 'lightblue' for sig in excess_df['is_significant']]
    
    plt.barh(range(len(excess_df)), excess_df['excess_risk'], color=colors, alpha=0.7)
    plt.yticks(range(len(excess_df)), [s[:50] + '...' if len(s) > 50 else s for s in excess_df['stratum']])
    plt.xlabel('Excess Risk (Mean Residual)')
    plt.title(f'Excess Risk by Stratum (n={len(excess_df)})')
    plt.axvline(0, color='black', linestyle='--', alpha=0.5)
    
    # Add significance legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Significant (p<0.05)'),
        Patch(facecolor='lightblue', alpha=0.7, label='Not significant')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'excess_risk_by_stratum.png'), dpi=300, bbox_inches='tight')
    plt.close()  # Changed from plt.show() to plt.close()
    
    # 2. Significance plot
    if excess_df['is_significant'].sum() > 0:
        sig_data = excess_df[excess_df['is_significant']].copy()
        
        plt.figure(figsize=(12, 6))
        plt.scatter(sig_data['excess_risk'], -np.log10(sig_data['p_value']), 
                   s=sig_data['n_observations']*5, alpha=0.6, c='red')
        plt.xlabel('Excess Risk')
        plt.ylabel('-log10(p-value)')
        plt.title(f'Significant Excess Risk (n={len(sig_data)})')
        plt.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=0.5, label='p=0.05')
        plt.axvline(0, color='black', linestyle='--', alpha=0.5)
        plt.legend()
        
        # Annotate top results
        for i, row in sig_data.head(5).iterrows():
            plt.annotate(row['stratum'][:30], 
                        (row['excess_risk'], -np.log10(row['p_value'])),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, 'excess_risk_significance.png'), dpi=300, bbox_inches='tight')
        plt.close()  # Changed from plt.show() to plt.close()
    
    # Export to Excel if path provided
    if save_excel_path:
        try:
            with pd.ExcelWriter(save_excel_path, engine="xlsxwriter") as writer:
                excess_df.to_excel(writer, sheet_name="Excess_Risk_AllStrata", index=False)
                
                # Summary sheet
                summary_data = {
                    'Metric': ['Total Strata', 'Significant Strata', 'Mean Excess Risk', 
                              'Max Excess Risk', 'Min Excess Risk', 'Std Excess Risk'],
                    'Value': [len(excess_df), excess_df['is_significant'].sum(), 
                             excess_df['excess_risk'].mean(), excess_df['excess_risk'].max(),
                             excess_df['excess_risk'].min(), excess_df['excess_risk'].std()]
                }
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name="Summary", index=False)
                
            print(f" Results saved to: {save_excel_path}")
        except Exception as e:
            print(f" Failed to save Excel file: {e}")
    
    # Print summary
    print(f"\n SUMMARY")
    print(f" Total intersections: {len(df_sorted)}")
    if len(df_sorted) > 0:
        print(f" Top 5 highest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.head(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
        
        print(f" Top 5 lowest {sort_by}:")
        for i, (idx, row) in enumerate(df_sorted.tail(5).iterrows()):
            stratum_name = row['stratum'] if 'stratum' in row else str(idx)
            value = row[sort_by] if sort_by in row else 'N/A'
            print(f"   {i+1}. {stratum_name[:60]}: {value:.4f}")
    
    return tables


def create_proper_stratum_id(df, vignette_cols, exclude_val="not_mentioned"):
    """
    Create proper stratum IDs that only combine truly present dimensions
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    vignette_cols : list
        List of vignette column names
    exclude_val : str
        Value to exclude (typically "not_mentioned")
    
    Returns:
    --------
    list : List of stratum IDs
    """
    
    stratum_ids = []
    
    for _, row in df.iterrows():
        present_dims = []
        
        for col in vignette_cols:
            if col in row and pd.notna(row[col]) and row[col] != exclude_val:
                present_dims.append(f"{col}:{row[col]}")
        
        if len(present_dims) == 0:
            stratum_ids.append("baseline")
        else:
            stratum_ids.append("|".join(present_dims))
    
    return stratum_ids


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


import pandas as pd, numpy as np, os
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

def run_maihda_simple(
    df,
    response_var="response_recoded",
    fixed_effects=("race", "gender", "religion", "transness", "concept"),
    stratum_col="stratum",
    out_dir="maihda_output",
    reml=True,
    optimizer="powell",
    disp=False,
    save_excel=True,
    make_plots=True,
):
    """
    Erweiterte MAIHDA Analyse mit allen Features aus dem R-Tutorial:
      - Deskriptive Statistiken (Table 1 & 2)
      - Nullmodell (1A) & Additivmodell (1B)
      - Alle Visualisierungen (Figure 1, 2, 3)
      - Random Effects mit CIs
      - Stratum-Level Analysen
    """

    import os
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    import matplotlib.pyplot as plt
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    print("\n" + "="*60)
    print(" MAIHDA ANALYSE GESTARTET")
    print("="*60)

    # ==========================================
    # TABLE 1: Deskriptive Statistiken (Individual Level)
    # ==========================================
    print("\n TABLE 1: Individual-Level Deskriptive Statistiken")
    print("-" * 60)
    
    table1_list = []
    
    # Outcome-Variable Statistiken
    outcome_stats = {
        "Variable": response_var,
        "N": len(df),
        "Mean": df[response_var].mean(),
        "SD": df[response_var].std(),
        "Min": df[response_var].min(),
        "Max": df[response_var].max()
    }
    table1_list.append(outcome_stats)
    print(f"\n{response_var}:")
    print(f"  N = {outcome_stats['N']}")
    print(f"  Mean (Sample Mean) = {outcome_stats['Mean']:.2f}")
    print(f"  SD = {outcome_stats['SD']:.2f}")
    print(f"  Range: [{outcome_stats['Min']:.2f}, {outcome_stats['Max']:.2f}]")
    
    # Kovariaten-Häufigkeiten
    print("\nKovariaten-Verteilung:")
    for var in fixed_effects:
        if var != "concept":  # concept ist meist konstant
            freq = df[var].value_counts().sort_index()
            print(f"\n{var}:")
            for idx, count in freq.items():
                pct = (count / len(df)) * 100
                print(f"  {idx}: {count} ({pct:.1f}%)")

    # ==========================================
    # TABLE 2: Stratum-Level Deskriptive Statistiken
    # ==========================================
    print("\n" + "="*60)
    print(" TABLE 2: Stratum-Level Deskriptive Statistiken")
    print("-" * 60)
    
    # Stratum-Größen berechnen
    stratum_sizes = df.groupby(stratum_col).size().reset_index(name='strataN')
    
    # Kategorisierung nach Größe
    size_categories = {
        'n100plus': (stratum_sizes['strataN'] >= 100).sum(),
        'n50plus': (stratum_sizes['strataN'] >= 50).sum(),
        'n30plus': (stratum_sizes['strataN'] >= 30).sum(),
        'n20plus': (stratum_sizes['strataN'] >= 20).sum(),
        'n10plus': (stratum_sizes['strataN'] >= 10).sum(),
        'nlessthan10': (stratum_sizes['strataN'] < 10).sum()
    }
    
    print(f"\nAnzahl Strata gesamt: {len(stratum_sizes)}")
    print("\nStratum-Größen-Verteilung:")
    for cat, count in size_categories.items():
        pct = (count / len(stratum_sizes)) * 100
        print(f"  {cat}: {count} strata ({pct:.1f}%)")
    
    print(f"\nStratum-Größe Statistiken:")
    print(f"  Mean: {stratum_sizes['strataN'].mean():.1f}")
    print(f"  Median: {stratum_sizes['strataN'].median():.0f}")
    print(f"  Min: {stratum_sizes['strataN'].min()}")
    print(f"  Max: {stratum_sizes['strataN'].max()}")
    
    # Observed stratum means
    observed_stratum_means = df.groupby(stratum_col)[response_var].mean()
    grand_mean = observed_stratum_means.mean()
    
    print(f"\nObserved Stratum Means:")
    print(f"  Grand Mean: {grand_mean:.2f}")
    print(f"  SD: {observed_stratum_means.std():.2f}")
    print(f"  Range: [{observed_stratum_means.min():.2f}, {observed_stratum_means.max():.2f}]")

    # ==========================================
    # MODELL 1A: Nullmodell
    # ==========================================
    print("\n" + "="*60)
    print("⚙️  MODELL 1A: Nullmodell (nur Random Intercept)")
    print("-" * 60)
    
    m1A = smf.mixedlm(f"{response_var} ~ 1", df, groups=df[stratum_col])
    m1A_fit = m1A.fit(reml=reml, method=optimizer, disp=disp)
    
    var_u_1A = float(m1A_fit.cov_re.iloc[0, 0])
    var_e_1A = m1A_fit.scale
    vpc_1A = var_u_1A / (var_u_1A + var_e_1A)
    
    # Predictions für 1A
    df['m1A_pred'] = m1A_fit.fittedvalues
    predicted_stratum_means_1A = df.groupby(stratum_col)['m1A_pred'].mean()
    precision_weighted_grand_mean = predicted_stratum_means_1A.mean()
    
    print(f" Modell 1A erfolgreich gefittet")
    print(f"  Var(u0) = {var_u_1A:.4f}")
    print(f"  Var(e) = {var_e_1A:.4f}")
    print(f"  VPC = {vpc_1A:.3f} ({vpc_1A*100:.1f}%)")
    print(f"  Precision Weighted Grand Mean: {precision_weighted_grand_mean:.2f}")

    # ==========================================
    # MODELL 1B: Additivmodell
    # ==========================================
    print("\n" + "="*60)
    print(" MODELL 1B: Additivmodell (Fixed + Random Effects)")
    print("-" * 60)
    
    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects])
    formula_1B = f"{response_var} ~ {fixed_formula}"
    
    m1B = smf.mixedlm(formula_1B, df, groups=df[stratum_col])
    m1B_fit = m1B.fit(reml=reml, method=optimizer, disp=disp)

    var_u_1B = float(m1B_fit.cov_re.iloc[0, 0])
    var_e_1B = m1B_fit.scale
    vpc_1B = var_u_1B / (var_u_1B + var_e_1B)
    pcv = ((var_u_1A - var_u_1B) / var_u_1A) * 100 if var_u_1A > 0 else np.nan

    print(f" Modell 1B erfolgreich gefittet")
    print(f"  Var(u0) = {var_u_1B:.4f}")
    print(f"  Var(e) = {var_e_1B:.4f}")
    print(f"  VPC = {vpc_1B:.3f} ({vpc_1B*100:.1f}%)")
    print(f"  PCV = {pcv:.2f}%")

    # Predictions für 1B
    df['m1B_pred'] = m1B_fit.fittedvalues

    # ==========================================
    # Random Effects mit CIs
    # ==========================================
    re_df = pd.DataFrame({
        stratum_col: list(m1B_fit.random_effects.keys()),
        "random_intercept": [v[0] for v in m1B_fit.random_effects.values()]
    })
    se_u = np.sqrt(var_u_1B)
    re_df["ci_lower"] = re_df["random_intercept"] - 1.96 * se_u
    re_df["ci_upper"] = re_df["random_intercept"] + 1.96 * se_u
    re_df["significant"] = ~((re_df["ci_lower"] <= 0) & (re_df["ci_upper"] >= 0))

    # Fixed Effects mit CIs
    fe_df = pd.DataFrame({
        "coef": m1B_fit.params,
        "pvalue": m1B_fit.pvalues,
        "ci_lower": m1B_fit.conf_int()[0],
        "ci_upper": m1B_fit.conf_int()[1]
    })

    # ==========================================
    # Stratum-Level Predictions
    # ==========================================
    stratum_df = (
        df.groupby(stratum_col)
        .agg({
            response_var: ['mean', 'size'],
            'm1A_pred': 'mean',
            'm1B_pred': 'mean'
        })
        .reset_index()
    )
    stratum_df.columns = [stratum_col, 'observed_mean', 'n_obs', 
                          'predicted_mean_1A', 'predicted_mean_1B']
    
    stratum_df = stratum_df.merge(re_df, on=stratum_col, how="left")
    
    # Approximate CIs für predicted means (vereinfachte Methode)
    # In R wird predictInterval verwendet, hier approximieren wir
    stratum_df['pred_se'] = np.sqrt(var_u_1B / stratum_df['n_obs'] + var_e_1B / stratum_df['n_obs'])
    stratum_df['pred_ci_lower'] = stratum_df['predicted_mean_1B'] - 1.96 * stratum_df['pred_se']
    stratum_df['pred_ci_upper'] = stratum_df['predicted_mean_1B'] + 1.96 * stratum_df['pred_se']

    # ==========================================
    # TABLE 3: Model Summary
    # ==========================================
    summary = pd.DataFrame({
        "Model": ["1A (Null)", "1B (Additiv)"],
        "Var_u0": [var_u_1A, var_u_1B],
        "Var_e": [var_e_1A, var_e_1B],
        "VPC": [vpc_1A, vpc_1B],
        "PCV_%": [np.nan, pcv]
    })

    # ==========================================
    # TABLE 4: Top/Bottom Strata
    # ==========================================
    stratum_df_sorted = stratum_df.sort_values('predicted_mean_1B')
    top6 = stratum_df_sorted.tail(6)
    bottom6 = stratum_df_sorted.head(6)
    
    print("\n" + "="*60)
    print(" TABLE 4: Top & Bottom 6 Strata (nach Model 1B)")
    print("-" * 60)
    print("\nBottom 6 (niedrigste predicted means):")
    print(bottom6[[stratum_col, 'predicted_mean_1B', 'observed_mean', 'n_obs']].to_string(index=False))
    print("\nTop 6 (höchste predicted means):")
    print(top6[[stratum_col, 'predicted_mean_1B', 'observed_mean', 'n_obs']].to_string(index=False))
    table2 = pd.DataFrame({
        'Metric': ['Total Strata', 'Mean Stratum Size', 'Median Stratum Size',
                'Min Size', 'Max Size', 'n>=100', 'n>=50', 'n>=30', 
                'n>=20', 'n>=10', 'n<10',
                'Grand Mean', 'SD of Stratum Means', 'Min Stratum Mean', 
                'Max Stratum Mean'],
        'Value': [len(stratum_sizes), stratum_sizes['strataN'].mean(),
                stratum_sizes['strataN'].median(), stratum_sizes['strataN'].min(),
                stratum_sizes['strataN'].max(),
                size_categories['n100plus'], size_categories['n50plus'],
                size_categories['n30plus'], size_categories['n20plus'],
                size_categories['n10plus'], size_categories['nlessthan10'],
                grand_mean, observed_stratum_means.std(),
                observed_stratum_means.min(), observed_stratum_means.max()]
    })
    # ==========================================
    # Excel Export
    # ==========================================
    if save_excel:
        out_path = os.path.join(out_dir, "maihda_results_full.xlsx")
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            # Table 1: Individual Stats
            pd.DataFrame(table1_list).to_excel(writer, sheet_name="Table1_Individual", index=False)
            
            # Table 2: Stratum Stats
            table2 = pd.DataFrame({
                'Metric': ['Total Strata', 'Mean Stratum Size', 'Median Stratum Size',
                          'Min Size', 'Max Size', 'n>=100', 'n>=50', 'n>=30', 
                          'n>=20', 'n>=10', 'n<10',
                          'Grand Mean', 'SD of Stratum Means', 'Min Stratum Mean', 
                          'Max Stratum Mean'],
                'Value': [len(stratum_sizes), stratum_sizes['strataN'].mean(),
                         stratum_sizes['strataN'].median(), stratum_sizes['strataN'].min(),
                         stratum_sizes['strataN'].max(),
                         size_categories['n100plus'], size_categories['n50plus'],
                         size_categories['n30plus'], size_categories['n20plus'],
                         size_categories['n10plus'], size_categories['nlessthan10'],
                         grand_mean, observed_stratum_means.std(),
                         observed_stratum_means.min(), observed_stratum_means.max()]
            })
            table2.to_excel(writer, sheet_name="Table2_Stratum_Stats", index=False)
            
            # Table 3: Model Summary
            summary.to_excel(writer, sheet_name="Table3_Model_Summary", index=False)
            
            # Table 4: Top/Bottom
            pd.concat([bottom6, top6]).to_excel(writer, sheet_name="Table4_TopBottom6", index=False)
            
            # Random Effects
            re_df.to_excel(writer, sheet_name="Random_Effects", index=False)
            
            # Fixed Effects
            fe_df.to_excel(writer, sheet_name="Fixed_Effects", index=True)
            
            # Stratum Predictions
            stratum_df.to_excel(writer, sheet_name="Stratum_Predictions", index=False)
            
        print(f"\n💾 Excel gespeichert: {out_path}")

    # ==========================================
    # PLOTS
    # ==========================================
    if make_plots:
        print("\n" + "="*60)
        print("📈 Erstelle Visualisierungen...")
        print("-" * 60)
        
        # FIGURE 1: Histogramme (3 Panels)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Panel A: Individual Outcomes
        ax = axes[0]
        ax.hist(df[response_var], bins=20, density=True, alpha=0.7, edgecolor='black')
        ax.axvline(df[response_var].mean(), color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(response_var)
        ax.set_ylabel('Density')
        ax.set_title('Panel A: Individual Outcomes')
        ax.text(0.98, 0.95, f'Sample Mean = {df[response_var].mean():.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel B: Observed Stratum Means
        ax = axes[1]
        ax.hist(observed_stratum_means, bins=20, density=True, alpha=0.7, 
                edgecolor='black', color='orange')
        ax.axvline(grand_mean, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Observed Stratum Mean {response_var}')
        ax.set_ylabel('Density')
        ax.set_title('Panel B: Observed Stratum Means')
        ax.text(0.98, 0.95, f'Grand Mean = {grand_mean:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Panel C: Predicted Stratum Means (Model 1A)
        ax = axes[2]
        ax.hist(predicted_stratum_means_1A, bins=20, density=True, alpha=0.7,
                edgecolor='black', color='green')
        ax.axvline(precision_weighted_grand_mean, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel(f'Predicted Stratum Mean {response_var} (Model 1A)')
        ax.set_ylabel('Density')
        ax.set_title('Panel C: Predicted Stratum Means (1A)')
        ax.text(0.98, 0.95, f'Precision Weighted\nGrand Mean = {precision_weighted_grand_mean:.2f}',
                transform=ax.transAxes, ha='right', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Figure1_Histograms.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(" Figure 1 gespeichert: Figure1_Histograms.png")
        
        # FIGURE 2: Caterpillar Plots - Predicted Means
        fig, ax = plt.subplots(figsize=(10, max(6, len(stratum_df) * 0.15)))
        
        stratum_df_sorted = stratum_df.sort_values('predicted_mean_1B').reset_index(drop=True)
        stratum_df_sorted['rank'] = range(len(stratum_df_sorted))
        
        ax.errorbar(
            stratum_df_sorted['predicted_mean_1B'],
            stratum_df_sorted['rank'],
            xerr=[
                stratum_df_sorted['predicted_mean_1B'] - stratum_df_sorted['pred_ci_lower'],
                stratum_df_sorted['pred_ci_upper'] - stratum_df_sorted['predicted_mean_1B']
            ],
            fmt='o', color='black', ecolor='gray', capsize=2, markersize=4
        )
        ax.set_xlabel(f'Predicted {response_var} (Model 1B)')
        ax.set_ylabel('Stratum Rank')
        ax.set_title('Figure 2: Caterpillar Plot - Predicted Stratum Means (Model 1B)')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Figure2_Caterpillar_Predicted_Means.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(" Figure 2 gespeichert: Figure2_Caterpillar_Predicted_Means.png")
        
        # FIGURE 3: Random Effects (3 Panels)
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(re_df) * 0.12)))
        
        # Panel A: Alle Random Effects
        ax = axes[0]
        re_df_sorted = re_df.sort_values('random_intercept').reset_index(drop=True)
        re_df_sorted['rank'] = range(len(re_df_sorted))
        
        ax.errorbar(
            re_df_sorted['random_intercept'],
            re_df_sorted['rank'],
            xerr=[
                re_df_sorted['random_intercept'] - re_df_sorted['ci_lower'],
                re_df_sorted['ci_upper'] - re_df_sorted['random_intercept']
            ],
            fmt='o', color='black', ecolor='tab:blue', capsize=2, markersize=3
        )
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Random Intercept (u₀ⱼ)')
        ax.set_ylabel('Stratum Rank')
        ax.set_title('Panel A: All Random Effects')
        ax.grid(axis='x', alpha=0.3)
        
        # Panel B: Nur signifikante Random Effects
        ax = axes[1]
        re_sig = re_df_sorted[re_df_sorted['significant']].reset_index(drop=True)
        
        if len(re_sig) > 0:
            re_sig['rank_sig'] = range(len(re_sig))
            ax.errorbar(
                re_sig['random_intercept'],
                re_sig['rank_sig'],
                xerr=[
                    re_sig['random_intercept'] - re_sig['ci_lower'],
                    re_sig['ci_upper'] - re_sig['random_intercept']
                ],
                fmt='o', color='darkred', ecolor='red', capsize=3, markersize=4
            )
            
            # Stratum IDs als Labels (nur für wenige Strata lesbar)
            if len(re_sig) <= 30:
                for idx, row in re_sig.iterrows():
                    ax.text(row['random_intercept'], row['rank_sig'], 
                           f" {row[stratum_col]}", fontsize=6, va='center')
        
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Random Intercept (u₀ⱼ)')
        ax.set_ylabel('Rank (Significant Only)')
        ax.set_title(f'Panel B: Significant Random Effects (n={len(re_sig)})')
        ax.grid(axis='x', alpha=0.3)
        
        # Panel C: Random Effects Verteilung
        ax = axes[2]
        ax.hist(re_df['random_intercept'], bins=30, density=True, 
               alpha=0.7, edgecolor='black', color='purple')
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        
        # Normalverteilung overlay
        mu, sigma = 0, se_u
        x = np.linspace(re_df['random_intercept'].min(), 
                       re_df['random_intercept'].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, 
               label=f'N(0, {sigma:.3f})')
        
        ax.set_xlabel('Random Intercept (u₀ⱼ)')
        ax.set_ylabel('Density')
        ax.set_title('Panel C: Distribution of Random Effects')
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'Figure3_Random_Effects_Analysis.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(" Figure 3 gespeichert: Figure3_Random_Effects_Analysis.png")
        
        print(f"\n✨ Alle {3} Visualisierungen erfolgreich erstellt!")

    # ==========================================
    # Final Summary
    # ==========================================
    print("\n" + "="*60)
    print(" MAIHDA ANALYSE ABGESCHLOSSEN")
    print("="*60)
    print(f"📁 Alle Ergebnisse gespeichert in: {out_dir}/")
    print(f"   - Excel: maihda_results_full.xlsx")
    if make_plots:
        print(f"   - Plots: Figure1-3 (PNG)")
    print(f"\n Wichtigste Ergebnisse:")
    print(f"   - VPC (Model 1A): {vpc_1A:.3f} ({vpc_1A*100:.1f}%)")
    print(f"   - VPC (Model 1B): {vpc_1B:.3f} ({vpc_1B*100:.1f}%)")
    print(f"   - PCV: {pcv:.2f}%")
    print(f"   - Signifikante Strata: {re_df['significant'].sum()} von {len(re_df)}")
    print("="*60 + "\n")

    return {
        "summary": summary,
        "fixed_effects": fe_df,
        "random_effects": re_df,
        "stratum_predictions": stratum_df,
        "model_1A": m1A_fit,
        "model_1B": m1B_fit,
        "table1": table1_list,
        "table2": table2,
        "top_bottom_6": pd.concat([bottom6, top6])
    }


def run_maihda_simple_long(
    df,
    response_var="response_recoded",
    fixed_effects=("race", "gender", "religion", "transness", "concept"),
    stratum_col="stratum",
    item_col=None,
    use_item_vc=False,
    out_dir="maihda_output",
    reml=True,
    optimizer="powell",
    disp=False,
    save_excel=True,
    make_plots=True,
):
    import os
    import numpy as np
    import pandas as pd
    import statsmodels.formula.api as smf
    import matplotlib.pyplot as plt
    from scipy import stats

    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print(" MAIHDA ANALYSE GESTARTET")
    print("=" * 60)

    # --------------------------
    # Helpers
    # --------------------------
    def _safe_var_u(fit):
        """Group random-intercept variance (stratum)."""
        try:
            cov_re = fit.cov_re
            if cov_re is None or getattr(cov_re, "empty", False):
                return 0.0
            return float(cov_re.iloc[0, 0])
        except Exception:
            return 0.0

    def _safe_var_item(fit, use_item_vc):
        """Variance component from vc_formula (item)."""
        if not use_item_vc:
            return 0.0
        try:
            return float(fit.vcomp[0]) if hasattr(fit, "vcomp") and len(fit.vcomp) > 0 else 0.0
        except Exception:
            return 0.0

    # ==========================================
    # TABLE 1: Individual-Level Deskriptive Statistiken
    # ==========================================
    print("\n TABLE 1: Individual-Level Deskriptive Statistiken")
    print("-" * 60)

    table1_list = []
    outcome_stats = {
        "Variable": response_var,
        "N": len(df),
        "Mean": df[response_var].mean(),
        "SD": df[response_var].std(),
        "Min": df[response_var].min(),
        "Max": df[response_var].max(),
    }
    table1_list.append(outcome_stats)

    print(f"\n{response_var}:")
    print(f"  N = {outcome_stats['N']}")
    print(f"  Mean (Sample Mean) = {outcome_stats['Mean']:.2f}")
    print(f"  SD = {outcome_stats['SD']:.2f}")
    print(f"  Range: [{outcome_stats['Min']:.2f}, {outcome_stats['Max']:.2f}]")

    print("\nKovariaten-Verteilung:")
    for var in fixed_effects:
        if var in df.columns and var != "concept":
            freq = df[var].value_counts().sort_index()
            print(f"\n{var}:")
            for idx, count in freq.items():
                pct = (count / len(df)) * 100
                print(f"  {idx}: {count} ({pct:.1f}%)")

    # ==========================================
    # TABLE 2: Stratum-Level Deskriptive Statistiken
    # ==========================================
    print("\n" + "=" * 60)
    print(" TABLE 2: Stratum-Level Deskriptive Statistiken")
    print("-" * 60)

    stratum_sizes = df.groupby(stratum_col).size().reset_index(name="strataN")
    size_categories = {
        "n100plus": (stratum_sizes["strataN"] >= 100).sum(),
        "n50plus": (stratum_sizes["strataN"] >= 50).sum(),
        "n30plus": (stratum_sizes["strataN"] >= 30).sum(),
        "n20plus": (stratum_sizes["strataN"] >= 20).sum(),
        "n10plus": (stratum_sizes["strataN"] >= 10).sum(),
        "nlessthan10": (stratum_sizes["strataN"] < 10).sum(),
    }

    print(f"\nAnzahl Strata gesamt: {len(stratum_sizes)}")
    print("\nStratum-Größen-Verteilung:")
    for cat, count in size_categories.items():
        pct = (count / len(stratum_sizes)) * 100
        print(f"  {cat}: {count} strata ({pct:.1f}%)")

    print(f"\nStratum-Größe Statistiken:")
    print(f"  Mean: {stratum_sizes['strataN'].mean():.1f}")
    print(f"  Median: {stratum_sizes['strataN'].median():.0f}")
    print(f"  Min: {stratum_sizes['strataN'].min()}")
    print(f"  Max: {stratum_sizes['strataN'].max()}")

    observed_stratum_means = df.groupby(stratum_col)[response_var].mean()
    grand_mean = observed_stratum_means.mean()

    print(f"\nObserved Stratum Means:")
    print(f"  Grand Mean: {grand_mean:.2f}")
    print(f"  SD: {observed_stratum_means.std():.2f}")
    print(f"  Range: [{observed_stratum_means.min():.2f}, {observed_stratum_means.max():.2f}]")

    table2 = pd.DataFrame({
        "Metric": [
            "Total Strata", "Mean Stratum Size", "Median Stratum Size",
            "Min Size", "Max Size", "n>=100", "n>=50", "n>=30",
            "n>=20", "n>=10", "n<10",
            "Grand Mean", "SD of Stratum Means", "Min Stratum Mean", "Max Stratum Mean"
        ],
        "Value": [
            len(stratum_sizes), stratum_sizes["strataN"].mean(),
            stratum_sizes["strataN"].median(), stratum_sizes["strataN"].min(),
            stratum_sizes["strataN"].max(),
            size_categories["n100plus"], size_categories["n50plus"], size_categories["n30plus"],
            size_categories["n20plus"], size_categories["n10plus"], size_categories["nlessthan10"],
            grand_mean, observed_stratum_means.std(),
            observed_stratum_means.min(), observed_stratum_means.max()
        ]
    })

    # ==========================================
    # Variance component formula (item)
    # ==========================================
    vc = None
    if use_item_vc:
        if item_col is None:
            raise ValueError("If use_item_vc=True, you must provide item_col.")
        if item_col not in df.columns:
            raise ValueError(f"item_col '{item_col}' not found in df columns.")
        vc = {"item": f"0 + C({item_col})"}

    # ==========================================
    # MODELL 1A: Nullmodell
    # ==========================================
    print("\n" + "=" * 60)
    print("⚙️  MODELL 1A: Nullmodell (Random Intercept Stratum + optional item VC)")
    print("-" * 60)

    m1A = smf.mixedlm(
        f"{response_var} ~ 1",
        df,
        groups=df[stratum_col],
        re_formula="1",
        vc_formula=vc,
    )
    m1A_fit = m1A.fit(reml=reml, method=optimizer, disp=disp)

    var_u_1A = _safe_var_u(m1A_fit)
    var_item_1A = _safe_var_item(m1A_fit, use_item_vc)
    var_e_1A = float(m1A_fit.scale)
    denom_1A = var_u_1A + var_item_1A + var_e_1A
    vpc_1A = (var_u_1A / denom_1A) if denom_1A > 0 else np.nan

    # Predictions for 1A (needed for plots/table)
    df = df.copy()
    df["m1A_pred"] = m1A_fit.fittedvalues
    predicted_stratum_means_1A = df.groupby(stratum_col)["m1A_pred"].mean()
    precision_weighted_grand_mean = predicted_stratum_means_1A.mean()

    print(" Modell 1A erfolgreich gefittet")
    print(f"  Var(u0) = {var_u_1A:.4f}")
    if use_item_vc:
        print(f"  Var(item) = {var_item_1A:.4f}")
    print(f"  Var(e) = {var_e_1A:.4f}")
    print(f"  VPC_stratum = {vpc_1A:.3f} ({vpc_1A*100:.1f}%)")
    print(f"  Precision Weighted Grand Mean: {precision_weighted_grand_mean:.2f}")

    # ==========================================
    # MODELL 1B: Additivmodell
    # ==========================================
    print("\n" + "=" * 60)
    print(" MODELL 1B: Additivmodell (Fixed + Random Effects)")
    print("-" * 60)

    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects if f in df.columns])
    formula_1B = f"{response_var} ~ {fixed_formula}" if fixed_formula else f"{response_var} ~ 1"

    m1B = smf.mixedlm(
        formula_1B,
        df,
        groups=df[stratum_col],
        re_formula="1",
        vc_formula=vc,
    )
    m1B_fit = m1B.fit(reml=reml, method=optimizer, disp=disp)

    var_u_1B = _safe_var_u(m1B_fit)
    var_item_1B = _safe_var_item(m1B_fit, use_item_vc)
    var_e_1B = float(m1B_fit.scale)
    denom_1B = var_u_1B + var_item_1B + var_e_1B
    vpc_1B = (var_u_1B / denom_1B) if denom_1B > 0 else np.nan
    pcv = ((var_u_1A - var_u_1B) / var_u_1A) * 100 if var_u_1A > 0 else np.nan

    print(" Modell 1B erfolgreich gefittet")
    print(f"  Var(u0) = {var_u_1B:.4f}")
    if use_item_vc:
        print(f"  Var(item) = {var_item_1B:.4f}")
    print(f"  Var(e) = {var_e_1B:.4f}")
    print(f"  VPC_stratum = {vpc_1B:.3f} ({vpc_1B*100:.1f}%)")
    print(f"  PCV = {pcv:.2f}%")

    df["m1B_pred"] = m1B_fit.fittedvalues

    # ==========================================
    # Random Effects mit CIs
    # ==========================================
    re_df = pd.DataFrame({
        stratum_col: list(m1B_fit.random_effects.keys()),
        "random_intercept": [v[0] for v in m1B_fit.random_effects.values()],
    })
    se_u = np.sqrt(max(var_u_1B, 0.0))
    re_df["ci_lower"] = re_df["random_intercept"] - 1.96 * se_u
    re_df["ci_upper"] = re_df["random_intercept"] + 1.96 * se_u
    re_df["significant"] = ~((re_df["ci_lower"] <= 0) & (re_df["ci_upper"] >= 0))

    # Fixed Effects with CIs
    fe_ci = m1B_fit.conf_int()
    fe_df = pd.DataFrame({
        "coef": m1B_fit.params,
        "pvalue": m1B_fit.pvalues,
        "ci_lower": fe_ci[0],
        "ci_upper": fe_ci[1],
    })

    # ==========================================
    # Stratum-Level Predictions
    # ==========================================
    stratum_df = (
        df.groupby(stratum_col)
        .agg({
            response_var: ["mean", "size"],
            "m1A_pred": "mean",
            "m1B_pred": "mean",
        })
        .reset_index()
    )
    stratum_df.columns = [
        stratum_col,
        "observed_mean",
        "n_obs",
        "predicted_mean_1A",
        "predicted_mean_1B",
    ]
    stratum_df = stratum_df.merge(re_df, on=stratum_col, how="left")

    # Approximate CI for predicted means
    stratum_df["pred_se"] = np.sqrt(var_u_1B / stratum_df["n_obs"] + var_e_1B / stratum_df["n_obs"])
    stratum_df["pred_ci_lower"] = stratum_df["predicted_mean_1B"] - 1.96 * stratum_df["pred_se"]
    stratum_df["pred_ci_upper"] = stratum_df["predicted_mean_1B"] + 1.96 * stratum_df["pred_se"]

    # ==========================================
    # TABLE 3: Model Summary
    # ==========================================
    summary = pd.DataFrame({
        "Model": ["1A (Null)", "1B (Additiv)"],
        "Var_u0": [var_u_1A, var_u_1B],
        "Var_item": [var_item_1A, var_item_1B],
        "Var_e": [var_e_1A, var_e_1B],
        "VPC_stratum": [vpc_1A, vpc_1B],
        "PCV_%": [np.nan, pcv],
    })

    # ==========================================
    # TABLE 4: Top/Bottom strata
    # ==========================================
    stratum_df_sorted = stratum_df.sort_values("predicted_mean_1B")
    top6 = stratum_df_sorted.tail(6)
    bottom6 = stratum_df_sorted.head(6)

    print("\n" + "=" * 60)
    print(" TABLE 4: Top & Bottom 6 Strata (nach Model 1B)")
    print("-" * 60)
    print("\nBottom 6 (niedrigste predicted means):")
    print(bottom6[[stratum_col, "predicted_mean_1B", "observed_mean", "n_obs"]].to_string(index=False))
    print("\nTop 6 (höchste predicted means):")
    print(top6[[stratum_col, "predicted_mean_1B", "observed_mean", "n_obs"]].to_string(index=False))

    # ==========================================
    # Excel Export
    # ==========================================
    if save_excel:
        out_path = os.path.join(out_dir, "maihda_results_full.xlsx")
        with pd.ExcelWriter(out_path, engine="xlsxwriter") as writer:
            pd.DataFrame(table1_list).to_excel(writer, sheet_name="Table1_Individual", index=False)
            table2.to_excel(writer, sheet_name="Table2_Stratum_Stats", index=False)
            summary.to_excel(writer, sheet_name="Table3_Model_Summary", index=False)
            pd.concat([bottom6, top6]).to_excel(writer, sheet_name="Table4_TopBottom6", index=False)
            re_df.to_excel(writer, sheet_name="Random_Effects", index=False)
            fe_df.to_excel(writer, sheet_name="Fixed_Effects", index=True)
            stratum_df.to_excel(writer, sheet_name="Stratum_Predictions", index=False)
        print(f"\n💾 Excel gespeichert: {out_path}")

    # ==========================================
    # PLOTS
    # ==========================================
    if make_plots:
        print("\n" + "=" * 60)
        print("📈 Erstelle Visualisierungen...")
        print("-" * 60)

        # FIGURE 1: Histograms (3 panels)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        ax = axes[0]
        ax.hist(df[response_var], bins=20, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(df[response_var].mean(), linestyle="--", linewidth=2)
        ax.set_xlabel(response_var)
        ax.set_ylabel("Density")
        ax.set_title("Panel A: Individual Outcomes")

        ax = axes[1]
        ax.hist(observed_stratum_means, bins=20, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(grand_mean, linestyle="--", linewidth=2)
        ax.set_xlabel(f"Observed Stratum Mean {response_var}")
        ax.set_ylabel("Density")
        ax.set_title("Panel B: Observed Stratum Means")

        ax = axes[2]
        ax.hist(predicted_stratum_means_1A, bins=20, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(precision_weighted_grand_mean, linestyle="--", linewidth=2)
        ax.set_xlabel(f"Predicted Stratum Mean {response_var} (Model 1A)")
        ax.set_ylabel("Density")
        ax.set_title("Panel C: Predicted Stratum Means (1A)")

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure1_Histograms.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # FIGURE 2: Caterpillar predicted means
        fig, ax = plt.subplots(figsize=(10, max(6, len(stratum_df) * 0.15)))
        sdf = stratum_df.sort_values("predicted_mean_1B").reset_index(drop=True)
        sdf["rank"] = range(len(sdf))
        ax.errorbar(
            sdf["predicted_mean_1B"],
            sdf["rank"],
            xerr=[sdf["predicted_mean_1B"] - sdf["pred_ci_lower"], sdf["pred_ci_upper"] - sdf["predicted_mean_1B"]],
            fmt="o",
            ecolor="gray",
            capsize=2,
            markersize=4,
        )
        ax.set_xlabel(f"Predicted {response_var} (Model 1B)")
        ax.set_ylabel("Stratum Rank")
        ax.set_title("Figure 2: Caterpillar Plot - Predicted Stratum Means (Model 1B)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure2_Caterpillar_Predicted_Means.png"), dpi=300, bbox_inches="tight")
        plt.close()

        # FIGURE 3: Random effects
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(re_df) * 0.12)))

        ax = axes[0]
        rsorted = re_df.sort_values("random_intercept").reset_index(drop=True)
        rsorted["rank"] = range(len(rsorted))
        ax.errorbar(
            rsorted["random_intercept"],
            rsorted["rank"],
            xerr=[rsorted["random_intercept"] - rsorted["ci_lower"], rsorted["ci_upper"] - rsorted["random_intercept"]],
            fmt="o",
            capsize=2,
            markersize=3,
        )
        ax.axvline(0, linestyle="--", linewidth=2)
        ax.set_xlabel("Random Intercept (u₀ⱼ)")
        ax.set_ylabel("Stratum Rank")
        ax.set_title("Panel A: All Random Effects")
        ax.grid(axis="x", alpha=0.3)

        ax = axes[1]
        re_sig = rsorted[rsorted["significant"]].reset_index(drop=True)
        if len(re_sig) > 0:
            re_sig["rank_sig"] = range(len(re_sig))
            ax.errorbar(
                re_sig["random_intercept"],
                re_sig["rank_sig"],
                xerr=[re_sig["random_intercept"] - re_sig["ci_lower"], re_sig["ci_upper"] - re_sig["random_intercept"]],
                fmt="o",
                capsize=3,
                markersize=4,
            )
        ax.axvline(0, linestyle="--", linewidth=2)
        ax.set_xlabel("Random Intercept (u₀ⱼ)")
        ax.set_ylabel("Rank (Significant Only)")
        ax.set_title(f"Panel B: Significant Random Effects (n={len(re_sig)})")
        ax.grid(axis="x", alpha=0.3)

        ax = axes[2]
        ax.hist(re_df["random_intercept"], bins=30, density=True, alpha=0.7, edgecolor="black")
        ax.axvline(0, linestyle="--", linewidth=2)
        # Normal overlay
        mu, sigma = 0, se_u if se_u > 0 else 1e-9
        x = np.linspace(re_df["random_intercept"].min(), re_df["random_intercept"].max(), 100)
        ax.plot(x, stats.norm.pdf(x, mu, sigma), linewidth=2, label=f"N(0, {sigma:.3f})")
        ax.set_xlabel("Random Intercept (u₀ⱼ)")
        ax.set_ylabel("Density")
        ax.set_title("Panel C: Distribution of Random Effects")
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "Figure3_Random_Effects_Analysis.png"), dpi=300, bbox_inches="tight")
        plt.close()

        print("✨ Plots gespeichert.")

    print("\n" + "=" * 60)
    print(" MAIHDA ANALYSE ABGESCHLOSSEN")
    print("=" * 60)
    print(f"📁 Alle Ergebnisse gespeichert in: {out_dir}/")
    print(f"   - Excel: maihda_results_full.xlsx")
    if make_plots:
        print(f"   - Plots: Figure1-3 (PNG)")
    print("\n Wichtigste Ergebnisse:")
    print(f"   - VPC_stratum (Model 1A): {vpc_1A:.3f} ({vpc_1A*100:.1f}%)")
    print(f"   - VPC_stratum (Model 1B): {vpc_1B:.3f} ({vpc_1B*100:.1f}%)")
    print(f"   - PCV: {pcv:.2f}%")
    print(f"   - Signifikante Strata: {re_df['significant'].sum()} von {len(re_df)}")
    print("=" * 60 + "\n")

    return {
        "summary": summary,
        "fixed_effects": fe_df,
        "random_effects": re_df,
        "stratum_predictions": stratum_df,
        "model_1A": m1A_fit,
        "model_1B": m1B_fit,
        "table1": table1_list,
        "table2": table2,
        "top_bottom_6": pd.concat([bottom6, top6]),
    }





import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# ------------------------------------------------------------
# 1) ROLL-UP: mittlere Excess-Risks pro Faktorlevel
# ------------------------------------------------------------
def rollup_excess(re_stratum, df_strata, factors, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    rollup_results = {}

    # --- NEU: robustes Umwandeln in Series
    if isinstance(re_stratum, pd.DataFrame):
        if re_stratum.shape[1] == 1:
            re_stratum = re_stratum.squeeze()
        else:
            raise ValueError("re_stratum DataFrame hat mehr als eine Spalte – bitte spezifizieren, welche verwendet werden soll.")
    elif not isinstance(re_stratum, pd.Series):
        raise TypeError(f"re_stratum muss Series oder 1-spaltiger DataFrame sein, ist aber {type(re_stratum)}")

    re_series = re_stratum.rename('excess_risk')

    # --- Rollup pro Faktor
    for f in factors:
        tmp = (
            df_strata[['stratum', f]]
            .drop_duplicates()
            .merge(re_series, left_on='stratum', right_index=True, how='left')
        )
        agg = (
            tmp.groupby(f)['excess_risk']
            .agg(['mean', 'std', 'count'])
            .sort_values('mean', ascending=False)
        )
        rollup_results[f] = agg

        # Export Excel
        agg.to_excel(os.path.join(out_dir, f"rollup_{f}.xlsx"))

        # Plot
        plt.figure(figsize=(6, max(3, 0.3 * len(agg))))
        plt.errorbar(
            agg['mean'], np.arange(len(agg)),
            xerr=agg['std'], fmt='o', color='black',
            ecolor='gray', capsize=4
        )
        plt.yticks(np.arange(len(agg)), agg.index)
        plt.axvline(0, color='k', linestyle='--', lw=1)
        plt.title(f"Excess Risk (mean ± sd) by {f}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"rollup_{f}.png"))
        plt.close()

    return rollup_results


# ------------------------------------------------------------
# 2) HEATMAPS: 2-Wege-Kombinationen der Factors
# ------------------------------------------------------------
def heatmaps_excess(re_stratum, df_strata, factors, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # --- NEU: gleiche Robustheit wie oben
    if isinstance(re_stratum, pd.DataFrame):
        re_stratum = re_stratum.squeeze()
    re_series = re_stratum.rename('excess_risk')

    for a, b in combinations(factors, 2):
        tmp = (
            df_strata[['stratum', a, b]]
            .drop_duplicates()
            .merge(re_series, left_on='stratum', right_index=True, how='left')
        )
        pivot = tmp.pivot_table(values='excess_risk', index=a, columns=b, aggfunc='mean')
        plt.figure(figsize=(6, 5))
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap='coolwarm', center=0)
        plt.title(f"Mean Excess Risk by {a} × {b}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"heatmap_{a}_{b}.png"))
        plt.close()


# ------------------------------------------------------------
# 3) Pfad-Analyse: additive vs. verschachtelte Effekte
# ------------------------------------------------------------
def additive_prediction(row, beta, baseline='not_mentioned', factors=None):
    pred = beta.get('Intercept', 0.0)
    for col in factors:
        lvl = row[col]
        if lvl == baseline:
            continue
        key = f"C({col}, Treatment('{baseline}'))[T.{lvl}]"
        pred += beta.get(key, 0.0)
    return pred


def nested_path_deltas(levels, beta, baseline='not_mentioned', order=None):
    steps, cur = [], {k: baseline for k in order}
    base = additive_prediction(cur, beta, baseline, order)
    acc = base
    for k in order:
        cur[k] = levels[k]
        new = additive_prediction(cur, beta, baseline, order)
        steps.append((k, levels[k], new - acc))
        acc = new
    return base, steps, acc


def example_path_plot(beta, example, order, out_dir, baseline='not_mentioned'):
    base, steps, final = nested_path_deltas(example, beta, baseline, order)
    labels = [f"{k}={v}" for k, v, _ in steps]
    deltas = [d for *_, d in steps]
    cumulative = np.cumsum([base] + deltas)
    plt.figure(figsize=(7, 4))
    plt.bar(range(len(cumulative)), cumulative, color='skyblue')
    plt.xticks(range(len(cumulative)), ['baseline'] + labels, rotation=45, ha='right')
    plt.ylabel("Predicted value")
    plt.title("Nested additive path")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "nested_path_example.png"))
    plt.close()


# ------------------------------------------------------------
# 4) MASTER-FUNKTION: alles zusammen
# ------------------------------------------------------------
def summarize_maihda(res, df_strata, out_dir="maihda_summary", baseline='not_mentioned'):
    os.makedirs(out_dir, exist_ok=True)

    # --- Random Effects extrahieren (immer 'random_intercept' verwenden)
    re_full = res['stratum_random_effects'] if 'stratum_random_effects' in res else res['random_effects']
    if isinstance(re_full, pd.DataFrame) and 'random_intercept' in re_full.columns:
        re = re_full.set_index('stratum')['random_intercept']
    else:
        re = re_full.squeeze()

    # --- Fixed Effects
    beta = (
        res['fixed_effects']['coef']
        if isinstance(res['fixed_effects'], pd.DataFrame)
        else res['fixed_effects']
    )
    factors = ['race', 'gender', 'religion', 'transness']

    print(f"\n📈 Creating MAIHDA summary for {len(df_strata)} strata...")
    print(f"Random effects type: {type(re)}")

    #  Roll-ups
    rollup_excess(re, df_strata, factors, out_dir)

    #  Heatmaps
    heatmaps_excess(re, df_strata, factors, out_dir)

    #  Beispielpfad
    example = {'gender': 'man', 'race': 'Black', 'religion': 'christian', 'transness': 'cis'}
    nested_path_deltas(example, beta, baseline, factors)
    example_path_plot(beta, example, factors, out_dir)

    print(f" MAIHDA summary created in: {out_dir}")



# bootstrap_maihda.py
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

# ---------------------------
# Utilities
# ---------------------------

def _fit_mixed(formula, df, group_col, reml=True, optimizer="powell", disp=False):
    return smf.mixedlm(formula, df, groups=df[group_col]).fit(
        reml=reml, method=optimizer, disp=disp
    )

def _simulate_from_fit(fit, df, group_col, rng):
    """
    Parametrische Simulation für *kontinuierliches* Outcome (random intercept + iid errors).
    y* = X beta + u_j[group] + eps
    """
    beta = fit.fe_params.values
    
    # Handle case where random effects variance might be zero or very small
    try:
        var_u = float(fit.cov_re.iloc[0, 0])     # Varianz random intercept
        if var_u < 1e-10:  # Practically zero variance
            var_u = 0.0
    except (IndexError, AttributeError):
        var_u = 0.0  # No random effects variance
        
    var_e = float(fit.scale)                 # Residualvarianz

    X = fit.model.exog
    groups = df[group_col].values

    # Map group -> index
    uniq = pd.Index(np.unique(groups))
    gpos = {g: i for i, g in enumerate(uniq)}

    # Generate random intercepts (will be zero if var_u is zero)
    u = rng.normal(0.0, np.sqrt(var_u), size=len(uniq)) if var_u > 0 else np.zeros(len(uniq))
    u_i = np.array([u[gpos[g]] for g in groups])
    eps = rng.normal(0.0, np.sqrt(var_e), size=X.shape[0])

    y_star = X @ beta + u_i + eps
    return y_star

# ---------------------------
# 1) Observed stratum means (nonparametrisch, innerhalb Stratum)
# ---------------------------

import numpy as np
import pandas as pd

def bootstrap_observed_stratum_means(
    df,
    response_var: str,
    stratum_col: str,
    B: int = 3000,
    alpha: float = 0.05,
    seed: int | None = None,
    clip_bounds: tuple[float, float] | None = (-5.0, 5.0),
    chunk_size: int | None = None,  # z.B. 500 für RAM-Entlastung; None = alles auf einmal
):
    """
    Bootstrap-CIs für observed stratum means (nichtparametrisch, innerhalb Stratum).

    - Entfernt NaN-Werte von response_var innerhalb jedes Strata.
    - Optional: CI auf bekannte Outcome-Grenzen clippen (z. B. [-5, 5]).
    - Optional: chunked Bootstrap (RAM-sparend).

    Returns: DataFrame mit Spalten:
      [stratum_col, observed_mean, obs_ci_lower, obs_ci_upper, n_obs]
    """
    rng = np.random.default_rng(seed)
    out_rows = []

    # Gruppieren einmal, dann je Stratum arbeiten
    grouped = df.groupby(stratum_col, sort=False)

    for s, sub in grouped:
        # Werte säubern -> numerisch & ohne NaN
        vals = pd.to_numeric(sub[response_var], errors="coerce").dropna().to_numpy()
        n = len(vals)

        if n == 0:
            # Falls alles NaN war: Mittel/CI nicht sinnvoll
            out_rows.append({
                stratum_col: s,
                "observed_mean": np.nan,
                "obs_ci_lower": np.nan,
                "obs_ci_upper": np.nan,
                "n_obs": 0
            })
            continue

        observed_mean = float(vals.mean())

        if n == 1:
            # Kein Bootstrap-CI möglich
            lo = hi = np.nan
        else:
            if chunk_size is None:
                # Einmalige Allokation (schnell, RAM-intensiv)
                idx = rng.integers(0, n, size=(B, n))
                boot = vals[idx].mean(axis=1)
            else:
                # Chunked (RAM-schonend)
                k = 0
                boots = []
                while k < B:
                    b_now = min(chunk_size, B - k)
                    idx = rng.integers(0, n, size=(b_now, n))
                    boots.append(vals[idx].mean(axis=1))
                    k += b_now
                boot = np.concatenate(boots, axis=0)

            lo, hi = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])

            # Optional auf bekannte Grenzen clippen
            if clip_bounds is not None:
                lo = float(np.clip(lo, clip_bounds[0], clip_bounds[1]))
                hi = float(np.clip(hi, clip_bounds[0], clip_bounds[1]))

        out_rows.append({
            stratum_col: s,
            "observed_mean": observed_mean,
            "obs_ci_lower": float(lo) if not np.isnan(lo) else np.nan,
            "obs_ci_upper": float(hi) if not np.isnan(hi) else np.nan,
            "n_obs": int(n)
        })

    return pd.DataFrame(out_rows)

def _fit_mixed(formula, df, group_col, reml=True, optimizer="powell", disp=False):
    return smf.mixedlm(formula, df, groups=df[group_col]).fit(
        reml=reml, method=optimizer, disp=disp
    )

def _simulate_from_fit(base, df, stratum_col, rng):
    """
    Parametric simulation from a random-intercept mixed model:
        y* = X beta + u_g + eps
    where u_g ~ N(0, var_u) and eps ~ N(0, var_e).
    """
    # Fixed part prediction (includes only fixed effects)
    mu_fixed = base.model.predict(base.params, exog=base.model.exog)

    # Estimate random-intercept variance (1x1 for random intercept models)
    # base.cov_re is the estimated covariance of random effects
    var_u = float(np.asarray(base.cov_re)[0, 0])

    # Residual variance
    var_e = float(base.scale)

    # Draw random intercept per stratum
    strata = df[stratum_col].to_numpy()
    uniq = pd.unique(strata)
    u_draws = rng.normal(loc=0.0, scale=np.sqrt(var_u), size=len(uniq))
    u_map = dict(zip(uniq, u_draws))
    u = np.array([u_map[g] for g in strata], dtype=float)

    # Draw observation noise
    eps = rng.normal(loc=0.0, scale=np.sqrt(var_e), size=len(df))

    # Simulated response
    y_star = mu_fixed + u + eps
    return y_star



import numpy as np
import pandas as pd
import statsmodels.api as sm
import patsy

import numpy as np
import pandas as pd
import patsy

def cluster_bootstrap_ols_minimal(
    formula: str,
    df: pd.DataFrame,
    cluster_col: str = "vignette_id",
    B: int = 2000,
    ci: float = 0.95,
    seed: int = 42
) -> pd.DataFrame:
    """
    Minimal cluster (pairs) bootstrap for OLS.
    Resamples clusters (vignette_id) with replacement, refits OLS via least squares.

    Returns a DataFrame with: coef, boot_se, ci_low, ci_high, p_boot, n_boot.
    """
    rng = np.random.default_rng(seed)

    # Stable design matrices (keeps categorical coding fixed)
    y_df, X_df = patsy.dmatrices(formula, df, return_type="dataframe")
    y = y_df.iloc[:, 0].to_numpy()
    X = X_df.to_numpy()
    terms = list(X_df.columns)

    # Align cluster ids to the patsy row order
    clusters = df.loc[X_df.index, cluster_col].to_numpy()
    uniq = pd.unique(clusters)
    m = len(uniq)
    idx_map = {c: np.where(clusters == c)[0] for c in uniq}

    # Base OLS
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

    # Bootstrap draws
    k = X.shape[1]
    draws = np.full((B, k), np.nan)

    for b in range(B):
        sampled = rng.choice(uniq, size=m, replace=True)
        idx = np.concatenate([idx_map[c] for c in sampled])
        try:
            draws[b], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        except Exception:
            continue

    draws = draws[~np.isnan(draws).any(axis=1)]
    n_eff = draws.shape[0]

    alpha = 1 - ci
    ci_low = np.percentile(draws, 100 * (alpha / 2), axis=0)
    ci_high = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
    boot_se = draws.std(axis=0, ddof=1)

    # Two-sided bootstrap p-value for H0: beta=0 (sign-based)
    p_boot = np.array([
        min(1.0, 2 * min((draws[:, j] >= 0).mean(), (draws[:, j] <= 0).mean()))
        for j in range(k)
    ])

    return pd.DataFrame(
        {"coef": beta_hat, "boot_se": boot_se, "ci_low": ci_low, "ci_high": ci_high,
         "p_boot": p_boot, "n_boot": n_eff},
        index=terms
    )

def plot_2d_coefficients_with_bootstrap_ellipses(
    df,
    model_name="LLM",
    cluster_col="vignette_id",
    B=2000,
    ci=0.95,
    seed=42,
    n_std=1.96,  # keep 1.96 for intuitive "95%" scaling
    auto_save=True,
    save_dir="plots",
    save_formats=("png", "pdf", "svg"),
    dpi=300,
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot warmth vs competence OLS coefficients with *bootstrap-based* confidence ellipses.

    - Fits two OLS models:
        warmth_score ~ C(gender) + C(race) + C(religion) + C(transness)
        competence_score ~ C(gender) + C(race) + C(religion) + C(transness)
    - Uses cluster (pairs) bootstrap on cluster_col to estimate uncertainty.
    - Ellipse covariance for each term is computed from joint bootstrap draws of (beta_w, beta_c).

    Returns dict with saved_files, coefficients, and bootstrap summaries.
    """

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import matplotlib.patheffects as patheffects
    from pathlib import Path
    import patsy
    import statsmodels.formula.api as smf
    from matplotlib.patches import Ellipse

    # ----------------------------
    # Helper: robust ellipse draw
    # ----------------------------
    def draw_confidence_ellipse(ax, xy, cov, n_std=1.96, **kwargs):
        cov = np.asarray(cov, dtype=float)
        vals, vecs = np.linalg.eigh(cov)
        vals = np.clip(vals, a_min=0.0, a_max=None)

        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        width, height = 2 * n_std * np.sqrt(vals)
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        ax.add_patch(Ellipse(xy=xy, width=width, height=height, angle=angle, **kwargs))

    # ---------------------------------------------------
    # Internal: minimal cluster bootstrap that keeps draws
    # ---------------------------------------------------
    def _cluster_bootstrap_with_draws(formula, df, cluster_col, B, ci, seed):
        rng = np.random.default_rng(seed)

        y_df, X_df = patsy.dmatrices(formula, df, return_type="dataframe")
        y = y_df.iloc[:, 0].to_numpy()
        X = X_df.to_numpy()
        terms = list(X_df.columns)

        clusters = df.loc[X_df.index, cluster_col].to_numpy()
        uniq = pd.unique(clusters)
        m = len(uniq)
        idx_map = {c: np.where(clusters == c)[0] for c in uniq}

        beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)

        k = X.shape[1]
        draws = np.full((B, k), np.nan)

        for b in range(B):
            sampled = rng.choice(uniq, size=m, replace=True)
            idx = np.concatenate([idx_map[c] for c in sampled])
            try:
                draws[b], *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
            except Exception:
                continue

        draws = draws[~np.isnan(draws).any(axis=1)]
        n_eff = draws.shape[0]

        alpha = 1 - ci
        ci_low = np.percentile(draws, 100 * (alpha / 2), axis=0)
        ci_high = np.percentile(draws, 100 * (1 - alpha / 2), axis=0)
        boot_se = draws.std(axis=0, ddof=1)

        p_boot = np.array([
            min(1.0, 2 * min((draws[:, j] >= 0).mean(), (draws[:, j] <= 0).mean()))
            for j in range(k)
        ])

        summary = pd.DataFrame(
            {"coef": beta_hat, "boot_se": boot_se, "ci_low": ci_low, "ci_high": ci_high,
             "p_boot": p_boot, "n_boot": n_eff},
            index=terms
        )

        draws_df = pd.DataFrame(draws, columns=terms)
        return summary, draws_df

    # ----------------------------
    # Saving setup
    # ----------------------------
    save_dir = Path(save_dir)
    if auto_save:
        save_dir.mkdir(exist_ok=True, parents=True)

    original_rc = dict(mpl.rcParams)

    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["figure.edgecolor"] = "white"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    # ----------------------------
    # Fit OLS models (points)
    # ----------------------------
    f_w = "warmth_score ~ C(gender) + C(race) + C(religion) + C(transness)"
    f_c = "competence_score ~ C(gender) + C(race) + C(religion) + C(transness)"

    model_w = smf.ols(f_w, data=df).fit()
    model_c = smf.ols(f_c, data=df).fit()

    coef_w = model_w.params.rename("warmth_coef").to_frame()
    coef_c = model_c.params.rename("competence_coef").to_frame()
    coef = coef_w.join(coef_c)
    coef = coef.drop(index="Intercept", errors="ignore")

    # Identify predictor family
    def family(term):
        if term.startswith("C(gender)"):
            return "gender"
        if term.startswith("C(race)"):
            return "race"
        if term.startswith("C(religion)"):
            return "religion"
        if term.startswith("C(transness)"):
            return "transness"
        return "other"

    coef["family"] = coef.index.map(family)

    # Clean labels
    def clean_ols_label(term):
        # Patsy style: C(gender)[T.woman]
        if "[" in term and "]" in term:
            inside = term.split("[", 1)[1].split("]", 1)[0]  # e.g. T.woman
            if inside.startswith("T."):
                return inside[2:]
            if "." in inside:
                return inside.split(".", 1)[1]
            return inside
        return term

    # Colors per family
    color_map = {
        "gender": "#1f77b4",
        "race": "#d62728",
        "religion": "#2ca02c",
        "transness": "#9467bd",
        "other": "gray",
    }

    # ----------------------------
    # Bootstrap for ellipses
    # ----------------------------
    if cluster_col not in df.columns:
        raise ValueError(f"cluster_col='{cluster_col}' not found in df.")

    sum_w, draws_w = _cluster_bootstrap_with_draws(f_w, df, cluster_col, B, ci, seed)
    sum_c, draws_c = _cluster_bootstrap_with_draws(f_c, df, cluster_col, B, ci, seed + 1)

    # Covariance per term from joint draws
    cov = {}
    for term in coef.index:
        if term not in draws_w.columns or term not in draws_c.columns:
            continue
        w = draws_w[term].to_numpy()
        c = draws_c[term].to_numpy()
        cov[term] = np.cov(np.vstack([w, c]), ddof=1)

    # ----------------------------
    # Plot
    # ----------------------------
    fig, ax = plt.subplots(figsize=figsize)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    for term, row in coef.iterrows():
        x = row["warmth_coef"]
        y = row["competence_coef"]
        fam = row["family"]

        if term in cov:
            draw_confidence_ellipse(
                ax=ax,
                xy=(x, y),
                cov=cov[term],
                n_std=n_std,
                alpha=0.2,
                color=color_map.get(fam, "gray"),
                zorder=1
            )

        ax.scatter(
            x, y, s=120,
            color=color_map.get(fam, "gray"),
            edgecolor="black", linewidth=1.2,
            alpha=0.9, zorder=3
        )

        ax.text(
            x + 0.02, y + 0.02,
            clean_ols_label(term),
            fontsize=11, weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")],
            zorder=4
        )

    ax.set_xlabel("Warmth Coefficient", fontsize=14, weight="bold")
    ax.set_ylabel("Competence Coefficient", fontsize=14, weight="bold")
    ax.set_title(
        f"{model_name}: Warmth vs Competence Coefficients (Cluster Bootstrap Ellipses)",
        fontsize=16, weight="bold", pad=20
    )

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=key,
                   markerfacecolor=color_map[key], markersize=12,
                   markeredgecolor="black", markeredgewidth=1.2)
        for key in ("gender", "race", "religion", "transness")
        if key in color_map
    ]
    ax.legend(handles=handles, title="Identity Dimensions", loc="best",
              framealpha=0.95, fontsize=12, title_fontsize=13,
              frameon=True, fancybox=True, shadow=True)

    ax.grid(alpha=0.25, zorder=0)
    ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=6)

    plt.tight_layout()

    # ----------------------------
    # Saving
    # ----------------------------
    saved_files = []

    def _save_one(path: Path, fmt: str):
        save_kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.2)
        if fmt.lower() in ("png", "tiff", "jpg", "jpeg"):
            save_kwargs["dpi"] = dpi
        plt.savefig(path, **save_kwargs)

    if save_path:
        p = Path(save_path)
        _save_one(p, p.suffix.lstrip("."))
        saved_files.append(str(p))

    if auto_save:
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (" ", "-", "_")).strip().replace(" ", "_")
        base = f"warmth_competence_coefficients_bootstrap_{safe_model_name}"
        for fmt in save_formats:
            p = save_dir / f"{base}.{fmt}"
            try:
                _save_one(p, fmt)
                saved_files.append(str(p))
            except Exception as e:
                print(f"Failed to save {p}: {e}")

    # restore rcParams
    mpl.rcParams.update(original_rc)

    plt.show()

    return {
        "saved_files": saved_files,
        "coefficients": coef,
        "models": {"warmth": model_w, "competence": model_c},
        "bootstrap": {"warmth": sum_w, "competence": sum_c}
    }


import re

def simplify_term(term: str, keep_prefix: bool = False) -> str:
    """
    Simplify statsmodels/patsy term names.
    Examples:
      'C(gender)[T.man]' -> 'man'  (or 'gender: man' if keep_prefix=True)
      'Intercept'        -> 'Intercept'
    """
    if term == "Intercept":
        return "Intercept"

    m = re.match(r"C\((?P<var>[^)]+)\)\[T\.(?P<lvl>.+)\]", term)
    if m:
        var = m.group("var")
        lvl = m.group("lvl")
        return f"{var}: {lvl}" if keep_prefix else lvl

    # fallback: return original
    return term


def stars(p: float) -> str:
    """
    Conventional star coding.
    """
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""




def bootstrap_fixed_effects(
    df, response_var, fixed_effects, stratum_col,
    B=1500, alpha=0.05, mode="cluster", reml=True,
    optimizer="powell", disp=False, seed=None
):
    """
    Bootstrap-CIs für Fixed-Effects eines MixedLM (Random Intercept).
    mode: 'cluster' (nichtparametrisch) oder 'parametric'.
    """
    rng = np.random.default_rng(seed)
    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects])
    formula = f"{response_var} ~ {fixed_formula}"

    # Basismodell (liefert fe_names und Startwerte für param. Bootstrap)
    base = _fit_mixed(formula, df, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
    fe_names = list(base.fe_params.index)
    k = len(fe_names)

    boots = np.empty((B, k), dtype=float)
    fail = 0

    if mode == "cluster":
        groups = df[stratum_col].unique()
        for b in range(B):
            g_star = rng.choice(groups, size=len(groups), replace=True)
            df_star = pd.concat([df[df[stratum_col] == g] for g in g_star], ignore_index=True)
            try:
                fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
                boots[b, :] = fit_b.fe_params.values
            except Exception:
                boots[b, :] = np.nan
                fail += 1

    elif mode == "parametric":
        for b in range(B):
            try:
                y_star = _simulate_from_fit(base, df, stratum_col, rng)
                df_star = df.copy()
                df_star[response_var] = y_star
                fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
                boots[b, :] = fit_b.fe_params.values
            except Exception:
                boots[b, :] = np.nan
                fail += 1
    else:
        raise ValueError("mode must be 'cluster' or 'parametric'")

    # fehlgeschlagene Fits entfernen
    ok = ~np.isnan(boots).any(axis=1)
    boots = boots[ok]
    if boots.size == 0:
        raise RuntimeError(f"Alle Bootstrap-Fits fehlgeschlagen (fail={fail}). Optimizer wechseln oder B reduzieren.")

    lo = np.percentile(boots, 100*alpha/2, axis=0)
    hi = np.percentile(boots, 100*(1-alpha/2), axis=0)
    mean_ = boots.mean(axis=0)

    # Nur Fixed-Effects aus asymptotischen Größen herausziehen
    ci_all = base.conf_int()
    ci_fe = ci_all.loc[fe_names]
    pvals_fe = base.pvalues.loc[fe_names]

    out = pd.DataFrame({
        "coef": fe_names,
        "coef_hat": base.fe_params.values,
        "boot_mean": mean_,
        "boot_ci_lower": lo,
        "boot_ci_upper": hi,
        "asym_ci_lower": ci_fe[0].values,
        "asym_ci_upper": ci_fe[1].values,
        "pvalue_asym": pvals_fe.values,
        "B_used": boots.shape[0],
        "B_failed": fail
    })
    return out, base


def bootstrap_random_intercepts(
    df, response_var, fixed_effects, stratum_col,
    B=1500, alpha=0.05, mode="parametric", reml=True,
    optimizer="powell", disp=False, seed=None, verbose=True
):
    """
    Bootstrap confidence intervals for random intercepts in MAIHDA models.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    response_var : str
        Name of response variable (e.g., "response_recoded")
    fixed_effects : tuple/list
        Fixed effects variables (e.g., ("race", "gender", "religion", "transness", "concept"))
    stratum_col : str
        Column name for strata/groups (e.g., "stratum")
    B : int, default=1500
        Number of bootstrap samples
    alpha : float, default=0.05
        Significance level for confidence intervals (1-alpha = confidence level)
    mode : str, default="parametric"
        Bootstrap method:
        - "parametric": Simulate from fitted model (recommended for random effects)
        - "cluster": Resample entire strata (may be unstable for random effects)
    reml : bool, default=True
        Use REML estimation
    optimizer : str, default="powell"
        Optimization method
    disp : bool, default=False
        Display convergence information
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    tuple: (bootstrap_results_df, base_model)
        - bootstrap_results_df: DataFrame with random intercept bootstrap results
        - base_model: Original fitted model
    """
    if verbose:
        print(f"\n🔄 BOOTSTRAPPING RANDOM INTERCEPTS")
        print(f"   Method: {mode}")
        print(f"   Bootstrap samples: {B}")
        print(f"   Confidence level: {100*(1-alpha):.1f}%")
    
    rng = np.random.default_rng(seed)
    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects])
    formula = f"{response_var} ~ {fixed_formula}"

    # Fit base model
    if verbose:
        print(f"   Fitting base model...")
    base = _fit_mixed(formula, df, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
    
    # Get original strata and their random effects
    original_strata = pd.Index(df[stratum_col].unique()).sort_values()
    n_strata = len(original_strata)
    
    if verbose:
        print(f"   Number of strata: {n_strata}")
    
    # Initialize bootstrap matrix: B x n_strata
    boots = np.empty((B, n_strata), dtype=float)
    fail = 0

    if verbose:
        print(f"   Running bootstrap...")
        
    for b in range(B):
        if verbose and (b + 1) % max(1, B // 10) == 0:
            print(f"   Progress: {b + 1}/{B} ({100*(b+1)/B:.1f}%)")
            
        try:
            if mode == "parametric":
                # Parametric bootstrap: simulate new response from fitted model
                y_star = _simulate_from_fit(base, df, stratum_col, rng)
                df_star = df.copy()
                df_star[response_var] = y_star
                
            elif mode == "cluster":
                # Cluster bootstrap: resample entire strata
                groups = df[stratum_col].unique()
                g_star = rng.choice(groups, size=len(groups), replace=True)
                df_star = pd.concat([df[df[stratum_col] == g] for g in g_star], ignore_index=True)
                
            else:
                raise ValueError("mode must be 'parametric' or 'cluster'")
            
            # Fit model to bootstrap sample
            fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
            
            # Extract random effects for original strata
            re_dict = fit_b.random_effects
            
            # For each original stratum, get its random intercept (0 if not present in bootstrap sample)
            for i, stratum in enumerate(original_strata):
                if stratum in re_dict:
                    # Extract random intercept (first element of random effects array)
                    boots[b, i] = re_dict[stratum][0] if hasattr(re_dict[stratum], '__len__') else re_dict[stratum]
                else:
                    # Stratum not present in this bootstrap sample -> set to 0
                    boots[b, i] = 0.0
                    
        except Exception as e:
            if verbose and fail == 0:  # Print first error
                print(f"   Warning: Bootstrap iteration {b+1} failed: {e}")
            boots[b, :] = np.nan
            fail += 1

    # Remove failed bootstrap samples
    valid_mask = ~np.isnan(boots).any(axis=1)
    boots_clean = boots[valid_mask]
    
    if boots_clean.size == 0:
        raise RuntimeError(f"All {B} bootstrap iterations failed. Try different optimizer or reduce model complexity.")
    
    if verbose:
        print(f"   Bootstrap completed: {boots_clean.shape[0]}/{B} successful iterations")
        if fail > 0:
            print(f"   Failed iterations: {fail}")

    # Calculate bootstrap statistics
    boot_mean = boots_clean.mean(axis=0)
    boot_ci_lower = np.percentile(boots_clean, 100 * alpha / 2, axis=0)
    boot_ci_upper = np.percentile(boots_clean, 100 * (1 - alpha / 2), axis=0)
    boot_se = boots_clean.std(axis=0, ddof=1)

    # Get original random effects
    base_re_dict = base.random_effects
    original_re = np.array([
        base_re_dict.get(stratum, [0.0])[0] if stratum in base_re_dict else 0.0
        for stratum in original_strata
    ])

    # Create results DataFrame
    results_df = pd.DataFrame({
        stratum_col: original_strata,
        "random_intercept": original_re,
        "boot_mean": boot_mean,
        "boot_se": boot_se,
        "boot_ci_lower": boot_ci_lower,
        "boot_ci_upper": boot_ci_upper,
        "significant": ~((boot_ci_lower <= 0) & (boot_ci_upper >= 0)),  # CI doesn't include 0
        "B_used": boots_clean.shape[0],
        "B_failed": fail
    })
    
    # Sort by absolute magnitude of random intercept
    results_df = results_df.reindex(results_df['random_intercept'].abs().sort_values(ascending=False).index)
    
    if verbose:
        n_significant = results_df['significant'].sum()
        print(f"    Results: {n_significant}/{n_strata} strata have significant random intercepts")
        
        # Show top 5 most extreme random intercepts
        print(f"\n   Top 5 largest (absolute) random intercepts:")
        top5 = results_df.head(5)
        for _, row in top5.iterrows():
            sig_mark = "***" if row['significant'] else "   "
            print(f"   {sig_mark} {row[stratum_col][:40]:40} {row['random_intercept']:8.4f} [{row['boot_ci_lower']:7.4f}, {row['boot_ci_upper']:7.4f}]")

    return results_df, base


def bootstrap_maihda_complete(
    df,
    response_var="response_recoded",
    fixed_effects=("race", "gender", "religion", "transness", "concept"),
    stratum_col="stratum",
    B_fixed=1500,
    B_random=1500,
    B_means=1000,
    alpha=0.05,
    mode_fixed="cluster",
    mode_random="parametric",
    reml=True,
    optimizer="powell",
    seed=None,
    verbose=True
):
    """
    Complete bootstrap analysis for MAIHDA models.
    
    Performs bootstrap confidence intervals for:
    1. Fixed effects (covariates)
    2. Random intercepts (stratum-specific deviations) 
    3. Observed stratum means
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with MAIHDA data
    response_var : str, default="response_recoded"
        Name of response variable
    fixed_effects : tuple, default=("race", "gender", "religion", "transness", "concept")
        Fixed effects variables
    stratum_col : str, default="stratum"
        Column name for intersectional strata
    B_fixed : int, default=1500
        Bootstrap samples for fixed effects
    B_random : int, default=1500
        Bootstrap samples for random intercepts  
    B_means : int, default=1000
        Bootstrap samples for observed stratum means
    alpha : float, default=0.05
        Significance level for CIs
    mode_fixed : str, default="cluster"
        Bootstrap method for fixed effects ("cluster" or "parametric")
    mode_random : str, default="parametric"
        Bootstrap method for random intercepts ("parametric" recommended)
    reml : bool, default=True
        Use REML estimation
    optimizer : str, default="powell"
        Optimization method
    seed : int, optional
        Random seed for reproducibility
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    dict: Complete bootstrap results containing:
        - "fixed_effects": DataFrame with fixed effects bootstrap CIs
        - "random_intercepts": DataFrame with random intercept bootstrap CIs  
        - "observed_means": DataFrame with observed stratum means bootstrap CIs
        - "base_model": Original fitted model
        - "summary": Summary statistics
    """
    if verbose:
        print("🚀 COMPREHENSIVE MAIHDA BOOTSTRAP ANALYSIS")
        print("=" * 60)
        print(f"Dataset: {len(df):,} observations, {df[stratum_col].nunique()} strata")
        print(f"Confidence level: {100*(1-alpha):.1f}%")
        print(f"Bootstrap samples: Fixed={B_fixed}, Random={B_random}, Means={B_means}")
        
    results = {}
    
    # Set consistent seed for reproducibility
    if seed is not None:
        np.random.seed(seed)
        
    # 1. Bootstrap Fixed Effects
    if verbose:
        print("\n FIXED EFFECTS BOOTSTRAP")
    try:
        fe_results, base_model = bootstrap_fixed_effects(
            df=df,
            response_var=response_var,
            fixed_effects=fixed_effects,
            stratum_col=stratum_col,
            B=B_fixed,
            alpha=alpha,
            mode=mode_fixed,
            reml=reml,
            optimizer=optimizer,
            seed=seed
        )
        results["fixed_effects"] = fe_results
        results["base_model"] = base_model
        if verbose:
            n_sig_fe = (fe_results["boot_ci_lower"] * fe_results["boot_ci_upper"] > 0).sum()
            print(f"    {n_sig_fe}/{len(fe_results)} fixed effects significant")
    except Exception as e:
        if verbose:
            print(f"    Fixed effects bootstrap failed: {e}")
        results["fixed_effects"] = pd.DataFrame()
        
    # 2. Bootstrap Random Intercepts
    if verbose:
        print("\n RANDOM INTERCEPTS BOOTSTRAP")
    try:
        re_results, _ = bootstrap_random_intercepts(
            df=df,
            response_var=response_var,
            fixed_effects=fixed_effects,
            stratum_col=stratum_col,
            B=B_random,
            alpha=alpha,
            mode=mode_random,
            reml=reml,
            optimizer=optimizer,
            seed=seed,
            verbose=verbose
        )
        results["random_intercepts"] = re_results
    except Exception as e:
        if verbose:
            print(f"    Random intercepts bootstrap failed: {e}")
        results["random_intercepts"] = pd.DataFrame()
        
    # 3. Bootstrap Observed Stratum Means
    if verbose:
        print("\n OBSERVED STRATUM MEANS BOOTSTRAP")
    try:
        means_results = bootstrap_observed_stratum_means(
            df=df,
            response_var=response_var,
            stratum_col=stratum_col,
            B=B_means,
            alpha=alpha,
            seed=seed
        )
        results["observed_means"] = means_results
        if verbose:
            print(f"    Bootstrap CIs for {len(means_results)} stratum means")
    except Exception as e:
        if verbose:
            print(f"    Observed means bootstrap failed: {e}")
        results["observed_means"] = pd.DataFrame()
    
    # Create summary
    summary = {
        "n_observations": len(df),
        "n_strata": df[stratum_col].nunique(),
        "n_fixed_effects": len(results.get("fixed_effects", [])),
        "n_random_intercepts": len(results.get("random_intercepts", [])),
        "confidence_level": 100 * (1 - alpha),
        "bootstrap_samples": {
            "fixed_effects": B_fixed,
            "random_intercepts": B_random, 
            "observed_means": B_means
        },
        "methods": {
            "fixed_effects": mode_fixed,
            "random_intercepts": mode_random,
            "observed_means": "nonparametric"
        }
    }
    
    # Add significance counts
    if not results.get("fixed_effects", pd.DataFrame()).empty:
        fe_df = results["fixed_effects"]
        summary["n_significant_fixed_effects"] = int((fe_df["boot_ci_lower"] * fe_df["boot_ci_upper"] > 0).sum())
        
    if not results.get("random_intercepts", pd.DataFrame()).empty:
        re_df = results["random_intercepts"]
        summary["n_significant_random_intercepts"] = int(re_df["significant"].sum())
    
    results["summary"] = summary
    
    if verbose:
        print("\n🎯 BOOTSTRAP ANALYSIS COMPLETE")
        print("=" * 60)
        print(f" Fixed effects: {summary.get('n_significant_fixed_effects', 0)}/{summary.get('n_fixed_effects', 0)} significant")
        print(f" Random intercepts: {summary.get('n_significant_random_intercepts', 0)}/{summary.get('n_random_intercepts', 0)} significant")
        print(f" Stratum means: {len(results.get('observed_means', []))} computed")
        
    return results


def plot_random_intercept_bootstrap(
    bootstrap_results,
    stratum_col="stratum",
    n_top=20,
    n_bottom=20,
    figsize=(12, 10),
    save_path=None,
    title_suffix="",
    show_insignificant=True
):
    """
    Create visualization of bootstrap confidence intervals for random intercepts.
    
    Parameters:
    -----------
    bootstrap_results : pd.DataFrame
        Results from bootstrap_random_intercepts()
    stratum_col : str, default="stratum"
        Column name for strata
    n_top : int, default=20
        Number of highest intercepts to show
    n_bottom : int, default=20
        Number of lowest intercepts to show
    figsize : tuple, default=(12, 10)
        Figure size
    save_path : str, optional
        Path to save the plot
    title_suffix : str, default=""
        Suffix for plot title
    show_insignificant : bool, default=True
        Whether to show non-significant intercepts
        
    Returns:
    --------
    matplotlib figure
    """
    df = bootstrap_results.copy()
    
    if df.empty:
        print("Warning: Empty bootstrap results")
        return None
        
    # Filter based on significance if requested
    if not show_insignificant:
        df = df[df['significant']].copy()
        
    if df.empty:
        print("Warning: No significant random intercepts found")
        return None
    
    # Select top and bottom intercepts
    df_sorted = df.sort_values('random_intercept')
    
    n_total = len(df_sorted)
    n_bottom_actual = min(n_bottom, n_total // 2)
    n_top_actual = min(n_top, n_total // 2)
    
    df_bottom = df_sorted.head(n_bottom_actual)
    df_top = df_sorted.tail(n_top_actual)
    df_plot = pd.concat([df_bottom, df_top])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create y positions
    y_pos = np.arange(len(df_plot))
    
    # Colors based on significance
    colors = ['#d62728' if sig else '#7f7f7f' for sig in df_plot['significant']]
    
    # Plot confidence intervals as horizontal lines
    for i, (_, row) in enumerate(df_plot.iterrows()):
        ax.plot([row['boot_ci_lower'], row['boot_ci_upper']], [i, i], 
                color=colors[i], linewidth=2, alpha=0.7)
    
    # Plot point estimates
    ax.scatter(df_plot['random_intercept'], y_pos, 
               c=colors, s=50, zorder=3, edgecolors='black', linewidth=0.5)
    
    # Reference line at zero
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
    
    # Formatting
    ax.set_yticks(y_pos)
    
    # Create shortened labels
    labels = []
    for stratum in df_plot[stratum_col]:
        if len(str(stratum)) > 40:
            label = str(stratum)[:37] + "..."
        else:
            label = str(stratum)
        labels.append(label)
    
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    
    ax.set_xlabel('Random Intercept (95% Bootstrap CI)', fontsize=12)
    ax.set_ylabel('Intersectional Strata', fontsize=12)
    
    title = f'Random Intercepts with Bootstrap Confidence Intervals{title_suffix}'
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#d62728', label='Significant (CI excludes 0)'),
        Patch(facecolor='#7f7f7f', label='Not significant')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add grid
    ax.grid(axis='x', alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    return fig


def export_maihda_bootstrap_results(
    bootstrap_results,
    output_dir="maihda_bootstrap_output",
    base_name="maihda_bootstrap",
    create_plots=True,
    save_excel=True,
    save_csv=True
):
    """
    Export and visualize complete MAIHDA bootstrap results.
    
    Parameters:
    -----------
    bootstrap_results : dict
        Results from bootstrap_maihda_complete()
    output_dir : str, default="maihda_bootstrap_output"
        Output directory
    base_name : str, default="maihda_bootstrap"
        Base name for output files
    create_plots : bool, default=True
        Whether to create visualization plots
    save_excel : bool, default=True
        Whether to save Excel file
    save_csv : bool, default=True
        Whether to save CSV files
        
    Returns:
    --------
    dict: Paths to created files
    """
    import os
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    
    # Export CSV files
    if save_csv:
        if "fixed_effects" in bootstrap_results and not bootstrap_results["fixed_effects"].empty:
            fe_path = os.path.join(output_dir, f"{base_name}_fixed_effects.csv")
            bootstrap_results["fixed_effects"].to_csv(fe_path, index=False)
            output_paths["fixed_effects_csv"] = fe_path
            
        if "random_intercepts" in bootstrap_results and not bootstrap_results["random_intercepts"].empty:
            re_path = os.path.join(output_dir, f"{base_name}_random_intercepts.csv")
            bootstrap_results["random_intercepts"].to_csv(re_path, index=False)
            output_paths["random_intercepts_csv"] = re_path
            
        if "observed_means" in bootstrap_results and not bootstrap_results["observed_means"].empty:
            means_path = os.path.join(output_dir, f"{base_name}_observed_means.csv")
            bootstrap_results["observed_means"].to_csv(means_path, index=False)
            output_paths["observed_means_csv"] = means_path
    
    # Export Excel file with multiple sheets
    if save_excel:
        excel_path = os.path.join(output_dir, f"{base_name}_complete.xlsx")
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            
            if "fixed_effects" in bootstrap_results and not bootstrap_results["fixed_effects"].empty:
                bootstrap_results["fixed_effects"].to_excel(writer, sheet_name='Fixed_Effects', index=False)
                
            if "random_intercepts" in bootstrap_results and not bootstrap_results["random_intercepts"].empty:
                bootstrap_results["random_intercepts"].to_excel(writer, sheet_name='Random_Intercepts', index=False)
                
            if "observed_means" in bootstrap_results and not bootstrap_results["observed_means"].empty:
                bootstrap_results["observed_means"].to_excel(writer, sheet_name='Observed_Means', index=False)
                
            if "summary" in bootstrap_results:
                summary_df = pd.DataFrame.from_dict(bootstrap_results["summary"], orient='index', columns=['Value'])
                summary_df.to_excel(writer, sheet_name='Summary')
        
        output_paths["excel"] = excel_path
        
    # Create plots
    if create_plots:
        if "random_intercepts" in bootstrap_results and not bootstrap_results["random_intercepts"].empty:
            # Plot random intercepts
            plot_path = os.path.join(output_dir, f"{base_name}_random_intercepts_plot.png")
            fig = plot_random_intercept_bootstrap(
                bootstrap_results["random_intercepts"],
                save_path=plot_path,
                title_suffix=f" ({base_name})"
            )
            if fig:
                plt.close(fig)
                output_paths["random_intercepts_plot"] = plot_path
    
    # Print summary
    print(f"\n📁 BOOTSTRAP RESULTS EXPORTED")
    print(f"Output directory: {output_dir}")
    for key, path in output_paths.items():
        print(f"  {key}: {os.path.basename(path)}")
    
    if "summary" in bootstrap_results:
        summary = bootstrap_results["summary"]
        print(f"\n📊 SUMMARY")
        print(f"  Observations: {summary.get('n_observations', 'N/A'):,}")
        print(f"  Strata: {summary.get('n_strata', 'N/A')}")
        print(f"  Significant fixed effects: {summary.get('n_significant_fixed_effects', 0)}/{summary.get('n_fixed_effects', 0)}")
        print(f"  Significant random intercepts: {summary.get('n_significant_random_intercepts', 0)}/{summary.get('n_random_intercepts', 0)}")
    
    return output_paths




# export_bootstrap_results.py
import os, json
from typing import Dict, Any
import pandas as pd

def export_bootstrap_results(results: Dict[str, Any],
                             out_dir: str = "maihda_bootstrap_output",
                             base_name: str = "maihda_bootstrap",
                             to_csv: bool = True,
                             to_excel: bool = True,
                             excel_engine: str = "xlsxwriter",
                             float_format: str | None = None,
                             include_metadata: bool = True,
                             metadata: Dict[str, Any] | None = None) -> Dict[str, str]:
    """
    Speichert die Bootstrap-Ergebnisse:
      - CSVs: fe_df_boot, re_df_boot, obs_means_boot
      - Excel: alle drei in separaten Sheets
      - Optional: metadata.json mit Parametern

    Parameters
    ----------
    results : dict
        Rückgabe von bootstrap_maihda_all(...):
          keys: "fe_df_boot", "re_df_boot", "obs_means_boot", optional *base_fit*s.
    out_dir : str
        Zielordner.
    base_name : str
        Basis für Dateinamen.
    to_csv : bool
        Einzelne CSV-Dateien schreiben.
    to_excel : bool
        Eine Excel-Datei mit mehreren Sheet-Tabs schreiben.
    excel_engine : str
        z.B. "xlsxwriter" (empfohlen) oder "openpyxl".
    float_format : str | None
        Z.B. "%.6f" für CSV/Excel-Werteformatierung.
    include_metadata : bool
        metadata.json schreiben (nützlich für Reproduzierbarkeit).
    metadata : dict | None
        Zusätzliche Infos/Parameter, die mitgespeichert werden.

    Returns
    -------
    dict : Pfade der erzeugten Dateien.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths = {}

    # Erwartete DataFrames holen (kopieren, um Seiteneffekte zu vermeiden)
    fe = results.get("fe_df_boot", pd.DataFrame()).copy()
    re = results.get("re_df_boot", pd.DataFrame()).copy()
    obs = results.get("obs_means_boot", pd.DataFrame()).copy()

    # CSVs
    if to_csv:
        p_fe = os.path.join(out_dir, f"{base_name}_fixed_effects.csv")
        p_re = os.path.join(out_dir, f"{base_name}_random_effects.csv")
        p_obs = os.path.join(out_dir, f"{base_name}_observed_means.csv")
        fe.to_csv(p_fe, index=False, float_format=float_format)
        re.to_csv(p_re, index=False, float_format=float_format)
        obs.to_csv(p_obs, index=False, float_format=float_format)
        paths["csv_fixed_effects"] = p_fe
        paths["csv_random_effects"] = p_re
        paths["csv_observed_means"] = p_obs

    # Excel (mehrere Sheets)
    if to_excel:
        p_xlsx = os.path.join(out_dir, f"{base_name}.xlsx")
        with pd.ExcelWriter(p_xlsx, engine=excel_engine) as writer:
            fe.to_excel(writer, sheet_name="Fixed_Effects", index=False)
            re.to_excel(writer, sheet_name="Random_Effects", index=False)
            obs.to_excel(writer, sheet_name="Observed_Means", index=False)

            # Optional: auch die asymptotischen Summaries aus Base-Fits anhängen
            fe_base = results.get("fe_base_fit", None)
            re_base = results.get("re_base_fit", None)
            if fe_base is not None:
                # Statsmodels Summary als Textsheet
                pd.DataFrame({"summary": [str(fe_base.summary())]}) \
                    .to_excel(writer, sheet_name="FE_Base_Summary", index=False)
            if re_base is not None:
                pd.DataFrame({"summary": [str(re_base.summary())]}) \
                    .to_excel(writer, sheet_name="RE_Base_Summary", index=False)
        paths["excel"] = p_xlsx

    # Metadaten
    if include_metadata:
        meta = dict(metadata or {})
        # Minimal sinnvolle Defaults
        meta.setdefault("n_fixed_effects", int(fe.shape[0]))
        meta.setdefault("n_random_effects", int(re.shape[0]))
        meta.setdefault("n_strata", int(obs.shape[0]))
        p_meta = os.path.join(out_dir, f"{base_name}_metadata.json")
        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        paths["metadata"] = p_meta

    return paths



def check_maihda_assumptions(model, df=None, out_dir="maihda_diagnostics", model_name="Model"):
    """
    Automatische Diagnostik für MixedLM / MAIHDA-Modelle.
    
    Args:
        model: Fitted MixedLMResults (z. B. m1B)
        df: optional, Originaldaten (für fitted/residual Plots)
        out_dir: Zielordner für Plots
        model_name: Bezeichnung des Modells
    Returns:
        summary_dict: Dictionary mit Var(u0), Var(e), VPC etc.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"\n🩺 Running MAIHDA diagnostics for {model_name}...")
    
    # -------------------------------
    #  Varianzkomponenten & VPC
    # -------------------------------
    var_u0 = float(model.cov_re.iloc[0, 0]) if hasattr(model, "cov_re") else np.nan
    var_e = model.scale
    vpc = var_u0 / (var_u0 + var_e) if (var_u0 + var_e) > 0 else np.nan
    
    print(f"Var(u0): {var_u0:.4f}")
    print(f"Var(e):  {var_e:.4f}")
    print(f"VPC:     {vpc:.3f}")

    # -------------------------------
    #  Residuen
    # -------------------------------
    resid = model.resid
    fitted = model.fittedvalues if hasattr(model, "fittedvalues") else None
    
    # Histogram & QQ-Plot
    plt.figure(figsize=(6,4))
    sns.histplot(resid, kde=True)
    plt.title(f"{model_name} – Residual Distribution")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{model_name}_residual_hist.png")
    plt.close()
    
    plt.figure(figsize=(5,5))
    st.probplot(resid, dist="norm", plot=plt)
    plt.title(f"{model_name} – Q-Q Plot of Residuals")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{model_name}_residual_qq.png")
    plt.close()

    # Residuals vs Fitted
    if fitted is not None:
        plt.figure(figsize=(6,4))
        plt.scatter(fitted, resid, alpha=0.6)
        plt.axhline(0, color='red', linestyle='--')
        plt.xlabel("Fitted values")
        plt.ylabel("Residuals")
        plt.title(f"{model_name} – Residuals vs Fitted")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{model_name}_resid_vs_fitted.png")
        plt.close()

    # -------------------------------
    #  Random Effects
    # -------------------------------
    try:
        re_dict = model.random_effects
        u0 = np.array([v[0] for v in re_dict.values()])
        
        plt.figure(figsize=(6,4))
        sns.histplot(u0, kde=True)
        plt.title(f"{model_name} – Random Intercepts Distribution")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{model_name}_random_effects_hist.png")
        plt.close()

        st.probplot(u0, dist="norm", plot=plt)
        plt.title(f"{model_name} – Q-Q Plot of Random Intercepts")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{model_name}_random_effects_qq.png")
        plt.close()

        # Einflussreiche Strata
        re_df = pd.DataFrame({
            "stratum": list(re_dict.keys()),
            "u0": u0
        })
        re_df["abs_z"] = (re_df["u0"] - re_df["u0"].mean()) / re_df["u0"].std()
        outliers = re_df.loc[re_df["abs_z"].abs() > 3]

        plt.figure(figsize=(8,4))
        sns.boxplot(x=re_df["u0"])
        plt.title(f"{model_name} – Random Intercepts Boxplot")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{model_name}_random_effects_box.png")
        plt.close()
        
        if len(outliers) > 0:
            print(f" Influential strata detected (>3 SD): {len(outliers)}")
            print(outliers.sort_values('abs_z', ascending=False).head())
    except Exception as e:
        print(f" Could not extract random effects: {e}")

    # -------------------------------
    # 4️⃣ Zusammenfassung
    # -------------------------------
    summary_dict = {
        "model": model_name,
        "Var(u0)": var_u0,
        "Var(e)": var_e,
        "VPC": vpc,
        "n_groups": getattr(model, "k_re", np.nan),
        "n_obs": model.nobs
    }

    print(f" Diagnostics saved to: {out_dir}/")
    return summary_dict



## Export Function for Tables

import pandas as pd
from pathlib import Path

# === SETTINGS ===
OUTPUT_DIR = Path("tables")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Safe LaTeX escaper ---
def escape_latex(text):
    """Safely escape LaTeX special characters for Overleaf."""
    if text is None:
        return ""
    text_str = str(text)
    replacements = {
        "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
        "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}", "\\": r"\textbackslash{}",
    }
    for char, repl in replacements.items():
        text_str = text_str.replace(char, repl)
    return text_str


# --- Flatten MAIHDA result dictionaries ---
def flatten_model_dict(model_dict):
    """
    Takes your res_* dicts and extracts key numeric summaries for LaTeX.
    Returns a flat dictionary of scalar values.
    """
    flat = {}

    #  Pull from summary DataFrame
    if isinstance(model_dict.get("summary"), pd.DataFrame):
        summary_df = model_dict["summary"]
        # get first row if it exists
        if not summary_df.empty:
            for col in summary_df.columns:
                val = summary_df.iloc[0][col]
                if pd.api.types.is_number(val):
                    flat[col] = round(float(val), 4)
                else:
                    flat[col] = str(val)

    #  Extract stats from model objects
    for key in ["model_1A", "model_1B"]:
        m = model_dict.get(key)
        if m is not None:
            flat[f"{key}_AIC"] = getattr(m, "aic", None)
            flat[f"{key}_BIC"] = getattr(m, "bic", None)
            flat[f"{key}_LogLik"] = getattr(m, "llf", None)
            flat[f"{key}_N"] = getattr(m, "nobs", None)

    #  Add counts or shapes of big DataFrames (instead of dumping them)
    for key in ["fixed_effects", "random_effects", "stratum_predictions",
                "table2", "top_bottom_6"]:
        obj = model_dict.get(key)
        if isinstance(obj, pd.DataFrame):
            flat[f"{key}_rows"] = obj.shape[0]
            flat[f"{key}_cols"] = obj.shape[1]

    # 4️⃣ Add list length for table1
    if isinstance(model_dict.get("table1"), list):
        flat["table1_len"] = len(model_dict["table1"])

    return flat


# --- Export LaTeX ---
def export_model_to_latex(result_dict, model_name):
    import pandas as pd
    from pathlib import Path

    # replace % → perc in keys
    cleaned = {k.replace('%', 'perc'): v for k, v in result_dict.items()}

    df = pd.DataFrame(list(cleaned.items()), columns=["Metric", "Value"])

    # round numeric values
    df["Value"] = df["Value"].apply(
        lambda x: f"{x:.3f}" if isinstance(x, (float, int)) else str(x)
    )

    # basic LaTeX export
    latex_str = df.to_latex(
        index=False,
        caption=f"MAIHDA summary: {model_name}",
        label=f"tab:maihda_{model_name.lower()}",
        column_format="l r",
        escape=True,
    )

    Path("tables").mkdir(exist_ok=True)
    path = Path(f"tables/maihda_{model_name.lower()}.tex")
    path.write_text(latex_str, encoding="utf-8")
    print(f" Exported pretty table: {path}")



# --- Main function ---
def export_all_models(namespace):
    """Find all res_* variables in the current namespace and export clean tables."""
    res_vars = {k: v for k, v in namespace.items() if k.startswith("res_")}
    if not res_vars:
        print("  No variables starting with 'res_' found.")
        return

    for name, res in res_vars.items():
        if isinstance(res, dict):
            flat_res = flatten_model_dict(res)
            export_model_to_latex(flat_res, name)
        else:
            print(f"  Skipped {name} (not a dict)")



def _calculate_bca_intervals(boots, theta_hat, jackknife_values, alpha):
    """
    Calculate BCa (bias-corrected and accelerated) confidence intervals.
    
    Parameters:
    -----------
    boots : np.ndarray
        Bootstrap distribution (B x k array)
    theta_hat : np.ndarray
        Original parameter estimates (length k)
    jackknife_values : np.ndarray
        Jackknife estimates (n x k array), or None to skip acceleration
    alpha : float
        Significance level
        
    Returns:
    --------
    tuple: (lower_bounds, upper_bounds) as arrays of length k
    """
    B, k = boots.shape
    lower = np.zeros(k)
    upper = np.zeros(k)
    
    # Standard normal quantiles
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)
    
    for j in range(k):
        boot_j = boots[:, j]
        
        # Bias correction: proportion of bootstrap estimates < original
        z0 = stats.norm.ppf(np.mean(boot_j < theta_hat[j]))
        
        # Acceleration factor from jackknife
        if jackknife_values is not None:
            jack_j = jackknife_values[:, j]
            jack_mean = np.mean(jack_j)
            num = np.sum((jack_mean - jack_j) ** 3)
            den = 6 * (np.sum((jack_mean - jack_j) ** 2) ** 1.5)
            a = num / den if den != 0 else 0
        else:
            a = 0
        
        # Adjusted percentiles
        alpha_lower = stats.norm.cdf(z0 + (z0 + z_alpha_lower) / (1 - a * (z0 + z_alpha_lower)))
        alpha_upper = stats.norm.cdf(z0 + (z0 + z_alpha_upper) / (1 - a * (z0 + z_alpha_upper)))
        
        # Handle edge cases
        alpha_lower = np.clip(alpha_lower, 0.001, 0.999)
        alpha_upper = np.clip(alpha_upper, 0.001, 0.999)
        
        lower[j] = np.percentile(boot_j, 100 * alpha_lower)
        upper[j] = np.percentile(boot_j, 100 * alpha_upper)
    
    return lower, upper


def _bootstrap_diagnostics(boots, param_names, alpha=0.05):
    """
    Calculate diagnostic statistics for bootstrap distributions.
    
    Returns:
    --------
    pd.DataFrame with diagnostics for each parameter
    """
    B, k = boots.shape
    diagnostics = []
    
    for j in range(k):
        boot_j = boots[:, j]
        
        # Normality test (Shapiro-Wilk if B < 5000, else Kolmogorov-Smirnov)
        if B < 5000:
            stat, p_val = stats.shapiro(boot_j)
            test_name = "Shapiro-Wilk"
        else:
            # Standardize and test against normal
            z = (boot_j - boot_j.mean()) / boot_j.std()
            stat, p_val = stats.kstest(z, 'norm')
            test_name = "KS-test"
        
        # Skewness and kurtosis
        skew = stats.skew(boot_j)
        kurt = stats.kurtosis(boot_j)
        
        diagnostics.append({
            'parameter': param_names[j],
            'boot_mean': boot_j.mean(),
            'boot_se': boot_j.std(ddof=1),
            'boot_skewness': skew,
            'boot_kurtosis': kurt,
            'normality_test': test_name,
            'normality_p': p_val,
            'normal': p_val > alpha,  # Fail to reject normality
            'effective_B': B
        })
    
    return pd.DataFrame(diagnostics)


def bootstrap_fixed_effects_improved(
    df, response_var, fixed_effects, stratum_col,
    B=1500, alpha=0.05, mode="cluster", reml=True,
    optimizer="powell", disp=False, seed=None,
    ci_method="percentile", verbose=True
):
    """
    Improved bootstrap CIs for fixed effects with multiple CI methods and diagnostics.
    
    New parameters:
    ---------------
    ci_method : str, default="percentile"
        Method for CI construction:
        - "percentile": Standard percentile method
        - "bca": Bias-corrected and accelerated (more accurate but slower)
        - "both": Calculate both methods
    verbose : bool, default=True
        Print diagnostic information
        
    Returns:
    --------
    tuple: (results_df, base_model, diagnostics_df)
    """
    if verbose:
        print(f"\n🔄 BOOTSTRAPPING FIXED EFFECTS")
        print(f"   Method: {mode}, CI method: {ci_method}")
        print(f"   Bootstrap samples: {B}")
    
    rng = np.random.default_rng(seed)
    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects])
    formula = f"{response_var} ~ {fixed_formula}"

    # Fit base model
    base = _fit_mixed(formula, df, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
    fe_names = list(base.fe_params.index)
    k = len(fe_names)
    theta_hat = base.fe_params.values

    boots = np.empty((B, k), dtype=float)
    fail = 0

    if mode == "cluster":
        groups = df[stratum_col].unique()
        for b in range(B):
            g_star = rng.choice(groups, size=len(groups), replace=True)
            df_star = pd.concat([df[df[stratum_col] == g] for g in g_star], ignore_index=True)
            try:
                fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
                boots[b, :] = fit_b.fe_params.values
            except Exception:
                boots[b, :] = np.nan
                fail += 1

    elif mode == "parametric":
        for b in range(B):
            try:
                y_star = _simulate_from_fit(base, df, stratum_col, rng)
                df_star = df.copy()
                df_star[response_var] = y_star
                fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
                boots[b, :] = fit_b.fe_params.values
            except Exception:
                boots[b, :] = np.nan
                fail += 1
    else:
        raise ValueError("mode must be 'cluster' or 'parametric'")

    # Remove failed fits
    ok = ~np.isnan(boots).any(axis=1)
    boots = boots[ok]
    
    if boots.size == 0:
        raise RuntimeError(f"All bootstrap fits failed (fail={fail}). Try different optimizer or reduce model complexity.")
    
    B_success = boots.shape[0]
    failure_rate = fail / B
    
    if verbose:
        print(f"   Success rate: {B_success}/{B} ({100*B_success/B:.1f}%)")
    
    # Warning for high failure rate
    if failure_rate > 0.1:
        warnings.warn(f"High bootstrap failure rate: {100*failure_rate:.1f}%. Results may be unreliable.")

    # Calculate percentile CIs
    lo_pct = np.percentile(boots, 100*alpha/2, axis=0)
    hi_pct = np.percentile(boots, 100*(1-alpha/2), axis=0)
    mean_ = boots.mean(axis=0)
    se_ = boots.std(axis=0, ddof=1)

    # Calculate BCa CIs if requested
    if ci_method in ["bca", "both"]:
        if verbose:
            print(f"   Calculating BCa intervals (this may take a moment)...")
        # Note: Full jackknife would be expensive; using simplified version
        lo_bca, hi_bca = _calculate_bca_intervals(boots, theta_hat, None, alpha)
    
    # Get asymptotic CIs
    ci_all = base.conf_int()
    ci_fe = ci_all.loc[fe_names]
    pvals_fe = base.pvalues.loc[fe_names]

    # Build results dataframe
    out = pd.DataFrame({
        "coef": fe_names,
        "coef_hat": theta_hat,
        "boot_mean": mean_,
        "boot_se": se_,
        "boot_ci_lower": lo_pct,
        "boot_ci_upper": hi_pct,
    })
    
    if ci_method in ["bca", "both"]:
        out["bca_ci_lower"] = lo_bca
        out["bca_ci_upper"] = hi_bca
    
    out["asym_ci_lower"] = ci_fe[0].values
    out["asym_ci_upper"] = ci_fe[1].values
    out["pvalue_asym"] = pvals_fe.values
    out["B_used"] = B_success
    out["B_failed"] = fail
    
    # Calculate diagnostics
    diagnostics = _bootstrap_diagnostics(boots, fe_names, alpha)
    
    if verbose:
        non_normal = (~diagnostics['normal']).sum()
        if non_normal > 0:
            print(f"   ⚠️  Warning: {non_normal}/{k} parameters have non-normal bootstrap distributions")
    
    return out, base, diagnostics

def _as_scalar_random_intercept(val):
    if isinstance(val, (pd.Series, pd.DataFrame)):
        return float(val.iloc[0])
    arr = np.asarray(val)
    return float(arr.ravel()[0]) if arr.ndim > 0 else float(arr)


def bootstrap_random_intercepts_improved(
    df, response_var, fixed_effects, stratum_col,
    B=1500, alpha=0.05, mode="parametric", reml=True,
    optimizer="powell", disp=False, seed=None, 
    min_appearance_rate=0.5, ci_method="percentile", verbose=True
):
    """
    Improved bootstrap CIs for random intercepts with better handling of missing strata.
    
    New parameters:
    ---------------
    min_appearance_rate : float, default=0.5
        Minimum proportion of bootstrap samples in which a stratum must appear
        to calculate a CI. Strata appearing less often will be flagged.
    ci_method : str, default="percentile"
        CI construction method ("percentile" or "bca")
    verbose : bool, default=True
        Print diagnostic information
        
    Returns:
    --------
    tuple: (results_df, base_model, diagnostics_df)
    """
    if verbose:
        print(f"\n🔄 BOOTSTRAPPING RANDOM INTERCEPTS")
        print(f"   Method: {mode}, CI method: {ci_method}")
        print(f"   Bootstrap samples: {B}")
        print(f"   Minimum appearance rate: {100*min_appearance_rate:.0f}%")
    
    rng = np.random.default_rng(seed)
    fixed_formula = " + ".join([f"C({f})" for f in fixed_effects])
    formula = f"{response_var} ~ {fixed_formula}"

    # Fit base model
    base = _fit_mixed(formula, df, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
    
    # Get original strata
    original_strata = pd.Index(df[stratum_col].unique()).sort_values()
    n_strata = len(original_strata)
    
    if verbose:
        print(f"   Number of strata: {n_strata}")
    
    # Initialize: B x n_strata, but track appearances separately
    boots = np.full((B, n_strata), np.nan, dtype=float)
    appearance_count = np.zeros(n_strata, dtype=int)
    fail = 0

    if verbose:
        print(f"   Running bootstrap...")
        
    for b in range(B):
        if verbose and (b + 1) % max(1, B // 10) == 0:
            print(f"   Progress: {b + 1}/{B} ({100*(b+1)/B:.1f}%)")
            
        try:
            if mode == "parametric":
                y_star = _simulate_from_fit(base, df, stratum_col, rng)
                df_star = df.copy()
                df_star[response_var] = y_star
                
            elif mode == "cluster":
                groups = df[stratum_col].unique()
                g_star = rng.choice(groups, size=len(groups), replace=True)
                df_star = pd.concat([df[df[stratum_col] == g] for g in g_star], ignore_index=True)
                
            else:
                raise ValueError("mode must be 'parametric' or 'cluster'")
            
            fit_b = _fit_mixed(formula, df_star, stratum_col, reml=reml, optimizer=optimizer, disp=disp)
            re_dict = fit_b.random_effects
            
            # IMPROVED: Track which strata actually appear
            for i, stratum in enumerate(original_strata):
                if stratum in re_dict:
                    boots[b, i] = re_dict[stratum][0] if hasattr(re_dict[stratum], '__len__') else re_dict[stratum]
                    appearance_count[i] += 1
                # else: leave as np.nan (not 0!)
                    
        except Exception as e:
            if verbose and fail == 0:
                print(f"   Warning: Bootstrap iteration {b+1} failed: {e}")
            fail += 1

    B_success = B - fail
    appearance_rate = appearance_count / B_success
    
    if verbose:
        print(f"   Bootstrap completed: {B_success}/{B} successful iterations")
        if fail > 0:
            print(f"   Failed iterations: {fail}")
        
        # Report strata with low appearance
        low_appearance = (appearance_rate < min_appearance_rate).sum()
        if low_appearance > 0:
            print(f"   ⚠️  Warning: {low_appearance}/{n_strata} strata appear in <{100*min_appearance_rate:.0f}% of samples")

    # Get original random effects
    base_re_dict = base.random_effects
    original_re = np.array([
        _as_scalar_random_intercept(base_re_dict[stratum])
        for stratum in original_strata
    ])


    # Calculate CIs only using valid bootstrap samples (where stratum appeared)
    boot_mean = np.zeros(n_strata)
    boot_se = np.zeros(n_strata)
    boot_ci_lower = np.zeros(n_strata)
    boot_ci_upper = np.zeros(n_strata)
    sufficient_data = np.zeros(n_strata, dtype=bool)
    
    for i in range(n_strata):
        valid_samples = boots[:, i][~np.isnan(boots[:, i])]
        
        if len(valid_samples) >= B_success * min_appearance_rate:
            sufficient_data[i] = True
            boot_mean[i] = valid_samples.mean()
            boot_se[i] = valid_samples.std(ddof=1)
            boot_ci_lower[i] = np.percentile(valid_samples, 100 * alpha / 2)
            boot_ci_upper[i] = np.percentile(valid_samples, 100 * (1 - alpha / 2))
        else:
            # Insufficient data for reliable CI
            sufficient_data[i] = False
            boot_mean[i] = np.nan
            boot_se[i] = np.nan
            boot_ci_lower[i] = np.nan
            boot_ci_upper[i] = np.nan

    # Build results
    results_df = pd.DataFrame({
        stratum_col: original_strata,
        "random_intercept": original_re,
        "boot_mean": boot_mean,
        "boot_se": boot_se,
        "boot_ci_lower": boot_ci_lower,
        "boot_ci_upper": boot_ci_upper,
        "appearance_rate": appearance_rate,
        "appearance_count": appearance_count,
        "sufficient_data": sufficient_data,
        "significant": sufficient_data & ~((boot_ci_lower <= 0) & (boot_ci_upper >= 0)),
        "B_used": B_success,
        "B_failed": fail
    })
    
    # Sort by absolute magnitude
    results_df = results_df.reindex(results_df['random_intercept'].abs().sort_values(ascending=False).index)
    
    # Create diagnostics for strata with sufficient data
    strata_with_data = results_df[results_df['sufficient_data']]
    if len(strata_with_data) > 0:
        # Sample diagnostics for top strata
        top_strata_indices = results_df[results_df['sufficient_data']].head(10).index
        sample_boots = np.array([
            boots[:, original_strata.get_loc(results_df.loc[idx, stratum_col])] 
            for idx in top_strata_indices
        ]).T
        sample_boots = sample_boots[~np.isnan(sample_boots).any(axis=1)]
        
        diagnostics = _bootstrap_diagnostics(
            sample_boots, 
            [f"RE_{results_df.loc[idx, stratum_col][:20]}" for idx in top_strata_indices],
            alpha
        )
    else:
        diagnostics = pd.DataFrame()
    
    if verbose:
        n_sufficient = sufficient_data.sum()
        n_significant = results_df['significant'].sum()
        print(f"   ✓ Results: {n_sufficient}/{n_strata} strata have sufficient bootstrap data")
        print(f"   ✓ Significant: {n_significant}/{n_sufficient} strata with CIs excluding zero")
        
        # Show top 5
        print(f"\n   Top 5 largest (absolute) random intercepts:")
        top5 = results_df[results_df['sufficient_data']].head(5)
        for _, row in top5.iterrows():
            sig_mark = "***" if row['significant'] else "   "
            app_rate = f"{100*row['appearance_rate']:.0f}%"
            print(f"   {sig_mark} {str(row[stratum_col])[:30]:30} {row['random_intercept']:8.4f} [{row['boot_ci_lower']:7.4f}, {row['boot_ci_upper']:7.4f}] (appears {app_rate})")

    return results_df, base, diagnostics


def compare_ci_methods(bootstrap_results, param_name="coef"):
    """
    Compare different CI methods visually.
    
    Parameters:
    -----------
    bootstrap_results : pd.DataFrame
        Results from bootstrap_fixed_effects_improved with multiple CI methods
    param_name : str
        Column name for parameter labels
        
    Returns:
    --------
    pd.DataFrame with CI comparison
    """
    if "bca_ci_lower" not in bootstrap_results.columns:
        print("BCa CIs not available. Run bootstrap with ci_method='both'")
        return None
    
    comparison = pd.DataFrame({
        'parameter': bootstrap_results[param_name],
        'estimate': bootstrap_results['coef_hat'],
        'percentile_lower': bootstrap_results['boot_ci_lower'],
        'percentile_upper': bootstrap_results['boot_ci_upper'],
        'bca_lower': bootstrap_results['bca_ci_lower'],
        'bca_upper': bootstrap_results['bca_ci_upper'],
        'asymptotic_lower': bootstrap_results['asym_ci_lower'],
        'asymptotic_upper': bootstrap_results['asym_ci_upper'],
    })
    
    # Calculate widths
    comparison['percentile_width'] = comparison['percentile_upper'] - comparison['percentile_lower']
    comparison['bca_width'] = comparison['bca_upper'] - comparison['bca_lower']
    comparison['asymptotic_width'] = comparison['asymptotic_upper'] - comparison['asymptotic_lower']
    
    # Check agreement on significance
    comparison['percentile_sig'] = ~((comparison['percentile_lower'] <= 0) & (comparison['percentile_upper'] >= 0))
    comparison['bca_sig'] = ~((comparison['bca_lower'] <= 0) & (comparison['bca_upper'] >= 0))
    comparison['asymptotic_sig'] = ~((comparison['asymptotic_lower'] <= 0) & (comparison['asymptotic_upper'] >= 0))
    comparison['all_agree'] = (comparison['percentile_sig'] == comparison['bca_sig']) & (comparison['bca_sig'] == comparison['asymptotic_sig'])
    
    return comparison




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

def compute_scm_scores_long(
    df,
    response_col="response",
    concept_col="concept",
    keep_cols=None,
    use_absolute=False,
    keep_only_measuring=True,  # drop items that do not measure a dimension (polarity==0)
):
    """
    Compute SCM-polarity-coded item values and KEEP THEM at the item level.

    Output is LONG: one row per (groupby_cols), typically (stratum x concept).
    For each row you get:
      - mean_response (optionally abs)
      - warmth_polarity, competence_polarity
      - warmth_item   = warmth_polarity * mean_response
      - competence_item = competence_polarity * mean_response
      - n_obs (how many raw rows contributed)

    This is the right representation if you want to compare "valence-only" vs "SCM-structured"
    models on the SAME data level (stratum × item), e.g. ~ 175*72 rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw individual responses (e.g., 72 items × 100 repeats × stratum)
    response_col : str
        Column with raw response values (-5..+5)
    concept_col : str
        Column with item/category code: {'ADM','PIT','ENV','CON','AF','AH','PF','PH'}
    groupby_cols : tuple/list of str
        Columns defining the item-level aggregation. Default keeps both stratum and concept.
    keep_cols : list[str] or None
        Additional columns to keep (assumed constant within the group); aggregated with 'first'
    use_absolute : bool
        If True, uses abs(response) before polarity coding
    keep_only_measuring : bool
        If True, drops rows where polarity==0 for a given dimension by setting those item-values to NaN
        (so you can later filter per dimension cleanly). Keeps the row, but those fields are NaN.

    Returns
    -------
    pd.DataFrame
        Long dataframe with item-level SCM-coded values.
    """

    import numpy as np
    import pandas as pd

    # --- Polarity mappings ---
    warmth_polarity = {
        "ADM": +1,  # high warmth
        "PIT": +1,  # high warmth
        "ENV": -1,  # low warmth
        "CON": -1,  # low warmth
        "AF": +1,   # warmth behavior (facilitation)
        "AH": -1,   # warmth behavior (harm)
        "PF": 0,    # not warmth
        "PH": 0,    # not warmth
    }

    competence_polarity = {
        "ADM": +1,  # high competence
        "PIT": -1,  # low competence
        "ENV": +1,  # high competence
        "CON": -1,  # low competence
        "AF": 0,    # not competence
        "AH": 0,    # not competence
        "PF": +1,   # competence behavior (facilitation)
        "PH": -1,   # competence behavior (harm)
    }

    out = df.copy()

    # IMPORTANT: use your *concept* column (AF/PH/...), not item_id ("item_1"...)
    out["warmth_polarity"] = out[concept_col].map(warmth_polarity)
    out["competence_polarity"] = out[concept_col].map(competence_polarity)

    if out["warmth_polarity"].isna().any():
        bad = out.loc[out["warmth_polarity"].isna(), concept_col].unique()
        raise ValueError(f"Unmapped concepts in {concept_col}: {bad}")

    resp = out[response_col].abs() if use_absolute else out[response_col]

    out["warmth_item"] = out["warmth_polarity"] * resp
    out["competence_item"] = out["competence_polarity"] * resp

    out["measures_warmth"] = out["warmth_polarity"] != 0
    out["measures_competence"] = out["competence_polarity"] != 0

    return out






import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, normaltest, jarque_bera, kstest
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

import matplotlib.pyplot as plt
import statsmodels.api as sm

class BaseModelDiagnostics:
    def __init__(self, model, df=None, response_var=None):
        self.model = model
        self.df = df
        self.response_var = response_var
        self.results = {}

    def check_residual_normality(self, plot=True):
        residuals = self.model.resid
        tests = {}

        # Shapiro
        if len(residuals) < 5000:
            stat, p = shapiro(residuals)
            tests["Shapiro-Wilk"] = {"stat": stat, "p": p}

        # Jarque-Bera
        jb_stat, jb_p = jarque_bera(residuals)
        skew = stats.skew(residuals)
        kurt = stats.kurtosis(residuals, fisher=False)
        tests["Jarque-Bera"] = {
            "stat": jb_stat, "p": jb_p,
            "skew": skew, "kurtosis": kurt
        }

        # D’Agostino-Pearson (guard for very small n)
        if len(residuals) >= 8:
            dp_stat, dp_p = normaltest(residuals)
            tests["D'Agostino-Pearson"] = {"stat": dp_stat, "p": dp_p}

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Histogram
            axes[0].hist(residuals, bins=30, density=True)
            axes[0].set_title("Residuals histogram")

            # QQ-plot
            sm.qqplot(residuals, line="45", ax=axes[1])
            axes[1].set_title("Residuals QQ-plot")

            plt.tight_layout()

        self.results["residual_normality"] = tests
        return tests


class OLSDiagnostics:
    """
    Diagnostics for OLS models, tuned for aggregated data (e.g. vignette means).

    - Uses *standardized* residuals for normality checks / QQ plot
    - Adds heteroskedasticity tests (Breusch–Pagan, White)
    - Adds leverage / influence summary (hat values, Cook's distance)
    """

    def __init__(self, model, df=None, response_var=None, model_name=None):
        self.model = model
        self.df = df
        self.response_var = response_var
        self.model_name = model_name
        self.results = {}

    # ---------- Helpers ----------

    @property
    def resid(self):
        r = self.model.resid
        return np.asarray(r)

    @property
    def resid_std(self):
        r = self.resid
        r = r - np.mean(r)
        sd = np.std(r, ddof=1)
        if sd == 0:
            return r * 0.0
        return r / sd

    # ---------- 1. Normality / shape ----------

    def check_residual_normality(self, plot=True):
        r = self.resid_std  # standardized residuals

        tests = {}

        # Shapiro-Wilk only for small-ish n
        if len(r) < 5000:
            stat, p = stats.shapiro(r)
            tests["Shapiro-Wilk"] = {"stat": float(stat), "p": float(p)}

        # Jarque–Bera
        jb_stat, jb_p = stats.jarque_bera(r)
        skew = stats.skew(r)
        # Fisher=False => normal has kurtosis=3, but here std residuals ~ N(0,1), so kurtosis ~3
        kurt = stats.kurtosis(r, fisher=False)
        tests["Jarque-Bera"] = {
            "stat": float(jb_stat),
            "p": float(jb_p),
            "skew": float(skew),
            "kurtosis": float(kurt),
        }

        # D’Agostino-Pearson (needs n>=8)
        if len(r) >= 8:
            dp_stat, dp_p = stats.normaltest(r)
            tests["D'Agostino-Pearson"] = {
                "stat": float(dp_stat),
                "p": float(dp_p),
            }

        if plot:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Histogram of standardized residuals
            axes[0].hist(r, bins=20, density=True)
            axes[0].set_title("Standardized residuals")
            axes[0].set_xlabel("Residual")
            axes[0].set_ylabel("Density")

            # QQ plot of standardized residuals
            sm.qqplot(r, line="45", ax=axes[1])
            axes[1].set_title("QQ-plot (standardized residuals)")

            fig.suptitle(self.model_name or "OLS diagnostics", y=1.05)
            plt.tight_layout()

        self.results["residual_normality"] = tests
        return tests

    # ---------- 2. Heteroskedasticity ----------

    def check_heteroskedasticity(self, plot=True):
        r = self.resid
        exog = self.model.model.exog

        bp_stat, bp_p, _, _ = het_breuschpagan(r, exog)
        white_stat, white_p, _, _ = het_white(r, exog)

        tests = {
            "Breusch-Pagan": {
                "stat": float(bp_stat),
                "p": float(bp_p),
            },
            "White": {
                "stat": float(white_stat),
                "p": float(white_p),
            },
        }

        if plot:
            fitted = self.model.fittedvalues
            plt.figure(figsize=(5, 4))
            plt.scatter(fitted, r, alpha=0.7)
            plt.axhline(0, color="black", lw=1)
            plt.xlabel("Fitted values")
            plt.ylabel("Residuals")
            plt.title("Residuals vs. fitted")
            plt.tight_layout()

        self.results["heteroskedasticity"] = tests
        return tests

    # ---------- 3. Leverage / influence ----------

    def check_influence(self, top_n=10, plot=True):
        infl = self.model.get_influence()
        hat = infl.hat_matrix_diag
        cooks = infl.cooks_distance[0]
        stud_resid = infl.resid_studentized_external

        n = len(hat)
        p = self.model.df_model + 1  # incl. intercept

        # Simple rules of thumb
        lev_thresh = 2 * p / n
        cook_thresh = 4 / n

        high_lev = np.where(hat > lev_thresh)[0]
        high_cook = np.where(cooks > cook_thresh)[0]

        summary = {
            "n": int(n),
            "p": int(p),
            "leverage_threshold": float(lev_thresh),
            "cook_threshold": float(cook_thresh),
            "max_leverage": float(hat.max()),
            "max_cooks": float(cooks.max()),
            "n_high_leverage": int(len(high_lev)),
            "n_high_cooks": int(len(high_cook)),
            "top_points": [
                {
                    "index": int(i),
                    "leverage": float(hat[i]),
                    "cooks": float(cooks[i]),
                    "stud_resid": float(stud_resid[i]),
                }
                for i in np.argsort(cooks)[-top_n:][::-1]
            ],
        }

        if plot:
            plt.figure(figsize=(5, 4))
            plt.scatter(hat, stud_resid, alpha=0.7)
            plt.xlabel("Leverage (hat values)")
            plt.ylabel("Studentized residuals")
            plt.title("Influence: leverage vs. studentized residuals")
            plt.tight_layout()

        self.results["influence"] = summary
        return summary

    # ---------- 4. Master report ----------

    def report(self, plot=False):
        """
        Run all diagnostics and return a nested dict.
        `plot=True` will produce the key figures.
        """
        self.results = {}
        self.check_residual_normality(plot=plot)
        self.check_heteroskedasticity(plot=plot)
        self.check_influence(plot=plot)
        return self.results


class MAIHDADiagnostics(BaseModelDiagnostics):
    """
    Diagnostics for multilevel MAIHDA models.
    """

    def calculate_icc(self):
        re_var = self.model.cov_re.iloc[0, 0]
        res_var = self.model.scale
        icc = re_var / (re_var + res_var)

        self.results["icc"] = {
            "icc": icc,
            "re_var": re_var,
            "res_var": res_var
        }
        return icc

    def assess_variance_components(self):
        re_var = self.model.cov_re.iloc[0, 0]
        re_sd = np.sqrt(re_var)
        res_var = self.model.scale

        self.results["variance_components"] = {
            "re_var": re_var,
            "re_sd": re_sd,
            "res_var": res_var
        }
        return self.results["variance_components"]

    def check_convergence_warnings(self):
        if hasattr(self.model, "converged"):
            status = "CONVERGED" if self.model.converged else "FAILED"
        else:
            status = "UNKNOWN"

        self.results["convergence"] = status
        return status

    def report(self):
        print("\n=== MAIHDA DIAGNOSTICS REPORT ===")

        # Residual normality
        norm = self.check_residual_normality(plot=True)

        print("\nResidual normality tests:")
        for test_name, vals in norm.items():
            p = vals["p"]
            print(f"  {test_name:20} p = {p:.4f}")

        # ICC
        icc_res = self.calculate_icc()
        icc = icc_res["icc"] if isinstance(icc_res, dict) else icc_res

        print(f"\nICC (ρ): {float(icc):.3f}")
        if icc < 0.05:
            print("Interpretation: negligible between-stratum clustering.")
        elif icc < 0.10:
            print("Interpretation: small clustering.")
        elif icc < 0.20:
            print("Interpretation: moderate clustering.")
        else:
            print("Interpretation: strong clustering — MAIHDA is justified.")

        # Variance components
        vc = self.assess_variance_components()
        print("\nVariance components:")
        print(f"  RE variance (τ²): {vc['re_var']:.6f}")
        print(f"  Residual variance (σ²): {vc['res_var']:.6f}")
        print(f"  RE SD: {vc['re_sd']:.4f}")

        # Convergence
        conv = self.check_convergence_warnings()
        print(f"\nConvergence status: {conv}")

        self.plot_random_effects()
        return self.results
    
    def plot_random_effects(self):
        re = np.array([
            float(v[0]) if hasattr(v, '__iter__') else float(v)
            for v in self.model.random_effects.values()
        ])

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        # Histogram
        axes[0].hist(re, bins=30, edgecolor='black', alpha=0.7)
        axes[0].set_title("Random Effects Distribution")
        axes[0].set_xlabel("Random intercept estimate")
        axes[0].set_ylabel("Count")

        # Q-Q plot of random effects
        stats.probplot(re, plot=axes[1])
        axes[1].set_title("Q-Q Plot of Random Effects")

        plt.tight_layout()
        plt.show()


def extract_variance_components(model):
    """Extract tau^2 and omega^2 from a MixedLMResultsWrapper."""
    tau2 = float(model.cov_re.iloc[0,0])
    omega2 = float(model.scale)
    icc = tau2 / (tau2 + omega2)
    return tau2, omega2, icc

    # optional: dict of null models for PCV
    null_models = {
        # "comp_llama_maihda": model_c_llama_maihda_null,
        # "warmth_llama_maihda": model_w_llama_maihda_null,
        # ... fill in once you have them
    }

    rows = []

    for name, (model, df, yvar, modeltype) in all_models.items():
        if modeltype != "mai":
            continue  # skip OLS
        
        tau2, omega2, icc = extract_variance_components(model)
        vpc = icc

        # compute PCV only if null model exists
        if name in null_models:
            tau2_null, _, _ = extract_variance_components(null_models[name])
            pcv = 100 * (tau2_null - tau2) / tau2_null
        else:
            pcv = None
        
        rows.append({
            "Model": name,
            "ICC": icc,
            "tau2": tau2,
            "omega2": omega2,
            "VPC": vpc,
            "PCV": pcv
        })

    df_results = pd.DataFrame(rows)

def compare_ols_and_maihda_diagnostics(ols_model, maihda_model, df_ols, df_maihda,
                                       stratum_col, response_var,
                                       model_name="",
                                       save_plots=True, output_dir='.'):
    """
    Run diagnostics on both OLS and MAIHDA models and compare.
    
    Parameters:
    -----------
    ols_model : fitted OLS model
    maihda_model : fitted MixedLM model
    df_ols : pd.DataFrame for OLS model
    df_maihda : pd.DataFrame for MAIHDA model
    stratum_col : str
    response_var : str
    model_name : str, identifier for this comparison
    save_plots : bool
    output_dir : str
    """
    print("\n" + "🔬"*35)
    print(f"COMPARATIVE DIAGNOSTICS: {model_name}")
    print("🔬"*35)
    
    # OLS diagnostics
    print("\n" + "="*70)
    print("PART 1: OLS MODEL DIAGNOSTICS")
    print("="*70)
    ols_diag = OLSDiagnostics(ols_model, df_ols, response_var)
    ols_results = ols_diag.report(
        save_plots=save_plots, 
        output_dir=f"{output_dir}/ols_{model_name}"
    )
    
    # MAIHDA diagnostics
    print("\n" + "="*70)
    print("PART 2: MAIHDA MODEL DIAGNOSTICS")
    print("="*70)
    maihda_diag = MAIHDADiagnostics(maihda_model, df_maihda, stratum_col, response_var)
    maihda_results = maihda_diag.report(
        save_plots=save_plots,
        output_dir=f"{output_dir}/maihda_{model_name}"
    )
    
    # Comparison
    print("\n" + "="*70)
    print("PART 3: MODEL COMPARISON")
    print("="*70)
    
    print("\nModel Fit Comparison:")
    print(f"{'Metric':<25} {'OLS':>15} {'MAIHDA':>15} {'Difference':>15}")
    print("-"*70)
    
    if hasattr(ols_model, 'aic') and hasattr(maihda_model, 'aic'):
        aic_diff = ols_model.aic - maihda_model.aic
        better_aic = "MAIHDA" if aic_diff > 0 else "OLS"
        print(f"{'AIC':<25} {ols_model.aic:>15.2f} {maihda_model.aic:>15.2f} {aic_diff:>15.2f} ← {better_aic}")
    
    if hasattr(ols_model, 'bic') and hasattr(maihda_model, 'bic'):
        bic_diff = ols_model.bic - maihda_model.bic
        better_bic = "MAIHDA" if bic_diff > 0 else "OLS"
        print(f"{'BIC':<25} {ols_model.bic:>15.2f} {maihda_model.bic:>15.2f} {bic_diff:>15.2f} ← {better_bic}")
    
    if hasattr(ols_model, 'llf') and hasattr(maihda_model, 'llf'):
        ll_diff = maihda_model.llf - ols_model.llf
        print(f"{'Log-Likelihood':<25} {ols_model.llf:>15.2f} {maihda_model.llf:>15.2f} {ll_diff:>15.2f}")
    
    # R-squared comparison (if available)
    if hasattr(ols_model, 'rsquared'):
        print(f"{'OLS R-squared':<25} {ols_model.rsquared:>15.4f}")
    
    # Final recommendation
    icc = maihda_results.get('icc', {}).get('icc', None)
    
    print("\n" + "="*70)
    print("🎯 FINAL RECOMMENDATION")
    print("="*70)
    
    if icc is not None and icc < 0.05:
        print(f"""
CHOOSE: OLS MODEL
Rationale:
- ICC = {icc:.1%} (negligible clustering)
- MAIHDA adds complexity without substantial benefit
- Parsimony principle favors simpler model
- OLS coefficients are more interpretable
Action:
→ Report OLS model as primary analysis
→ Mention MAIHDA was tested but not needed
→ Focus interpretation on main effects
        """)
        recommendation = "OLS"
    elif icc is not None and icc < 0.10:
        print(f"""
CHOOSE: Report both, but emphasize OLS
Rationale:
- ICC = {icc:.1%} (modest clustering)
- Both models likely give similar conclusions
- Check if fixed effects estimates differ substantially
Action:
→ Report both models in supplementary materials
→ Discuss why OLS is preferred (parsimony)
→ Note that conclusions are robust to model choice
        """)
        recommendation = "BOTH"
    else:
        icc_display = f"{icc:.1%}" if icc is not None else "N/A"
        print(f"""
CHOOSE: MAIHDA MODEL
Rationale:
- ICC = {icc_display} (substantial clustering)
- Ignoring clustering biases standard errors
- MAIHDA appropriately models hierarchical structure
Action:
→ Report MAIHDA as primary analysis
→ Interpret both fixed effects AND random effects
→ Discuss intersectional heterogeneity substantively
        """)
        recommendation = "MAIHDA"
    
    print("="*70)
    
    return {
        'ols_diagnostics': ols_results,
        'maihda_diagnostics': maihda_results,
        'recommendation': recommendation,
        'icc': icc,
        'aic_ols': getattr(ols_model, 'aic', None),
        'aic_maihda': getattr(maihda_model, 'aic', None),
        'bic_ols': getattr(ols_model, 'bic', None),
        'bic_maihda': getattr(maihda_model, 'bic', None)
    }

def run_maihda_per_concept(
    df,
    response_var="response_recoded",
    fixed_effects=("race", "gender", "religion", "transness"),
    stratum_col="stratum",
    concept_col="concept",
    out_dir="maihda_per_concept",
    reml=True,
    optimizer="powell",
    disp=False,
    save_excel=True,
    make_plots=True,
    verbose=True
):
    """
    Run separate MAIHDA models for each concept in the dataset.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe with MAIHDA data
    response_var : str, default="response_recoded"
        Name of response variable
    fixed_effects : tuple, default=("race", "gender", "religion", "transness")
        Fixed effects variables (concept will be excluded from these)
    stratum_col : str, default="stratum"
        Column name for intersectional strata
    concept_col : str, default="concept"
        Column name for concepts
    out_dir : str, default="maihda_per_concept"
        Output directory for results
    reml : bool, default=True
        Use REML estimation
    optimizer : str, default="powell"
        Optimization method
    disp : bool, default=False
        Display convergence information
    save_excel : bool, default=True
        Save results to Excel files
    make_plots : bool, default=True
        Create visualization plots
    verbose : bool, default=True
        Print progress information
        
    Returns:
    --------
    dict: Results for each concept containing MAIHDA model outputs
    """
    import os
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    if verbose:
        print("\n" + "🔬"*35)
        print("MAIHDA ANALYSIS PER CONCEPT")
        print("🔬"*35)
    
    # Get unique concepts
    concepts = df[concept_col].unique()
    n_concepts = len(concepts)
    
    if verbose:
        print(f"\nDataset overview:")
        print(f"  Total observations: {len(df):,}")
        print(f"  Number of concepts: {n_concepts}")
        print(f"  Concepts: {list(concepts)}")
        print(f"  Strata: {df[stratum_col].nunique()}")
    
    # Store results for each concept
    concept_results = {}
    summary_data = []
    
    # Run MAIHDA for each concept
    for i, concept in enumerate(concepts, 1):
        if verbose:
            print(f"\n" + "="*70)
            print(f"CONCEPT {i}/{n_concepts}: {concept}")
            print("="*70)
        
        # Filter data for this concept
        df_concept = df[df[concept_col] == concept].copy()
        n_obs_concept = len(df_concept)
        n_strata_concept = df_concept[stratum_col].nunique()
        
        if verbose:
            print(f"  Observations: {n_obs_concept:,}")
            print(f"  Strata: {n_strata_concept}")
        
        # Check minimum requirements
        if n_obs_concept < 50:
            if verbose:
                print(f"  ⚠️  Skipping {concept}: Too few observations ({n_obs_concept})")
            concept_results[concept] = {"status": "skipped", "reason": "insufficient_observations"}
            continue
            
        if n_strata_concept < 5:
            if verbose:
                print(f"  ⚠️  Skipping {concept}: Too few strata ({n_strata_concept})")
            concept_results[concept] = {"status": "skipped", "reason": "insufficient_strata"}
            continue
        
        try:
            # Create concept-specific output directory
            concept_out_dir = os.path.join(out_dir, f"concept_{concept}".replace("/", "_").replace(" ", "_"))
            
            # Run MAIHDA for this concept (excluding concept from fixed effects)
            concept_fixed_effects = [fe for fe in fixed_effects if fe != concept_col]
            
            if verbose:
                print(f"  Running MAIHDA with fixed effects: {concept_fixed_effects}")
            
            maihda_results = run_maihda_simple(
                df=df_concept,
                response_var=response_var,
                fixed_effects=tuple(concept_fixed_effects),  # No concept in fixed effects
                stratum_col=stratum_col,
                out_dir=concept_out_dir,
                reml=reml,
                optimizer=optimizer,
                disp=disp,
                save_excel=save_excel,
                make_plots=make_plots
            )
            
            # Extract key metrics from results
            if isinstance(maihda_results.get("summary"), pd.DataFrame) and len(maihda_results["summary"]) > 0:
                model_1B_summary = maihda_results["summary"].iloc[1] if len(maihda_results["summary"]) > 1 else maihda_results["summary"].iloc[0]
                vpc_1B = model_1B_summary.get("VPC", np.nan)
                pcv = model_1B_summary.get("PCV_%", np.nan)
            else:
                vpc_1B = np.nan
                pcv = np.nan
            
            # Count significant random effects
            if "random_effects" in maihda_results and not maihda_results["random_effects"].empty:
                n_significant_re = maihda_results["random_effects"]["significant"].sum()
                total_re = len(maihda_results["random_effects"])
            else:
                n_significant_re = 0
                total_re = 0
            
            # Store results
            concept_results[concept] = {
                "status": "success",
                "maihda_results": maihda_results,
                "n_observations": n_obs_concept,
                "n_strata": n_strata_concept,
                "vpc": vpc_1B,
                "pcv": pcv,
                "n_significant_re": n_significant_re,
                "total_re": total_re,
                "output_dir": concept_out_dir
            }
            
            # Add to summary
            summary_data.append({
                "concept": concept,
                "n_observations": n_obs_concept,
                "n_strata": n_strata_concept,
                "vpc": vpc_1B,
                "pcv": pcv,
                "n_significant_re": n_significant_re,
                "total_re": total_re,
                "prop_significant_re": n_significant_re / total_re if total_re > 0 else 0,
                "status": "success"
            })
            
            if verbose:
                print(f"  ✅ MAIHDA completed successfully")
                print(f"     VPC: {vpc_1B:.3f}" if not np.isnan(vpc_1B) else "     VPC: N/A")
                print(f"     Significant RE: {n_significant_re}/{total_re}")
                
        except Exception as e:
            if verbose:
                print(f"  ❌ MAIHDA failed: {str(e)}")
            
            concept_results[concept] = {
                "status": "failed", 
                "error": str(e),
                "n_observations": n_obs_concept,
                "n_strata": n_strata_concept
            }
            
            summary_data.append({
                "concept": concept,
                "n_observations": n_obs_concept,
                "n_strata": n_strata_concept,
                "vpc": np.nan,
                "pcv": np.nan,
                "n_significant_re": np.nan,
                "total_re": np.nan,
                "prop_significant_re": np.nan,
                "status": "failed"
            })
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Generate comparative analysis
    if verbose:
        print(f"\n" + "="*70)
        print("COMPARATIVE SUMMARY ACROSS CONCEPTS")
        print("="*70)
    
    successful_concepts = summary_df[summary_df["status"] == "success"]
    
    if len(successful_concepts) > 0:
        if verbose:
            print(f"\nSuccessful models: {len(successful_concepts)}/{len(summary_df)}")
            print(f"\nVPC (Variance Partition Coefficient) by concept:")
            vpc_sorted = successful_concepts.sort_values("vpc", ascending=False)
            for _, row in vpc_sorted.iterrows():
                vpc_val = row["vpc"]
                if not np.isnan(vpc_val):
                    print(f"  {row['concept']:15} VPC = {vpc_val:.3f} ({vpc_val*100:.1f}%)")
                else:
                    print(f"  {row['concept']:15} VPC = N/A")
            
            print(f"\nSignificant random effects by concept:")
            re_sorted = successful_concepts.sort_values("prop_significant_re", ascending=False)
            for _, row in re_sorted.iterrows():
                prop = row["prop_significant_re"]
                if not np.isnan(prop):
                    print(f"  {row['concept']:15} {row['n_significant_re']}/{row['total_re']} ({prop*100:.1f}%)")
        
        # Create comparison plots
        if make_plots and len(successful_concepts) > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # VPC comparison
            ax = axes[0, 0]
            vpc_data = successful_concepts.dropna(subset=["vpc"])
            if len(vpc_data) > 0:
                ax.bar(range(len(vpc_data)), vpc_data["vpc"], color='steelblue', alpha=0.7)
                ax.set_xticks(range(len(vpc_data)))
                ax.set_xticklabels(vpc_data["concept"], rotation=45, ha='right')
                ax.set_ylabel('VPC')
                ax.set_title('Variance Partition Coefficient by Concept')
                ax.grid(True, alpha=0.3)
            
            # Number of observations
            ax = axes[0, 1]
            ax.bar(range(len(successful_concepts)), successful_concepts["n_observations"], color='orange', alpha=0.7)
            ax.set_xticks(range(len(successful_concepts)))
            ax.set_xticklabels(successful_concepts["concept"], rotation=45, ha='right')
            ax.set_ylabel('Number of Observations')
            ax.set_title('Sample Size by Concept')
            ax.grid(True, alpha=0.3)
            
            # Proportion of significant random effects
            ax = axes[1, 0]
            prop_data = successful_concepts.dropna(subset=["prop_significant_re"])
            if len(prop_data) > 0:
                ax.bar(range(len(prop_data)), prop_data["prop_significant_re"], color='green', alpha=0.7)
                ax.set_xticks(range(len(prop_data)))
                ax.set_xticklabels(prop_data["concept"], rotation=45, ha='right')
                ax.set_ylabel('Proportion Significant RE')
                ax.set_title('Proportion of Significant Random Effects')
                ax.grid(True, alpha=0.3)
            
            # Number of strata
            ax = axes[1, 1]
            ax.bar(range(len(successful_concepts)), successful_concepts["n_strata"], color='purple', alpha=0.7)
            ax.set_xticks(range(len(successful_concepts)))
            ax.set_xticklabels(successful_concepts["concept"], rotation=45, ha='right')
            ax.set_ylabel('Number of Strata')
            ax.set_title('Number of Strata by Concept')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle('MAIHDA Results Comparison Across Concepts', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            comparison_plot_path = os.path.join(out_dir, "concept_comparison.png")
            plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            if verbose:
                print(f"\n📊 Comparison plot saved: {comparison_plot_path}")
    
    # Save summary to Excel
    if save_excel:
        summary_excel_path = os.path.join(out_dir, "concept_summary.xlsx")
        
        with pd.ExcelWriter(summary_excel_path, engine='xlsxwriter') as writer:
            # Summary sheet
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual concept sheets (key metrics only)
            for concept, results in concept_results.items():
                if results["status"] == "success" and "maihda_results" in results:
                    sheet_name = f"{concept}"[:31]  # Excel sheet name limit
                    
                    # Extract key results for this sheet
                    maihda_res = results["maihda_results"]
                    
                    # Model summary
                    if "summary" in maihda_res:
                        maihda_res["summary"].to_excel(writer, sheet_name=f"{sheet_name}_Summary", index=False)
                    
                    # Random effects (first 50 to avoid huge sheets)
                    if "random_effects" in maihda_res and not maihda_res["random_effects"].empty:
                        re_subset = maihda_res["random_effects"].head(50)
                        re_subset.to_excel(writer, sheet_name=f"{sheet_name}_RE", index=False)
        
        if verbose:
            print(f"\n💾 Summary Excel saved: {summary_excel_path}")
    
    # Final summary
    if verbose:
        print(f"\n" + "🎯"*35)
        print("ANALYSIS COMPLETE")
        print("🎯"*35)
        
        successful = len([r for r in concept_results.values() if r["status"] == "success"])
        failed = len([r for r in concept_results.values() if r["status"] == "failed"])
        skipped = len([r for r in concept_results.values() if r["status"] == "skipped"])
        
        print(f"\nResults:")
        print(f"  ✅ Successful: {successful}")
        print(f"  ❌ Failed: {failed}")
        print(f"  ⏭️  Skipped: {skipped}")
        print(f"  📁 Output directory: {out_dir}")
        
        if successful > 0:
            high_vpc_concepts = successful_concepts[successful_concepts["vpc"] > 0.1]["concept"].tolist()
            if high_vpc_concepts:
                print(f"\nConcepts with high intersectional effects (VPC > 10%):")
                for concept in high_vpc_concepts:
                    vpc_val = successful_concepts[successful_concepts["concept"] == concept]["vpc"].iloc[0]
                    print(f"  🔸 {concept}: VPC = {vpc_val:.3f}")
            
            low_vpc_concepts = successful_concepts[successful_concepts["vpc"] < 0.05]["concept"].tolist()
            if low_vpc_concepts:
                print(f"\nConcepts with minimal intersectional effects (VPC < 5%):")
                for concept in low_vpc_concepts:
                    vpc_val = successful_concepts[successful_concepts["concept"] == concept]["vpc"].iloc[0]
                    if not np.isnan(vpc_val):
                        print(f"  🔹 {concept}: VPC = {vpc_val:.3f}")
    
    return {
        "concept_results": concept_results,
        "summary": summary_df,
        "output_directory": out_dir,
        "successful_concepts": len([r for r in concept_results.values() if r["status"] == "success"]),
        "total_concepts": len(concepts)
    }




def export_llm_results_to_excel(
    results_per_concept,
    output_path,
    llm_name="LLM",
    include_global_model=None,
    verbose=True
):
    """
    Export all MAIHDA results for one LLM into a comprehensive Excel file.
    
    Parameters:
    -----------
    results_per_concept : dict
        Results from run_maihda_per_concept function
    output_path : str
        Path for the output Excel file
    llm_name : str
        Name of the LLM for labeling
    include_global_model : dict, optional
        Results from global MAIHDA model (e.g., res_llama) to include as additional column
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict: Summary of exported data
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if verbose:
        print(f"\n📊 EXPORTING {llm_name.upper()} RESULTS TO EXCEL")
        print("="*60)
        print(f"Output file: {output_path}")
    
    # Extract concept results
    concept_results = results_per_concept.get("concept_results", {})
    successful_concepts = [k for k, v in concept_results.items() if v.get("status") == "success"]
    
    if not successful_concepts:
        print("❌ No successful concept results to export")
        return None
    
    if verbose:
        print(f"Exporting results for {len(successful_concepts)} concepts:")
        print(f"  {', '.join(successful_concepts)}")
    
    # Initialize data containers
    fixed_effects_data = {}
    random_intercepts_data = {}
    expected_values_data = {}
    
    # Process each concept
    for concept in successful_concepts:
        concept_res = concept_results[concept]["maihda_results"]
        
        # 1. FIXED EFFECTS
        if "fixed_effects" in concept_res and not concept_res["fixed_effects"].empty:
            fe_df = concept_res["fixed_effects"].copy()
            # Use parameter names as index, coefficient as values
            fe_series = fe_df.set_index('parameter')['coefficient']
            fixed_effects_data[f"{concept}_coeff"] = fe_series
            fixed_effects_data[f"{concept}_pvalue"] = fe_df.set_index('parameter')['p_value']
            fixed_effects_data[f"{concept}_ci_lower"] = fe_df.set_index('parameter')['ci_lower']
            fixed_effects_data[f"{concept}_ci_upper"] = fe_df.set_index('parameter')['ci_upper']
        
        # 2. RANDOM INTERCEPTS
        if "random_effects" in concept_res and not concept_res["random_effects"].empty:
            re_df = concept_res["random_effects"].copy()
            # Use stratum as index, random intercept as values
            re_series = re_df.set_index('stratum')['random_intercept']
            random_intercepts_data[f"{concept}_intercept"] = re_series
            random_intercepts_data[f"{concept}_significant"] = re_df.set_index('stratum')['significant']
            if 'p_value' in re_df.columns:
                random_intercepts_data[f"{concept}_pvalue"] = re_df.set_index('stratum')['p_value']
        
        # 3. EXPECTED VALUES (stratum predictions)
        if "stratum_predictions" in concept_res and not concept_res["stratum_predictions"].empty:
            sp_df = concept_res["stratum_predictions"].copy()
            # Use stratum as index, expected mean as values
            exp_series = sp_df.set_index('stratum')['expected_mean']
            expected_values_data[f"{concept}_expected"] = exp_series
            expected_values_data[f"{concept}_observed"] = sp_df.set_index('stratum')['observed_mean']
            expected_values_data[f"{concept}_n_obs"] = sp_df.set_index('stratum')['n_obs']
            expected_values_data[f"{concept}_residual"] = sp_df.set_index('stratum')['residual']
    
    # Add global model results if provided
    if include_global_model:
        concept = "GLOBAL"
        
        # Fixed effects
        if "fixed_effects" in include_global_model and not include_global_model["fixed_effects"].empty:
            fe_df = include_global_model["fixed_effects"].copy()
            fe_series = fe_df.set_index('parameter')['coefficient']
            fixed_effects_data[f"{concept}_coeff"] = fe_series
            fixed_effects_data[f"{concept}_pvalue"] = fe_df.set_index('parameter')['p_value']
            fixed_effects_data[f"{concept}_ci_lower"] = fe_df.set_index('parameter')['ci_lower']
            fixed_effects_data[f"{concept}_ci_upper"] = fe_df.set_index('parameter')['ci_upper']
        
        # Random intercepts
        if "random_effects" in include_global_model and not include_global_model["random_effects"].empty:
            re_df = include_global_model["random_effects"].copy()
            re_series = re_df.set_index('stratum')['random_intercept']
            random_intercepts_data[f"{concept}_intercept"] = re_series
            random_intercepts_data[f"{concept}_significant"] = re_df.set_index('stratum')['significant']
            if 'p_value' in re_df.columns:
                random_intercepts_data[f"{concept}_pvalue"] = re_df.set_index('stratum')['p_value']
        
        # Expected values
        if "stratum_predictions" in include_global_model and not include_global_model["stratum_predictions"].empty:
            sp_df = include_global_model["stratum_predictions"].copy()
            exp_series = sp_df.set_index('stratum')['expected_mean']
            expected_values_data[f"{concept}_expected"] = exp_series
            expected_values_data[f"{concept}_observed"] = sp_df.set_index('stratum')['observed_mean']
            expected_values_data[f"{concept}_n_obs"] = sp_df.set_index('stratum')['n_obs']
            expected_values_data[f"{concept}_residual"] = sp_df.set_index('stratum')['residual']
    
    # Create DataFrames
    fixed_effects_df = pd.DataFrame(fixed_effects_data)
    random_intercepts_df = pd.DataFrame(random_intercepts_data)
    expected_values_df = pd.DataFrame(expected_values_data)
    
    if verbose:
        print(f"\nData dimensions:")
        print(f"  Fixed effects: {fixed_effects_df.shape}")
        print(f"  Random intercepts: {random_intercepts_df.shape}")
        print(f"  Expected values: {expected_values_df.shape}")
    
    # Create summary statistics
    summary_stats = []
    
    for concept in successful_concepts:
        concept_res = concept_results[concept]
        
        # Get VPC and other metrics
        maihda_res = concept_res["maihda_results"]
        if "summary" in maihda_res and not maihda_res["summary"].empty:
            summary_row = maihda_res["summary"].iloc[1] if len(maihda_res["summary"]) > 1 else maihda_res["summary"].iloc[0]
            vpc = summary_row.get("VPC", np.nan)
            pcv = summary_row.get("PCV_%", np.nan)
            icc = summary_row.get("ICC", np.nan)
        else:
            vpc = np.nan
            pcv = np.nan
            icc = np.nan
        
        summary_stats.append({
            "concept": concept,
            "n_observations": concept_res.get("n_observations", 0),
            "n_strata": concept_res.get("n_strata", 0),
            "vpc": vpc,
            "icc": icc,
            "pcv_percent": pcv,
            "n_significant_re": concept_res.get("n_significant_re", 0),
            "total_re": concept_res.get("total_re", 0),
            "prop_significant_re": concept_res.get("n_significant_re", 0) / max(concept_res.get("total_re", 1), 1)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Write to Excel
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D7E4BC',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '0.000'})
            pvalue_format = workbook.add_format({'num_format': '0.000', 'bg_color': '#FFF2CC'})
            
            # Sheet 1: Fixed Effects
            if not fixed_effects_df.empty:
                fixed_effects_df.to_excel(writer, sheet_name='Fixed_Effects', startrow=1)
                worksheet = writer.sheets['Fixed_Effects']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Fixed Effects by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(fixed_effects_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format p-value columns
                for col_num, col_name in enumerate(fixed_effects_df.columns):
                    if 'pvalue' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 30)  # Parameter names column
            
            # Sheet 2: Random Intercepts
            if not random_intercepts_df.empty:
                random_intercepts_df.to_excel(writer, sheet_name='Random_Intercepts', startrow=1)
                worksheet = writer.sheets['Random_Intercepts']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Random Intercepts by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(random_intercepts_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num, col_name in enumerate(random_intercepts_df.columns):
                    if 'pvalue' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                    elif 'significant' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12)
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 40)  # Stratum names column
            
            # Sheet 3: Expected Values
            if not expected_values_df.empty:
                expected_values_df.to_excel(writer, sheet_name='Expected_Values', startrow=1)
                worksheet = writer.sheets['Expected_Values']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Expected Values by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(expected_values_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num, col_name in enumerate(expected_values_df.columns):
                    if 'n_obs' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 10, 
                                           workbook.add_format({'num_format': '0'}))
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 40)  # Stratum names column
            
            # Sheet 4: Summary
            summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=1)
            worksheet = writer.sheets['Summary']
            
            # Add title
            worksheet.write(0, 0, f'{llm_name.upper()} - Summary Statistics by Concept', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
            
            # Format headers
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(1, col_num, value, header_format)
            
            # Auto-adjust column widths
            for col_num, col_name in enumerate(summary_df.columns):
                max_len = max(
                    len(str(col_name)),
                    summary_df[col_name].astype(str).str.len().max() if not summary_df.empty else 0
                )
                worksheet.set_column(col_num, col_num, min(max_len + 2, 20))
        
        if verbose:
            print(f"✅ Excel file exported successfully: {output_path}")
            print(f"\nSheets created:")
            print(f"  1. Fixed_Effects ({fixed_effects_df.shape[0]} parameters × {fixed_effects_df.shape[1]} columns)")
            print(f"  2. Random_Intercepts ({random_intercepts_df.shape[0]} strata × {random_intercepts_df.shape[1]} columns)")
            print(f"  3. Expected_Values ({expected_values_df.shape[0]} strata × {expected_values_df.shape[1]} columns)")
            print(f"  4. Summary ({summary_df.shape[0]} concepts × {summary_df.shape[1]} metrics)")
    
    except Exception as e:
        print(f"❌ Failed to export Excel file: {e}")
        return None
    
    return {
        "output_path": output_path,
        "fixed_effects_shape": fixed_effects_df.shape,
        "random_intercepts_shape": random_intercepts_df.shape,
        "expected_values_shape": expected_values_df.shape,
        "summary_shape": summary_df.shape,
        "concepts": successful_concepts
    }

def export_llm_results_to_excel(
    results_per_concept,
    output_path,
    llm_name="LLM",
    include_global_model=None,
    verbose=True
):
    """
    Export all MAIHDA results for one LLM into a comprehensive Excel file.
    
    Parameters:
    -----------
    results_per_concept : dict
        Results from run_maihda_per_concept function
    output_path : str
        Path for the output Excel file
    llm_name : str
        Name of the LLM for labeling
    include_global_model : dict, optional
        Results from global MAIHDA model (e.g., res_llama) to include as additional column
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict: Summary of exported data
    """
    import pandas as pd
    import numpy as np
    import os
    
    # Create output directory if needed - FIXED to handle files in current directory
    output_dir = os.path.dirname(output_path)
    if output_dir:  # Only create directory if there is one specified
        os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"\n📊 EXPORTING {llm_name.upper()} RESULTS TO EXCEL")
        print("="*60)
        print(f"Output file: {output_path}")
    
    # Extract concept results
    concept_results = results_per_concept.get("concept_results", {})
    successful_concepts = [k for k, v in concept_results.items() if v.get("status") == "success"]
    
    if not successful_concepts:
        print("❌ No successful concept results to export")
        return None
    
    if verbose:
        print(f"Exporting results for {len(successful_concepts)} concepts:")
        print(f"  {', '.join(successful_concepts)}")
    
    # Initialize data containers
    fixed_effects_data = {}
    random_intercepts_data = {}
    expected_values_data = {}
    
    # Process each concept
    for concept in successful_concepts:
        concept_res = concept_results[concept]["maihda_results"]
        
        # 1. FIXED EFFECTS - Check for different possible column names
        if "fixed_effects" in concept_res and not concept_res["fixed_effects"].empty:
            fe_df = concept_res["fixed_effects"].copy()
            
            # Determine the correct column names
            param_col = None
            coeff_col = None
            pval_col = None
            ci_lower_col = None
            ci_upper_col = None
            
            # Check for parameter column
            for col in ['parameter', 'param', 'variable', 'predictor']:
                if col in fe_df.columns:
                    param_col = col
                    break
            
            # Check for coefficient column
            for col in ['coefficient', 'coef', 'estimate', 'beta']:
                if col in fe_df.columns:
                    coeff_col = col
                    break
            
            # Check for p-value column
            for col in ['p_value', 'pvalue', 'p', 'p_val']:
                if col in fe_df.columns:
                    pval_col = col
                    break
            
            # Check for CI columns
            for col in ['ci_lower', 'conf_int_lower', 'lower_ci', 'ci_low']:
                if col in fe_df.columns:
                    ci_lower_col = col
                    break
            
            for col in ['ci_upper', 'conf_int_upper', 'upper_ci', 'ci_high']:
                if col in fe_df.columns:
                    ci_upper_col = col
                    break
            
            if verbose:
                print(f"  {concept} fixed effects columns detected:")
                print(f"    Parameter: {param_col}")
                print(f"    Coefficient: {coeff_col}")
                print(f"    P-value: {pval_col}")
                print(f"    CI Lower: {ci_lower_col}")
                print(f"    CI Upper: {ci_upper_col}")
            
            # Export data if we found the essential columns
            if param_col and coeff_col:
                try:
                    fe_series = fe_df.set_index(param_col)[coeff_col]
                    fixed_effects_data[f"{concept}_coeff"] = fe_series
                    
                    if pval_col:
                        fixed_effects_data[f"{concept}_pvalue"] = fe_df.set_index(param_col)[pval_col]
                    if ci_lower_col:
                        fixed_effects_data[f"{concept}_ci_lower"] = fe_df.set_index(param_col)[ci_lower_col]
                    if ci_upper_col:
                        fixed_effects_data[f"{concept}_ci_upper"] = fe_df.set_index(param_col)[ci_upper_col]
                except Exception as e:
                    print(f"    Warning: Could not process fixed effects for {concept}: {e}")
            else:
                print(f"    Warning: Could not find essential columns for {concept} fixed effects")
        
        # 2. RANDOM INTERCEPTS - Check for different possible column names
        if "random_effects" in concept_res and not concept_res["random_effects"].empty:
            re_df = concept_res["random_effects"].copy()
            
            # Determine the correct column names
            stratum_col = None
            intercept_col = None
            sig_col = None
            pval_col = None
            
            # Check for stratum column
            for col in ['stratum', 'group', 'cluster', 'level']:
                if col in re_df.columns:
                    stratum_col = col
                    break
            
            # Check for random intercept column
            for col in ['random_intercept', 'intercept', 'estimate', 'effect']:
                if col in re_df.columns:
                    intercept_col = col
                    break
            
            # Check for significance column
            for col in ['significant', 'sig', 'is_significant']:
                if col in re_df.columns:
                    sig_col = col
                    break
            
            # Check for p-value column
            for col in ['p_value', 'pvalue', 'p', 'p_val']:
                if col in re_df.columns:
                    pval_col = col
                    break
            
            if verbose:
                print(f"  {concept} random effects columns detected:")
                print(f"    Stratum: {stratum_col}")
                print(f"    Intercept: {intercept_col}")
                print(f"    Significant: {sig_col}")
                print(f"    P-value: {pval_col}")
            
            # Export data if we found the essential columns
            if stratum_col and intercept_col:
                try:
                    re_series = re_df.set_index(stratum_col)[intercept_col]
                    random_intercepts_data[f"{concept}_intercept"] = re_series
                    
                    if sig_col:
                        random_intercepts_data[f"{concept}_significant"] = re_df.set_index(stratum_col)[sig_col]
                    if pval_col:
                        random_intercepts_data[f"{concept}_pvalue"] = re_df.set_index(stratum_col)[pval_col]
                except Exception as e:
                    print(f"    Warning: Could not process random effects for {concept}: {e}")
            else:
                print(f"    Warning: Could not find essential columns for {concept} random effects")
        
        # 3. EXPECTED VALUES (stratum predictions) - Check for different possible column names
        if "stratum_predictions" in concept_res and not concept_res["stratum_predictions"].empty:
            sp_df = concept_res["stratum_predictions"].copy()
            
            # Determine the correct column names
            stratum_col = None
            expected_col = None
            observed_col = None
            n_obs_col = None
            residual_col = None
            
            # Check for stratum column
            for col in ['stratum', 'group', 'cluster', 'level']:
                if col in sp_df.columns:
                    stratum_col = col
                    break
            
            # Check for expected mean column
            for col in ['expected_mean', 'predicted', 'fitted', 'expected']:
                if col in sp_df.columns:
                    expected_col = col
                    break
            
            # Check for observed mean column
            for col in ['observed_mean', 'observed', 'actual', 'mean']:
                if col in sp_df.columns:
                    observed_col = col
                    break
            
            # Check for n_obs column
            for col in ['n_obs', 'n', 'count', 'sample_size']:
                if col in sp_df.columns:
                    n_obs_col = col
                    break
            
            # Check for residual column
            for col in ['residual', 'resid', 'difference', 'error']:
                if col in sp_df.columns:
                    residual_col = col
                    break
            
            if verbose:
                print(f"  {concept} stratum predictions columns detected:")
                print(f"    Stratum: {stratum_col}")
                print(f"    Expected: {expected_col}")
                print(f"    Observed: {observed_col}")
                print(f"    N_obs: {n_obs_col}")
                print(f"    Residual: {residual_col}")
            
            # Export data if we found the essential columns
            if stratum_col and expected_col:
                try:
                    exp_series = sp_df.set_index(stratum_col)[expected_col]
                    expected_values_data[f"{concept}_expected"] = exp_series
                    
                    if observed_col:
                        expected_values_data[f"{concept}_observed"] = sp_df.set_index(stratum_col)[observed_col]
                    if n_obs_col:
                        expected_values_data[f"{concept}_n_obs"] = sp_df.set_index(stratum_col)[n_obs_col]
                    if residual_col:
                        expected_values_data[f"{concept}_residual"] = sp_df.set_index(stratum_col)[residual_col]
                except Exception as e:
                    print(f"    Warning: Could not process stratum predictions for {concept}: {e}")
            else:
                print(f"    Warning: Could not find essential columns for {concept} stratum predictions")
    
    # Add global model results if provided (using same flexible column detection)
    if include_global_model:
        concept = "GLOBAL"
        
        # Fixed effects
        if "fixed_effects" in include_global_model and not include_global_model["fixed_effects"].empty:
            fe_df = include_global_model["fixed_effects"].copy()
            
            # Use the same flexible column detection as above
            param_col = None
            coeff_col = None
            pval_col = None
            ci_lower_col = None
            ci_upper_col = None
            
            for col in ['parameter', 'param', 'variable', 'predictor']:
                if col in fe_df.columns:
                    param_col = col
                    break
            for col in ['coefficient', 'coef', 'estimate', 'beta']:
                if col in fe_df.columns:
                    coeff_col = col
                    break
            for col in ['p_value', 'pvalue', 'p', 'p_val']:
                if col in fe_df.columns:
                    pval_col = col
                    break
            for col in ['ci_lower', 'conf_int_lower', 'lower_ci', 'ci_low']:
                if col in fe_df.columns:
                    ci_lower_col = col
                    break
            for col in ['ci_upper', 'conf_int_upper', 'upper_ci', 'ci_high']:
                if col in fe_df.columns:
                    ci_upper_col = col
                    break
            
            if param_col and coeff_col:
                try:
                    fe_series = fe_df.set_index(param_col)[coeff_col]
                    fixed_effects_data[f"{concept}_coeff"] = fe_series
                    if pval_col:
                        fixed_effects_data[f"{concept}_pvalue"] = fe_df.set_index(param_col)[pval_col]
                    if ci_lower_col:
                        fixed_effects_data[f"{concept}_ci_lower"] = fe_df.set_index(param_col)[ci_lower_col]
                    if ci_upper_col:
                        fixed_effects_data[f"{concept}_ci_upper"] = fe_df.set_index(param_col)[ci_upper_col]
                except Exception as e:
                    print(f"Warning: Could not process global model fixed effects: {e}")
        
        # Apply same logic for random effects and stratum predictions...
        # (Similar flexible column detection for global model)
    
    # Create DataFrames
    fixed_effects_df = pd.DataFrame(fixed_effects_data) if fixed_effects_data else pd.DataFrame()
    random_intercepts_df = pd.DataFrame(random_intercepts_data) if random_intercepts_data else pd.DataFrame()
    expected_values_df = pd.DataFrame(expected_values_data) if expected_values_data else pd.DataFrame()
    
    if verbose:
        print(f"\nData dimensions:")
        print(f"  Fixed effects: {fixed_effects_df.shape}")
        print(f"  Random intercepts: {random_intercepts_df.shape}")
        print(f"  Expected values: {expected_values_df.shape}")
    
    # Create summary statistics
    summary_stats = []
    
    for concept in successful_concepts:
        concept_res = concept_results[concept]
        
        # Get VPC and other metrics
        maihda_res = concept_res["maihda_results"]
        
        # Flexible summary extraction
        vpc = np.nan
        pcv = np.nan
        icc = np.nan
        
        if "summary" in maihda_res and not maihda_res["summary"].empty:
            summary_df = maihda_res["summary"]
            
            # Look for VPC in different possible columns/rows
            if "VPC" in summary_df.columns:
                vpc_values = summary_df["VPC"].dropna()
                if not vpc_values.empty:
                    vpc = vpc_values.iloc[-1]  # Take last non-null value
            
            # Look for ICC in different possible columns/rows
            if "ICC" in summary_df.columns:
                icc_values = summary_df["ICC"].dropna()
                if not icc_values.empty:
                    icc = icc_values.iloc[-1]
            
            # Look for PCV in different possible columns/rows
            for col in ["PCV_%", "PCV", "PCV_percent"]:
                if col in summary_df.columns:
                    pcv_values = summary_df[col].dropna()
                    if not pcv_values.empty:
                        pcv = pcv_values.iloc[-1]
                        break
        
        summary_stats.append({
            "concept": concept,
            "n_observations": concept_res.get("n_observations", 0),
            "n_strata": concept_res.get("n_strata", 0),
            "vpc": vpc,
            "icc": icc,
            "pcv_percent": pcv,
            "n_significant_re": concept_res.get("n_significant_re", 0),
            "total_re": concept_res.get("total_re", 0),
            "prop_significant_re": concept_res.get("n_significant_re", 0) / max(concept_res.get("total_re", 1), 1)
        })
    
    summary_df = pd.DataFrame(summary_stats)
    
    # Write to Excel
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#D7E4BC',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '0.000'})
            pvalue_format = workbook.add_format({'num_format': '0.000', 'bg_color': '#FFF2CC'})
            
            # Sheet 1: Fixed Effects
            if not fixed_effects_df.empty:
                fixed_effects_df.to_excel(writer, sheet_name='Fixed_Effects', startrow=1)
                worksheet = writer.sheets['Fixed_Effects']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Fixed Effects by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(fixed_effects_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format p-value columns
                for col_num, col_name in enumerate(fixed_effects_df.columns):
                    if 'pvalue' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 30)  # Parameter names column
            
            # Sheet 2: Random Intercepts
            if not random_intercepts_df.empty:
                random_intercepts_df.to_excel(writer, sheet_name='Random_Intercepts', startrow=1)
                worksheet = writer.sheets['Random_Intercepts']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Random Intercepts by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(random_intercepts_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num, col_name in enumerate(random_intercepts_df.columns):
                    if 'pvalue' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                    elif 'significant' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12)
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 40)  # Stratum names column
            
            # Sheet 3: Expected Values
            if not expected_values_df.empty:
                expected_values_df.to_excel(writer, sheet_name='Expected_Values', startrow=1)
                worksheet = writer.sheets['Expected_Values']
                
                # Add title
                worksheet.write(0, 0, f'{llm_name.upper()} - Expected Values by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(expected_values_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num, col_name in enumerate(expected_values_df.columns):
                    if 'n_obs' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 10, 
                                           workbook.add_format({'num_format': '0'}))
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 40)  # Stratum names column
            
            # Sheet 4: Summary
            summary_df.to_excel(writer, sheet_name='Summary', index=False, startrow=1)
            worksheet = writer.sheets['Summary']
            
            # Add title
            worksheet.write(0, 0, f'{llm_name.upper()} - Summary Statistics by Concept', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
            
            # Format headers
            for col_num, value in enumerate(summary_df.columns.values):
                worksheet.write(1, col_num, value, header_format)
            
            # Auto-adjust column widths
            for col_num, col_name in enumerate(summary_df.columns):
                max_len = max(
                    len(str(col_name)),
                    summary_df[col_name].astype(str).str.len().max() if not summary_df.empty else 0
                )
                worksheet.set_column(col_num, col_num, min(max_len + 2, 20))
        
        if verbose:
            print(f"✅ Excel file exported successfully: {output_path}")
            print(f"\nSheets created:")
            print(f"  1. Fixed_Effects ({fixed_effects_df.shape[0]} parameters × {fixed_effects_df.shape[1]} columns)")
            print(f"  2. Random_Intercepts ({random_intercepts_df.shape[0]} strata × {random_intercepts_df.shape[1]} columns)")
            print(f"  3. Expected_Values ({expected_values_df.shape[0]} strata × {expected_values_df.shape[1]} columns)")
            print(f"  4. Summary ({summary_df.shape[0]} concepts × {summary_df.shape[1]} metrics)")
    
    except Exception as e:
        print(f"❌ Failed to export Excel file: {e}")
        return None
    
    return {
        "output_path": output_path,
        "fixed_effects_shape": fixed_effects_df.shape,
        "random_intercepts_shape": random_intercepts_df.shape,
        "expected_values_shape": expected_values_df.shape,
        "summary_shape": summary_df.shape,
        "concepts": successful_concepts
    }




def run_ols_per_concept(
    df,
    response_var="response_recoded",
    fixed_effects=("race", "gender", "religion", "transness"),
    concept_col="concept",
    out_dir=None,
    save_excel=True,
    make_plots=False,
    verbose=True
):
    """
    Run separate OLS models for each concept.
    
    Parameters:
    -----------
    df : DataFrame
        Input data with all concepts
    response_var : str
        Name of response variable
    fixed_effects : tuple
        Fixed effect variables (concept excluded automatically)
    concept_col : str
        Column name for concept grouping
    out_dir : str, optional
        Directory for outputs
    save_excel : bool
        Save results to Excel
    make_plots : bool
        Create diagnostic plots
    verbose : bool
        Print progress
        
    Returns:
    --------
    dict: Results for each concept
    """
    import pandas as pd
    import numpy as np
    import statsmodels.formula.api as smf
    import os
    from pathlib import Path
    
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(exist_ok=True)
    
    # Get unique concepts
    concepts = df[concept_col].unique()
    
    if verbose:
        print(f"\n🔍 RUNNING OLS PER CONCEPT")
        print("="*50)
        print(f"Total concepts: {len(concepts)}")
        print(f"Concepts: {', '.join(concepts)}")
        print(f"Fixed effects: {', '.join(fixed_effects)}")
    
    results = {
        "concept_results": {},
        "summary": {
            "n_concepts_total": len(concepts),
            "n_concepts_successful": 0,
            "n_concepts_failed": 0,
            "failed_concepts": []
        }
    }
    
    # Build formula (exclude concept from fixed effects)
    formula_vars = [f"C({var})" for var in fixed_effects if var != concept_col]
    formula = f"{response_var} ~ " + " + ".join(formula_vars)
    
    if verbose:
        print(f"Formula: {formula}")
    
    # Run OLS for each concept
    for concept in concepts:
        if verbose:
            print(f"\n📊 Processing concept: {concept}")
        
        # Filter data for this concept
        df_concept = df[df[concept_col] == concept].copy()
        df_concept = df_concept.dropna(subset=[response_var])
        
        if df_concept.empty:
            if verbose:
                print(f"   ❌ No valid data for {concept}")
            results["concept_results"][concept] = {
                "status": "failed",
                "error": "No valid data",
                "n_observations": 0
            }
            results["summary"]["failed_concepts"].append(concept)
            results["summary"]["n_concepts_failed"] += 1
            continue
        
        try:
            # Fit OLS model
            model = smf.ols(formula=formula, data=df_concept).fit()
            
            # Extract results
            ols_results = extract_ols_results(model, df_concept)
            
            # Store results
            results["concept_results"][concept] = {
                "status": "success",
                "ols_results": ols_results,
                "model": model,
                "n_observations": len(df_concept),
                "formula": formula
            }
            
            results["summary"]["n_concepts_successful"] += 1
            
            if verbose:
                print(f"   ✅ Success: {len(df_concept)} observations, R² = {model.rsquared:.4f}")
            
            # Save individual results if requested
            if out_dir and save_excel:
                concept_excel = out_dir / f"ols_{concept.lower()}_results.xlsx"
                export_single_ols_results(ols_results, concept_excel, concept, verbose=False)
            
            # Make plots if requested
            if out_dir and make_plots:
                plot_dir = out_dir / "plots"
                plot_dir.mkdir(exist_ok=True)
                create_ols_diagnostic_plots(model, df_concept, concept, plot_dir)
                
        except Exception as e:
            if verbose:
                print(f"   ❌ Failed: {str(e)}")
            results["concept_results"][concept] = {
                "status": "failed",
                "error": str(e),
                "n_observations": len(df_concept)
            }
            results["summary"]["failed_concepts"].append(concept)
            results["summary"]["n_concepts_failed"] += 1
    
    if verbose:
        print(f"\n📈 SUMMARY:")
        print(f"   Successful: {results['summary']['n_concepts_successful']}")
        print(f"   Failed: {results['summary']['n_concepts_failed']}")
        if results["summary"]["failed_concepts"]:
            print(f"   Failed concepts: {', '.join(results['summary']['failed_concepts'])}")
    
    # Save comprehensive results
    if out_dir and save_excel:
        summary_excel = out_dir / "ols_all_concepts_summary.xlsx"
        export_ols_concept_summary(results, summary_excel, verbose=verbose)
    
    return results


def extract_ols_results(model, df_concept):
    """
    Extract standardized results from OLS model.
    """
    import pandas as pd
    import numpy as np
    
    # Fixed effects (coefficients)
    coef_summary = model.summary2().tables[1]  # Coefficient table
    fixed_effects = pd.DataFrame({
        "parameter": coef_summary.index,
        "coefficient": coef_summary["Coef."],
        "std_error": coef_summary["Std.Err."],
        "t_value": coef_summary["t"],
        "p_value": coef_summary["P>|t|"],
        "ci_lower": coef_summary["[0.025"],
        "ci_upper": coef_summary["0.975]"]
    }).reset_index(drop=True)
    
    # Model fit statistics
    summary_stats = pd.DataFrame({
        "metric": ["R_squared", "R_squared_adj", "F_statistic", "F_pvalue", "AIC", "BIC", "N_obs"],
        "value": [
            model.rsquared,
            model.rsquared_adj, 
            model.fvalue,
            model.f_pvalue,
            model.aic,
            model.bic,
            model.nobs
        ]
    })
    
    # Residual analysis
    residuals = model.resid
    fitted_values = model.fittedvalues
    
    residual_stats = pd.DataFrame({
        "statistic": ["mean", "std", "min", "max", "skewness", "kurtosis"],
        "value": [
            residuals.mean(),
            residuals.std(),
            residuals.min(),
            residuals.max(),
            residuals.skew(),
            residuals.kurtosis()
        ]
    })
    
    return {
        "fixed_effects": fixed_effects,
        "summary": summary_stats,
        "residual_stats": residual_stats,
        "residuals": residuals,
        "fitted_values": fitted_values,
        "raw_model": model
    }


def export_single_ols_results(ols_results, output_path, concept_name, verbose=True):
    """
    Export results for a single OLS concept to Excel.
    """
    import pandas as pd
    
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            # Sheet 1: Fixed Effects
            ols_results["fixed_effects"].to_excel(
                writer, sheet_name='Fixed_Effects', index=False
            )
            
            # Sheet 2: Model Summary  
            ols_results["summary"].to_excel(
                writer, sheet_name='Model_Summary', index=False
            )
            
            # Sheet 3: Residual Statistics
            ols_results["residual_stats"].to_excel(
                writer, sheet_name='Residual_Stats', index=False
            )
        
        if verbose:
            print(f"   💾 Saved {concept_name} results: {output_path}")
            
    except Exception as e:
        if verbose:
            print(f"   ❌ Failed to save {concept_name}: {e}")


def export_ols_concept_summary(results, output_path, verbose=True):
    """
    Export summary of all OLS concept results to Excel.
    """
    import pandas as pd
    
    concept_results = results.get("concept_results", {})
    successful_concepts = [k for k, v in concept_results.items() if v.get("status") == "success"]
    
    if not successful_concepts:
        if verbose:
            print("❌ No successful OLS results to export")
        return None
    
    try:
        with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#E1D5E7',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '0.000'})
            pvalue_format = workbook.add_format({'num_format': '0.000', 'bg_color': '#FCE4D6'})
            
            # Sheet 1: Coefficients
            coef_data = {}
            for concept in successful_concepts:
                concept_res = concept_results[concept]["ols_results"]
                fe_df = concept_res["fixed_effects"]
                
                # Use parameter as index, coefficient as values
                coef_series = fe_df.set_index('parameter')['coefficient']
                coef_data[f"{concept}_coeff"] = coef_series
                coef_data[f"{concept}_pvalue"] = fe_df.set_index('parameter')['p_value']
                coef_data[f"{concept}_ci_lower"] = fe_df.set_index('parameter')['ci_lower']
                coef_data[f"{concept}_ci_upper"] = fe_df.set_index('parameter')['ci_upper']
            
            coef_df = pd.DataFrame(coef_data)
            
            if not coef_df.empty:
                coef_df.to_excel(writer, sheet_name='Coefficients', startrow=1)
                worksheet = writer.sheets['Coefficients']
                
                # Add title
                worksheet.write(0, 0, 'OLS Coefficients by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(coef_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num, col_name in enumerate(coef_df.columns):
                    if 'pvalue' in col_name.lower():
                        worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                    else:
                        worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                
                worksheet.set_column(0, 0, 30)  # Parameter names column
            
            # Sheet 2: Model Summary Statistics
            summary_data = {}
            for concept in successful_concepts:
                concept_res = concept_results[concept]["ols_results"]
                summary_df = concept_res["summary"]
                
                summary_series = summary_df.set_index('metric')['value']
                summary_data[concept] = summary_series
            
            summary_stats_df = pd.DataFrame(summary_data)
            
            if not summary_stats_df.empty:
                summary_stats_df.to_excel(writer, sheet_name='Model_Statistics', startrow=1)
                worksheet = writer.sheets['Model_Statistics']
                
                # Add title
                worksheet.write(0, 0, 'Model Statistics by Concept', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(summary_stats_df.columns.values):
                    worksheet.write(1, col_num + 1, value, header_format)
                
                # Format columns
                for col_num in range(len(summary_stats_df.columns)):
                    worksheet.set_column(col_num + 1, col_num + 1, 15, number_format)
                
                worksheet.set_column(0, 0, 20)  # Metric names column
            
            # Sheet 3: Overall Summary
            overall_summary = []
            for concept in successful_concepts:
                concept_res = concept_results[concept]
                ols_res = concept_res["ols_results"]
                model = ols_res["raw_model"]
                
                overall_summary.append({
                    "concept": concept,
                    "n_observations": concept_res.get("n_observations", 0),
                    "r_squared": model.rsquared,
                    "r_squared_adj": model.rsquared_adj,
                    "f_statistic": model.fvalue,
                    "f_pvalue": model.f_pvalue,
                    "aic": model.aic,
                    "bic": model.bic,
                    "n_parameters": len(model.params)
                })
            
            overall_df = pd.DataFrame(overall_summary)
            overall_df.to_excel(writer, sheet_name='Summary', index=False, startrow=1)
            worksheet = writer.sheets['Summary']
            
            # Add title
            worksheet.write(0, 0, 'OLS Summary by Concept', 
                          workbook.add_format({'bold': True, 'font_size': 14}))
            
            # Format headers
            for col_num, value in enumerate(overall_df.columns.values):
                worksheet.write(1, col_num, value, header_format)
            
            # Auto-adjust column widths
            for col_num, col_name in enumerate(overall_df.columns):
                max_len = max(
                    len(str(col_name)),
                    overall_df[col_name].astype(str).str.len().max() if not overall_df.empty else 0
                )
                worksheet.set_column(col_num, col_num, min(max_len + 2, 20))
        
        if verbose:
            print(f"✅ OLS summary exported: {output_path}")
            print(f"   Sheets: Coefficients, Model_Statistics, Summary")
            print(f"   Concepts: {len(successful_concepts)}")
            
    except Exception as e:
        if verbose:
            print(f"❌ Failed to export OLS summary: {e}")
        return None


def export_all_llm_ols_results(
    ols_results_dict,
    output_dir="ols_complete_results",
    verbose=True
):
    """
    Export OLS results for all LLMs into one comprehensive Excel file.
    
    Parameters:
    -----------
    ols_results_dict : dict
        Dictionary with LLM names as keys and OLS results as values
    output_dir : str
        Directory for outputs
    verbose : bool
        Print progress
    """
    import pandas as pd
    import os
    from pathlib import Path
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if verbose:
        print(f"\n📊 EXPORTING ALL LLM OLS RESULTS")
        print("="*60)
        print(f"Output directory: {output_dir}")
    
    # Main comprehensive file
    main_output = output_dir / "all_llm_ols_results.xlsx"
    
    try:
        with pd.ExcelWriter(main_output, engine='xlsxwriter') as writer:
            workbook = writer.book
            
            # Define formats
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'bg_color': '#E1D5E7',
                'border': 1
            })
            
            number_format = workbook.add_format({'num_format': '0.000'})
            pvalue_format = workbook.add_format({'num_format': '0.000', 'bg_color': '#FCE4D6'})
            
            # Create one sheet per LLM for coefficients
            for llm_name, ols_results in ols_results_dict.items():
                if verbose:
                    print(f"   Processing {llm_name.upper()}...")
                
                concept_results = ols_results.get("concept_results", {})
                successful_concepts = [k for k, v in concept_results.items() if v.get("status") == "success"]
                
                if not successful_concepts:
                    if verbose:
                        print(f"     ❌ No successful results for {llm_name}")
                    continue
                
                # Coefficients sheet for this LLM
                coef_data = {}
                for concept in successful_concepts:
                    concept_res = concept_results[concept]["ols_results"]
                    fe_df = concept_res["fixed_effects"]
                    
                    coef_series = fe_df.set_index('parameter')['coefficient']
                    coef_data[f"{concept}_coeff"] = coef_series
                    coef_data[f"{concept}_pvalue"] = fe_df.set_index('parameter')['p_value']
                
                coef_df = pd.DataFrame(coef_data)
                
                if not coef_df.empty:
                    sheet_name = f"{llm_name.upper()}_Coefficients"
                    coef_df.to_excel(writer, sheet_name=sheet_name, startrow=1)
                    worksheet = writer.sheets[sheet_name]
                    
                    # Add title
                    worksheet.write(0, 0, f'{llm_name.upper()} - OLS Coefficients by Concept', 
                                  workbook.add_format({'bold': True, 'font_size': 14}))
                    
                    # Format headers
                    for col_num, value in enumerate(coef_df.columns.values):
                        worksheet.write(1, col_num + 1, value, header_format)
                    
                    # Format columns
                    for col_num, col_name in enumerate(coef_df.columns):
                        if 'pvalue' in col_name.lower():
                            worksheet.set_column(col_num + 1, col_num + 1, 12, pvalue_format)
                        else:
                            worksheet.set_column(col_num + 1, col_num + 1, 12, number_format)
                    
                    worksheet.set_column(0, 0, 30)  # Parameter names column
                
                if verbose:
                    print(f"     ✅ {len(successful_concepts)} concepts exported")
            
            # Summary comparison sheet across all LLMs
            summary_comparison = []
            for llm_name, ols_results in ols_results_dict.items():
                concept_results = ols_results.get("concept_results", {})
                successful_concepts = [k for k, v in concept_results.items() if v.get("status") == "success"]
                
                for concept in successful_concepts:
                    concept_res = concept_results[concept]
                    ols_res = concept_res["ols_results"]
                    model = ols_res["raw_model"]
                    
                    summary_comparison.append({
                        "llm": llm_name,
                        "concept": concept,
                        "n_observations": concept_res.get("n_observations", 0),
                        "r_squared": model.rsquared,
                        "r_squared_adj": model.rsquared_adj,
                        "f_statistic": model.fvalue,
                        "f_pvalue": model.f_pvalue,
                        "aic": model.aic,
                        "bic": model.bic
                    })
            
            if summary_comparison:
                summary_df = pd.DataFrame(summary_comparison)
                summary_df.to_excel(writer, sheet_name='Summary_All_LLMs', index=False, startrow=1)
                worksheet = writer.sheets['Summary_All_LLMs']
                
                # Add title
                worksheet.write(0, 0, 'OLS Summary - All LLMs and Concepts', 
                              workbook.add_format({'bold': True, 'font_size': 14}))
                
                # Format headers
                for col_num, value in enumerate(summary_df.columns.values):
                    worksheet.write(1, col_num, value, header_format)
                
                # Auto-adjust column widths
                for col_num, col_name in enumerate(summary_df.columns):
                    max_len = max(
                        len(str(col_name)),
                        summary_df[col_name].astype(str).str.len().max() if not summary_df.empty else 0
                    )
                    worksheet.set_column(col_num, col_num, min(max_len + 2, 20))
        
        if verbose:
            print(f"✅ Comprehensive OLS results exported: {main_output}")
            
        return {
            "main_file": str(main_output),
            "n_llms": len(ols_results_dict),
            "total_concepts": len(summary_comparison)
        }
        
    except Exception as e:
        if verbose:
            print(f"❌ Failed to export comprehensive OLS results: {e}")
        return None


def create_ols_diagnostic_plots(model, df_concept, concept_name, plot_dir):
    """
    Create diagnostic plots for OLS model.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from pathlib import Path
    
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'OLS Diagnostics - {concept_name}', fontsize=16)
    
    # Residuals vs Fitted
    axes[0, 0].scatter(model.fittedvalues, model.resid, alpha=0.6)
    axes[0, 0].axhline(y=0, color='red', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # Q-Q plot
    from scipy import stats
    stats.probplot(model.resid, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Histogram of residuals
    axes[1, 0].hist(model.resid, bins=30, alpha=0.7, density=True)
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Residual Distribution')
    
    # Cook's distance (simplified)
    from statsmodels.stats.outliers_influence import OLSInfluence
    influence = OLSInfluence(model)
    cooks_d = influence.cooks_distance[0]
    axes[1, 1].stem(range(len(cooks_d)), cooks_d, markerfmt=",")
    axes[1, 1].set_xlabel('Observation')
    axes[1, 1].set_ylabel("Cook's Distance")
    axes[1, 1].set_title("Cook's Distance")
    
    plt.tight_layout()
    
    plot_file = plot_dir / f"ols_diagnostics_{concept_name.lower()}.png"
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()


    
from matplotlib.patches import Ellipse


# -----------------------------------------------------
# Helper: draw a confidence ellipse from two models
# -----------------------------------------------------
def draw_confidence_ellipse(ax, xy, cov, n_std=1.96, **kwargs):
    """
    Draw covariance ellipse for (warmth_coef, competence_coef)
    xy = [warmth, competence]
    cov = 2x2 covariance matrix from the two models (cross-cov assumed 0)
    """
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_) * n_std  # scale by desired CI
    
    # angle of rotation
    angle = np.degrees(np.arctan2(v[1, 0], v[0, 0]))
    
    ellipse = Ellipse(
        xy=xy,
        width=lambda_[0] * 2,
        height=lambda_[1] * 2,
        angle=angle,
        **kwargs
    )
    ax.add_patch(ellipse)


# -----------------------------------------------------
# Main plotting function
# -----------------------------------------------------


def plot_2d_coefficients_with_ellipses(
    df, 
    model_name="LLM",
    auto_save=True,
    save_dir="plots",
    save_formats=["png", "pdf", "svg"],
    dpi=300,
    save_path=None,
    figsize=(10, 8)
):
    """
    Plot warmth vs competence coefficients with confidence ellipses and automatic high-quality saving.
    
    NEW PARAMETERS:
    ---------------
    auto_save : bool, default=True
        Automatically save high-quality versions of the plot
    save_dir : str, default="plots"
        Directory to save auto-generated plots
    save_formats : list, default=["png", "pdf", "svg"]
        List of formats to save. Options: "png", "pdf", "svg", "eps", "tiff"
    dpi : int, default=300
        DPI for raster formats (png, tiff). Higher = better quality but larger files
        Recommended: 150 (draft), 300 (publication), 600 (high-res)
    save_path : str, optional
        If provided, save figure to this exact path (e.g., 'my_plot.png')
    figsize : tuple, default=(10, 8)
        Figure size in inches
    
    EXISTING PARAMETERS:
    -------------------
    df : pd.DataFrame
        DataFrame with warmth_score and competence_score
    model_name : str, default="LLM"
        Model name for plot title and filename
    """
    
    import os
    import matplotlib as mpl
    from pathlib import Path
    
    # Create save directory if auto_save is enabled
    if auto_save or save_path:
        if auto_save:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
    
    # Store original matplotlib settings to restore later
    original_dpi = mpl.rcParams['figure.dpi']
    original_facecolor = mpl.rcParams['figure.facecolor']
    original_edgecolor = mpl.rcParams['figure.edgecolor']
    
    # Set high-quality matplotlib parameters
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.edgecolor'] = 'white'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    
    # Fit warmth and competence models
    model_w = smf.ols(
        "warmth_score ~ gender + race + religion + transness",
        data=df
    ).fit()
    model_c = smf.ols(
        "competence_score ~ gender + race + religion + transness",
        data=df
    ).fit()

    # Extract coefficients
    coef_w = model_w.params.rename("warmth_coef").to_frame()
    coef_c = model_c.params.rename("competence_coef").to_frame()

    coef = coef_w.join(coef_c)
    coef = coef.drop(index="Intercept")

    # Identify predictor family by name patterns
    def family(term):
        if term.startswith("gender"):
            return "gender"
        if term.startswith("race"):
            return "race"
        if term.startswith("religion"):
            return "religion"
        if term.startswith("transness"):
            return "transness"
        return "other"
    
    coef["family"] = coef.index.map(family)

    # Cleanup labels
    def clean_ols_label(term):
        """
        Clean OLS categorical term names for readability.
        Examples:
        'gender[T.woman]'   -> 'woman'
        'race[T.Black]'     -> 'Black'
        'religion[T.muslim]'-> 'muslim'
        'transness[T.trans]'-> 'trans'
        """
        if "[" in term and "]" in term:
            inside = term.split("[")[1].split("]")[0]   # e.g. T.woman
            if "." in inside:
                return inside.split(".")[1]             # keep only 'woman'
            return inside
        return term

    # Colors per predictor family
    color_map = {
        "gender": "#1f77b4",
        "race": "#d62728",
        "religion": "#2ca02c",
        "transness": "#9467bd"
    }

    # Extract standard errors for ellipses
    # (Assume zero covariance between warmth and competence estimates)
    se_w = model_w.bse.rename("se_w")
    se_c = model_c.bse.rename("se_c")
    se = se_w.to_frame().join(se_c.to_frame())
    se = se.drop(index="Intercept")

    # Covariance matrices per point (diagonal only)
    # Using zero cross-covariance between warmth and competence betas
    cov = {
        term: np.array([[se.loc[term, "se_w"]**2, 0],
                        [0, se.loc[term, "se_c"]**2]])
        for term in coef.index
    }

    # Create figure with enhanced quality settings
    fig, ax = plt.subplots(figsize=figsize)

    # SCM quadrant lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)

    # Plot confidence ellipses + points
    for term, row in coef.iterrows():
        x = row["warmth_coef"]
        y = row["competence_coef"]
        
        # ellipse
        draw_confidence_ellipse(
            ax=ax,
            xy=(x, y),
            cov=cov[term],
            n_std=1.96,
            alpha=0.2,  # Slightly more visible
            color=color_map[row["family"]],
        )
        
        # point
        ax.scatter(x, y, s=120, color=color_map[row["family"]], 
                  edgecolor="black", linewidth=1.2, alpha=0.9, zorder=3)
        
        # label with enhanced styling
        ax.text(
            x + 0.02, y + 0.02,
            clean_ols_label(term),
            fontsize=11, weight="bold",
            path_effects=[patheffects.withStroke(linewidth=2.5, foreground="white")]
        )

    # Enhanced styling
    ax.set_xlabel("Warmth Coefficient", fontsize=14, weight="bold")
    ax.set_ylabel("Competence Coefficient", fontsize=14, weight="bold")
    ax.set_title(f"{model_name}: Warmth vs Competence Coefficients with 95% Confidence Ellipses", 
                fontsize=16, weight="bold", pad=20)

    # Legend for families with enhanced styling
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=key,
                   markerfacecolor=color_map[key], markersize=12, 
                   markeredgecolor="black", markeredgewidth=1.2)
        for key in color_map
    ]
    ax.legend(handles=handles, title="Identity Dimensions", loc="best", 
             framealpha=0.95, fontsize=12, title_fontsize=13, 
             frameon=True, fancybox=True, shadow=True)
    
    # Enhanced grid and tick styling
    ax.grid(alpha=0.25, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)

    plt.tight_layout()

    # ---------------------------------------------------
    # HIGH-QUALITY SAVING
    # ---------------------------------------------------
    saved_files = []
    
    # Save to specific path if provided
    if save_path:
        try:
            # Determine format from extension
            ext = Path(save_path).suffix.lower().lstrip('.')
            save_kwargs = {
                'bbox_inches': 'tight',
                'facecolor': 'white',
                'edgecolor': 'none',
                'pad_inches': 0.2
            }
            
            if ext in ['png', 'tiff', 'jpg', 'jpeg']:
                save_kwargs['dpi'] = dpi
            
            plt.savefig(save_path, **save_kwargs)
            saved_files.append(save_path)
            print(f"💾 Figure saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save to {save_path}: {e}")
    
    # Auto-save in multiple formats if enabled
    if auto_save:
        # Create filename base
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_model_name = safe_model_name.replace(' ', '_')
        filename_base = f"warmth_competence_coefficients_{safe_model_name}"
        
        print(f"\n💾 AUTO-SAVING HIGH-QUALITY PLOTS...")
        print(f"   Directory: {save_dir}")
        print(f"   Formats: {', '.join(save_formats)}")
        print(f"   DPI: {dpi} (for raster formats)")
        
        for fmt in save_formats:
            try:
                file_path = save_dir / f"{filename_base}.{fmt}"
                
                # Format-specific settings
                save_kwargs = {
                    'bbox_inches': 'tight',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'pad_inches': 0.2
                }
                
                if fmt in ['png', 'tiff', 'jpg', 'jpeg']:
                    save_kwargs['dpi'] = dpi
                elif fmt == 'pdf':
                    save_kwargs['dpi'] = 600  # High resolution for PDF
                elif fmt == 'svg':
                    save_kwargs['format'] = 'svg'
                elif fmt == 'eps':
                    save_kwargs['format'] = 'eps'
                
                plt.savefig(file_path, **save_kwargs)
                saved_files.append(str(file_path))
                print(f"   ✅ Saved: {file_path.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to save {fmt}: {e}")
        
        print(f"\n📁 Total files saved: {len(saved_files)}")
    
    # Restore original matplotlib settings
    mpl.rcParams['figure.dpi'] = original_dpi
    mpl.rcParams['figure.facecolor'] = original_facecolor
    mpl.rcParams['figure.edgecolor'] = original_edgecolor
    
    plt.show()
    
    return {
        'saved_files': saved_files,
        'coefficients': coef,
        'models': {'warmth': model_w, 'competence': model_c}
    }



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ======================================================
# 1. COLOR CODING FOR OLS FAMILIES
# ======================================================

color_map = {
    "gender":   "#1f77b4",   # blue
    "race":     "#d62728",   # red
    "religion": "#2ca02c",   # green
    "transness":"#9467bd"    # purple
}



# ======================================================
# 2. LABEL CLEANUP
# ======================================================

def clean_label_parts(s):
    """Clean label into meaningful parts, removing all not_mentioned variants."""
    # Normalize separators
    s = s.replace("-", "_").replace(" ", "_")

    # Split into tokens
    parts = s.split("_")

    # Remove all "not" and all "mention*" variants
    cleaned = [p for p in parts if ("not" not in p and "mention" not in p and p != "")]
    return cleaned    # return list of components

def clean_ols_label(term):
    """
    Clean OLS categorical term names for readability.
    Examples:
      'gender[T.woman]'   -> 'woman'
      'race[T.Black]'     -> 'Black'
      'religion[T.muslim]'-> 'muslim'
      'transness[T.trans]'-> 'trans'
    """
    if "[" in term and "]" in term:
        inside = term.split("[")[1].split("]")[0]   # e.g. T.woman
        if "." in inside:
            return inside.split(".")[1]             # keep only 'woman'
        return inside
    return term


# ======================================================
# 3. REORDER LABEL COMPONENTS
#    Order: transness → race → religion → gender
# ======================================================

def reorder_label_parts(parts):
    order_trans = {"cis", "trans"}
    order_race = {"white", "Black", "asian"}
    order_religion = {"muslim", "christian", "jewish"}
    order_gender = {"woman", "man", "nonbinary"}

    ordered = []
    for p in parts:
        if p in order_trans:
            ordered.append(p)
    for p in parts:
        if p in order_race:
            ordered.append(p)
    for p in parts:
        if p in order_religion:
            ordered.append(p)
    for p in parts:
        if p in order_gender:
            ordered.append(p)

    return ordered



# ======================================================
# 4. ABBREVIATIONS
# ======================================================

abbr = {
    "woman": "woman",
    "man": "man",
    "nonbinary": "nb",

    "white": "white",
    "Black": "Black",
    "asian": "asian",

    "muslim": "Mus",
    "christian": "Chr",
    "jewish": "Jew",

    "cis": "cis",
    "trans": "trans",
}

def abbreviate(parts):
    return "-".join([abbr.get(p, p) for p in parts])



# ======================================================
# 5. CENTER MAIHDA PREDICTIONS
# ======================================================

def center_predictions(strata_pred, warm_intercept, comp_intercept):
    sp = strata_pred.copy()
    sp["warmth_centered"] = sp["warmth_pred"] - warm_intercept
    sp["competence_centered"] = sp["competence_pred"] - comp_intercept
    return sp



# ======================================================
# 6. ROBUST INTERSECTIONAL GROUP SELECTION
# ======================================================

def select_spread_groups_robust(strata_raw, k=4):
    """
    Always select a meaningful spread of extreme intersectional groups.
    Uses raw predictions for stability, then is centered later.
    """

    # top & bottom raw warmth
    top_w = strata_raw.nlargest(k, "warmth_pred")
    bot_w = strata_raw.nsmallest(k, "warmth_pred")

    # top & bottom raw competence
    top_c = strata_raw.nlargest(k, "competence_pred")
    bot_c = strata_raw.nsmallest(k, "competence_pred")

    # Combine & dedupe
    out = pd.concat([top_w, bot_w, top_c, bot_c]).drop_duplicates("stratum")

    # Guarantee at least 6 groups
    if len(out) < 6:
        d = np.sqrt(
            (strata_raw["warmth_pred"] - strata_raw["warmth_pred"].mean())**2 +
            (strata_raw["competence_pred"] - strata_raw["competence_pred"].mean())**2
        )
        out = strata_raw.loc[d.nlargest(6).index]

    return out



# ======================================================
# 7. LABEL POSITION HELPER (avoid overlap)
# ======================================================

def label_offset(x, i):
    """
    Dynamically position labels to avoid overlap:
    - If x >= 0 → place to the right
    - If x < 0 → place to the left
    - Even/odd indices alternate vertical offsets
    """
    dx = 0.015 if x >= 0 else -0.10
    dy = 0.02 if i % 2 else -0.02
    return dx, dy



# ======================================================
# 8. FINAL PLOT FUNCTION
# ======================================================

def plot_bias_space_minimal(
    coef,
    res_warmth_maihda,
    res_comp_maihda,
    model_name="LLM",
    k_extremes=4
):

    # ---------------------------------------------------
    # A. OLS coefficients
    # ---------------------------------------------------
    coef = coef.copy()
    x_ols = coef["warmth_coef"]
    y_ols = coef["competence_coef"]


    # ---------------------------------------------------
    # B. MAIHDA predictions
    # ---------------------------------------------------
    warm_df = res_warmth_maihda["stratum_predictions"][
        ["stratum", "predicted_mean_1B"]
    ].rename(columns={"predicted_mean_1B": "warmth_pred"})

    comp_df = res_comp_maihda["stratum_predictions"][
        ["stratum", "predicted_mean_1B"]
    ].rename(columns={"predicted_mean_1B": "competence_pred"})

    strata_pred = warm_df.merge(comp_df, on="stratum", how="inner")


    # ---------------------------------------------------
    # C. select intersectional extremes (raw)
    # ---------------------------------------------------
    important = select_spread_groups_robust(strata_pred, k=k_extremes)


    # ---------------------------------------------------
    # D. center AFTER selecting
    # ---------------------------------------------------
    warm_intercept = res_warmth_maihda["fixed_effects"].loc["Intercept", "coef"]
    comp_intercept = res_comp_maihda["fixed_effects"].loc["Intercept", "coef"]

    important = center_predictions(important, warm_intercept, comp_intercept)


    # ---------------------------------------------------
    # E. clean + reorder + abbreviate labels
    # ---------------------------------------------------
    final_labels = []
    for s in important["stratum"]:
        parts = clean_label_parts(s)
        ordered = reorder_label_parts(parts)
        final_labels.append(abbreviate(ordered))
    important["clean"] = final_labels


    # ===================================================
    # F. PLOTTING
    # ===================================================

    fig, ax = plt.subplots(figsize=(10, 8))

    # Axes lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1)


    # -----------------------------
    # Plot OLS main effects
    # -----------------------------
    for term, row in coef.iterrows():
        fam = row["family"]
        col = color_map.get(fam, "gray")

        ax.scatter(
            row["warmth_coef"],
            row["competence_coef"],
            s=130,
            color=col,
            edgecolor="black",
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )

        ax.text(
            row["warmth_coef"] + 0.015,
            row["competence_coef"] + 0.015,
            clean_ols_label(term),
            fontsize=9,
        )

    # -----------------------------
    # Plot intersectional extreme groups
    # -----------------------------
    import matplotlib.patheffects as patheffects

    for i, row in important.iterrows():
        gx = row["warmth_centered"]
        gy = row["competence_centered"]
        label = row["clean"]

        # point
        ax.scatter(
            gx, gy,
            s=120, marker="D",
            color="#F2C94C",
            edgecolor="black",
            linewidth=1.1, alpha=0.92,
            zorder=5
        )

        # offsets
        dx, dy = label_offset(gx, i)

        # clamp text inside axes
        x_text = np.clip(gx + dx, ax.get_xlim()[0] + 0.01, ax.get_xlim()[1] - 0.01)
        y_text = np.clip(gy + dy, ax.get_ylim()[0] + 0.01, ax.get_ylim()[1] - 0.01)

        # callout line
        ax.plot(
            [gx, x_text], [gy, y_text],
            color="black", linewidth=0.6, alpha=0.7, zorder=4
        )

        # label with halo
        ax.text(
            x_text, y_text, label,
            fontsize=9, weight="medium",
            path_effects=[patheffects.withStroke(linewidth=2, foreground="white")],
            zorder=6
        )



    # -----------------------------
    # Axis scaling
    # -----------------------------
    pad_x = (x_ols.max() - x_ols.min()) * 2.5
    pad_y = (y_ols.max() - y_ols.min()) * 0.5

    ax.set_xlim(x_ols.min() - pad_x, x_ols.max() + pad_x)
    ax.set_ylim(y_ols.min() - pad_y, y_ols.max() + pad_y)


    # -----------------------------
    # Labels and title
    # -----------------------------
    ax.set_xlabel("Warmth (centered)", fontsize=13)
    ax.set_ylabel("Competence (centered)", fontsize=13)
    ax.set_title(f"{model_name}: Additive vs. Intersectional Bias", fontsize=15)

    # Legend
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=color_map[f], markeredgecolor="black",
                   markersize=10, label=f)
        for f in color_map
    ]

    handles.append(
        plt.Line2D([0], [0], marker="D", color="w",
                   markerfacecolor="#F2C94C", markeredgecolor="black",
                   markersize=10, label="Intersectional strata")
    )

    ax.legend(handles=handles, loc="upper right")

    plt.tight_layout()
    plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects

# ======================================================
# Plot most informative intersectional groups
# ======================================================

def plot_intersectional_bias_space(
    res_warmth_maihda,
    res_comp_maihda,
    model_name="LLM",
    n_extremes=3,  # top/bottom per dimension
    show_all_points=True,
    min_stratum_size=0,  # filter small strata (0 = no filter)
    save_path=None,  # optional: save figure to file
    auto_save=True,  # NEW: automatically save high-quality versions
    save_dir="plots",  # NEW: directory for auto-saved plots
    save_formats=["png", "pdf", "svg"],  # NEW: formats to save
    dpi=600,  # NEW: resolution for raster formats
    additional_strata=None,  # list of stratum names to always include
    label_single_identity=False  # automatically label all single-identity strata
):
    """
    Plot intersectional groups in SCM bias space using MAIHDA predictions.
    
    NEW PARAMETERS:
    ---------------
    auto_save : bool, default=True
        Automatically save high-quality versions of the plot
    save_dir : str, default="plots"
        Directory to save auto-generated plots
    save_formats : list, default=["png", "pdf", "svg"]
        List of formats to save. Options: "png", "pdf", "svg", "eps", "tiff"
    dpi : int, default=300
        DPI for raster formats (png, tiff). Higher = better quality but larger files
        Recommended: 150 (draft), 300 (publication), 600 (high-res)
    
    EXISTING PARAMETERS:
    -------------------
    res_warmth_maihda : dict
        Results from run_maihda_simple() for warmth
    res_comp_maihda : dict
        Results from run_maihda_simple() for competence
    model_name : str
        Model name for plot title
    n_extremes : int
        Number of top/bottom groups per dimension to label
    show_all_points : bool
        Whether to show all strata as background points
    min_stratum_size : int
        Minimum stratum size to consider for labeling (0 = no filter)
    save_path : str, optional
        If provided, save figure to this exact path (e.g., 'my_plot.png')
    additional_strata : list, optional
        List of stratum names to always include in labeling
    label_single_identity : bool
        If True, automatically label all single-identity strata
    """
    
    import os
    from pathlib import Path
    
    # Create save directory if auto_save is enabled
    if auto_save or save_path:
        if auto_save:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)
    
    # ---------------------------------------------------
    # 1. Extract MAIHDA predictions & random effects
    # ---------------------------------------------------
    warm_df = res_warmth_maihda["stratum_predictions"][
        ["stratum", "predicted_mean_1B"]
    ].rename(columns={"predicted_mean_1B": "warmth_pred"})

    comp_df = res_comp_maihda["stratum_predictions"][
        ["stratum", "predicted_mean_1B"]
    ].rename(columns={"predicted_mean_1B": "competence_pred"})

    # Random effects significance
    warm_re = res_warmth_maihda["random_effects"][["stratum", "significant"]].rename(
        columns={"significant": "warmth_significant"}
    )
    comp_re = res_comp_maihda["random_effects"][["stratum", "significant"]].rename(
        columns={"significant": "competence_significant"}
    )

    # Merge everything
    strata = (warm_df
              .merge(comp_df, on="stratum", how="inner")
              .merge(warm_re, on="stratum", how="left")
              .merge(comp_re, on="stratum", how="left"))
    
    # Keep only one n_obs column (they should be identical)
    strata = strata.loc[:, ~strata.columns.duplicated()]
    
    # ---------------------------------------------------
    # 2. Center predictions around grand mean (intercept)
    # ---------------------------------------------------
    warm_intercept = res_warmth_maihda["fixed_effects"].loc["Intercept", "coef"]
    comp_intercept = res_comp_maihda["fixed_effects"].loc["Intercept", "coef"]
    
    strata["warmth_centered"] = strata["warmth_pred"] - warm_intercept
    strata["competence_centered"] = strata["competence_pred"] - comp_intercept
    
    # ---------------------------------------------------
    # 3. Smart selection: extremes + significant + custom
    # ---------------------------------------------------
    selected_groups = []
    
    # A) Extreme warmth
    selected_groups.append(strata.nlargest(n_extremes, "warmth_centered"))
    selected_groups.append(strata.nsmallest(n_extremes, "warmth_centered"))
    
    # B) Extreme competence
    selected_groups.append(strata.nlargest(n_extremes, "competence_centered"))
    selected_groups.append(strata.nsmallest(n_extremes, "competence_centered"))
    
    # C) Significant in BOTH dimensions (strong intersectionality)
    sig_both = strata[
        strata["warmth_significant"] & strata["competence_significant"]
    ]
    if len(sig_both) > 0:
        # From significant, pick the most extreme
        sig_both["extremity"] = np.sqrt(
            sig_both["warmth_centered"]**2 + sig_both["competence_centered"]**2
        )
        selected_groups.append(sig_both.nlargest(min(n_extremes, len(sig_both)), "extremity"))
    
    # D) Additional user-specified strata
    if additional_strata is not None and len(additional_strata) > 0:
        custom_selection = strata[strata["stratum"].isin(additional_strata)]
        if len(custom_selection) > 0:
            selected_groups.append(custom_selection)
            print(f"📍 Added {len(custom_selection)} custom strata")
        else:
            print(f"⚠️  Warning: None of the specified additional_strata found in data")
    
    # E) Single-identity strata (if requested)
    if label_single_identity:
        def count_identity_components(stratum_name):
            """Count how many identity dimensions are specified"""
            # Clean and split
            s = stratum_name.replace("-", "_").replace(" ", "_")
            parts = [p for p in s.split("_") if "not" not in p and "mention" not in p and p]
            
            # Count unique dimensions
            dimensions = set()
            for p in parts:
                if p in ["cis", "trans"]:
                    dimensions.add("transness")
                elif p in ["white", "Black", "asian"]:
                    dimensions.add("race")
                elif p in ["muslim", "christian", "jewish"]:
                    dimensions.add("religion")
                elif p in ["woman", "man", "nonbinary"]:
                    dimensions.add("gender")
            
            return len(dimensions)
        
        strata["n_dimensions"] = strata["stratum"].apply(count_identity_components)
        single_identity = strata[strata["n_dimensions"] == 1]
        
        if len(single_identity) > 0:
            selected_groups.append(single_identity)
            print(f"📍 Added {len(single_identity)} single-identity strata")
    
    # Combine and deduplicate
    labeled_strata = pd.concat(selected_groups).drop_duplicates("stratum")
    
    # Mark selection reason for color coding
    labeled_strata["selection_reason"] = ""
    for idx, row in labeled_strata.iterrows():
        reasons = []
        if row["stratum"] in strata.nlargest(n_extremes, "warmth_centered")["stratum"].values:
            reasons.append("High W")
        if row["stratum"] in strata.nsmallest(n_extremes, "warmth_centered")["stratum"].values:
            reasons.append("Low W")
        if row["stratum"] in strata.nlargest(n_extremes, "competence_centered")["stratum"].values:
            reasons.append("High C")
        if row["stratum"] in strata.nsmallest(n_extremes, "competence_centered")["stratum"].values:
            reasons.append("Low C")
        if row["warmth_significant"] and row["competence_significant"]:
            reasons.append("Sig both")
        if additional_strata is not None and row["stratum"] in additional_strata:
            reasons.append("Custom")
        if label_single_identity and "n_dimensions" in row and row["n_dimensions"] == 1:
            reasons.append("Single-ID")
        labeled_strata.at[idx, "selection_reason"] = ", ".join(reasons)
    
    print(f"📌 Labeled strata: {len(labeled_strata)}")
    print(f"   - Significant in warmth: {labeled_strata['warmth_significant'].sum()}")
    print(f"   - Significant in competence: {labeled_strata['competence_significant'].sum()}")
    print(f"   - Significant in both: {(labeled_strata['warmth_significant'] & labeled_strata['competence_significant']).sum()}")
    
    # ---------------------------------------------------
    # 4. Clean labels (abbreviated)
    # ---------------------------------------------------
    def clean_stratum_label(s):
        """Clean and abbreviate stratum labels"""
        abbr = {
            "woman": "Woman", "man": "Man", "nonbinary": "NB",
            "white": "White", "Black": "Black", "asian": "Asian",
            "muslim": "Muslim", "christian": "Christian", "jewish": "Jewish",
            "cis": "cis", "trans": "trans"
        }
        
        # Remove "not_mentioned" variants and split
        s = s.replace("-", "_").replace(" ", "_")
        parts = [p for p in s.split("_") if "not" not in p and "mention" not in p and p]
        
        # Order: transness → race → religion → gender
        ordered = []
        
        for p in parts:
            if p in ["white", "Black", "asian"]:
                ordered.append(abbr.get(p, p))
        for p in parts:
            if p in ["muslim", "christian", "jewish"]:
                ordered.append(abbr.get(p, p))
        for p in parts:
            if p in ["cis", "trans"]:
                ordered.append(abbr.get(p, p))
        for p in parts:
            if p in ["woman", "man", "nonbinary"]:
                ordered.append(abbr.get(p, p))
        
        return "-".join(ordered) if ordered else s
    
    labeled_strata["label"] = labeled_strata["stratum"].apply(clean_stratum_label)
    
    # ---------------------------------------------------
    # 5. Create plot with extra padding and high quality settings
    # ---------------------------------------------------
    # NEW: Set high-quality matplotlib parameters
    import matplotlib as mpl
    
    # Store original settings to restore later
    original_dpi = mpl.rcParams['figure.dpi']
    original_facecolor = mpl.rcParams['figure.facecolor']
    original_edgecolor = mpl.rcParams['figure.edgecolor']
    
    # Set high-quality parameters
    mpl.rcParams['figure.dpi'] = dpi
    mpl.rcParams['figure.facecolor'] = 'white'
    mpl.rcParams['figure.edgecolor'] = 'white'
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
    mpl.rcParams['axes.linewidth'] = 1.2
    mpl.rcParams['axes.spines.left'] = True
    mpl.rcParams['axes.spines.bottom'] = True
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    
    fig, ax = plt.subplots(figsize=(16, 12))  # Large figure for high quality
    
    # Quadrant lines
    ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
    
    # ---------------------------------------------------
    # 6. Plot points with smart color coding
    # ---------------------------------------------------
    
    # Background: all unlabeled strata
    if show_all_points:
        unlabeled = strata[~strata["stratum"].isin(labeled_strata["stratum"])]
        ax.scatter(
            unlabeled["warmth_centered"],
            unlabeled["competence_centered"],
            s=45, alpha=0.25, color="lightgray",
            edgecolor="gray", linewidth=0.5,
            zorder=2, label=f"Other strata (n={len(unlabeled)})"
        )
    
    # Labeled groups - color by significance
    colors = []
    for _, row in labeled_strata.iterrows():
        if row["warmth_significant"] and row["competence_significant"]:
            colors.append("#FF4500")  # Red-orange: significant in BOTH
        elif row["warmth_significant"] or row["competence_significant"]:
            colors.append("#FFD700")  # Gold: significant in ONE
        else:
            colors.append("#87CEEB")  # Light blue: not significant (just extreme)
    
    ax.scatter(
        labeled_strata["warmth_centered"],
        labeled_strata["competence_centered"],
        s=200, marker="D",  # Slightly larger for clarity
        c=colors,
        edgecolor="black",
        linewidth=1.8,  # Thicker edges for print quality
        alpha=0.95,
        zorder=4
    )
    
    # ---------------------------------------------------
    # 7. Smart label positioning with adjustText
    # ---------------------------------------------------
    
    # Add padding to axes for label space
    x_lim = ax.get_xlim()
    y_lim = ax.get_ylim()
    x_range = x_lim[1] - x_lim[0]
    y_range = y_lim[1] - y_lim[0]
    
    ax.set_xlim(x_lim[0] - 0.15*x_range, x_lim[1] + 0.15*x_range)
    ax.set_ylim(y_lim[0] - 0.15*y_range, y_lim[1] + 0.15*y_range)
    
    print(f"\n📍 Placing {len(labeled_strata)} labels with adjustText...")
    
    # Create text objects for all labels
    texts = []
    for _, row in labeled_strata.iterrows():
        x = row["warmth_centered"]
        y = row["competence_centered"]
        label_text = row["label"]
        
        # Create text object with white halo for readability
        txt = ax.text(
            x, y, label_text,
            fontsize=14,
            weight="normal",
            ha='center',
            va='center',
            path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
            zorder=5
        )
        texts.append(txt)
    
    # Use adjustText to optimize positions automatically
    adjust_text(
        texts,
        ax=ax,
        arrowprops=dict(arrowstyle='->', color='black', lw=0.8, alpha=0.7),
        expand_points=(1.5, 1.5),      # Space around data points
        expand_text=(1.2, 1.2),        # Space around text boxes
        force_points=(0.4, 0.4),       # Repulsion from data points
        force_text=(0.6, 0.6),         # Repulsion between text labels
        lim=1000,                      # Maximum iterations
        precision=0.001,               # Convergence precision
        time_lim=10,                   # Time limit in seconds
        only_move={'points': 'xy', 'text': 'xy'},
        avoid_self=True,               # Avoid overlap with own point
        save_steps=False,
        save_prefix='',
        save_format='png'
    )
    
    print(f"✅ All {len(texts)} labels placed with automatic collision avoidance")
    
    # ---------------------------------------------------
    # 8. Labels, title, legend with enhanced styling
    # ---------------------------------------------------
    ax.set_xlabel("Warmth (centered at grand mean)", fontsize=16, weight="bold")
    ax.set_ylabel("Competence (centered at grand mean)", fontsize=16, weight="bold")
    ax.set_title(
        f"{model_name}: Intersectional Groups in Stereotype Content Model Space\n"
        f"MAIHDA Model 1B: Extremes & Significant Random Effects" + 
        (f" (n≥{min_stratum_size})" if min_stratum_size > 0 else ""),
        fontsize=18, weight="bold", pad=25
    )
    
    # Custom legend with enhanced styling
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#FF4500',
               markersize=14, markeredgecolor='black', markeredgewidth=1.5,
               label='Sig. in both W & C'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#FFD700',
               markersize=14, markeredgecolor='black', markeredgewidth=1.5,
               label='Sig. in W or C'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#87CEEB',
               markersize=14, markeredgecolor='black', markeredgewidth=1.5,
               label='Extreme (not sig.)'),
    ]
    if show_all_points:
        legend_elements.append(
            Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=10, markeredgecolor='gray', 
                   label=f'Other strata (n={len(strata)-len(labeled_strata)})')
        )
    
    ax.legend(handles=legend_elements, loc="lower right", framealpha=0.95, 
              fontsize=13, frameon=True, fancybox=True, shadow=True)
    ax.grid(alpha=0.25, zorder=0)
    
    # Enhanced tick parameters
    ax.tick_params(axis='both', which='major', labelsize=12, width=1.2, length=6)
    
    plt.tight_layout()
    
    # ---------------------------------------------------
    # 9. HIGH-QUALITY SAVING
    # ---------------------------------------------------
    saved_files = []
    
    # Save to specific path if provided
    if save_path:
        try:
            # Determine format from extension
            ext = Path(save_path).suffix.lower().lstrip('.')
            save_kwargs = {
                'bbox_inches': 'tight',
                'facecolor': 'white',
                'edgecolor': 'none',
                'pad_inches': 0.2
            }
            
            if ext in ['png', 'tiff', 'jpg', 'jpeg']:
                save_kwargs['dpi'] = dpi
            
            plt.savefig(save_path, **save_kwargs)
            saved_files.append(save_path)
            print(f"💾 Figure saved to: {save_path}")
        except Exception as e:
            print(f"❌ Failed to save to {save_path}: {e}")
    
    # Auto-save in multiple formats if enabled
    if auto_save:
        # Create filename base
        safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_model_name = safe_model_name.replace(' ', '_')
        filename_base = f"intersectional_bias_space_{safe_model_name}"
        
        print(f"\n💾 AUTO-SAVING HIGH-QUALITY PLOTS...")
        print(f"   Directory: {save_dir}")
        print(f"   Formats: {', '.join(save_formats)}")
        print(f"   DPI: {dpi} (for raster formats)")
        
        for fmt in save_formats:
            try:
                file_path = save_dir / f"{filename_base}.{fmt}"
                
                # Format-specific settings
                save_kwargs = {
                    'bbox_inches': 'tight',
                    'facecolor': 'white',
                    'edgecolor': 'none',
                    'pad_inches': 0.2
                }
                
                if fmt in ['png', 'tiff', 'jpg', 'jpeg']:
                    save_kwargs['dpi'] = dpi
                elif fmt == 'pdf':
                    save_kwargs['dpi'] = 600  # High resolution for PDF
                elif fmt == 'svg':
                    save_kwargs['format'] = 'svg'
                elif fmt == 'eps':
                    save_kwargs['format'] = 'eps'
                
                plt.savefig(file_path, **save_kwargs)
                saved_files.append(str(file_path))
                print(f"   ✅ Saved: {file_path.name}")
                
            except Exception as e:
                print(f"   ❌ Failed to save {fmt}: {e}")
        
        print(f"\n📁 Total files saved: {len(saved_files)}")
    
    # Restore original matplotlib settings
    mpl.rcParams['figure.dpi'] = original_dpi
    mpl.rcParams['figure.facecolor'] = original_facecolor
    mpl.rcParams['figure.edgecolor'] = original_edgecolor
    
    plt.show()
    
    # ---------------------------------------------------
    # 10. Print summary table
    # ---------------------------------------------------
    print("\n" + "="*80)
    print("📋 LABELED INTERSECTIONAL GROUPS SUMMARY")
    print("="*80)
    
    summary_table = labeled_strata[[
        "label", "warmth_centered", "competence_centered", 
        "warmth_significant", "competence_significant", "selection_reason"
    ]].sort_values("warmth_centered", ascending=False)
    
    print(summary_table.to_string(index=False))
    print("="*80 + "\n")
    
    return strata, labeled_strata
    
  


# ======================================================
# BONUS: Compare additive vs. intersectional predictions
# ======================================================

def identify_non_additive_groups(
    res_warmth_maihda,
    res_comp_maihda,
    top_n=10
):
    """
    Identify groups showing strongest NON-ADDITIVE intersectional effects.
    
    Returns groups with significant random effects that deviate most from
    additive predictions.
    """
    
    # Extract random effects
    warm_re = res_warmth_maihda["random_effects"]
    comp_re = res_comp_maihda["random_effects"]
    
    # Merge
    re_combined = warm_re.merge(
        comp_re, 
        on="stratum", 
        suffixes=("_warmth", "_competence")
    )
    
    # Calculate "intersectionality score" (magnitude of random effects)
    re_combined["intersectionality_score"] = np.sqrt(
        re_combined["random_intercept_warmth"]**2 + 
        re_combined["random_intercept_competence"]**2
    )
    
    # Filter: significant in at least one dimension
    re_combined["any_significant"] = (
        re_combined["significant_warmth"] | re_combined["significant_competence"]
    )
    
    significant_groups = re_combined[re_combined["any_significant"]].copy()
    top_non_additive = significant_groups.nlargest(top_n, "intersectionality_score")
    
    print("\n" + "="*80)
    print(f"🎯 TOP {top_n} NON-ADDITIVE INTERSECTIONAL GROUPS")
    print("="*80)
    print("(Groups where random effects show strongest deviations from additive model)\n")
    
    for _, row in top_non_additive.iterrows():
        print(f"📍 {row['stratum']}")
        print(f"   Warmth random effect: {row['random_intercept_warmth']:+.3f} "
              f"({'SIG' if row['significant_warmth'] else 'ns'})")
        print(f"   Competence random effect: {row['random_intercept_competence']:+.3f} "
              f"({'SIG' if row['significant_competence'] else 'ns'})")
        print(f"   Intersectionality score: {row['intersectionality_score']:.3f}\n")
    
    print("="*80 + "\n")
    
    return top_non_additive

import warnings
import pandas as pd
import numpy as np
from statsmodels.tools.sm_exceptions import ConvergenceWarning


def bootstrap_random_intercepts_from_raw(
    df_raw: pd.DataFrame,
    response_var: str,  # "warmth_score" oder "competence_score"
    fixed_effects=("race", "gender", "religion", "transness"),
    stratum_col="stratum",
    concept_col="concept",
    response_col_raw="response",
    score_groupby_col=None,     # default: stratum_col
    keep_cols=None,             # default: fixed_effects
    B=1500,
    alpha=0.05,
    reml=True,
    optimizer="powell",
    disp=False,
    seed=None,
    verbose=True,
    use_absolute=False,
    resample_within=None,       # default: (stratum_col, concept_col)
    dropna_in_fit=True,         # robust gegen Patsy/groups mismatch
):
    """
    Bootstrap-CIs für Random Intercepts (MAIHDA Random Intercepts),
    wobei Unsicherheit aus LLM-Stochastik über Rohantworten propagiert wird:

      (1) Resample Rohantworten innerhalb resample_within (z.B. stratum × concept)
      (2) Recompute SCM Scores (warmth/competence)
      (3) Fit Mixed Model auf Scores
      (4) Extract random intercepts und berechne Bootstrap-CIs

    Returns
    -------
    results_df, base_model, base_scores_df
    """

    # -----------------------
    # Defaults & checks
    # -----------------------
    if score_groupby_col is None:
        score_groupby_col = stratum_col

    if resample_within is None:
        resample_within = (stratum_col, concept_col)

    if keep_cols is None:
        keep_cols = list(fixed_effects)

    needed_cols = {stratum_col, concept_col, response_col_raw, *resample_within, *keep_cols}
    missing = needed_cols - set(df_raw.columns)
    if missing:
        raise ValueError(f"df_raw missing required columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    
    def _as_scalar_random_intercept(val):
        # statsmodels liefert je nach Fall float, array, Series
        if isinstance(val, (pd.Series, pd.DataFrame)):
            return float(val.iloc[0])
        arr = np.asarray(val)
        return float(arr.ravel()[0]) if arr.ndim > 0 else float(arr)


    # Optional: nur benötigte Spalten behalten (schneller, weniger RAM)
    use_cols = sorted(set([stratum_col, concept_col, response_col_raw, *resample_within, *keep_cols]))
    df0 = df_raw.loc[:, use_cols].copy()

    # -----------------------
    # Precompute group indices for fast resampling
    # -----------------------
    grouped = df0.groupby(list(resample_within), sort=False)

    # grouped.indices.values() kann je nach Pandas-Version Index oder ndarray liefern.
    group_index_arrays = [np.asarray(idx, dtype=int) for idx in grouped.indices.values()]
    group_sizes = np.array([len(a) for a in group_index_arrays], dtype=int)

    if (group_sizes < 1).any():
        raise ValueError("Found an empty group in resample_within grouping (should not happen).")

    # -----------------------
    # Score function wrapper
    # -----------------------
    def _scores_from_raw(df_in: pd.DataFrame) -> pd.DataFrame:
        return compute_scm_scores(
            df_in,
            response_col=response_col_raw,
            concept_col=concept_col,
            groupby_col=score_groupby_col,
            keep_cols=keep_cols,
            use_absolute=use_absolute,
        )

    # -----------------------
    # Fit wrapper (robust against NA mismatches)
    # -----------------------
    def _fit_on_scores(df_scores: pd.DataFrame):
        fe_cols = [f for f in fixed_effects if f in df_scores.columns]
        if len(fe_cols) == 0:
            raise ValueError(
                "No fixed effects columns found in df_scores for formula. "
                f"fixed_effects={fixed_effects}, df_scores columns={list(df_scores.columns)}"
            )

        fixed_formula = " + ".join([f"C({f})" for f in fe_cols])
        formula = f"{response_var} ~ {fixed_formula}"

        if dropna_in_fit:
            needed = [response_var, stratum_col] + fe_cols
            df_fit = df_scores.dropna(subset=needed).reset_index(drop=True)
        else:
            df_fit = df_scores

        return _fit_mixed(formula, df_fit, stratum_col, reml=reml, optimizer=optimizer, disp=disp)

    # -----------------------
    # Base scores + base model
    # -----------------------
    base_scores = _scores_from_raw(df0)

    if response_var not in base_scores.columns:
        raise ValueError(
            f"response_var='{response_var}' not in scores df. "
            f"Available: {list(base_scores.columns)}"
        )
    if stratum_col not in base_scores.columns:
        raise ValueError(
            f"stratum_col='{stratum_col}' not in base_scores. "
            "Check score_groupby_col/keep_cols and your compute_scm_scores grouping."
        )

    original_strata = pd.Index(base_scores[stratum_col].unique()).sort_values()
    n_strata = len(original_strata)

    if verbose:
        print("\n🔄 BOOTSTRAP (FAST RAW → SCORES → MIXED) RANDOM INTERCEPTS")
        print(f"   Response (scores): {response_var}")
        print(f"   Resample within: {resample_within}")
        print(f"   Strata: {n_strata}")
        print(f"   Bootstrap samples: {B}")
        print(f"   Confidence level: {100*(1-alpha):.1f}%")
        print("   Fitting base model on base scores...")

    base_model = _fit_on_scores(base_scores)

    base_re_dict = base_model.random_effects
    original_re = np.array([_as_scalar_random_intercept(base_re_dict[s]) for s in original_strata], dtype=float)


    # -----------------------
    # Bootstrap loop (fast)
    # -----------------------
    boots = np.empty((B, n_strata), dtype=float)
    fail = 0

    if verbose:
        print("   Running bootstrap...")

    for b in range(B):
        if verbose and (b + 1) % max(1, B // 10) == 0:
            print(f"   Progress: {b + 1}/{B} ({100*(b+1)/B:.1f}%)")

        try:
            # 1) Resample indices within each group (no groupby.apply)
            sampled_idx = np.concatenate([
                rng.choice(idx, size=len(idx), replace=True)
                for idx in group_index_arrays
            ])

            # 2) Build bootstrap raw df
            df_star = df0.iloc[sampled_idx].reset_index(drop=True)

            # 3) Recompute scores
            scores_star = _scores_from_raw(df_star)

            # 4) Fit model
            fit_b = _fit_on_scores(scores_star)

            # 5) Extract random intercepts in fixed order
            re_dict = fit_b.random_effects
            for i, s in enumerate(original_strata):
                boots[b, i] = _as_scalar_random_intercept(re_dict[s])

        except Exception as e:
            if verbose and fail == 0:
                print(f"   Warning: Bootstrap iteration {b+1} failed: {e}")
            boots[b, :] = np.nan
            fail += 1

    valid_mask = ~np.isnan(boots).any(axis=1)
    boots_clean = boots[valid_mask]

    if boots_clean.size == 0:
        raise RuntimeError(
            f"All {B} bootstrap iterations failed. "
            "Try different optimizer, reduce fixed effects, or inspect missingness."
        )

    if verbose:
        print(f"   Bootstrap completed: {boots_clean.shape[0]}/{B} successful iterations")
        if fail > 0:
            print(f"   Failed iterations: {fail}")

    # -----------------------
    # Bootstrap stats
    # -----------------------
    boot_mean = boots_clean.mean(axis=0)
    boot_se = boots_clean.std(axis=0, ddof=1)
    boot_ci_lower = np.percentile(boots_clean, 100 * alpha / 2, axis=0)
    boot_ci_upper = np.percentile(boots_clean, 100 * (1 - alpha / 2), axis=0)

    results_df = pd.DataFrame({
        stratum_col: original_strata,
        "random_intercept": original_re,
        "boot_mean": boot_mean,
        "boot_se": boot_se,
        "boot_ci_lower": boot_ci_lower,
        "boot_ci_upper": boot_ci_upper,
        "significant": ~((boot_ci_lower <= 0) & (boot_ci_upper >= 0)),
        "B_used": boots_clean.shape[0],
        "B_failed": fail,
    })

    results_df = results_df.reindex(
        results_df["random_intercept"].abs().sort_values(ascending=False).index
    )

    if verbose:
        n_sig = int(results_df["significant"].sum())
        print(f"   Results: {n_sig}/{n_strata} strata have significant random intercepts")
        print("\n   Top 5 |random_intercept|:")
        for _, row in results_df.head(5).iterrows():
            sig_mark = "***" if row["significant"] else "   "
            sname = str(row[stratum_col])
            print(f"   {sig_mark} {sname[:40]:40} {row['random_intercept']:8.4f} "
                  f"[{row['boot_ci_lower']:7.4f}, {row['boot_ci_upper']:7.4f}]")

    return results_df, base_model, base_scores


import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

def _fit_mixedlm_robust(formula, df, group_col,
                        reml=True,
                        optimizers=("lbfgs", "powell", "nm"),
                        disp=False):
    last_err = None
    for opt in optimizers:
        try:
            res = smf.mixedlm(
                formula,
                df,
                groups=df[group_col]
            ).fit(reml=reml, method=opt, disp=disp)

            if hasattr(res, "converged") and res.converged:
                return res
            last_err = RuntimeError(f"Non-converged with {opt}")
        except Exception as e:
            last_err = e
    raise last_err


def build_fixed_formula(fixed_effects, reference="not_mentioned"):
    return " + ".join(
        f"C({c}, Treatment(reference='{reference}'))"
        for c in fixed_effects
    )

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def bootstrap_ols_from_raw_classic(
    df_raw: pd.DataFrame,
    response_dim: str,  # "warmth_score" oder "competence_score"
    fixed_effects=("race", "gender", "religion", "transness"),
    item_col="item_id",
    stratum_col="vignette_id",
    rep_col="response_index",
    response_col="response",
    concept_col="concept",
    B=1000,
    alpha=0.05,
    seed=42,
    use_absolute=False,
    dropna_in_fit=True,
    fill_not_mentioned=True,
    not_mentioned_label="not_mentioned",
    verbose=True,
    check_rep_counts=True,
    rep_count_min=20,
):
    """
    Klassisches (nicht-parametrisches) Bootstrap für OLS:
    - Resample MIT replacement innerhalb jeder (stratum × item)-Zelle über LLM-Replikationen
    - Zellmittelwerte -> compute_scm_scores -> OLS auf Stratum-Level
    - Unsicherheit = LLM-Stochastik (Reps), Strata & Items fix
    """

    rng = np.random.default_rng(seed)

    # -----------------------
    # 0) Cleaning / Typen
    # -----------------------
    df0 = df_raw.copy()

    # response muss numerisch sein (niemals Strings)
    df0[response_col] = pd.to_numeric(df0[response_col], errors="coerce")
    df0 = df0.dropna(subset=[response_col, stratum_col, item_col]).reset_index(drop=True)

    if fill_not_mentioned:
        for c in fixed_effects:
            if c in df0.columns:
                df0[c] = df0[c].fillna(not_mentioned_label)

    # NumPy view für speed & korrektes positionsbasiertes Indexing
    y = df0[response_col].to_numpy()

    # -----------------------
    # 1) item -> concept Mapping (m:1) sichern
    # -----------------------
    concept_nunique = df0.groupby(item_col)[concept_col].nunique()
    bad_items = concept_nunique[concept_nunique > 1]
    if len(bad_items) > 0:
        ex = bad_items.index[0]
        raise ValueError(
            f"{len(bad_items)} item_ids haben mehrere concepts. Beispiel: {ex} -> {bad_items.iloc[0]}"
        )

    item2concept = df0[[item_col, concept_col]].drop_duplicates(subset=[item_col]).copy()

    # -----------------------
    # 2) Fixed effects pro Stratum extrahieren
    # -----------------------
    fe_df = (
        df0.groupby(stratum_col)[list(fixed_effects)]
           .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else not_mentioned_label)
           .reset_index()
    )

    # -----------------------
    # 3) Zell-Indices vorbereiten (positionsbasiert!)
    # -----------------------
    grouped = df0.groupby([stratum_col, item_col], sort=False)
    cell_indices = grouped.indices  # dict: (s,i) -> np.ndarray positions

    if check_rep_counts:
        if rep_col in df0.columns:
            rep_counts = df0.groupby([stratum_col, item_col])[rep_col].nunique()
        else:
            rep_counts = df0.groupby([stratum_col, item_col])[response_col].size()
        low = rep_counts[rep_counts < rep_count_min]
        if len(low) > 0:
            ex = low.index[0]
            raise ValueError(
                f"Einige (stratum,item)-Zellen haben zu wenige Reps (<{rep_count_min}). "
                f"Beispiel: {ex} hat {int(low.iloc[0])}."
            )

    keys = list(cell_indices.keys())
    idx_arrays = [cell_indices[k] for k in keys]
    sizes = np.array([len(a) for a in idx_arrays], dtype=int)

    # -----------------------
    # 4) Base: Zellmittel aus allen Reps -> Scores -> Base-OLS
    # -----------------------
    cell_means = (
        df0.groupby([stratum_col, item_col], as_index=False)[response_col]
           .mean()
    )

    df_items_base = (
        cell_means
        .merge(item2concept, on=item_col, how="left", validate="m:1")
        .merge(fe_df, on=stratum_col, how="left", validate="m:1")
    )

    if df_items_base[concept_col].isna().any():
        n_na = int(df_items_base[concept_col].isna().sum())
        raise RuntimeError(f"Base merge erzeugt {n_na} NA concepts. Prüfe item_id dtype / merge keys.")

    base_scores = compute_scm_scores(
        df_items_base,
        response_col=response_col,
        concept_col=concept_col,
        groupby_col=stratum_col,
        keep_cols=list(fixed_effects),
        use_absolute=use_absolute
    )

    fe_cols = [c for c in fixed_effects if c in base_scores.columns]
    fixed_formula = build_fixed_formula(fixed_effects)
    formula = f"{response_dim} ~ {fixed_formula}"

    df_fit_base = base_scores.copy()
    if dropna_in_fit:
        df_fit_base = df_fit_base.dropna(subset=[response_dim, stratum_col] + fe_cols).reset_index(drop=True)

    if df_fit_base.empty:
        raise RuntimeError("Base-Fit DF ist leer (nach dropna). Prüfe response_dim/FE-Spalten.")

    # Kategorien-Level fixieren (stabile Dummies)
    levels = {}
    for c in fe_cols:
        df_fit_base[c] = df_fit_base[c].astype("category")
        levels[c] = df_fit_base[c].cat.categories

    base_fit = smf.ols(formula, data=df_fit_base).fit()
    coef_names = list(base_fit.params.index)

    # -----------------------
    # 5) Bootstrap-Loop
    # -----------------------
    boot = np.empty((B, len(coef_names)), dtype=float)
    fail = 0
    n_empty_fit = 0
    n_missing_coef = 0

    if verbose:
        print(f"OLS classic cell-bootstrap: B={B}, cells={len(keys)}, strata={df0[stratum_col].nunique()}, items={df0[item_col].nunique()}")
        print("Formula:", formula)

    for b in range(B):
        if verbose and (b + 1) % max(1, B // 10) == 0:
            print(f"Bootstrap {b + 1}/{B}")

        try:
            means = np.empty(len(keys), dtype=float)

            # schnell & korrekt: positionsbasiert über NumPy
            for j, idx in enumerate(idx_arrays):
                draw = rng.choice(idx, size=sizes[j], replace=True)
                means[j] = y[draw].mean()

            df_items_b = pd.DataFrame({
                stratum_col: [k[0] for k in keys],
                item_col:    [k[1] for k in keys],
                response_col: means
            })

            df_items_b = (
                df_items_b
                .merge(item2concept, on=item_col, how="left", validate="m:1")
                .merge(fe_df, on=stratum_col, how="left", validate="m:1")
            )

            if df_items_b[concept_col].isna().any():
                n_na = int(df_items_b[concept_col].isna().sum())
                raise RuntimeError(f"Missing concept after merge: {n_na}")

            scores_b = compute_scm_scores(
                df_items_b,
                response_col=response_col,
                concept_col=concept_col,
                groupby_col=stratum_col,
                keep_cols=list(fixed_effects),
                use_absolute=use_absolute
            )

            if response_dim not in scores_b.columns:
                raise RuntimeError(f"{response_dim} fehlt in scores_b. Spalten: {list(scores_b.columns)}")

            df_fit_b = scores_b
            if dropna_in_fit:
                df_fit_b = df_fit_b.dropna(subset=[response_dim, stratum_col] + fe_cols).reset_index(drop=True)

            if df_fit_b.empty:
                n_empty_fit += 1
                raise RuntimeError("df_fit_b empty after dropna")

            # gleiche Kategorien-Level erzwingen
            for c in fe_cols:
                df_fit_b[c] = pd.Categorical(df_fit_b[c], categories=levels[c])

            fit_b = smf.ols(formula, data=df_fit_b).fit()

            # fehlende Koeffizienten hart als Fail zählen (sonst boot-row wird NaN und später gelöscht)
            params = fit_b.params
            missing = [t for t in coef_names if t not in params.index]
            if missing:
                n_missing_coef += 1
                raise RuntimeError(f"Missing coef(s) in this iteration: {missing[:5]}{'...' if len(missing) > 5 else ''}")

            boot[b, :] = params.reindex(coef_names).to_numpy()

        except Exception as e:
            if verbose and fail == 0:
                print("First bootstrap failure (OLS):", repr(e))
            boot[b, :] = np.nan
            fail += 1

    ok = ~np.isnan(boot).any(axis=1)
    boot = boot[ok]

    if boot.shape[0] == 0:
        raise RuntimeError("Alle Bootstrap-Iterationen fehlgeschlagen (OLS).")

    # -----------------------
    # 6) Percentile CI
    # -----------------------
    ci = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

    out = pd.DataFrame({
        "term": coef_names,
        "estimate": base_fit.params.reindex(coef_names).to_numpy(),
        "ci_lower": ci[0],
        "ci_upper": ci[1],
        "significant": ~((ci[0] <= 0) & (ci[1] >= 0)),
        "B_used": int(boot.shape[0]),
        "B_failed": int(fail),
        "n_empty_fit": int(n_empty_fit),
        "n_missing_coef": int(n_missing_coef),
    }).sort_values("term")

    summary = {
        "bootstrap_type": "classic_within_cell_resample",
        "model": "OLS",
        "formula": formula,
        "response_dim": response_dim,
        "B": int(B),
        "B_used": int(boot.shape[0]),
        "B_failed": int(fail),
        "alpha": float(alpha),
        "n_cells": int(len(keys)),
        "n_strata": int(df0[stratum_col].nunique()),
        "n_items": int(df0[item_col].nunique()),
        "n_empty_fit": int(n_empty_fit),
        "n_missing_coef": int(n_missing_coef),
    }

    return {
        "ols": out,
        "base_fit": base_fit,
        "base_scores": base_scores,
        "summary": summary
    }


import warnings


import os

def _safe_mkdir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)


def _safe_to_csv(df, path):
    try:
        df.to_csv(path, index=False)
    except Exception as e:
        print(f"[WARN] Failed to save CSV {path}: {e}")


def _safe_to_excel(writer, df, sheet_name):
    try:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    except Exception as e:
        print(f"[WARN] Failed to save sheet {sheet_name}: {e}")





def bootstrap_maihda_from_raw_classic(
    df_raw: pd.DataFrame,
    response_dim: str,  # "warmth_score" oder "competence_score"
    fixed_effects=("race", "gender", "religion", "transness"),
    item_col="item_id",
    stratum_col="vignette_id",
    rep_col="response_index",
    response_col="response",
    concept_col="concept",
    B=1000,
    alpha=0.05,
    seed=42,
    reml=True,
    optimizers=("lbfgs", "powell", "nm"),
    use_absolute=False,
    dropna_in_fit=True,
    verbose=True,
    check_rep_counts=True,
    rep_count_min=20,
    fill_not_mentioned=True,
    not_mentioned_label="not_mentioned",
    require_converged=False,  # MixedLM in statsmodels hat nicht immer verlässlich converged Flag
):
    """
    Klassisches Bootstrap für MAIHDA:
    - Resample MIT replacement innerhalb jeder (stratum × item)-Zelle über LLM-Reps
    - Zellmittelwerte -> compute_scm_scores -> MixedLM random intercept pro Stratum
    """

    rng = np.random.default_rng(seed)

    # -----------------------
    # 0) Cleaning / Typen
    # -----------------------
    df0 = df_raw.copy()
    df0[response_col] = pd.to_numeric(df0[response_col], errors="coerce")
    df0 = df0.dropna(subset=[response_col, stratum_col, item_col]).reset_index(drop=True)

    if fill_not_mentioned:
        for c in fixed_effects:
            if c in df0.columns:
                df0[c] = df0[c].fillna(not_mentioned_label)

    y = df0[response_col].to_numpy()

    # -----------------------
    # 1) item -> concept Mapping (m:1) sichern
    # -----------------------
    concept_nunique = df0.groupby(item_col)[concept_col].nunique()
    bad_items = concept_nunique[concept_nunique > 1]
    if len(bad_items) > 0:
        ex = bad_items.index[0]
        raise ValueError(
            f"{len(bad_items)} item_ids haben mehrere concepts. Beispiel: {ex} -> {bad_items.iloc[0]}"
        )

    item2concept = df0[[item_col, concept_col]].drop_duplicates(subset=[item_col]).copy()

    # -----------------------
    # 2) Fixed effects pro Stratum extrahieren
    # -----------------------
    fe_df = (
        df0.groupby(stratum_col)[list(fixed_effects)]
           .agg(lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else not_mentioned_label)
           .reset_index()
    )

    # -----------------------
    # 3) Zell-Indices vorbereiten
    # -----------------------
    grouped = df0.groupby([stratum_col, item_col], sort=False)
    cell_indices = grouped.indices

    if check_rep_counts:
        if rep_col in df0.columns:
            rep_counts = df0.groupby([stratum_col, item_col])[rep_col].nunique()
        else:
            rep_counts = df0.groupby([stratum_col, item_col])[response_col].size()
        low = rep_counts[rep_counts < rep_count_min]
        if len(low) > 0:
            ex = low.index[0]
            raise ValueError(
                f"Einige (stratum,item)-Zellen haben zu wenige Reps (<{rep_count_min}). "
                f"Beispiel: {ex} hat {int(low.iloc[0])}."
            )

    keys = list(cell_indices.keys())
    idx_arrays = [cell_indices[k] for k in keys]
    sizes = np.array([len(a) for a in idx_arrays], dtype=int)

    # -----------------------
    # 4) Base: Zellmittel -> Scores -> MixedLM
    # -----------------------
    cell_means = (
        df0.groupby([stratum_col, item_col], as_index=False)[response_col]
           .mean()
    )

    df_items_base = (
        cell_means
        .merge(item2concept, on=item_col, how="left", validate="m:1")
        .merge(fe_df, on=stratum_col, how="left", validate="m:1")
    )

    if df_items_base[concept_col].isna().any():
        n_na = int(df_items_base[concept_col].isna().sum())
        raise RuntimeError(f"Base merge erzeugt {n_na} NA concepts. Prüfe item_id dtype / merge keys.")

    base_scores = compute_scm_scores(
        df_items_base,
        response_col=response_col,
        concept_col=concept_col,
        groupby_col=stratum_col,
        keep_cols=list(fixed_effects),
        use_absolute=use_absolute
    )

    fe_cols = [c for c in fixed_effects if c in base_scores.columns]
    fixed_formula = build_fixed_formula(fixed_effects)
    formula = f"{response_dim} ~ {fixed_formula}"

    df_fit_base = base_scores.copy()
    if dropna_in_fit:
        df_fit_base = df_fit_base.dropna(subset=[response_dim, stratum_col] + fe_cols).reset_index(drop=True)

    if df_fit_base.empty:
        raise RuntimeError("Base-Fit DF ist leer (nach dropna). Prüfe response_dim/FE-Spalten.")

    # Kategorien-Level fixieren (stabilisiert Fixed-Effects Design)
    levels = {}
    for c in fe_cols:
        df_fit_base[c] = df_fit_base[c].astype("category")
        levels[c] = df_fit_base[c].cat.categories

    base_model = _fit_mixedlm_robust(
        formula, df_fit_base, group_col=stratum_col, reml=reml, optimizers=optimizers
    )

    strata = pd.Index(df_fit_base[stratum_col].unique()).sort_values()
    fe_names = list(base_model.fe_params.index)

    # -----------------------
    # 5) Bootstrap-Loop
    # -----------------------
    fe_boot = np.empty((B, len(fe_names)), dtype=float)
    re_boot = np.empty((B, len(strata)), dtype=float)
    fail = 0
    n_empty_fit = 0
    n_nonconverged = 0

    def _as_scalar_random_intercept(val):
        if isinstance(val, pd.Series):
            return float(val.iloc[0])
        arr = np.asarray(val)
        return float(arr.ravel()[0]) if arr.ndim > 0 else float(arr)

    if verbose:
        print(f"MAIHDA classic cell-bootstrap: B={B}, cells={len(keys)}, strata={df0[stratum_col].nunique()}, items={df0[item_col].nunique()}")
        print("Formula:", formula)

    for b in range(B):
        if verbose and (b + 1) % max(1, B // 10) == 0:
            print(f"Bootstrap {b + 1}/{B}")

        try:
            means = np.empty(len(keys), dtype=float)
            for j, idx in enumerate(idx_arrays):
                draw = rng.choice(idx, size=sizes[j], replace=True)
                means[j] = y[draw].mean()

            df_items_b = pd.DataFrame({
                stratum_col: [k[0] for k in keys],
                item_col:    [k[1] for k in keys],
                response_col: means
            })

            df_items_b = (
                df_items_b
                .merge(item2concept, on=item_col, how="left", validate="m:1")
                .merge(fe_df, on=stratum_col, how="left", validate="m:1")
            )

            if df_items_b[concept_col].isna().any():
                n_na = int(df_items_b[concept_col].isna().sum())
                raise RuntimeError(f"Missing concept after merge: {n_na}")

            scores_b = compute_scm_scores(
                df_items_b,
                response_col=response_col,
                concept_col=concept_col,
                groupby_col=stratum_col,
                keep_cols=list(fixed_effects),
                use_absolute=use_absolute
            )

            if response_dim not in scores_b.columns:
                raise RuntimeError(f"{response_dim} fehlt in scores_b. Spalten: {list(scores_b.columns)}")

            df_fit_b = scores_b
            if dropna_in_fit:
                df_fit_b = df_fit_b.dropna(subset=[response_dim, stratum_col] + fe_cols).reset_index(drop=True)

            if df_fit_b.empty:
                n_empty_fit += 1
                raise RuntimeError("df_fit_b empty after dropna")

            for c in fe_cols:
                df_fit_b[c] = pd.Categorical(df_fit_b[c], categories=levels[c])

            # MixedLM Fit – ConvergenceWarnings nicht automatisch fatal machen
            fit_b = _fit_mixedlm_robust(
                formula, df_fit_b, group_col=stratum_col, reml=reml, optimizers=optimizers
            )

            if require_converged and hasattr(fit_b, "converged") and (fit_b.converged is False):
                n_nonconverged += 1
                raise RuntimeError("non-converged")

            fe_boot[b, :] = fit_b.fe_params.reindex(fe_names).to_numpy()

            re_dict = fit_b.random_effects
            for i, s in enumerate(strata):
                re_boot[b, i] = _as_scalar_random_intercept(re_dict.get(s, 0.0))

        except Exception as e:
            if verbose and fail == 0:
                print("First bootstrap failure (MAIHDA):", repr(e))
            fe_boot[b, :] = np.nan
            re_boot[b, :] = np.nan
            fail += 1

    ok = ~np.isnan(fe_boot).any(axis=1) & ~np.isnan(re_boot).any(axis=1)
    fe_boot = fe_boot[ok]
    re_boot = re_boot[ok]

    if fe_boot.shape[0] == 0:
        raise RuntimeError("Alle Bootstrap-Iterationen fehlgeschlagen (MAIHDA).")

    # -----------------------
    # 6) Percentile CIs
    # -----------------------
    fe_ci = np.percentile(fe_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)
    fe_df_out = pd.DataFrame({
        "term": fe_names,
        "estimate": base_model.fe_params.reindex(fe_names).to_numpy(),
        "ci_lower": fe_ci[0],
        "ci_upper": fe_ci[1],
        "significant": ~((fe_ci[0] <= 0) & (fe_ci[1] >= 0)),
        "B_used": int(fe_boot.shape[0]),
        "B_failed": int(fail),
        "n_empty_fit": int(n_empty_fit),
        "n_nonconverged": int(n_nonconverged),
    }).sort_values("term")

    re_ci = np.percentile(re_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)], axis=0)

    base_re = base_model.random_effects
    re_hat = np.array([_as_scalar_random_intercept(base_re.get(s, 0.0)) for s in strata], dtype=float)

    re_df_out = pd.DataFrame({
        stratum_col: strata,
        "random_intercept": re_hat,
        "ci_lower": re_ci[0],
        "ci_upper": re_ci[1],
        "significant": ~((re_ci[0] <= 0) & (re_ci[1] >= 0)),
        "B_used": int(re_boot.shape[0]),
        "B_failed": int(fail),
        "n_empty_fit": int(n_empty_fit),
        "n_nonconverged": int(n_nonconverged),
    }).sort_values("random_intercept", key=lambda x: x.abs(), ascending=False)

    summary = {
        "bootstrap_type": "classic_within_cell_resample",
        "model": "MAIHDA",
        "formula": formula,
        "response_dim": response_dim,
        "B": int(B),
        "B_used": int(fe_boot.shape[0]),
        "B_failed": int(fail),
        "alpha": float(alpha),
        "n_cells": int(len(keys)),
        "n_strata_used": int(len(strata)),
        "n_items": int(df0[item_col].nunique()),
        "n_empty_fit": int(n_empty_fit),
        "n_nonconverged": int(n_nonconverged),
    }

    return {
        "fixed_effects": fe_df_out,
        "random_intercepts": re_df_out,
        "base_model": base_model,
        "base_scores": base_scores,
        "summary": summary,
    }

import os
import pandas as pd

def _collect_dataframes(obj, prefix=""):
    """
    Recursively collect (name, df) from nested dicts/lists/tuples.
    Returns list of (sheet_name, DataFrame).
    """
    out = []
    if isinstance(obj, pd.DataFrame):
        out.append((prefix or "data", obj))
        return out

    if isinstance(obj, dict):
        for k, v in obj.items():
            key = str(k)
            new_prefix = f"{prefix}_{key}" if prefix else key
            out.extend(_collect_dataframes(v, new_prefix))
        return out

    if isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}_{i}" if prefix else str(i)
            out.extend(_collect_dataframes(v, new_prefix))
        return out

    return out


def _safe_mkdir(path):
    if path is not None:
        os.makedirs(path, exist_ok=True)



def bootstrap_all_models_wrapper(
    df_raw,
    fixed_effects,
    item_col="item_id",
    stratum_col="vignette_id",
    rep_col="response_index",
    response_col="response",
    concept_col="concept",
    B=300,
    alpha=0.05,
    seed=42,
    use_absolute=False,
    dropna_in_fit=True,
    verbose=True,
    reml=True,
    optimizers=("lbfgs","powell","nm"),
    require_converged=True,
    fill_not_mentioned=True,
    not_mentioned_label="not_mentioned",
    # NEW ↓↓↓
    save_dir=None,
    save_prefix="bootstrap_results",
    save_excel=True,
    save_csv=True
):

    """
    Wrapper: führt 4 Bootstrap-Analysen aus (gemeinsames Raw-DF):
      - OLS Warmth
      - OLS Competence
      - MAIHDA Warmth
      - MAIHDA Competence

    continue_on_error=True speichert Fehler pro Block statt sofort abzubrechen.
    """

    seeds = {
        "ols_warmth": int(seed) + 1,
        "ols_competence": int(seed) + 2,
        "maihda_warmth": int(seed) + 3,
        "maihda_competence": int(seed) + 4,
    }

    results = {
        "meta": {
            "B": int(B),
            "alpha": float(alpha),
            "fixed_effects": tuple(fixed_effects),
            "use_absolute": bool(use_absolute),
            "dropna_in_fit": bool(dropna_in_fit),
            "seed_base": int(seed),
            "seeds": seeds,
            "columns": {
                "item_col": item_col,
                "stratum_col": stratum_col,
                "rep_col": rep_col,
                "response_col": response_col,
                "concept_col": concept_col,
            },
            "maihda": {
                "reml": bool(reml),
                "optimizers": tuple(optimizers),
                "require_converged": bool(require_converged),
            },
            "cleaning": {
                "fill_not_mentioned": bool(fill_not_mentioned),
                "not_mentioned_label": not_mentioned_label,
            },
        }
    }

    def _run(name, fn):
        try:
            out = fn()
            return {"ok": True, "result": out}
        except Exception as e:
            if verbose:
                print(f"[{name}] FAILED:", repr(e))
            if continue_on_error:
                return {"ok": False, "error": repr(e)}
            raise

    results["ols_warmth"] = _run(
        "ols_warmth",
        lambda: bootstrap_ols_from_raw_classic(
            df_raw=df_raw,
            response_dim="warmth_score",
            fixed_effects=fixed_effects,
            item_col=item_col,
            stratum_col=stratum_col,
            rep_col=rep_col,
            response_col=response_col,
            concept_col=concept_col,
            B=B,
            alpha=alpha,
            seed=seeds["ols_warmth"],
            use_absolute=use_absolute,
            dropna_in_fit=dropna_in_fit,
            fill_not_mentioned=fill_not_mentioned,
            not_mentioned_label=not_mentioned_label,
            verbose=verbose,
        )
    )

    results["ols_competence"] = _run(
        "ols_competence",
        lambda: bootstrap_ols_from_raw_classic(
            df_raw=df_raw,
            response_dim="competence_score",
            fixed_effects=fixed_effects,
            item_col=item_col,
            stratum_col=stratum_col,
            rep_col=rep_col,
            response_col=response_col,
            concept_col=concept_col,
            B=B,
            alpha=alpha,
            seed=seeds["ols_competence"],
            use_absolute=use_absolute,
            dropna_in_fit=dropna_in_fit,
            fill_not_mentioned=fill_not_mentioned,
            not_mentioned_label=not_mentioned_label,
            verbose=verbose,
        )
    )

    results["maihda_warmth"] = _run(
        "maihda_warmth",
        lambda: bootstrap_maihda_from_raw_classic(
            df_raw=df_raw,
            response_dim="warmth_score",
            fixed_effects=fixed_effects,
            item_col=item_col,
            stratum_col=stratum_col,
            rep_col=rep_col,
            response_col=response_col,
            concept_col=concept_col,
            B=B,
            alpha=alpha,
            seed=seeds["maihda_warmth"],
            reml=reml,
            optimizers=optimizers,
            use_absolute=use_absolute,
            dropna_in_fit=dropna_in_fit,
            fill_not_mentioned=fill_not_mentioned,
            not_mentioned_label=not_mentioned_label,
            require_converged=require_converged,
            verbose=verbose,
        )
    )

    results["maihda_competence"] = _run(
            "maihda_competence",
            lambda: bootstrap_maihda_from_raw_classic(
                df_raw=df_raw,
                response_dim="competence_score",
                fixed_effects=fixed_effects,
                item_col=item_col,
                stratum_col=stratum_col,
                rep_col=rep_col,
                response_col=response_col,
                concept_col=concept_col,
                B=B,
                alpha=alpha,
                seed=seeds["maihda_competence"],
                reml=reml,
                optimizers=optimizers,
                use_absolute=use_absolute,
                dropna_in_fit=dropna_in_fit,
                fill_not_mentioned=fill_not_mentioned,
                not_mentioned_label=not_mentioned_label,
                require_converged=require_converged,
                verbose=verbose,
            )
        )
        # --------------------------------------------------
    # Save results to disk (optional)
    # --------------------------------------------------
    # --------------------------------------------------
# Save results to disk (optional) — recursive
# --------------------------------------------------
    if save_dir is not None:
        _safe_mkdir(save_dir)

        dfs = _collect_dataframes(results)

        if verbose:
            print(f"[SAVE] Found {len(dfs)} DataFrames to export.")

        # ---------- Excel ----------
        if save_excel:
            excel_path = os.path.join(save_dir, f"{save_prefix}.xlsx")
            try:
                with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
                    used = set()
                    for name, df in dfs:
                        # Excel sheet name limit: 31 chars, must be unique
                        sheet = name[:31] if name else "data"
                        base = sheet
                        n = 1
                        while sheet in used:
                            suffix = f"_{n}"
                            sheet = (base[:31-len(suffix)] + suffix) if len(base) + len(suffix) > 31 else base + suffix
                            n += 1
                        used.add(sheet)

                        df.to_excel(writer, sheet_name=sheet, index=False)

                if verbose:
                    print(f"[OK] Results written to {excel_path}")
            except Exception as e:
                print(f"[WARN] Failed to write Excel file: {e}")

        # ---------- CSV ----------
        if save_csv:
            for name, df in dfs:
                fname = f"{save_prefix}_{name}.csv".replace(os.sep, "_")
                path = os.path.join(save_dir, fname)
                try:
                    df.to_csv(path, index=False)
                except Exception as e:
                    print(f"[WARN] Failed to save CSV {path}: {e}")


    return results
#


def format_coef_with_stars(coef, ci_lower, ci_upper, decimals=3):
    """
    Bootstrap-basierte Signifikanz:
    *  : 95%-CI schließt 0 aus
    ** : 99%-CI schließt 0 aus (wenn vorhanden)
    """
    if pd.isna(coef):
        return ""

    star = ""
    if ci_lower > 0 or ci_upper < 0:
        star = "*"

    return f"{coef:.{decimals}f}{star}"



def build_master_table(
    res_llama,
    res_qwen,
    res_mistral,
    key="ols",          # "ols" oder "fixed_effects"
    term_col="term",
    coef_col="estimate",
    ci_l="ci_lower",
    ci_u="ci_upper",
    decimals=3
):
    """
    Baut eine 3-LLM-Mastertabelle mit Sternen.
    Akzeptiert beide Rückgabeformen:
      A) {"ols": df, ...}
      B) {"ok": True, "result": {"ols": df, ...}}
    """

    import pandas as pd

    def _unwrap(res):
        # Fall B
        if isinstance(res, dict) and "result" in res and isinstance(res["result"], dict):
            return res["result"]
        # Fall A
        return res

    def format_coef_with_stars(coef, ci_lower, ci_upper):
        if pd.isna(coef):
            return ""
        star = "*" if (ci_lower > 0 or ci_upper < 0) else ""
        return f"{coef:.{decimals}f}{star}"

    def prep(res, name):
        res = _unwrap(res)

        if key not in res:
            raise KeyError(
                f"Key '{key}' not found. Available keys: {list(res.keys())}"
            )

        df = res[key]
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"res['{key}'] is not a DataFrame, got {type(df)}")

        out = df[[term_col, coef_col, ci_l, ci_u]].copy()
        out[name] = out.apply(
            lambda r: format_coef_with_stars(r[coef_col], r[ci_l], r[ci_u]),
            axis=1
        )
        return out[[term_col, name]]

    d1 = prep(res_llama, "llama")
    d2 = prep(res_qwen, "qwen")
    d3 = prep(res_mistral, "mistral")

    master = (
        d1.merge(d2, on=term_col, how="outer")
          .merge(d3, on=term_col, how="outer")
          .sort_values(term_col)
          .reset_index(drop=True)
    )
    return master



# ------------------------------------------------------------
# Helper: take wrapper block {"ok":..., "result":...} and return result dict
# ------------------------------------------------------------
def _unwrap_result(block: dict) -> dict:
    if not isinstance(block, dict):
        raise TypeError(f"Expected dict, got {type(block)}")
    if "result" in block and isinstance(block["result"], dict):
        return block["result"]
    return block

# ------------------------------------------------------------
# Helper: normalize vignette ids
#   - accepts "vignette_1"..."vignette_175" OR 1..175
#   - returns numeric id column
# ------------------------------------------------------------
def _make_vignette_id_num(series: pd.Series) -> pd.Series:
    # already numeric?
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64")

    s = series.astype(str)

    # common formats: "vignette_12", "vignette-12", "12"
    s = s.str.strip()
    s = s.str.replace("vignette", "", regex=False)
    s = s.str.replace("_", "", regex=False)
    s = s.str.replace("-", "", regex=False)

    # extract last integer just in case
    extracted = s.str.extract(r"(\d+)$")[0]
    return extracted.astype("Int64")

# ------------------------------------------------------------
# Core: merge vignette text onto random intercept bootstrap results
# Works for one model block (e.g., res_all_llama["maihda_warmth"])
# ------------------------------------------------------------
def attach_vignette_text_to_random_intercepts(
    model_block: dict,
    vignette_complete: pd.DataFrame,
    stratum_col_in_results: str = "vignette_id",  # column name in re_df
    vignette_id_col: str = "vignette_id",         # numeric id col in vignette_complete
    vignette_text_col: str = "vignette_complete", # text col in vignette_complete
    out_text_col: str = "vignette_text",
) -> pd.DataFrame:
    """
    Takes a wrapper block like res_all_llama["maihda_warmth"] and returns
    random intercept df with vignette text merged in.
    """
    res = _unwrap_result(model_block)

    if "random_intercepts" not in res:
        raise KeyError(f"'random_intercepts' not found. Available keys: {list(res.keys())}")

    re_df = res["random_intercepts"].copy()

    if stratum_col_in_results not in re_df.columns:
        raise KeyError(
            f"'{stratum_col_in_results}' not in random_intercepts columns: {list(re_df.columns)}"
        )

    # Prepare vignette_complete mapping
    vc = vignette_complete[[vignette_id_col, vignette_text_col]].copy()
    vc[vignette_id_col] = vc[vignette_id_col].astype(int)

    # Create numeric key in results
    re_df["_vignette_id_num"] = _make_vignette_id_num(re_df[stratum_col_in_results])

    # Merge
    re_df = re_df.merge(
        vc,
        left_on="_vignette_id_num",
        right_on=vignette_id_col,
        how="left",
        validate="many_to_one",
    )

    # Clean up and rename
    re_df = re_df.rename(columns={vignette_text_col: out_text_col})

    # Optional: keep original id column, drop helper cols
    drop_cols = ["_vignette_id_num"]
    if vignette_id_col in re_df.columns:
        drop_cols.append(vignette_id_col)
    re_df = re_df.drop(columns=[c for c in drop_cols if c in re_df.columns])

    # Sanity check
    n_missing = re_df[out_text_col].isna().sum()
    if n_missing > 0:
        print(f"[WARN] {n_missing} rows could not be matched to vignette text.")

    return re_df

# ------------------------------------------------------------
# Batch: do it for all LLMs and both MAIHDA dimensions
# ------------------------------------------------------------
def attach_vignette_text_all(
    res_all_llama: dict,
    res_all_qwen: dict,
    res_all_mistral: dict,
    vignette_complete: pd.DataFrame,
    out_text_col: str = "vignette_text",
):
    """
    Returns a dict with re_dfs for:
      llama_warmth, llama_competence, qwen_warmth, ...
    """
    out = {}

    out["llama_warmth_re"] = attach_vignette_text_to_random_intercepts(
        res_all_llama["maihda_warmth"], vignette_complete, out_text_col=out_text_col
    )
    out["llama_competence_re"] = attach_vignette_text_to_random_intercepts(
        res_all_llama["maihda_competence"], vignette_complete, out_text_col=out_text_col
    )

    out["qwen_warmth_re"] = attach_vignette_text_to_random_intercepts(
        res_all_qwen["maihda_warmth"], vignette_complete, out_text_col=out_text_col
    )
    out["qwen_competence_re"] = attach_vignette_text_to_random_intercepts(
        res_all_qwen["maihda_competence"], vignette_complete, out_text_col=out_text_col
    )

    out["mistral_warmth_re"] = attach_vignette_text_to_random_intercepts(
        res_all_mistral["maihda_warmth"], vignette_complete, out_text_col=out_text_col
    )
    out["mistral_competence_re"] = attach_vignette_text_to_random_intercepts(
        res_all_mistral["maihda_competence"], vignette_complete, out_text_col=out_text_col
    )

    return out


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from adjustText import adjust_text
from patsy import dmatrix

# ----------------------------
# Helpers
# ----------------------------
def _unwrap(block: dict) -> dict:
    return block["result"] if "result" in block else block

def _as_scalar_re(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(np.asarray(val).ravel()[0])

def _make_vignette_id_num(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("Int64")
    return series.astype(str).str.extract(r"(\d+)$")[0].astype("Int64")

# ----------------------------
# Pretty labels for fixed effects
# ----------------------------
def pretty_term(term: str) -> str:
    if term == "Intercept":
        return "Intercept"
    if "[T." not in term:
        return term

    level = term.split("[T.", 1)[1].rstrip("]")
    level = level.replace("_", " ").lower()

    mapping = {
        "black": "Black",
        "white": "White",
        "asian": "Asian",
        "man": "Man",
        "woman": "Woman",
        "nonbinary": "Non-binary",
        "cis": "Cis",
        "trans": "Trans",
        "muslim": "Muslim",
        "christian": "Christian",
        "jewish": "Jewish",
        "not mentioned": "Not mentioned"
    }

    return mapping.get(level, level.title())

# ----------------------------
# Stratum total deviation
# ----------------------------
def compute_stratum_total_deviation(res_all, dim):
    key = f"maihda_{dim}"
    res = _unwrap(res_all[key])

    base_model = res["base_model"]
    base_scores = res["base_scores"].copy()

    rhs = base_model.model.formula.split("~", 1)[1]
    X = dmatrix(rhs, base_scores, return_type="dataframe")
    fe = base_model.fe_params
    
    # Remove intercept from both X and fe
    X = X.reindex(columns=fe.index, fill_value=0.0)
    if "Intercept" in X.columns:
        X = X.drop(columns=["Intercept"])
    
    fe_no_intercept = fe.drop("Intercept", errors="ignore")
    X = X.reindex(columns=fe_no_intercept.index, fill_value=0.0)
    
    fixed_dev = np.asarray(X @ fe_no_intercept)  # Now this is just the deviations

    re_dict = base_model.random_effects
    strata = base_scores["vignette_id"].astype(str)
    re_vals = np.array([_as_scalar_re(re_dict[s]) for s in strata])

    return pd.DataFrame({
        "vignette_id": strata,
        f"{dim}_total": fixed_dev + re_vals,
        f"{dim}_re": re_vals
    })
from matplotlib.lines import Line2D

# ----------------------------
# Fixed effect space
# ----------------------------
def build_fixed_effect_space(res_all):
    rw = _unwrap(res_all["maihda_warmth"])
    rc = _unwrap(res_all["maihda_competence"])

    fe = (
        rw["fixed_effects"][["term", "estimate"]]
        .rename(columns={"estimate": "warmth"})
        .merge(
            rc["fixed_effects"][["term", "estimate"]]
            .rename(columns={"estimate": "competence"}),
            on="term"
        )
    )

    fe = fe[fe["term"] != "Intercept"].copy()
    fe["label"] = fe["term"].apply(pretty_term)
    return fe

# ----------------------------
# Plot function 
# ----------------------------
from adjustText import adjust_text
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_maihda_fixed_and_strata_adjusttext(
    res_all,
    llm_name,
    vignette_complete,
    n_extremes=5,
    figsize=(10, 9)
):
    fe = build_fixed_effect_space(res_all)

    w = compute_stratum_total_deviation(res_all, "warmth")
    c = compute_stratum_total_deviation(res_all, "competence")
    strata = w.merge(c, on="vignette_id", how="inner")

    # --- Auswahl: je Dimension n größte + n negativste Random-Intercepts
    w_pos = strata.nlargest(n_extremes, "warmth_re")
    w_neg = strata.nsmallest(n_extremes, "warmth_re")
    c_pos = strata.nlargest(n_extremes, "competence_re")
    c_neg = strata.nsmallest(n_extremes, "competence_re")

    strata_sel = (
        pd.concat([w_pos, w_neg, c_pos, c_neg], ignore_index=True)
        .drop_duplicates("vignette_id")
        .reset_index(drop=True)
    )

    # Merge vignette text
    vc = vignette_complete.copy()
    vc["vignette_id"] = vc["vignette_id"].astype(int)
    strata_sel["_vid"] = _make_vignette_id_num(strata_sel["vignette_id"])
    strata_sel = strata_sel.merge(
        vc[["vignette_id", "vignette_complete"]],
        left_on="_vid",
        right_on="vignette_id",
        how="left"
    )

    # Farbe: Projektion der REs (W+C) als Vorzeichen
    strata_sel["re_proj"] = (strata_sel["warmth_re"] + strata_sel["competence_re"]) / np.sqrt(2)
    strata_sel["color"] = np.where(strata_sel["re_proj"] >= 0, "red", "blue")

    fig, ax = plt.subplots(figsize=figsize)

    # --- Fixed effects points
    ax.scatter(
        fe["warmth"], fe["competence"],
        facecolors="none", edgecolors="black",
        s=90, linewidth=1.5, alpha=0.9, zorder=2,
        label="Fixed effects (additive)"
    )

    # --- FE labels (initial placement: slight offset so they don't sit on marker)
    texts_fe = []
    for _, r in fe.iterrows():
        texts_fe.append(
            ax.text(
                r["warmth"] + 0.01, r["competence"] + 0.01,
                r["label"],
                fontsize=12, color="black", zorder=3,
                ha="left", va="bottom"
            )
        )

    # --- Strata points (TOTAL location)
    ax.scatter(
        strata_sel["warmth_total"], strata_sel["competence_total"],
        c=strata_sel["color"],
        s=160, marker="^",
        edgecolor="black", linewidth=0.6,
        alpha=0.95, zorder=4,
        label=f"Top/Bottom {n_extremes} RE (W & C)"
    )

    # --- Strata labels
    texts_strata = []
    for _, r in strata_sel.iterrows():
        label = r.get("vignette_complete", np.nan)
        if pd.isna(label):
            label = str(r.get("vignette_id", ""))
        texts_strata.append(
            ax.text(
                r["warmth_total"], r["competence_total"],
                label,
                fontsize=12, color=r["color"],
                zorder=5, ha="center", va="center"
            )
        )
    for txt in texts_strata:
        x, y = txt.get_position()
        if abs(x) < 0.25 and abs(y) < 0.25:  # Zentrum-Box: anpassen
            txt.set_fontsize(10)

    # --- Padding BEFORE any adjustText
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.18*(x1-x0), x1 + 0.18*(x1-x0))
    ax.set_ylim(y0 - 0.18*(y1-y0), y1 + 0.18*(y1-y0))

    
    # ==========================================================
    # 1) FE-Labels sanft verteilen (NUR untereinander)
    #    - nah am Punkt halten (kleine Kräfte)
    #    - Pfeile optional: sehr kurz/dezent
    # ==========================================================
    # -----------------------
    # 1) FE labels zuerst: sanft verteilen, nah am Punkt halten
    # -----------------------
    adjust_text(
        texts_fe,
        ax=ax,
        expand_points=(1.15, 1.15),
        expand_text=(1.12, 1.12),
        force_points=(0.25, 0.25),   # hält Labels näher am Punkt
        force_text=(0.45, 0.45),     # verteilt FE-Labels besser
        lim=2000,
        only_move={'text': 'xy'},
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.6),
    )

    # -----------------------
    # 2) Strata labels danach: FE-Labels als Hindernisse
    # -----------------------
    adjust_text(
        texts_strata,
        ax=ax,
        objects=texts_fe,            # <- FE labels sind jetzt "Wände"
        expand_points=(1.05, 1.05),
        expand_text=(1.2, 1.2),    # mehr Platz zwischen Texten
        force_points=(1.2, 1.2),     # weg von Punkten im Zentrum
        force_text=(1.0, 1.0),       # weg voneinander
        lim=2500,
        precision=0.0001,
        only_move={'text': 'x', 'points': 'xy'},
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
    )


    # --- Axes / title
    ax.axvline(0, lw=1)
    ax.axhline(0, lw=1)
    ax.set_xlabel("Warmth (grand-mean centered)")
    ax.set_ylabel("Competence (grand-mean centered)")
    ax.set_title(f"{llm_name}: Fixed effects (distributed labels) + extreme strata (adjusted)")

    legend_elements = [
        Line2D([0], [0], marker="^", color="w",
               label="Positive random intercept proj (W+C)",
               markerfacecolor="red", markeredgecolor="black", markersize=10),
        Line2D([0], [0], marker="^", color="w",
               label="Negative random intercept proj (W+C)",
               markerfacecolor="blue", markeredgecolor="black", markersize=10),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles=h + legend_elements, frameon=True, loc="best")

    plt.tight_layout()
    return fig, ax, strata_sel



def plot_maihda_fixed_and_strata_adjusttext_fixedFE(
    res_all,
    llm_name,
    vignette_complete,
    n_extremes=5,
    figsize=(10, 9),
    center_box=(0.5, 0.25),
    show_fixed_effects=True
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from adjustText import adjust_text

    fe = build_fixed_effect_space(res_all)

    w = compute_stratum_total_deviation(res_all, "warmth")
    c = compute_stratum_total_deviation(res_all, "competence")
    strata = w.merge(c, on="vignette_id", how="inner")

    # Auswahl: Top/Bottom n nach RE pro Dimension
    w_pos = strata.nlargest(n_extremes, "warmth_re")
    w_neg = strata.nsmallest(n_extremes, "warmth_re")
    c_pos = strata.nlargest(n_extremes, "competence_re")
    c_neg = strata.nsmallest(n_extremes, "competence_re")

    strata_sel = (
        pd.concat([w_pos, w_neg, c_pos, c_neg], ignore_index=True)
        .drop_duplicates("vignette_id")
        .reset_index(drop=True)
    )

    # Merge vignette text
    vc = vignette_complete.copy()
    vc["vignette_id"] = vc["vignette_id"].astype(int)
    strata_sel["_vid"] = _make_vignette_id_num(strata_sel["vignette_id"])
    strata_sel = strata_sel.merge(
        vc[["vignette_id", "vignette_complete"]],
        left_on="_vid",
        right_on="vignette_id",
        how="left"
    )

    # Farbe nach RE-Projektion
    strata_sel["re_proj"] = (strata_sel["warmth_re"] + strata_sel["competence_re"]) / np.sqrt(2)
    strata_sel["color"] = np.where(strata_sel["re_proj"] >= 0, "red", "blue")

    fig, ax = plt.subplots(figsize=figsize)

    # Fixed effects points and labels (optional)
    texts_fe = []
    fe_obstacles = []
    
    if show_fixed_effects:
        # Fixed effects points
        ax.scatter(
            fe["warmth"], fe["competence"],
            facecolors="none", edgecolors="black",
            s=90, linewidth=1.5, alpha=0.9, zorder=2,
            label="Fixed effects (additive)"
        )
        
        # Fixed labels: NICHT adjusten, einfach setzen
        for _, r in fe.iterrows():
            txt = ax.text(
                float(r["warmth"]), float(r["competence"]), str(r["label"]),
                fontsize=12, color="black", zorder=3,
                ha="left", va="bottom"
            )
            texts_fe.append(txt)

    # Strata points (TOTAL position)
    ax.scatter(
        strata_sel["warmth_total"], strata_sel["competence_total"],
        c=strata_sel["color"], s=160,
        marker="^", edgecolor="black", linewidth=0.6,
        alpha=0.95, zorder=4,
        label=f"Top/Bottom {n_extremes} RE (W & C)"
    )

    # Strata labels: sollen ausweichen
    texts_strata = []
    for _, r in strata_sel.iterrows():
        label = r.get("vignette_complete", np.nan)
        if pd.isna(label):
            label = str(r.get("vignette_id", ""))
        texts_strata.append(
            ax.text(
                float(r["warmth_total"]), float(r["competence_total"]),
                str(label),
                fontsize=12, color=r["color"],
                zorder=5, ha="center", va="center"
            )
        )

    # Achsen-Padding (wichtig!)
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ax.set_xlim(x0 - 0.18*(x1-x0), x1 + 0.18*(x1-x0))
    ax.set_ylim(y0 - 0.18*(y1-y0), y1 + 0.18*(y1-y0))

    # ---------------------------------------------------------
    # Convert FE text bounding boxes into Rectangle obstacles (if FE shown)
    # ---------------------------------------------------------
    if show_fixed_effects and texts_fe:
        fig.canvas.draw()  # needed so text extents exist
        renderer = fig.canvas.get_renderer()

        for t in texts_fe:
            bb = t.get_window_extent(renderer=renderer).expanded(1.05, 1.15)  # a bit padded
            # transform from display -> data coords
            (x0d, y0d) = ax.transData.inverted().transform((bb.x0, bb.y0))
            (x1d, y1d) = ax.transData.inverted().transform((bb.x1, bb.y1))
            rect = Rectangle(
                (x0d, y0d),
                (x1d - x0d),
                (y1d - y0d),
                facecolor="none",
                edgecolor="none",
                linewidth=0,
                zorder=1
            )
            ax.add_patch(rect)
            fe_obstacles.append(rect)

    # adjustText nur für strata, mit FE obstacles falls vorhanden
    adjust_text(
        texts_strata,
        ax=ax,
        objects=fe_obstacles if fe_obstacles else None,  # <-- FE obstacles only if shown
        arrowprops=dict(arrowstyle="-", color="gray", lw=0.6, alpha=0.7),
        expand_points=(1.08, 1.08),
        expand_text=(1.2, 1.2),
        force_points=(1, 1),
        force_text=(0.7, 0.7),
        lim=2500,
        precision=0.0001,
        only_move={'text': 'xy'},
        ensure_inside_axes=True,
        explode_radius=0,
        force_explode=(0, 0),
    )

    # Achsen / Titel / Legende
    ax.axvline(0, lw=1)
    ax.axhline(0, lw=1)
    ax.set_xlabel("Warmth (grand-mean centered)")
    ax.set_ylabel("Competence (grand-mean centered)")
    
    # Update title based on whether FE is shown
    if show_fixed_effects:
        ax.set_title(f"{llm_name}: Fixed effects (fixed labels) + selected strata labels (adjusted)")
    else:
        ax.set_title(f"{llm_name}: Groups with extreme random intercepts (overall predictions)")

    legend_elements = [
        Line2D([0], [0], marker="^", color="w",
               label="Positive random intercept",
               markerfacecolor="red", markeredgecolor="black", markersize=8),
        Line2D([0], [0], marker="^", color="w",
               label="Negative random intercept",
               markerfacecolor="blue", markeredgecolor="black", markersize=8),
    ]
    h, l = ax.get_legend_handles_labels()
    ax.legend(handles= legend_elements, frameon=True, loc="best")

    plt.tight_layout()
    return fig, ax, strata_sel







def plot_intersectional_bias_space(
    res_warmth_maihda,
    res_comp_maihda,
    model_name="LLM",
    n_extremes=3,                 # top/bottom per dimension
    show_all_points=True,
    min_stratum_size=0,           # filter small strata (0 = no filter)
    save_path=None,               # optional: save figure to file
    auto_save=True,               # automatically save high-quality versions
    save_dir="plots",
    save_formats=("png", "pdf", "svg"),
    dpi=600,
    additional_strata=None,       # list of stratum names to always include
    label_single_identity=False,  # label all single-identity strata
    # NEW:
    label_quadrants=True,         # include UL/LR quadrant strata
    n_quadrant=5,                 # how many per quadrant to label
    quadrant_mode="distance",     # "distance" (default) or "abs_sum"
):
    """
    Plot intersectional groups in SCM bias space using MAIHDA predictions.

    Returns
    -------
    strata : pd.DataFrame
        All strata with centered coordinates and metadata.
    labeled_strata : pd.DataFrame
        Subset of strata selected for labeling (extremes/sig/quadrants/custom/etc.).
    """
    import os
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as patheffects
    from matplotlib.lines import Line2D
    from adjustText import adjust_text

    # ------------------------------------------------------------------
    # 0. Create save directory if needed
    # ------------------------------------------------------------------
    if auto_save or save_path:
        if auto_save:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True, parents=True)

    # ------------------------------------------------------------------
    # 1. Extract MAIHDA predictions & random effects
    # ------------------------------------------------------------------
    warm_pred = res_warmth_maihda["stratum_predictions"].copy()
    comp_pred = res_comp_maihda["stratum_predictions"].copy()

    # try to carry a size column if present (for min_stratum_size)
    size_candidates = ["n_obs", "n", "N", "count", "stratum_size"]
    size_col = next((c for c in size_candidates if c in warm_pred.columns), None)

    warm_cols = ["stratum", "predicted_mean_1B"]
    if size_col is not None:
        warm_cols.append(size_col)

    warm_df = warm_pred[warm_cols].rename(columns={"predicted_mean_1B": "warmth_pred"})
    comp_df = comp_pred[["stratum", "predicted_mean_1B"]].rename(
        columns={"predicted_mean_1B": "competence_pred"}
    )

    warm_re = res_warmth_maihda["random_effects"][["stratum", "significant"]].rename(
        columns={"significant": "warmth_significant"}
    )
    comp_re = res_comp_maihda["random_effects"][["stratum", "significant"]].rename(
        columns={"significant": "competence_significant"}
    )

    strata = (
        warm_df.merge(comp_df, on="stratum", how="inner")
        .merge(warm_re, on="stratum", how="left")
        .merge(comp_re, on="stratum", how="left")
    )

    # Ensure booleans (NaN -> False)
    strata["warmth_significant"] = strata["warmth_significant"].fillna(False).astype(bool)
    strata["competence_significant"] = strata["competence_significant"].fillna(False).astype(bool)

    # ------------------------------------------------------------------
    # 2. Center predictions around grand mean (intercept)
    # ------------------------------------------------------------------
    warm_intercept = res_warmth_maihda["fixed_effects"].loc["Intercept", "coef"]
    comp_intercept = res_comp_maihda["fixed_effects"].loc["Intercept", "coef"]

    strata["warmth_centered"] = strata["warmth_pred"] - warm_intercept
    strata["competence_centered"] = strata["competence_pred"] - comp_intercept

    # Overall extremity in plane
    strata["distance"] = np.sqrt(strata["warmth_centered"] ** 2 + strata["competence_centered"] ** 2)

    # ------------------------------------------------------------------
    # 2.5 Apply min_stratum_size if possible
    # ------------------------------------------------------------------
    if min_stratum_size > 0:
        if size_col is None:
            print(
                "⚠️ min_stratum_size was set, but no size column found in stratum_predictions. "
                "Skipping size filter."
            )
        else:
            before = len(strata)
            strata = strata[strata[size_col] >= min_stratum_size].copy()
            print(f"🔎 Size filter ({size_col} ≥ {min_stratum_size}): {before} -> {len(strata)} strata")

    # ------------------------------------------------------------------
    # 3. Smart selection: extremes + significant + quadrants + custom + single-ID
    # ------------------------------------------------------------------
    def pick_top(df, n, sort_col, ascending=False):
        df = df.dropna(subset=[sort_col])
        if df.empty:
            return df
        return df.sort_values(sort_col, ascending=ascending).head(n)

    selected = []

    # A/B) Axis extremes (your original intent)
    selected.append(pick_top(strata, n_extremes, "warmth_centered", ascending=False))  # high W
    selected.append(pick_top(strata, n_extremes, "warmth_centered", ascending=True))   # low W
    selected.append(pick_top(strata, n_extremes, "competence_centered", ascending=False))  # high C
    selected.append(pick_top(strata, n_extremes, "competence_centered", ascending=True))   # low C

    # C) Significant in BOTH -> most extreme overall
    sig_both = strata[strata["warmth_significant"] & strata["competence_significant"]].copy()
    selected.append(pick_top(sig_both, n_extremes, "distance", ascending=False))

    # NEW: Upper-left (W<0, C>0) and lower-right (W>0, C<0)
    if label_quadrants:
        ul = strata[(strata["warmth_centered"] < 0) & (strata["competence_centered"] > 0)].copy()
        lr = strata[(strata["warmth_centered"] > 0) & (strata["competence_centered"] < 0)].copy()

        if quadrant_mode == "abs_sum":
            ul["q_score"] = ul["warmth_centered"].abs() + ul["competence_centered"].abs()
            lr["q_score"] = lr["warmth_centered"].abs() + lr["competence_centered"].abs()
            selected.append(pick_top(ul, n_quadrant, "q_score", ascending=False))
            selected.append(pick_top(lr, n_quadrant, "q_score", ascending=False))
        else:
            selected.append(pick_top(ul, n_quadrant, "distance", ascending=False))
            selected.append(pick_top(lr, n_quadrant, "distance", ascending=False))

    # D) Additional user-specified strata
    if additional_strata is not None and len(additional_strata) > 0:
        custom = strata[strata["stratum"].isin(additional_strata)].copy()
        if custom.empty:
            print("⚠️  Warning: None of the specified additional_strata found after filtering")
        else:
            selected.append(custom)
            print(f"📍 Added {len(custom)} custom strata")

    # E) Single-identity strata
    if label_single_identity:
        def count_identity_components(stratum_name):
            s = str(stratum_name).replace("-", "_").replace(" ", "_")
            parts = [p for p in s.split("_") if "not" not in p and "mention" not in p and p]

            dims = set()
            for p in parts:
                p_low = p.lower()
                if p_low in ["cis", "trans"]:
                    dims.add("transness")
                elif p_low in ["white", "black", "asian"]:
                    dims.add("race")
                elif p_low in ["muslim", "christian", "jewish"]:
                    dims.add("religion")
                elif p_low in ["woman", "man", "nonbinary"]:
                    dims.add("gender")
            return len(dims)

        strata["n_dimensions"] = strata["stratum"].apply(count_identity_components)
        single = strata[strata["n_dimensions"] == 1].copy()
        if not single.empty:
            selected.append(single)
            print(f"📍 Added {len(single)} single-identity strata")

    labeled_strata = pd.concat(selected, ignore_index=True).drop_duplicates("stratum").copy()

    # Precompute sets for selection reasons (avoid recomputing nlargest inside a loop)
    highW = set(strata.nlargest(n_extremes, "warmth_centered")["stratum"])
    lowW  = set(strata.nsmallest(n_extremes, "warmth_centered")["stratum"])
    highC = set(strata.nlargest(n_extremes, "competence_centered")["stratum"])
    lowC  = set(strata.nsmallest(n_extremes, "competence_centered")["stratum"])

    ULset, LRset = set(), set()
    if label_quadrants:
        ULset = set(strata.loc[(strata["warmth_centered"] < 0) & (strata["competence_centered"] > 0), "stratum"])
        LRset = set(strata.loc[(strata["warmth_centered"] > 0) & (strata["competence_centered"] < 0), "stratum"])

    addset = set(additional_strata) if additional_strata else set()

    def compute_reason(row):
        s = row["stratum"]
        reasons = []
        if s in highW: reasons.append("High W")
        if s in lowW:  reasons.append("Low W")
        if s in highC: reasons.append("High C")
        if s in lowC:  reasons.append("Low C")
        if row["warmth_significant"] and row["competence_significant"]: reasons.append("Sig both")
        if s in addset: reasons.append("Custom")
        if label_quadrants:
            if s in ULset: reasons.append("Upper-left")
            if s in LRset: reasons.append("Lower-right")
        if label_single_identity and "n_dimensions" in labeled_strata.columns and row.get("n_dimensions", None) == 1:
            reasons.append("Single-ID")
        # unique, preserve order
        out = []
        for r in reasons:
            if r not in out:
                out.append(r)
        return ", ".join(out)

    labeled_strata["selection_reason"] = labeled_strata.apply(compute_reason, axis=1)

    print(f"📌 Labeled strata: {len(labeled_strata)}")
    print(f"   - Sig warmth: {int(labeled_strata['warmth_significant'].sum())}")
    print(f"   - Sig competence: {int(labeled_strata['competence_significant'].sum())}")
    print(f"   - Sig both: {int((labeled_strata['warmth_significant'] & labeled_strata['competence_significant']).sum())}")

    # ------------------------------------------------------------------
    # 4. Clean labels (abbreviated)
    # ------------------------------------------------------------------
    def clean_stratum_label(s):
        abbr = {
            "woman": "Woman", "man": "Man", "nonbinary": "NB",
            "white": "White", "black": "Black", "asian": "Asian",
            "muslim": "Muslim", "christian": "Christian", "jewish": "Jewish",
            "cis": "cis", "trans": "trans"
        }
        s0 = str(s).replace("-", "_").replace(" ", "_")
        parts = [p for p in s0.split("_") if "not" not in p and "mention" not in p and p]
        parts_low = [p.lower() for p in parts]

        ordered = []
        # Order: race → religion → transness → gender (keep consistent)
        for p in parts_low:
            if p in ["white", "black", "asian"]:
                ordered.append(abbr.get(p, p))
        for p in parts_low:
            if p in ["muslim", "christian", "jewish"]:
                ordered.append(abbr.get(p, p))
        for p in parts_low:
            if p in ["cis", "trans"]:
                ordered.append(abbr.get(p, p))
        for p in parts_low:
            if p in ["woman", "man", "nonbinary"]:
                ordered.append(abbr.get(p, p))

        return "-".join(ordered) if ordered else s0

    labeled_strata["label"] = labeled_strata["stratum"].apply(clean_stratum_label)

    # ------------------------------------------------------------------
    # 5. Create plot with high quality settings
    # ------------------------------------------------------------------
    original_rc = {
        "figure.dpi": mpl.rcParams.get("figure.dpi", 100),
        "figure.facecolor": mpl.rcParams.get("figure.facecolor", "white"),
        "figure.edgecolor": mpl.rcParams.get("figure.edgecolor", "white"),
    }

    mpl.rcParams["figure.dpi"] = dpi
    mpl.rcParams["figure.facecolor"] = "white"
    mpl.rcParams["figure.edgecolor"] = "white"
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
    mpl.rcParams["axes.linewidth"] = 1.2
    mpl.rcParams["axes.spines.left"] = True
    mpl.rcParams["axes.spines.bottom"] = True
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    # --- IMPORTANT: ensure the function always returns, even if plotting fails
    try:
        fig, ax = plt.subplots(figsize=(16, 12))

        # Quadrant lines
        ax.axhline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.5, alpha=0.7, zorder=1)

        # ------------------------------------------------------------------
        # 6. Plot points
        # ------------------------------------------------------------------
        if show_all_points:
            unlabeled = strata[~strata["stratum"].isin(labeled_strata["stratum"])]
            ax.scatter(
                unlabeled["warmth_centered"],
                unlabeled["competence_centered"],
                s=45, alpha=0.25, color="lightgray",
                edgecolor="gray", linewidth=0.5,
                zorder=2, label=f"Other strata (n={len(unlabeled)})"
            )

        # Color coding for labeled
        colors = []
        for _, row in labeled_strata.iterrows():
            if row["warmth_significant"] and row["competence_significant"]:
                colors.append("#FF4500")  # both
            elif row["warmth_significant"] or row["competence_significant"]:
                colors.append("#FFD700")  # one
            else:
                colors.append("#87CEEB")  # extreme/quadrant/custom only

        ax.scatter(
            labeled_strata["warmth_centered"],
            labeled_strata["competence_centered"],
            s=200, marker="D",
            c=colors,
            edgecolor="black",
            linewidth=1.8,
            alpha=0.95,
            zorder=4
        )

        # ------------------------------------------------------------------
        # 7. Label placement with adjustText
        # ------------------------------------------------------------------
        # Expand limits a bit for label breathing room
        x_min, x_max = strata["warmth_centered"].min(), strata["warmth_centered"].max()
        y_min, y_max = strata["competence_centered"].min(), strata["competence_centered"].max()
        x_rng = (x_max - x_min) if (x_max > x_min) else 1.0
        y_rng = (y_max - y_min) if (y_max > y_min) else 1.0
        ax.set_xlim(x_min - 0.20 * x_rng, x_max + 0.20 * x_rng)
        ax.set_ylim(y_min - 0.20 * y_rng, y_max + 0.20 * y_rng)

        print(f"\n📍 Placing {len(labeled_strata)} labels with adjustText...")

        texts = []
        for _, row in labeled_strata.iterrows():
            x = row["warmth_centered"]
            y = row["competence_centered"]
            t = ax.text(
                x, y, row["label"],
                fontsize=11,
                ha="center", va="center",
                path_effects=[patheffects.withStroke(linewidth=3.0, foreground="white")],
                zorder=5
            )
            texts.append(t)

        # Be conservative with adjustText args for compatibility
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8, alpha=0.7),
            expand_points=(1.4, 1.4),
            expand_text=(1.2, 1.2),
            force_points=(0.3, 0.3),
            force_text=(0.4, 0.4),
            lim=500
        )

        print(f"✅ All {len(texts)} labels placed")

        # ------------------------------------------------------------------
        # 8. Labels, title, legend
        # ------------------------------------------------------------------
        ax.set_xlabel("Warmth (centered at grand mean)", fontsize=16, weight="bold")
        ax.set_ylabel("Competence (centered at grand mean)", fontsize=16, weight="bold")
        ax.set_title(
            f"{model_name}: Intersectional Groups in Stereotype Content Model Space\n"
            f"MAIHDA Model 1B: Extremes, Significant REs"
            + (" + Quadrants" if label_quadrants else "")
            + (f" (n≥{min_stratum_size})" if min_stratum_size > 0 else ""),
            fontsize=18, weight="bold", pad=25
        )

        legend_elements = [
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#FF4500",
                   markersize=14, markeredgecolor="black", markeredgewidth=1.5,
                   label="Sig. in both W & C"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#FFD700",
                   markersize=14, markeredgecolor="black", markeredgewidth=1.5,
                   label="Sig. in W or C"),
            Line2D([0], [0], marker="D", color="w", markerfacecolor="#87CEEB",
                   markersize=14, markeredgecolor="black", markeredgewidth=1.5,
                   label="Extreme / custom / quadrant"),
        ]
        if show_all_points:
            legend_elements.append(
                Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgray",
                       markersize=10, markeredgecolor="gray",
                       label=f"Other strata (n={len(strata)-len(labeled_strata)})")
            )

        ax.legend(
            handles=legend_elements,
            loc="lower right",
            framealpha=0.95,
            fontsize=13,
            frameon=True,
            fancybox=True,
            shadow=True,
        )

        ax.grid(alpha=0.25, zorder=0)
        ax.tick_params(axis="both", which="major", labelsize=12, width=1.2, length=6)

        plt.tight_layout()

        # ------------------------------------------------------------------
        # 9. Saving
        # ------------------------------------------------------------------
        saved_files = []

        if save_path:
            try:
                ext = Path(save_path).suffix.lower().lstrip(".")
                save_kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.2)
                if ext in ["png", "tiff", "jpg", "jpeg"]:
                    save_kwargs["dpi"] = dpi
                plt.savefig(save_path, **save_kwargs)
                saved_files.append(str(save_path))
                print(f"💾 Figure saved to: {save_path}")
            except Exception as e:
                print(f"❌ Failed to save to {save_path}: {e}")

        if auto_save:
            safe_model_name = "".join(c for c in model_name if c.isalnum() or c in (" ", "-", "_")).strip()
            safe_model_name = safe_model_name.replace(" ", "_")
            filename_base = f"intersectional_bias_space_{safe_model_name}"

            print(f"\n💾 AUTO-SAVING HIGH-QUALITY PLOTS...")
            print(f"   Directory: {save_dir}")
            print(f"   Formats: {', '.join(save_formats)}")
            print(f"   DPI: {dpi} (for raster formats)")

            for fmt in save_formats:
                try:
                    file_path = Path(save_dir) / f"{filename_base}.{fmt}"
                    save_kwargs = dict(bbox_inches="tight", facecolor="white", edgecolor="none", pad_inches=0.2)

                    if fmt.lower() in ["png", "tiff", "jpg", "jpeg"]:
                        save_kwargs["dpi"] = dpi
                    elif fmt.lower() == "pdf":
                        save_kwargs["dpi"] = 600

                    plt.savefig(file_path, **save_kwargs)
                    saved_files.append(str(file_path))
                    print(f"   ✅ Saved: {file_path.name}")
                except Exception as e:
                    print(f"   ❌ Failed to save {fmt}: {e}")

            print(f"\n📁 Total files saved: {len(saved_files)}")

        plt.show()

        # ------------------------------------------------------------------
        # 10. Summary table
        # ------------------------------------------------------------------
        print("\n" + "=" * 80)
        print("📋 LABELED INTERSECTIONAL GROUPS SUMMARY")
        print("=" * 80)

        cols = [
            "label", "warmth_centered", "competence_centered",
            "warmth_significant", "competence_significant", "selection_reason"
        ]
        summary_table = labeled_strata[cols].sort_values("warmth_centered", ascending=False)
        print(summary_table.to_string(index=False))

        print("=" * 80 + "\n")

    finally:
        # Restore original matplotlib settings AND always return results
        mpl.rcParams["figure.dpi"] = original_rc["figure.dpi"]
        mpl.rcParams["figure.facecolor"] = original_rc["figure.facecolor"]
        mpl.rcParams["figure.edgecolor"] = original_rc["figure.edgecolor"]

        return strata, labeled_strata



def plot_ols_coefs_forest(
    df,
    term_col="term_simple",
    llm_col="llm",
    coef_col="coef",
    low_col="ci_low",
    high_col="ci_high",
    sort_by="mean_abs",   # "mean_abs", "mean", "none"
    figsize=(9, 12),
    title="OLS coefficients with 95% bootstrap CIs",
    xline_at=0.0,
    font_scale=1.5,       # NEW
):
    """
    Forest plot: one row per term_simple; 3 LLM estimates per row with CI.
    Expects df columns: llm, term_simple, coef, ci_low, ci_high (others allowed).
    """

    d = df.copy()

    # Basic sanity
    needed = {term_col, llm_col, coef_col, low_col, high_col}
    missing = needed - set(d.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Ensure stable ordering of terms
    if sort_by != "none":
        g = d.groupby(term_col)[coef_col].mean()
        if sort_by == "mean_abs":
            order = g.abs().sort_values(ascending=False).index.tolist()
        elif sort_by == "mean":
            order = g.sort_values(ascending=False).index.tolist()
        else:
            raise ValueError("sort_by must be 'mean_abs', 'mean', or 'none'")
        d[term_col] = pd.Categorical(d[term_col], categories=order, ordered=True)

    terms = (
        list(d[term_col].cat.categories)
        if isinstance(d[term_col].dtype, pd.CategoricalDtype)
        else sorted(d[term_col].unique())
    )

    llms = sorted(pd.unique(d[llm_col]))

    # y positions: one per term, with small offsets per llm
    y_base = np.arange(len(terms))
    offsets = np.linspace(-0.2, 0.2, num=len(llms)) if len(llms) > 1 else np.array([0.0])
    offset_map = dict(zip(llms, offsets))

    fig, ax = plt.subplots(figsize=figsize)

    # Vertical reference line at 0
    ax.axvline(xline_at, linewidth=1.5)

    # Plot each llm
    for llm in llms:
        sub = d[d[llm_col] == llm].copy()

        term_to_y = {t: i for i, t in enumerate(terms)}
        y = sub[term_col].map(term_to_y).astype(float) + offset_map[llm]

        x = sub[coef_col].to_numpy()
        lo = sub[low_col].to_numpy()
        hi = sub[high_col].to_numpy()
        xerr = np.vstack([x - lo, hi - x])

        ax.errorbar(
            x, y,
            xerr=xerr,
            fmt="o",
            capsize=4,
            elinewidth=2,
            markersize=9,
            label=str(llm),
        )

    # Axes + labels
    ax.set_yticks(y_base)
    ax.set_yticklabels(terms, fontsize=10 * font_scale)
    ax.invert_yaxis()

    ax.set_xlabel("Coefficient", fontsize=11 * font_scale)
    ax.set_title(title, fontsize=12 * font_scale, pad=10)

    ax.tick_params(axis="x", labelsize=10 * font_scale)

    ax.legend(
        title="LLM",
        frameon=False,
        fontsize=9 * font_scale,
        title_fontsize=10 * font_scale,
    )

    fig.tight_layout()
    return fig, ax
