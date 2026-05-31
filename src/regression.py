import pandas as pd
import statsmodels.api as sm
import numpy as np
import re
import graphviz
from io import StringIO
from typing import List, Dict, Tuple, Any, Optional

from src.helpers import logger, get_output_path
from src.common import convert_to_model_spec_dict


def create_model_spec(
    independent_dict: List[Dict], 
    dependent_dict: List[Dict], 
    relation_dict: List[Dict], 
    varcovar_dict: List[Dict]
) -> Tuple[str, Dict[str, Any]]:
    """Creates a model specification string and dictionary from variable definitions.

    Args:
        independent_dict: Independent variable definitions.
        dependent_dict: Dependent variable definitions.
        relation_dict: Relationship definitions.
        varcovar_dict: Variance-covariance definitions.

    Returns:
        Tuple[str, Dict[str, Any]]: Model spec string and parsed dictionary.
    """
    model_spec = "### Independent Variables\n"
    model_spec += "\n".join([
        f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(int(d['number_questions']))]) 
        for d in independent_dict
    ])
    
    model_spec += "\n\n### Dependent Variables\n"
    model_spec += "\n".join([
        f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(int(d['number_questions']))]) 
        for d in dependent_dict
    ])
    
    model_spec += "\n\n### Relations\n"
    model_spec += "\n".join([
        f"{r['Variable']} ~ {r['Related_Variables']}" 
        for r in relation_dict if r.get('Relation_Type') in ['direct', 'both']
    ])

    model_spec_dict = convert_to_model_spec_dict(model_spec)
    return model_spec, model_spec_dict


def extract_indicators_from_model_spec(model_spec: str) -> List[str]:
    """Extracts indicator names (e.g., Var_Q1) from the model spec.

    Args:
        model_spec: Model specification string.

    Returns:
        List[str]: Sorted list of unique indicator names.
    """
    regex = re.compile(r'\b\w+_Q\d+\b')
    indicators = list(set(re.findall(regex, model_spec)))
    indicators.sort()
    return indicators


def create_model_spec_graph_full(
    model_spec: str, 
    independent_dict: List[Dict], 
    dependent_dict: List[Dict], 
    graph_name: str
) -> str:
    """Generates a full Graphviz visualization of the model specification (indicators and latent variables).

    Args:
        model_spec: Model spec string.
        independent_dict: Independent variable definitions.
        dependent_dict: Dependent variable definitions.
        graph_name: Base name for the output file.

    Returns:
        str: Path to the generated PNG image.
    """
    output_path = get_output_path()
    g = graphviz.Digraph('Model Spec Full', format='png', engine='dot')
    g.attr(rankdir='RL', overlap='scale', splines='true', fontsize='12')

    dep_vars = [item['Variable'] for item in dependent_dict]
    indep_vars = [item['Variable'] for item in independent_dict]
    indicators = extract_indicators_from_model_spec(model_spec)

    # Nodes
    with g.subgraph() as s:
        for ind in indicators:
            s.node(ind, shape='box', fillcolor='#e6f2ff', style='filled')

    with g.subgraph() as s:
        for var in indep_vars:
            s.node(var, shape='ellipse', fillcolor='#cae6df', style='filled')

    with g.subgraph() as s:
        s.attr(rank='source')
        for var in dep_vars:
            s.node(var, shape='ellipse', fillcolor='#FFFF00', style='filled')

    # Edges from parsing spec
    lines = model_spec.strip().split("\n")
    for line in lines:
        if '~' in line and '=~' not in line:
            lhs, rhs = [x.strip() for x in line.split('~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                g.edge(var, lhs, dir='forward', color='blue', penwidth='2')
        elif '=~' in line:
            lhs, rhs = [x.strip() for x in line.split('=~', 1)]
            for ind in [x.strip() for x in rhs.split('+')]:
                g.edge(lhs, ind.strip(), dir='forward')

    g.attr(dpi='600')
    save_path = f"{output_path}/{graph_name}_full"
    g.render(save_path, cleanup=True)
    return f"{save_path}.png"


def create_model_spec_graph_short(
    model_spec: str, 
    independent_dict: List[Dict], 
    dependent_dict: List[Dict], 
    graph_name: str
) -> str:
    """Generates a simplified Graphviz visualization showing only relationships between latent variables.

    Args:
        model_spec: Model spec string.
        independent_dict: Independent variable definitions.
        dependent_dict: Dependent variable definitions.
        graph_name: Base name for the output file.

    Returns:
        str: Path to the generated PNG image.
    """
    output_path = get_output_path()
    g = graphviz.Digraph('Model Spec Short', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')

    dep_vars = [item['Variable'] for item in dependent_dict]
    indep_vars = [item['Variable'] for item in independent_dict]

    with g.subgraph() as s:
        s.attr(rank='same')
        for var in indep_vars:
            s.node(var, shape='ellipse', fillcolor='#cae6df', style='filled')

    with g.subgraph() as s:
        for var in dep_vars:
            s.node(var, shape='ellipse', fillcolor='#FFFF00', style='filled')

    lines = model_spec.strip().split("\n")
    for line in lines:
        if '~' in line and '=~' not in line:
            lhs, rhs = [x.strip() for x in line.split('~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                g.edge(var, lhs, dir='forward', color='blue', penwidth='2')

    g.attr(dpi='600')
    save_path = f"{output_path}/{graph_name}_short"
    g.render(save_path, cleanup=True)
    return f"{save_path}.png"


def create_composite_score(df: pd.DataFrame, questions: List[str]) -> pd.Series:
    """Calculates the mean score across a set of questions (indicators).

    Args:
        df: Input DataFrame.
        questions: List of column names.

    Returns:
        pd.Series: Composite score.
    """
    existing_cols = [q for q in questions if q in df.columns]
    if not existing_cols:
        return pd.Series(0, index=df.index)
    return df[existing_cols].mean(axis=1)


def run_regression(df: pd.DataFrame, relation: str) -> sm.regression.linear_model.RegressionResultsWrapper:
    """Executes an OLS regression for a single relationship string.

    Args:
        df: DataFrame containing composite scores.
        relation: Relation string (e.g., 'Y ~ X1 + X2').

    Returns:
        sm.regression.linear_model.RegressionResultsWrapper: Fitted model results.
    """
    lhs, rhs = [x.strip() for x in relation.split('~')]
    rhs_vars = [x.strip() for x in rhs.split('+')]

    X = df[rhs_vars]
    X = sm.add_constant(X)
    y = df[lhs]

    model = sm.OLS(y, X).fit()
    return model


def calculate_collinearity_diagnostics(df: pd.DataFrame, relation: str) -> pd.DataFrame:
    """Calculates Variance Inflation Factor (VIF) and Tolerance for regression predictors.

    Args:
        df: DataFrame containing composite scores.
        relation: Relation string.

    Returns:
        pd.DataFrame: Table with VIF and Tolerance for each predictor.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    
    _, rhs = [x.strip() for x in relation.split('~')]
    rhs_vars = [x.strip() for x in rhs.split('+')]
    
    X = df[rhs_vars]
    X = sm.add_constant(X)
    
    vifs = []
    # Index 0 is constant, we skip it for VIF usually or show it
    for i in range(1, X.shape[1]):
        vif = variance_inflation_factor(X.values, i)
        vifs.append(vif)
        
    diag_df = pd.DataFrame({
        'Variable': rhs_vars,
        'VIF': vifs,
        'Tolerance': [1/v if v > 0 else 0 for v in vifs]
    })
    
    return diag_df


def conduct_regression_analysis(df: pd.DataFrame, model_spec_dict: Dict[str, Any]) -> List[List[Any]]:
    """Performs regression analysis for all relations in the model spec.

    Args:
        df: Pre-processed survey data.
        model_spec_dict: Parsed model specification.

    Returns:
        List[List[Any]]: List of [relation_string, results_summary].
    """
    logger.info("Conducting regression analysis")
    data = df.copy()
    
    # Create composite scores for latents
    for construct, questions in {**model_spec_dict['independent'], **model_spec_dict['dependent']}.items():
        data[construct] = create_composite_score(data, questions)

    results = []
    for relation in model_spec_dict['relations']:
        try:
            model = run_regression(data, relation)
            results.append([relation, model.summary()])
        except Exception as e:
            logger.error(f"Regression failed for {relation}: {e}")

    return results


def conduct_regression_analysis_standardized(df: pd.DataFrame, model_spec_dict: Dict[str, Any]) -> List[List[Any]]:
    """Performs standardized regression (variables scaled by Z-score).

    Args:
        df: Pre-processed survey data.
        model_spec_dict: Parsed model specification.

    Returns:
        List[List[Any]]: List of [relation_string, results_summary].
    """
    logger.info("Conducting standardized regression analysis")
    data = df.copy()
    
    # Create composite scores
    for construct, questions in {**model_spec_dict['independent'], **model_spec_dict['dependent']}.items():
        data[construct] = create_composite_score(data, questions)

    # Standardize
    def standardize(series):
        if series.std() == 0:
            return series - series.mean()
        return (series - series.mean()) / series.std()

    results = []
    for relation in model_spec_dict['relations']:
        try:
            lhs, rhs = [x.strip() for x in relation.split('~')]
            rhs_vars = [x.strip() for x in rhs.split('+')]
            
            X = data[rhs_vars].apply(standardize)
            X = sm.add_constant(X)
            y = standardize(data[lhs])
            
            model = sm.OLS(y, X).fit()
            results.append([relation, model.summary()])
        except Exception as e:
            logger.error(f"Standardized regression failed for {relation}: {e}")

    return results


def get_parameter_thresholds(parameter_name: str, parameters_dict: List[Dict]) -> Tuple[Optional[float], Optional[float]]:
    """Retrieves thresholds for a given parameter from the configuration.

    Args:
        parameter_name: Name of the parameter.
        parameters_dict: Configuration list.

    Returns:
        Tuple[Optional[float], Optional[float]]: (high_threshold, moderate_threshold).
    """
    match = next((item for item in parameters_dict if item.get('parameter') == parameter_name), None)
    if match:
        return (
            pd.to_numeric(match.get('high_threshold'), errors='coerce'),
            pd.to_numeric(match.get('moderate_threshold'), errors='coerce')
        )
    return None, None


def extract_regression_results(summary: Any) -> Tuple[Dict, Dict, Dict]:
    """Extracts key statistics and coefficient information from a statsmodels summary object.

    Args:
        summary: statsmodels summary object.

    Returns:
        Tuple[Dict, Dict, Dict]: (general_stats, diagnostic_stats, coefficient_info).
    """
    summary_text = str(summary)
    
    # Helper for regex extraction
    def get_val(pattern, text, default=None):
        match = re.search(pattern, text)
        return match.group(1) if match else default

    general_stats = {
        'Dependent Variable': get_val(r"Dep\. Variable:\s+([^\s]+)", summary_text),
        'R-squared': get_val(r"R-squared:\s+([0-9.]+)", summary_text),
        'Adjusted R-squared': get_val(r"Adj\.\s*R-squared:\s+([-0-9.]+)", summary_text),
        'F-statistic': get_val(r"F-statistic:\s+([0-9.]+)", summary_text),
        'Prob (F-statistic)': get_val(r"Prob \(F-statistic\):\s+([0-9.e-]+)", summary_text)
    }

    diagnostic_stats = {
        'Durbin-Watson': float(get_val(r"Durbin-Watson:\s+(\d+\.\d+)", summary_text, 0)),
        'Omnibus': get_val(r"Omnibus:\s+(\d+\.\d+)", summary_text),
        'Skew': get_val(r"Skew:\s+([-]?\d+\.\d+)", summary_text),
        'Kurtosis': get_val(r"Kurtosis:\s+(\d+\.\d+)", summary_text)
    }

    # Use HTML tables for coefficient extraction as it's more structured than raw text
    try:
        tables = summary.tables
        coef_df = pd.read_html(StringIO(tables[1].as_html()), header=0, index_col=0)[0]
        
        coef_info = {}
        for var, row in coef_df.iterrows():
            if var != 'const':
                coef_info[var] = {
                    "Estimated Coefficient": row['coef'],
                    "Standard Error": row['std err'],
                    "t-Value": row['t'],
                    "P-Value": row['P>|t|']
                }
        return general_stats, diagnostic_stats, coef_info
    except Exception as e:
        logger.error(f"Failed to extract coefficient table: {e}")
        return general_stats, diagnostic_stats, {}


def create_ols_summary_table(general_stats: Dict) -> pd.DataFrame:
    """Creates a user-friendly summary table of OLS results.

    Args:
        general_stats: Dictionary of general statistics.

    Returns:
        pd.DataFrame: Formatted summary DataFrame.
    """
    dep_var = general_stats.get('Dependent Variable', 'Target')
    r2 = float(general_stats.get('R-squared', 0))
    adj_r2 = float(general_stats.get('Adjusted R-squared', 0))
    f_p = float(general_stats.get('Prob (F-statistic)', 1))

    summary_data = {
        "Aspect": ["Target", "Explaining Power", "Confidence", "Fit Quality", "Significance"],
        "Metric": [dep_var, "R-squared", "F-statistic Prob", "Adj. R-squared", "P-Value"],
        "Value": [
            "-", f"{r2:.1%}", f"{f_p:.3f}", f"{adj_r2:.1%}", f"{f_p:.3f}"
        ],
        "Interpretation": [
            f"Predicting {dep_var}",
            "Good" if r2 > 0.5 else "Moderate" if r2 > 0.3 else "Low",
            "Reliable" if f_p < 0.05 else "Unreliable",
            "Strong" if adj_r2 > 0.4 else "Fair",
            "Significant" if f_p < 0.05 else "Not Significant"
        ]
    }
    return pd.DataFrame(summary_data)


def interpret_regression_results(
    regression_results: List[List[Any]], 
    parameters_dict: List[Dict]
) -> Tuple[List[Any], List[pd.DataFrame], List[pd.DataFrame]]:
    """Provides high-level qualitative interpretation of multiple regression results.

    Args:
        regression_results: Output from conduct_regression_analysis.
        parameters_dict: Model parameters for thresholds.

    Returns:
        Tuple: (interpretations, coefficient_dfs, summary_tables)
    """
    _, dw_moderate = get_parameter_thresholds('durbin_watson_moderate', parameters_dict)
    dw_moderate = dw_moderate or 1.5
    _, p_threshold = get_parameter_thresholds('p_value_threshold', parameters_dict)
    p_threshold = p_threshold or 0.05

    interpretations = []
    coef_dfs = []
    summary_tables = []

    for relation, summary in regression_results:
        gen, diag, coefs = extract_regression_results(summary)
        
        interp = []
        r2 = float(gen.get('R-squared', 0))
        interp.append(f"Model explains {r2:.1%} of variance in target.")
        
        f_p = float(gen.get('Prob (F-statistic)', 1))
        interp.append("Model is statistically significant." if f_p < p_threshold else "Model is not significant.")
        
        dw = diag.get('Durbin-Watson', 2.0)
        if dw < dw_moderate:
            interp.append("Warning: Possible positive autocorrelation (DW < 1.5).")
        
        # Coefficients
        def safe_is_significant(info):
            try:
                return float(info.get('P-Value', 1.0)) < p_threshold
            except (ValueError, TypeError):
                return False

        sig_vars = [v for v, info in coefs.items() if safe_is_significant(info)]
        if sig_vars:
            interp.append(f"Significant predictors: {', '.join(sig_vars)}")
        else:
            interp.append("No significant individual predictors found.")

        interpretations.append([relation, "\n".join(interp)])
        
        # Coef DataFrame
        df_coef = pd.DataFrame(coefs).T.reset_index().rename(columns={'index': 'Variable'})
        coef_dfs.append(df_coef)
        
        summary_tables.append(create_ols_summary_table(gen))

    return interpretations, coef_dfs, summary_tables


def create_result_graph_short(
    model_spec: str, 
    independent_dict: List[Dict], 
    dependent_dict: List[Dict], 
    predictors_dfs: List[pd.DataFrame], 
    graph_name: str
) -> str:
    """Generates a result graph showing coefficients and significance levels on paths.

    Args:
        model_spec: Model spec string.
        independent_dict: Independent definitions.
        dependent_dict: Dependent definitions.
        predictors_dfs: List of DataFrames containing coefficient results.
        graph_name: Name for the output file.

    Returns:
        str: Path to the generated PNG image.
    """
    output_path = get_output_path()
    g = graphviz.Digraph('Regression Results', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')

    dep_vars = [item['Variable'] for item in dependent_dict]
    indep_vars = [item['Variable'] for item in independent_dict]

    for var in indep_vars:
        g.node(var, shape='ellipse', fillcolor='#cae6df', style='filled')
    for var in dep_vars:
        g.node(var, shape='ellipse', fillcolor='#FFFF00', style='filled')

    all_coefs = pd.concat(predictors_dfs, ignore_index=True)

    lines = model_spec.strip().split("\n")
    for line in lines:
        if '~' in line and '=~' not in line:
            lhs, rhs = [x.strip() for x in line.split('~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                var = var.strip()
                matches = all_coefs[all_coefs['Variable'] == var]
                if not matches.empty:
                    # We take the mean if a variable appears in multiple relations (though rare in basic OLS)
                    try:
                        coef = float(matches['Estimated Coefficient'].iloc[0])
                    except (ValueError, TypeError):
                        coef = 0.0
                        
                    try:
                        p_val = float(matches['P-Value'].iloc[0])
                    except (ValueError, TypeError):
                        p_val = 1.0
                        
                    label = f"c: {coef:.2f}\np: {p_val:.3f}"
                    color = 'blue' if p_val < 0.05 else 'red'
                    g.edge(var, lhs, label=label, dir='forward', color=color, penwidth='2')

    g.attr(dpi='600')
    save_path = f"{output_path}/{graph_name}"
    g.render(save_path, cleanup=True)
    return f"{save_path}.png"



def run_regression_engine(dataframe, target, features, indep_dict=None, dep_dict=None):
    """
    Executes an OLS regression and returns structured coefficients and p-values.

    Builds composite scores for latent constructs from indep_dict/dep_dict,
    then runs statsmodels OLS and extracts coef/p_val for each feature.
    """
    df = dataframe.copy()

    # Build composite scores for latent constructs referenced in this relation
    all_latents = [target] + features
    for d in (indep_dict or []) + (dep_dict or []):
        var = d.get('Variable')
        if var in all_latents:
            try:
                count = int(float(d.get('number_questions', 0)))
                indicators = [f"{var}_Q{i+1}" for i in range(count)]
                df[var] = create_composite_score(df, indicators)
            except (ValueError, TypeError):
                pass

    relation = f"{target} ~ {' + '.join(features)}"
    try:
        model = run_regression(df, relation)
        results = {}
        for feature in features:
            results[feature] = {
                'coef': model.params.get(feature, np.nan),
                'p_val': model.pvalues.get(feature, np.nan)
            }
        return results
    except Exception:
        return {feature: {'coef': np.nan, 'p_val': np.nan} for feature in features}

def generate_dynamic_multi_group_analysis(global_df, relations_config, demographic_config,
                                          indep_dict=None, dep_dict=None):
    """
    100% Dynamic, metadata-driven multi-group matrix engine.
    Ensures group columns align and appends them cleanly into wide markdown formats.

    Returns:
        Tuple[str, List[pd.DataFrame]]: (markdown_string, list_of_dataframes)
    """
    mga_markdown_output = []
    mga_dataframes = []
    
    # Locate all active moderators from config without mapping names explicitly
    moderator_columns = demographic_config[
        demographic_config['used_as_moderator'].str.lower() == 'yes'
    ]['Variable'].tolist()
    
    for idx, row in relations_config.iterrows():
        target_var = row['Variable'].strip()
        predictors = [p.strip() for p in row['Related_Variables'].split('+')]
        
        # Base Index tracking format: structural path mapping rows
        paths_index = [f"{pred} -> {target_var}" for pred in predictors]
        compiled_matrix_df = pd.DataFrame(index=paths_index)
        
        # 1. Compute and bind Global Baseline Data Series
        baseline_model = run_regression_engine(global_df, target_var, predictors,
                                               indep_dict, dep_dict)
        compiled_matrix_df['Baseline Coef'] = [baseline_model[pred]['coef'] for pred in predictors]
        compiled_matrix_df['Baseline P-Val'] = [baseline_model[pred]['p_val'] for pred in predictors]
        
        # 2. Iterate dynamically across metadata moderator features
        subgroup_n = {}
        for moderator in moderator_columns:
            if moderator not in global_df.columns:
                continue
            
            # Defensive encoding: ensure string-typed moderator columns are
            # converted to integer codes so pd.to_numeric downstream doesn't
            # destroy them (handles pyarrow/AcrowDtype backends transparently)
            if pd.api.types.is_string_dtype(global_df[moderator]):
                global_df[moderator] = pd.Categorical(global_df[moderator]).codes
            
            # Extract unique response categories present in your live survey data
            unique_subgroups = global_df[moderator].dropna().unique()
            
            for group in unique_subgroups:
                # Isolate sub-sample subset arrays
                sliced_subgroup_df = global_df[global_df[moderator] == group]
                
                # Algorithmic check: Adapt threshold to small sample limitations
                # Sets minimum boundary to 5 rows or 5% of total dataset dynamically
                min_threshold = max(5, int(len(global_df) * 0.05))
                
                if len(sliced_subgroup_df) < min_threshold:
                    continue  # Safely skip underrepresented strata
                
                # Compute subgroup parameters
                subgroup_model = run_regression_engine(sliced_subgroup_df, target_var, predictors,
                                                       indep_dict, dep_dict)
                
                # 3. Secure Column Binding Engine (Protects against lookup errors)
                group_clean_label = str(group).replace("[", "").replace("]", "").strip()
                subgroup_n[f"{moderator}[{group_clean_label}]"] = len(sliced_subgroup_df)
                coef_header = f"{moderator}[{group_clean_label}] Coef"
                pval_header = f"{moderator}[{group_clean_label}] P-Val"
                
                coef_values = []
                pval_values = []
                for pred in predictors:
                    # Defensive lookup step if path calculation exists
                    if pred in subgroup_model:
                        coef_values.append(subgroup_model[pred]['coef'])
                        pval_values.append(subgroup_model[pred]['p_val'])
                    else:
                        coef_values.append(np.nan)
                        pval_values.append(np.nan)
                
                # Commit new data vectors back directly into the tracking master frame
                compiled_matrix_df[coef_header] = coef_values
                compiled_matrix_df[pval_header] = pval_values
        
        # 4. Programmatic Delta Verification Steps (Vectorized across the full frame)
        coef_cols = [c for c in compiled_matrix_df.columns if 'Coef' in c and 'Baseline' not in c]
        if len(coef_cols) >= 2:
            for i in range(len(coef_cols)):
                for j in range(i + 1, len(coef_cols)):
                    col_a = coef_cols[i]
                    col_b = coef_cols[j]
                    
                    # Extract pure subgroup labels from string headers cleanly
                    label_a = col_a.split(']')[0].split('[')[-1]
                    label_b = col_b.split(']')[0].split('[')[-1]
                    delta_col_name = f"Δ ({label_a} vs {label_b})"
                    
                    # Single vectorized matrix delta evaluation loop
                    compiled_matrix_df[delta_col_name] = np.abs(
                        compiled_matrix_df[col_a] - compiled_matrix_df[col_b]
                    )

        # 5. Append sample size observation row
        n_row = {col: '' for col in compiled_matrix_df.columns}
        n_row['Baseline Coef'] = len(global_df)
        for key, count in subgroup_n.items():
            n_row[f"{key} Coef"] = count
        compiled_matrix_df.loc['Observations (N)'] = pd.Series(n_row)

        # 6. Compile matrix dataframe out to Markdown buffer block strings
        relation_header = f"### Subgroup Comparison Matrix: {target_var} ~ {' + '.join(predictors)}\n\n"
        markdown_table = (
            compiled_matrix_df.reset_index()
            .rename(columns={'index': 'Structural Path'})
            .to_markdown(index=False)
        )
        
        mga_markdown_output.append(relation_header + markdown_table + "\n\n")
        mga_dataframes.append(
            compiled_matrix_df.reset_index()
            .rename(columns={'index': 'Structural Path'})
        )
        
    return "".join(mga_markdown_output), mga_dataframes




