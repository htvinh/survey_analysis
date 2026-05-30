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



def interpret_moderator_results(comparison_data: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:
    """Generates qualitative observations based on subgroup coefficient differences.

    Args:
        comparison_data: Dictionary mapping relationships to comparison DataFrames.

    Returns:
        Dict[str, List[str]]: Mapping of relationship to list of observations.
    """
    observations = {}
    for rel, df in comparison_data.items():
        rel_obs = []
        baseline_col = 'Baseline Coef'
        
        # Columns that are coefficients (baseline or subgroup)
        subgroup_cols = [c for c in df.columns if 'Coef' in c]
        
        # Need the actual baseline coefficient column name
        baseline_col = 'Baseline Coef'
        
        for var in df.index:
            baseline_val = df.loc[var, baseline_col]
            for col in subgroup_cols:
                if col == baseline_col:
                    continue
                    
                sub_val = df.loc[var, col]
                if pd.notnull(sub_val):
                    diff = sub_val - baseline_val
                    if abs(diff) > 0.1:  # Threshold for "notable" difference
                        direction = "stronger" if diff > 0 else "weaker"
                        # Clean column name for better report readability (remove '_Coef')
                        group_name = col.replace('_Coef', '')
                        rel_obs.append(f"- **{var}**: Effect is {direction} for {group_name} (Δ={diff:.2f}) compared to baseline.")
        
        if rel_obs:
            observations[rel] = rel_obs
            
    return observations

def conduct_regression_analysis_with_moderators(
    model_spec_dict: Dict[str, Any], 
    data_normalized: pd.DataFrame, 
    label_mappings: Dict, 
    moderators: List[str], 
    baseline_results: List[List[Any]]
) -> Dict[str, pd.DataFrame]:
    """Performs subgroup analysis for each unique value of moderator variables.

    Args:
        model_spec_dict: Parsed model specification.
        data_normalized: Pre-processed data.
        label_mappings: Categorical label mappings.
        moderators: List of variable names to use as moderators.
        baseline_results: Results from the full dataset for comparison.

    Returns:
        Dict[str, pd.DataFrame]: Dictionary mapping relationship names to comparison tables.
    """
    logger.info(f"Conducting moderator analysis for: {moderators}")
    from src.helpers import get_back_original_label_from_numerical_label
    
    comparison_data = {}
    
    # Initialize comparison tables with baseline
    for relation, summary in baseline_results:
        _, _, coefs = extract_regression_results(summary)
        df_base = pd.DataFrame(coefs).T[['Estimated Coefficient', 'P-Value']]
        df_base.columns = ['Baseline Coef', 'Baseline P-Val']
        comparison_data[relation] = df_base

    for moderator in moderators:
        if moderator not in data_normalized.columns:
            continue
            
        unique_vals = data_normalized[moderator].unique()
        for val in unique_vals:
            label = get_back_original_label_from_numerical_label(label_mappings, moderator, val) or str(val)
            subset = data_normalized[data_normalized[moderator] == val]
            
            if len(subset) < 5: # Skip very small subgroups
                continue
                
            sub_results = conduct_regression_analysis(subset, model_spec_dict)
            for rel, sub_summary in sub_results:
                if rel in comparison_data:
                    _, _, sub_coefs = extract_regression_results(sub_summary)
                    col_name = f"{moderator}_{label}_Coef"
                    for var, info in sub_coefs.items():
                        if var in comparison_data[rel].index:
                            comparison_data[rel].loc[var, col_name] = info['Estimated Coefficient']

    return comparison_data



