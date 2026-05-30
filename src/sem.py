import pandas as pd
import numpy as np
from semopy import Model
import semopy
import graphviz
import re
from typing import List, Dict, Tuple, Any, Optional

from src.helpers import logger, get_output_path
from src.common import convert_to_model_spec_dict


ATTENTION_SEMOPY_INTERPRETATION = """
### **SEM Interpretation Notes**
Setting the estimate of the first construct to 1 in SEMopy establishes a baseline
for comparison, making one indicator a reference point for evaluating relationships.
"""


def create_model_spec(
    independent_dict: List[Dict], 
    mediator_dict: List[Dict], 
    dependent_dict: List[Dict], 
    relation_dict: List[Dict], 
    varcovar_dict: List[Dict]
) -> Tuple[str, Dict[str, Any]]:
    """Creates a SEM-compatible model specification string and dictionary.

    Args:
        independent_dict: Independent variable definitions.
        mediator_dict: Mediator variable definitions.
        dependent_dict: Dependent variable definitions.
        relation_dict: Relationship definitions.
        varcovar_dict: Variance-covariance definitions.

    Returns:
        Tuple[str, Dict[str, Any]]: SEM model spec string and parsed dictionary.
    """
    model_spec = "### Independent Variables\n"
    model_spec += "\n".join([
        f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(int(d['number_questions']))]) 
        for d in independent_dict
    ])

    model_spec += "\n\n### Mediator Variables\n"
    model_spec += "\n".join([f"{r['Variable']} =~ {r['Related_Variables']}" for r in mediator_dict])

    model_spec += "\n\n### Dependent Variables\n"
    model_spec += "\n".join([
        f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(int(d['number_questions']))]) 
        for d in dependent_dict
    ])

    model_spec += "\n\n### Relations\n"
    model_spec += "\n".join([f"{r['Variable']} ~ {r['Related_Variables']}" for r in relation_dict])

    model_spec_dict = convert_to_model_spec_dict(model_spec)
    return model_spec, model_spec_dict


def extract_indicators_from_sem_spec(sem_spec: str) -> List[str]:
    """Extracts indicator names (e.g., Var_Q1) from the SEM spec.

    Args:
        sem_spec: SEM specification string.

    Returns:
        List[str]: Sorted list of unique indicator names.
    """
    regex = re.compile(r'\b\w+_Q\d+\b')
    indicators = list(set(re.findall(regex, sem_spec)))
    indicators.sort()
    return indicators


def conduct_sem_analysis(
    data: pd.DataFrame, 
    sem_model_spec: str, 
    observable_dict: List[Dict], 
    mediator_dict: List[Dict], 
    dependent_dict: List[Dict]
) -> Tuple[Any, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str]:
    """Fits an SEM model and performs post-processing of results.

    Args:
        data: Pre-processed survey data.
        sem_model_spec: SEM spec string.
        observable_dict: Observable variable definitions.
        mediator_dict: Mediator variable definitions.
        dependent_dict: Dependent variable definitions.

    Returns:
        Tuple: (fit_result, stats_df, raw_inspect, enhanced_inspect, filtered_inspect, short_graph_path, full_graph_path).
    """
    logger.info("Conducting SEM analysis")
    try:
        sem_model = Model(sem_model_spec)
        sem_model.fit(data)
        
        sem_stats = semopy.calc_stats(sem_model)
        sem_inspect = sem_model.inspect(std_est=True)
        
        logger.info(f"SEM Inspection output:\n{sem_inspect.to_string()}")
        
        enhanced, filtered, g_short, g_full = post_process_sem_results(
            sem_model_spec, sem_inspect, observable_dict, mediator_dict, dependent_dict
        )
        
        return None, sem_stats, sem_inspect, enhanced, filtered, g_short, g_full
    except np.linalg.LinAlgError:
        logger.error("SEM Analysis failed: Non-positive definite matrix.")
        raise Exception("The model failed to fit because the data matrix is not positive definite. This is likely due to high multicollinearity or extreme redundancy between indicators. Please check correlations (Step 6) and consider pruning highly correlated redundant items.")
    except Exception as e:
        logger.error(f"SEM Analysis failed: {e}")
        raise


def post_process_sem_results(
    sem_model_spec: str, 
    sem_inspect: pd.DataFrame, 
    observable_dict: List[Dict], 
    mediator_dict: List[Dict], 
    dependent_dict: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Enhances inspection results and generates result visualizations.

    Args:
        sem_model_spec: SEM spec string.
        sem_inspect: Raw inspection DataFrame from semopy.
        observable_dict: Observable variable definitions.
        mediator_dict: Mediator definitions.
        dependent_dict: Dependent definitions.

    Returns:
        Tuple: (enhanced_df, filtered_df, short_graph_path, full_graph_path).
    """
    output_path = get_output_path()
    enhanced = enhance_inspection(sem_inspect)
    
    # Save results to Excel
    enhanced.to_excel(f'{output_path}SEM_Results.xlsx', index=True)

    filtered = filter_inspect_table_from_spec(sem_inspect, sem_model_spec)
    
    g_short = create_graph_for_sem_results_short(
        sem_model_spec, sem_inspect, observable_dict, mediator_dict, dependent_dict
    )
    g_full = create_graph_for_sem_results_full(
        sem_model_spec, sem_inspect, observable_dict, mediator_dict, dependent_dict
    )
    
    return enhanced, filtered, g_short, g_full


def filter_inspect_table_from_spec(sem_inspect: pd.DataFrame, sem_model_spec: str) -> pd.DataFrame:
    """Filters the inspection table to show only relationships defined in the structural model.

    Args:
        sem_inspect: Enhanced inspection DataFrame.
        sem_model_spec: SEM specification string.

    Returns:
        pd.DataFrame: Filtered inspection table.
    """
    lines = sem_model_spec.strip().split("\n")
    factors = set([line.split('=~')[0].strip() for line in lines if '=~' in line])
    
    # Structural relations are ~
    mask = (sem_inspect['op'] == '~') & (sem_inspect['lval'].isin(factors)) & (sem_inspect['rval'].isin(factors))
    return sem_inspect[mask].copy()


def enhance_inspection(df: pd.DataFrame) -> pd.DataFrame:
    """Adds significance and relation direction labels to the semopy inspection table.

    Args:
        df: Input inspection DataFrame.

    Returns:
        pd.DataFrame: Enhanced DataFrame.
    """
    df = df.copy()
    df['p-value'] = pd.to_numeric(df['p-value'], errors='coerce')
    df['Estimate'] = pd.to_numeric(df['Estimate'], errors='coerce')

    df['Significance'] = df['p-value'].apply(lambda p: 'Significant' if p < 0.05 else 'Not Significant')
    df['Relation'] = df['Estimate'].apply(lambda e: 'Positive' if e > 0 else 'Negative' if e < 0 else 'Neutral')
    return df


def create_label(row: Dict[str, Any]) -> str:
    """Creates a formatted string label for graph edges based on SEM results.
    Safely handles non-numeric values like '-'.

    Args:
        row: A row from the inspection DataFrame as a dictionary.

    Returns:
        str: Formatted label string.
    """
    parts = []
    
    def safe_float_format(val: Any, prefix: str) -> Optional[str]:
        if pd.isnull(val):
            return None
        try:
            f_val = float(val)
            return f"{prefix}: {f_val:.3f}"
        except (ValueError, TypeError):
            return None

    est_std_label = safe_float_format(row.get('Est. Std'), "Std Coef")
    if est_std_label:
        parts.append(est_std_label)
        
    p_val_label = safe_float_format(row.get('p-value'), "p-val")
    if p_val_label:
        parts.append(p_val_label)
        
    return ', '.join(parts)


def create_graph_for_sem_results_full(
    sem_model_spec: str, 
    sem_inspect: pd.DataFrame, 
    observable_dict: List[Dict], 
    mediator_dict: List[Dict], 
    dependent_dict: List[Dict]
) -> str:
    """Generates a full Graphviz visualization of SEM results including indicators.

    Args:
        sem_model_spec: SEM spec string.
        sem_inspect: Inspection DataFrame.
        observable_dict: Independent variable definitions.
        mediator_dict: Mediator definitions.
        dependent_dict: Dependent definitions.

    Returns:
        str: Path to the generated image.
    """
    output_path = get_output_path()
    g = graphviz.Digraph('SEM Full Results', format='png', engine='dot')
    g.attr(rankdir='RL', fontsize='12')

    med_vars = [item['Variable'] for item in mediator_dict]
    dep_vars = [item['Variable'] for item in dependent_dict]
    obs_vars = [item['Variable'] for item in observable_dict]
    indicators = extract_indicators_from_sem_spec(sem_model_spec)

    # Nodes
    for ind in indicators:
        g.node(ind, shape='box', fillcolor='#e6f2ff', style='filled')
    for var in obs_vars:
        g.node(var, shape='ellipse', fillcolor='#cae6df', style='filled')
    for var in med_vars:
        g.node(var, shape='ellipse', fillcolor='green', style='filled')
    for var in dep_vars:
        g.node(var, shape='ellipse', fillcolor='#FFFF00', style='filled')

    # Edge labels lookup
    edge_map = {(row['lval'], row['rval']): row for _, row in sem_inspect.iterrows()}

    lines = sem_model_spec.strip().split("\n")
    for line in lines:
        if '=~' in line:
            lhs, rhs = [x.strip() for x in line.split('=~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                row = edge_map.get((var, lhs))
                label = create_label(row) if row is not None else ""
                g.edge(lhs, var, dir='forward', label=label)
        elif '~' in line and '=~' not in line:
            lhs, rhs = [x.strip() for x in line.split('~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                row = edge_map.get((lhs, var))
                label = create_label(row) if row is not None else ""
                
                # Safely get p-value for coloring
                p_val = 1.0
                if row is not None:
                    try:
                        p_val = float(row.get('p-value', 1.0))
                    except (ValueError, TypeError):
                        pass
                
                color = 'blue' if p_val < 0.05 else 'red'
                g.edge(var, lhs, dir='forward', color=color, label=label, fontcolor=color)

    g.attr(dpi='600')
    save_path = f"{output_path}sem_graph_results_full"
    g.render(save_path, cleanup=True)
    return f"{save_path}.png"


def create_graph_for_sem_results_short(
    sem_model_spec: str, 
    sem_inspect: pd.DataFrame, 
    observable_dict: List[Dict], 
    mediator_dict: List[Dict], 
    dependent_dict: List[Dict]
) -> str:
    """Generates a simplified Graphviz visualization of SEM results showing only latent relationships.

    Args:
        sem_model_spec: SEM spec string.
        sem_inspect: Inspection DataFrame.
        observable_dict: Independent definitions.
        mediator_dict: Mediator definitions.
        dependent_dict: Dependent definitions.

    Returns:
        str: Path to the generated image.
    """
    output_path = get_output_path()
    g = graphviz.Digraph('SEM Short Results', format='png', engine='dot')
    g.attr(rankdir='LR', fontsize='12')

    med_vars = [item['Variable'] for item in mediator_dict]
    dep_vars = [item['Variable'] for item in dependent_dict]
    obs_vars = [item['Variable'] for item in observable_dict]

    for var in obs_vars:
        g.node(var, shape='ellipse', fillcolor='#cae6df', style='filled')
    for var in med_vars:
        g.node(var, shape='ellipse', fillcolor='green', style='filled')
    for var in dep_vars:
        g.node(var, shape='ellipse', fillcolor='#FFFF00', style='filled')

    edge_map = {(row['lval'], row['rval']): row for _, row in sem_inspect.iterrows()}

    lines = sem_model_spec.strip().split("\n")
    for line in lines:
        if '~' in line and '=~' not in line:
            lhs, rhs = [x.strip() for x in line.split('~', 1)]
            for var in [x.strip() for x in rhs.split('+')]:
                row = edge_map.get((lhs, var))
                label = create_label(row) if row is not None else ""
                
                # Safely get p-value for coloring
                p_val = 1.0
                if row is not None:
                    try:
                        p_val = float(row.get('p-value', 1.0))
                    except (ValueError, TypeError):
                        pass
                
                color = 'blue' if p_val < 0.05 else 'red'
                g.edge(var, lhs, dir='forward', color=color, label=label, fontcolor=color)

    g.attr(dpi='600')
    save_path = f"{output_path}sem_graph_results_short"
    g.render(save_path, cleanup=True)
    return f"{save_path}.png"


def interpret_sem_stats(sem_stats: pd.DataFrame, parameters_dict: List[Dict]) -> Tuple[pd.DataFrame, str]:
    """Interprets global fit indices for the SEM model.

    Args:
        sem_stats: stats DataFrame from semopy.
        parameters_dict: Configuration for thresholds.

    Returns:
        Tuple: (formatted_stats_df, overall_interpretation_message).
    """
    output_path = get_output_path()
    
    # Defaults
    t_chi2 = 0.05
    t_cfi = 0.9
    t_rmsea = 0.08

    # Format the stats table
    stats_df = sem_stats.T.reset_index()
    stats_df.columns = ['Metric', 'Value']
    
    metric_mapping = {
        'DoF': 'Degrees of Freedom',
        'chi2': 'Chi-Square',
        'chi2 p-value': 'Chi-Square p-value',
        'CFI': 'Comparative Fit Index',
        'GFI': 'Goodness of Fit Index',
        'RMSEA': 'Root Mean Square Error',
        'AIC': 'Akaike Information Criterion',
        'BIC': 'Bayesian Information Criterion'
    }
    stats_df['Full Name'] = stats_df['Metric'].map(lambda x: metric_mapping.get(x, x))
    stats_df.to_excel(f'{output_path}SEM_Model_Stats.xlsx', index=False)

    # Global interpretation
    chi2_p = sem_stats['chi2 p-value'].values[0]
    cfi = sem_stats['CFI'].values[0]
    rmsea = sem_stats['RMSEA'].values[0]

    reasons = []
    if chi2_p > t_chi2 and cfi > t_cfi and rmsea < t_rmsea:
        msg = "The model shows a GOOD fit to the data."
    elif cfi > 0.8 and rmsea < 0.1:
        msg = "The model shows an ACCEPTABLE fit to the data."
    else:
        if cfi <= 0.8: reasons.append(f"CFI ({cfi:.3f}) is low")
        if rmsea >= 0.1: reasons.append(f"RMSEA ({rmsea:.3f}) is high")
        msg = f"The model fit is POOR. Reasons: {', '.join(reasons)}"

    return stats_df, msg


def interepret_sem_inspect(
    enhanced_inspect: pd.DataFrame, 
    dependent_dict: List[Dict], 
    relation_dict: List[Dict], 
    parameters_dict: List[Dict]
) -> List[str]:
    """Generates qualitative interpretation of individual paths in the SEM model.

    Args:
        enhanced_inspect: Enhanced inspection DataFrame.
        dependent_dict: Dependent variable definitions.
        relation_dict: Relationship definitions.
        parameters_dict: Configuration for thresholds.

    Returns:
        List[str]: List of interpretation strings.
    """
    interpretations = [ATTENTION_SEMOPY_INTERPRETATION]
    
    # Group by dependent variable
    deps = [d['Variable'] for d in dependent_dict]
    for dep in deps:
        rel_rows = enhanced_inspect[(enhanced_inspect['lval'] == dep) & (enhanced_inspect['op'] == '~')]
        if rel_rows.empty:
            continue
            
        interp = f"### Analysis for **{dep}**\n"
        interp += f"- `{dep}` is modeled as being influenced by: "
        interp += ", ".join([f"`{r}`" for r in rel_rows['rval'].unique()]) + ".\n\n"
        
        for _, row in rel_rows.iterrows():
            pred = row['rval']
            sig = row['Significance']
            
            # Safely extract estimate and p-value
            try:
                est = float(row['Estimate'])
                est_str = f"{est:.3f}"
            except (ValueError, TypeError):
                est = 0.0
                est_str = "N/A"
                
            try:
                p_val = float(row['p-value'])
                p_str = f"{p_val:.3f}"
            except (ValueError, TypeError):
                p_val = 1.0
                p_str = "N/A"
            
            direction = "positive" if est >= 0 else "negative"
            sig_text = "statistically significant" if sig == 'Significant' else "NOT significant"
            
            interp += f"- **{pred} -> {dep}**: Has a {direction} influence (Est: {est_str}, p: {p_str}). "
            interp += f"This relationship is {sig_text}.\n"
            
        interpretations.append(interp)
        
    return interpretations



