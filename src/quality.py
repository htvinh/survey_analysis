import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any

from src.helpers import logger, get_output_path


def cronbach_alpha(items_df: pd.DataFrame) -> float:
    """Calculates Cronbach's Alpha for a given set of items.

    Args:
        items_df: DataFrame containing the items (questions) for a construct.

    Returns:
        float: The calculated Cronbach's Alpha coefficient.
    """
    item_count = len(items_df.columns)
    if item_count <= 1:
        return 0.0
        
    item_variances = items_df.var(axis=0)
    total_var = item_variances.sum()
    item_covariances_sum = items_df.cov().sum().sum() - total_var
    
    if (total_var + item_covariances_sum) == 0:
        return 0.0
        
    alpha = (item_count / (item_count - 1)) * (1 - (total_var / (total_var + item_covariances_sum)))
    return alpha


def compute_cronbach_alpha(
    data_normalized: pd.DataFrame, 
    observable_dict: List[Dict[str, Any]]
) -> Tuple[str, List[List[Any]]]:
    """Computes Cronbach's Alpha for each variable and overall.

    Args:
        data_normalized: The pre-processed survey data.
        observable_dict: Mapping of variables to their indicators.

    Returns:
        Tuple: (Overall alpha summary string, List of individual variable alphas and evaluations).
    """
    logger.info("Computing Cronbach's Alpha")
    alpha_table = []
    observable_col_names = []
    
    for col in observable_dict:
        col_names = []
        variable_name = col.get('Variable')
        nbr_questions = col.get('number_questions', 0)
        
        for idx in range(int(nbr_questions)):
            q_name = f'{variable_name}_Q{idx+1}'
            if q_name in data_normalized.columns:
                col_names.append(q_name)
                observable_col_names.append(q_name)
        
        if col_names:
            data_of_this_variable = data_normalized[col_names]
            alpha_variable = round(cronbach_alpha(data_of_this_variable), 2)
            alpha_evaluation = interpret_alpha(alpha_variable)
            alpha_table.append([alpha_variable, alpha_evaluation])
        else:
            alpha_table.append([0.0, "No Data"])

    if observable_col_names:
        overall_alpha = round(cronbach_alpha(data_normalized[observable_col_names]), 2)
        overall_alpha_evaluation = interpret_alpha(overall_alpha)
    else:
        overall_alpha = 0.0
        overall_alpha_evaluation = "No Data"

    logger.info(f"Overall Cronbach's Alpha: {overall_alpha}")

    return f"{overall_alpha}, '{overall_alpha_evaluation}'", alpha_table


def interpret_alpha(alpha: float) -> str:
    """Interprets the reliability level based on Cronbach's Alpha value.

    Args:
        alpha: The alpha coefficient.

    Returns:
        str: Qualitative evaluation string.
    """
    if alpha > 0.9:
        return "Excellent, >0.9"
    elif alpha > 0.8:
        return "Good, > 0.8"
    elif alpha > 0.7:
        return "Acceptable, > 0.7"
    elif alpha > 0.6:
        return "Questionable, < 0.7"
    elif alpha > 0.5:
        return "Poor, < 0.6"
    else:
        return "Unacceptable, <= 0.5"


def compute_cr_ave(loadings: List[float]) -> Tuple[float, float]:
    """Calculates Composite Reliability (CR) and Average Variance Extracted (AVE).

    Args:
        loadings: List of standardized factor loadings for a construct.

    Returns:
        Tuple: (Composite Reliability, Average Variance Extracted).
    """
    loadings_arr = np.array(loadings)
    sum_loadings_sq = np.sum(loadings_arr) ** 2
    sum_sq_loadings = np.sum(loadings_arr ** 2)
    n = len(loadings_arr)
    
    # Error variance: 1 - loading^2
    sum_error_var = np.sum(1 - loadings_arr ** 2)
    
    # CR = (sum loadings)^2 / ((sum loadings)^2 + sum error variance)
    cr = sum_loadings_sq / (sum_loadings_sq + sum_error_var) if (sum_loadings_sq + sum_error_var) > 0 else 0.0
    
    # AVE = sum(loading^2) / n
    ave = sum_sq_loadings / n if n > 0 else 0.0
    
    return cr, ave


def run_harman_single_factor_test(data: pd.DataFrame, observable_cols: List[str]) -> Tuple[float, str]:
    """Performs Harman's Single-Factor Test using PCA.

    Args:
        data: Pre-processed DataFrame.
        observable_cols: List of indicator column names.

    Returns:
        Tuple: (Variance explained by first factor, Interpretation message).
    """
    from sklearn.decomposition import PCA
    subset = data[observable_cols].dropna()
    if subset.empty:
        return 0.0, "No data available for Harman's test."
        
    pca = PCA(n_components=1)
    pca.fit(subset)
    variance_explained = pca.explained_variance_ratio_[0] * 100
    
    msg = f"First factor explains {variance_explained:.2f}% of variance. "
    if variance_explained < 50:
        msg += "Success: Common Method Bias (CMB) is not a major concern (< 50%)."
    else:
        msg += "Warning: Common Method Bias may be present (> 50%)."
        
    return variance_explained, msg


def check_discriminant_validity(construct_aves: Dict[str, float], construct_corrs: pd.DataFrame) -> List[str]:
    """Checks discriminant validity using the Fornell-Larcker criterion.
    The square root of AVE for each construct should be greater than its correlation with other constructs.

    Args:
        construct_aves: Dictionary mapping construct name to its AVE value.
        construct_corrs: Correlation matrix between constructs.

    Returns:
        List[str]: Interpretation comments.
    """
    results = []
    sq_root_aves = {c: np.sqrt(a) for c, a in construct_aves.items()}
    
    for c1 in sq_root_aves:
        for c2 in sq_root_aves:
            if c1 != c2:
                corr = abs(construct_corrs.loc[c1, c2])
                if sq_root_aves[c1] < corr:
                    results.append(f"Discriminant Validity issue: Corr({c1}, {c2}) = {corr:.3f} > Sqrt(AVE) of {c1} ({sq_root_aves[c1]:.3f})")
    
    if not results:
        results.append("Fornell-Larcker Criterion Met: All constructs show adequate discriminant validity.")
        
    return results


def compute_correlation(
    data_normalized: pd.DataFrame, 
    observable_dict: List[Dict[str, Any]]
) -> pd.DataFrame:
    """Computes the Pearson correlation matrix for observable variables and saves a heatmap.

    Args:
        data_normalized: Pre-processed survey data.
        observable_dict: Mapping of variables to indicators.

    Returns:
        pd.DataFrame: The correlation matrix.
    """
    logger.info("Computing correlation matrix")
    output_path = get_output_path()
    observable_col_names = []
    
    for col in observable_dict:
        variable_name = col.get('Variable')
        nbr_questions = col.get('number_questions', 0)
        for idx in range(int(nbr_questions)):
            q_name = f'{variable_name}_Q{idx+1}'
            if q_name in data_normalized.columns:
                observable_col_names.append(q_name)

    if not observable_col_names:
        logger.warning("No observable columns found for correlation analysis")
        return pd.DataFrame()

    observable_data = data_normalized[observable_col_names]
    correlation_matrix = observable_data.corr(method='pearson')

    # Save to Excel
    excel_file_path = f'{output_path}Correlation_Matrix.xlsx'
    correlation_matrix.to_excel(excel_file_path, index=True)

    # Generate Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1,
        linewidths=.5, fmt=".1f", annot_kws={"fontsize": 8}
    )
    plt.title('Correlation Matrix Heatmap', fontsize=16)
    
    save_path_heatmap = f'{output_path}correlation_heatmap.png'
    plt.savefig(save_path_heatmap, bbox_inches='tight')
    plt.close()

    return correlation_matrix


def interpret_correlation(
    correlation_matrix: pd.DataFrame, 
    threshold_strong: float, 
    threshold_moderate: float
) -> Tuple[List[str], List[str]]:
    """Interprets the correlation matrix identifying strong relationships and potential redundancy.

    Args:
        correlation_matrix: Pearson correlation matrix.
        threshold_strong: Threshold for strong correlation (redundancy check).
        threshold_moderate: Threshold for moderate correlation.

    Returns:
        Tuple: (List of interpretation comments, List of recommended items to drop).
    """
    if correlation_matrix.empty:
        return ["No correlation data available."], []

    # Exclude diagonal
    corr_no_diag = correlation_matrix.copy()
    for i in range(len(corr_no_diag)):
        corr_no_diag.iloc[i, i] = np.nan

    avg_correlation = corr_no_diag.abs().mean().mean()
    
    # Identify variables with weak overall relationships
    low_correlation_variables = correlation_matrix.columns[
        (corr_no_diag.abs().mean() < threshold_moderate)
    ]
    
    # Identify highly correlated pairs (potential redundancy)
    high_correlation_pairs = []
    recommended_drops = []
    visited_vars = set()
    
    for var1 in correlation_matrix.columns:
        for var2 in correlation_matrix.columns:
            if var1 != var2:
                # Use a higher strict threshold for auto-drop recommendations
                if correlation_matrix.loc[var1, var2] > 0.80:
                    pair = tuple(sorted((var1, var2)))
                    if pair not in high_correlation_pairs:
                        high_correlation_pairs.append(pair)
                        # Heuristic: if we haven't decided to drop var1, drop var2
                        if var1 not in visited_vars and var2 not in visited_vars:
                            recommended_drops.append(var2)
                            visited_vars.add(var2)

    interpretation = []
    if avg_correlation > threshold_strong:
        interpretation.append("Overall, variables show strong inter-relationships.")
    elif avg_correlation > threshold_moderate:
        interpretation.append("Overall, variables show moderate inter-relationships.")
    else:
        interpretation.append("Overall, variables show weak or no inter-relationships.")

    if not low_correlation_variables.empty:
        interpretation.append(
            f"Weak relationships detected for: {', '.join(low_correlation_variables)}. "
            "Consider reviewing their relevance."
        )
        
    if high_correlation_pairs:
        pairs_str = ', '.join([f"({p[0]}, {p[1]})" for p in high_correlation_pairs])
        interpretation.append(
            f"Critical redundancy (r > 0.80) detected in pairs: {pairs_str}. "
            "These items likely inflate standard errors and RMSEA."
        )
        
    return interpretation, recommended_drops
