
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler


output_path = './output/'

# Testing the reliability of the scale using Cronbach's alpha for observable variables
def cronbach_alpha(items_df):
    """
    Compute Cronbach's Alpha for a set of items (columns) in a DataFrame.

    :param items_df: DataFrame, where each column is an item and rows are observations
    :return: float, Cronbach's Alpha
    """
    # Number of items
    item_count = len(items_df.columns)
    
    # Variance for every individual item
    item_variances = items_df.var(axis=0)
    
    # Total variance for the item scores
    total_var = item_variances.sum()
    
    # Sum of the item-pair covariances
    item_covariances_sum = items_df.cov().sum().sum() - total_var
    
    # Cronbach's alpha formula
    alpha = (item_count / (item_count - 1)) * (1 - (total_var / (total_var + item_covariances_sum)))
    
    return alpha

def compute_cronbach_alpha(data_normalized, observable_dict):
    # Compute Cronbach Alpha for each Observable Variable
    alpha_table = []
    observable_col_names = []
    for col in observable_dict:
        col_names = []
        variable_name = col.get('Variable')
        nbr_questions = col.get('number_questions')
        for idx in range(nbr_questions):
            col_names.append(f'{variable_name}_Q{idx+1}')
            observable_col_names.append(f'{variable_name}_Q{idx+1}')
        data_of_this_variable = data_normalized[col_names]
        # Calcul Alphe for this variable
        alpha_variable = cronbach_alpha(data_of_this_variable)
        alpha_variable = round(alpha_variable, 2)
        alpha_evaluation = interpret_alpha(alpha_variable)
        alpha_table.append([alpha_variable, alpha_evaluation])


    # Overall Cronbach's Alpha
    overall_alpha = cronbach_alpha(data_normalized[observable_col_names])
    overall_alpha = round(overall_alpha, 2)
    overall_alpha_evaluation = interpret_alpha(overall_alpha)

    print(f"Overall Cronbach's Alpha: {overall_alpha}")

    return f"{overall_alpha}, '{overall_alpha_evaluation}'", alpha_table

def interpret_alpha(alpha):
    if alpha > 0.9:
        return "Excellent, >0.9"
    elif alpha > 0.8:
        return "Good, > 0.8"
    elif alpha > 0.7:
        return "Acceptable, > 0.7"
    elif alpha > 0.6:
        return "Questionable, > 0.6"
    elif alpha > 0.5:
        return "Poor, > 0.5"
    else:
        return "Unacceptable, <= 0.5"


def conduct_efa_analysis(data_normalized, observable_dict):
    observable_col_names = []
    for col in observable_dict:
        col_names = []
        variable_name = col.get('Variable')
        nbr_questions = col.get('number_questions')
        for idx in range(nbr_questions):
            observable_col_names.append(f'{variable_name}_Q{idx+1}')

    observable_data = data_normalized[observable_col_names]
    # Standardize the data
    scaler = StandardScaler()
    selected_data_standardized = scaler.fit_transform(observable_data)

    # Fit Factor Analysis model to the data to find eigenvalues
    fa = FactorAnalysis()
    fa.fit(selected_data_standardized)
    eigenvalues = np.linalg.eigvals(fa.get_covariance())
    
    # Determine the number of factors to retain based on eigenvalues greater than one
    num_factors = np.sum(eigenvalues > 1)

    # Initialize Factor Analysis
    fa = FactorAnalysis(n_components=num_factors)

    # Fit Factor Analysis to the standardized data
    fa.fit(selected_data_standardized)

    # Loadings are the coefficients of the original variables
    loadings = pd.DataFrame(fa.components_, columns=observable_data.columns)

    # Save the loadings to an Excel file
    filename = 'EFA_Analysis'
    excel_file_path = f'{output_path}{filename}.xlsx'
    loadings.to_excel(excel_file_path, index=True)  

    return loadings

def interpret_efa_results(efa_results, threshold_high, threshold_moderate):
   # Number of factors extracted
    num_factors = efa_results.shape[0]
    
    # Average loading
    avg_loading = efa_results.abs().mean().mean()

    # Identify questions with low loadings on all factors
    threshold_low = threshold_moderate  # Threshold for low loadings
    low_loading_questions = efa_results.columns[(efa_results.abs() < threshold_low).all(axis=0)]
    
    # Interpretation
    interpretation = []
    interpretation.append(f"The analysis identified {num_factors} latent main factors influencing the responses.")
    
    if avg_loading > threshold_high:
        interpretation.append("On average, the questions exhibit a strong association with at least one of the extracted factors. This indicates the identified factors capture significant variations in the data.")
    elif avg_loading > threshold_moderate:
        interpretation.append("On average, there's a moderate association between questions and the extracted factors. While the factors capture some variation, further refinement might improve clarity.")
    else:
        interpretation.append("The average associations between most questions and the extracted factors appear weak. It might be worthwhile to revisit the questions or factor extraction criteria for improvements.")

    if not low_loading_questions.empty:
        interpretation.append(f"The following questions demonstrate weak associations with all extracted factors and might need reconsideration: {', '.join(low_loading_questions)}.")
    else:
        interpretation.append("All questions have at least a moderate association with one or more factors, which is encouraging.")
    
    # Extract high and moderate loading questions
    # Iterate through factors (rows in your EFA results)
    loading_data = []
    for factor_idx, factor in efa_results.iterrows():
        # Find columns (questions) with high loadings for this factor
        high_loadings = factor[factor.abs() > threshold_high]
        high_loading_questions = [f"{question} ({loading:.2f})" for question, loading in high_loadings.items()]

        # Find columns (questions) with moderate loadings for this factor after excluding high loading ones
        remaining_factor = factor.drop(high_loadings.index)
        moderate_loadings = remaining_factor[(remaining_factor.abs() <= threshold_high) & (remaining_factor.abs() > threshold_moderate)]
        moderate_loading_questions = [f"{question} ({loading:.2f})" for question, loading in moderate_loadings.items()]

        # Construct a row for this factor in the new data
        factor_name = f"Factor {factor_idx + 1}"
        loading_data.append([factor_name, ', '.join(high_loading_questions), ', '.join(moderate_loading_questions)])

    # Convert the data to a DataFrame
    df_loadings = pd.DataFrame(loading_data, columns=['Factor', 'High Loading Questions', 'Moderate Loading Questions'])
    
    
    return interpretation, df_loadings
    

    
# Pearson correlation matrix
def compute_correlation(data_normalized, observable_dict):

    observable_col_names = []
    for col in observable_dict:
        col_names = []
        variable_name = col.get('Variable')
        nbr_questions = col.get('number_questions')
        for idx in range(nbr_questions):
            observable_col_names.append(f'{variable_name}_Q{idx+1}')

    observable_data = data_normalized[observable_col_names]

    correlation_matrix = observable_data.corr(method='pearson')

    print("Pearson correlation coefficient matrix: ")
    print(correlation_matrix)
    # Save the Correlation Table to an Excel file
    # Set index=False to exclude the DataFrame index
    filename = 'Correlation_Matrix'
    excel_file_path = f'{output_path}{filename}.xlsx'
    correlation_matrix.to_excel(excel_file_path, index=True)  

    figsize=(10, 8)
    plt.figure(figsize=figsize)
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1, 
                linewidths=.5, fmt=".2f") # , mask=~mask)
    # plt.title(f'Simplified Correlation Matrix with correlation > {threshold}', fontsize=16)
    plt.title(f'Correlation Matrix', fontsize=16)

    # Save the plot if save_path is provided
    save_path_heatmap = f'{output_path}correlation_heatmap.png'
    if save_path_heatmap:
        plt.savefig(save_path_heatmap, bbox_inches='tight')
    plt.close()  # Close the figure after saving

    return correlation_matrix

def interpret_correlation(correlation_matrix, threshold_strong,threshold_moderate):
    # Remove self-correlations by setting diagonal to NaN
    for i in range(correlation_matrix.shape[0]):
        correlation_matrix.iloc[i, i] = np.nan
    
    # Compute the average absolute correlation
    avg_correlation = correlation_matrix.abs().mean().mean()

    # Identify variables with low correlations with others
    threshold_low = threshold_moderate  # Threshold for low correlation
    low_correlation_variables = correlation_matrix.columns[(correlation_matrix.abs().mean() < threshold_low)]
    
    # Identify pairs of variables with high correlations
    threshold_high = threshold_strong  # Threshold for high correlation
    high_correlation_pairs = [(var1, var2) for var1 in correlation_matrix.columns \
            for var2 in correlation_matrix.columns \
                if (correlation_matrix.loc[var1, var2] > threshold_high) and (var1 != var2)]
    
    # Interpretation
    interpretation = []
    if avg_correlation > threshold_strong:
        comment = "Most variables in the dataset are strongly related to each other."
    elif avg_correlation > threshold_moderate:
        comment ="There are moderate relationships between variables in the dataset."
    else:
        comment = "The variables in the dataset are mostly weakly related or unrelated to each other."
    interpretation.append(comment)
    
    if not low_correlation_variables.empty:
        interpretation.append(f"The following variables have weak relationships with most other variables: {', '.join(low_correlation_variables)}. ")
        interpretation.append('Consider reviewing their relevance to the study.')
    if high_correlation_pairs:
        interpretation.append(f"Some pairs of variables are highly correlated, indicating potential redundancy. ") 
        interpretation.append(f"These pairs are: " + ', '.join([f"({var1}, {var2})" for var1, var2 in high_correlation_pairs]) + ".")
        interpretation.append('Consider reviewing whether both variables in each pair are necessary, or whether dimensionality reduction or feature engineering might be appropriate.')
    return interpretation


