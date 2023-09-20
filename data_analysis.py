from pingouin import cronbach_alpha
from factor_analyzer import FactorAnalyzer
from docx import Document
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder

import os

output_path = './output/'
# Check if the directory exists
if not os.path.exists(output_path):
    # If it doesn't exist, create the directory
    os.makedirs(output_path)
    print(f"Directory '{output_path}' created.")

def read_excel_to_dict(excel_path):
    # Read the Excel workbook
    xls = pd.ExcelFile(excel_path)
    
    # Initialize an empty dictionary to store all the data
    all_data = {}
    
    # Loop through each sheet and read data into a DataFrame, then convert it to a dictionary
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        all_data[sheet_name] = df.to_dict(orient='index')
        
    return all_data

# Display column names with index
def display_col_name_index(df):
  # Iterate through columns and display column names with indexes in brackets
  for i, column_name in enumerate(df.columns):
      print(f"Column {i} [{column_name}]")

# Rename columns for demographic data columns: to have shorter column names
def rename_demographic_columns_by_index(df, demographic_cols):
    demographic_cols_names = []  # Initialize a list to store independent column names
    for col in demographic_cols:
        col_index = col.get('col_index')
        new_column_name = col.get('name')
        df.rename(columns={df.columns[col_index]: new_column_name}, inplace=True)
        demographic_cols_names.append(new_column_name)
    return df, demographic_cols_names

# Rename Independent columns according to Independent_Cols
def rename_independent_columns(df, independent_cols):
    independent_cols_names = []  # Initialize a list to store independent column names
    for col in independent_cols:
        col_name = col.get('name')
        col_index_from = col.get('col_index_from')
        col_index_to = col_index_from + col.get('number_questions')

        # Rename columns based on Likert scale questions
        for i in range(col_index_from, col_index_to):
            # df.rename(columns={df.columns[i]: f'{col_name}_Q{i - col_index_from + 1}'}, inplace=True)
            old_col_name = df.columns[i]
            new_col_name = f'{col_name}_Q{i - col_index_from + 1}'
            df.rename(columns={old_col_name: new_col_name}, inplace=True)
            independent_cols_names.append(new_col_name)  # Append the new column name

    return df, independent_cols_names

# Rename Dependent/Target columns according to Dependent_Cols
def rename_target_columns(df, target_cols):
    target_cols_names = []  # Initialize a list to store Target column names
    for col in target_cols:
        col_name = col.get('name')
        col_index_from = col.get('col_index_from')

        # Rename the column
        df.rename(columns={df.columns[col_index_from]: col_name}, inplace=True)
        target_cols_names.append(col_name)  # Append the new column name

    return df, target_cols_names

# Create a statistics table for selected columns
def compute_selected_cols_statistics(df, selected_cols_names, output_file_name):
    nbr_data_points_cleaned = df.shape[0]
    stats_table = {}
    for col_name in selected_cols_names:
        freq = df[col_name].value_counts().sort_index()
        percent = (freq / nbr_data_points_cleaned) * 100
        percent = percent.round(1)  # Round to two decimal places
        stats_table[col_name] = pd.DataFrame({'Frequency': freq, 'Percent': percent})
        
        # Sort the DataFrame by 'Frequency' in descending order
        stats_table[col_name] = stats_table[col_name].sort_values(by='Frequency', ascending=False)
    
    # Save to Excel file
    output_file_name = f'{output_file_name}_Stats'
    save_to_excel(output_file_name, stats_table)

    del df
    return stats_table


# Convert the data in a nested dict into a dictionary of DataFrames
def convert_to_dataframe(table):
    dfs = {}
    for key, value in table.items():
        print(type(value))
        # lines = value.split("\n")
        # header = lines[0].split()
        # content = [line.split() for line in lines[1:]]
        # df = pd.DataFrame(content, columns=header)
        df = value
        dfs[key] = df

    # Concatenate the DataFrames vertically (along rows)
    combined_df = pd.concat(dfs.values(), axis=1, ignore_index=True)

    return combined_df

# Save the statistics table to an Excel file
def save_to_excel(filename, table):
    with pd.ExcelWriter(f'{output_path}{filename}.xlsx') as writer:
        for col, table in table.items():
            table.to_excel(writer, sheet_name=f'{col}')

# Extract data for selected columns
def extract_selected_colums_data(df, selected_cols_names):
    data_to_extract = df[selected_cols_names]
    # data_to_extract = convert_values_to_numeric(data_to_extract)
    data_to_extract = convert_likert_to_numerical(data_to_extract, extended_likert_mapping_all_languages)
    return data_to_extract

# Convert values of dataframe to numeric
def convert_values_to_numeric(df):
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode each column in the DataFrame
    for column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

# Function to manually convert Likert-scale responses to numerical values
# Function to manually convert Likert-scale responses to numerical values
def convert_likert_to_numerical(df, likert_mapping):
    df_numerical = df.copy()
    # Truncate leading and trailing spaces for consistent mapping
    # df_numerical = df_numerical.applymap(str.strip)
    # Convert all columns to string type for consistent mapping
    df_numerical = df_numerical.astype(str)
    # Convert the mapping keys to string type as well
    str_likert_mapping = {str(key): value for key, value in likert_mapping.items()}
    
    for column in df.columns:
        df_numerical[column] = df_numerical[column].map(str_likert_mapping)
        
    # Check for any remaining non-numeric values
    if df_numerical.applymap(np.isreal).all().all() == False:
        raise AssertionError("Not all columns are numeric after mapping. Check your mapping and DataFrame.")
        
    del df
    return df_numerical




# Sample Likert scale mapping (you can adjust this based on your actual survey Likert scale)
extended_likert_mapping_all_languages = {
    # English Labels
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 4,
    'Agree': 6,
    'Strongly Agree': 7,
    
    # Vietnamese Labels
    'Hoàn toàn không đồng ý': 1,
    'Không đồng ý': 2,
    'Không ý kiến': 4,
    'Đồng ý': 6,
    'Hoàn toàn đồng ý': 7,
    
    # Numerical Levels
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7
}




# Remove data points (rows) with missing data related to selected columns
def remove_rows_with_missing_data_related_to_selected_cols(df, selected_cols_names):
    total_num_data_points = df.shape[0]
    # Filter rows with missing data only in the selected columns
    df = df.dropna(subset=selected_cols_names)
    # Number of data points removed
    number_datapoints_removed = total_num_data_points - df.shape[0]
    print(f'\nNumber of data points removed: \n{number_datapoints_removed}')

    return df

# Testing the reliability of the scale using Cronbach's alpha for independent columns
def compute_cronbach_alpha(selected_data, independent_cols):
    # Calculate Cronbach's alpha
    overall_alpha = cronbach_alpha(selected_data)
    overall_alpha_evaluation = interpret_alpha(overall_alpha[0])

    # Compute Cronbach Alpha for each Independent Variables
    alpha_table = []
    for col in independent_cols:
        col_names = []
        variable_name = col.get('name')
        nbr_questions = col.get('number_questions')
        for idx in range(nbr_questions):
            col_names.append(f'{variable_name}_Q{idx+1}')
        print(col_names)
        data_of_this_variable = selected_data[col_names]
        # Calcul Alphe for this variable
        alpha_variable = cronbach_alpha(data_of_this_variable)
        alpha_evaluation = interpret_alpha(alpha_variable[0])
        alpha_table.append([alpha_variable, alpha_evaluation])

    # Print the result
    print(f"Overall Cronbach's Alpha: {overall_alpha}")

    return f"{overall_alpha}, '{overall_alpha_evaluation}'", alpha_table

def interpret_alpha(alpha):
    if alpha > 0.9:
        return "Excellent"
    elif alpha > 0.8:
        return "Good"
    elif alpha > 0.7:
        return "Acceptable"
    elif alpha > 0.6:
        return "Questionable"
    elif alpha > 0.5:
        return "Poor"
    else:
        return "Unacceptable"

# EFA analysis
def do_efa_analysis(selected_data, num_factors):
    fa = FactorAnalyzer(n_factors=num_factors, rotation='varimax')
    fa.fit(selected_data)
    factor_loadings = fa.loadings_
    # print(f"EFA loadings:\n{loadings}")
    # Visualize the factor loadings
    loadings_df = pd.DataFrame(factor_loadings, columns=[f'Factor {i+1}' for i in range(num_factors)], index=selected_data.columns)
    # print(loadings_df)
    # Save the EFA analysis to an Excel file
    filename = 'EFA_Analysis'
    # Set index=False to exclude the DataFrame index
    filename = 'EFA_Analysis'
    excel_file_path = f'{output_path}{filename}.xlsx'
    loadings_df.to_excel(excel_file_path, index=True)  

    return loadings_df

def interpret_based_on_loadings(loadings_df, threshold=0.5):
    interpretations = []
    for factor in loadings_df.columns:
        strong_vars = loadings_df.loc[loadings_df[factor].abs() >= threshold].index.tolist()
        
        if strong_vars:
            sentence = f"In simple terms, '{factor}' is mainly about {', '.join(strong_vars)}."
            interpretations.append({'Factor': factor, 'Interpretation': sentence})

    interpretation_df = pd.DataFrame(interpretations)
    return interpretation_df

def recommendations_based_on_loadings(loadings_df, threshold=0.5):
    recommendations = []
    for factor in loadings_df.columns:
        strong_vars = loadings_df.loc[loadings_df[factor].abs() >= threshold].index.tolist()
        
        if strong_vars:
            advice = f"To improve in '{factor}', consider focusing on {', '.join(strong_vars)}."
            recommendations.append({'Factor': factor, 'Recommendation': advice})

    recommendation_df = pd.DataFrame(recommendations)
    return recommendation_df

    
# Pearson correlation matrix
def compute_correlation(selected_data):
    correlation_table = selected_data.corr(method='pearson')
    print("Pearson correlation coefficient matrix: ")
    print(correlation_table)
    # Save the Correlation Table to an Excel file
    # Set index=False to exclude the DataFrame index
    filename = 'Correlation_Table'
    excel_file_path = f'{output_path}{filename}.xlsx'
    correlation_table.to_excel(excel_file_path, index=True)  

    return correlation_table

def interpret_and_recommend_correlation(correlation_table, threshold=0.5):
    interpretations = []
    recommendations = []

    # Loop over the upper triangle of the correlation matrix
    for i, row_var in enumerate(correlation_table.index):
        for j, col_var in enumerate(correlation_table.columns[i+1:]):  # i+1 to skip self-correlation
            corr_value = correlation_table.loc[row_var, col_var]

            # Only consider strong correlations (positive or negative)
            if abs(corr_value) >= threshold:
                if corr_value > 0:
                    interpretation = f"{row_var} and {col_var} have a strong positive relationship."
                    recommendation = f"Improving {row_var} could also positively impact {col_var}, and vice versa."
                else:
                    interpretation = f"{row_var} and {col_var} have a strong negative relationship."
                    recommendation = f"Focusing on {row_var} might have the opposite effect on {col_var}, so be cautious."

                interpretations.append({
                    'Variables': f"{row_var} & {col_var}",
                    'Correlation': corr_value,
                    'Interpretation': interpretation
                })

                recommendations.append({
                    'Variables': f"{row_var} & {col_var}",
                    'Recommendation': recommendation
                })

    interpretation_df = pd.DataFrame(interpretations)
    recommendation_df = pd.DataFrame(recommendations)

    return interpretation_df, recommendation_df

# Regression Analysis with OLS
def do_multivariate_regression_analysis_with_OLS(target_variable_data, independent_variable_data):
    # Add a constant term for the intercept
    independent_variable_data['intercept'] = 1

    # Perform multiple linear regression
    X = independent_variable_data
    Y = target_variable_data

    # Perform the regression the ordinal logistic regression model
    model = sm.OLS(Y, X)

    results = model.fit()
    print(results.summary())

    # Get the summary table as a text
    summary_str = results.summary().as_text()

    # Create a new Document
    doc = Document()
    doc.add_heading('Regression Summary', 0)

    # Add the summary to the document
    doc.add_paragraph(summary_str)

    # Save the document
    filename = 'Multivariate_Regression_Summary.docx'
    doc.save(f'{output_path}{filename}')

    return results # .summary()

# interpret_and_recommend_regression function here
def interpret_and_recommend_regression_with_OLS(results, alpha=0.05):
    interpretations = []
    recommendations = []
    
    # Convert the results summary to a DataFrame
    summary_df = pd.read_html(results.summary().tables[1].as_html(), header=0, index_col=0)[0]
    
    # R-squared value
    r_squared = results.rsquared
    interpretations.append(f"The model explains approximately {r_squared * 100:.2f}% (R-squared) of the variability in the target variable.")
    
    # Interpret the intercept
    intercept = summary_df.loc['intercept', 'coef']
    interpretations.append(f"The estimated intercept is {intercept:.2f}. This is the expected value of the target variable when all independent variables are zero.")

    for variable, row in summary_df.iterrows():
        coef = row['coef']
        p_value = row['P>|t|']
        conf_low, conf_high = row['[0.025'], row['0.975]']
        
        # Skip the intercept for recommendations (already interpreted)
        if variable == 'intercept':
            continue

        # Interpretation based on p-value and coefficient
        if p_value < alpha:
            interpretation = f"For {variable}, the estimated coefficient is {coef:.2f} with a 95% confidence interval between {conf_low:.2f} and {conf_high:.2f}. "
            if coef > 0:
                interpretation += f"A unit increase in {variable} is associated with an estimated increase of {coef:.2f} in the target variable."
                recommendation = f"Consider focusing on increasing {variable} to positively impact the target variable."
            else:
                interpretation += f"A unit increase in {variable} is associated with an estimated decrease of {abs(coef):.2f} in the target variable."
                recommendation = f"Be cautious when increasing {variable}, as it might negatively impact the target variable."
            
            interpretations.append(interpretation)
            recommendations.append(recommendation)
        else:
            recommendation = f"The variable {variable} is not statistically significant at the {alpha*100}% level, so it may not be a reliable predictor."
            recommendations.append(recommendation)

    interpretation_text = ' '.join(interpretations)
    recommendation_text = ' '.join(recommendations)
    
    return interpretation_text, recommendation_text

# Regression Analysis with Ordinal Logistic Regression Model
def do_multivariate_regression_analysis_with_MNLogit(target_variable_data, independent_variable_data):
    # Add a constant term for the intercept
    independent_variable_data['intercept'] = 1

    # Perform multiple linear regression
    X = independent_variable_data
    Y = target_variable_data

    # Perform the regression the ordinal logistic regression model
    model = sm.MNLogit(Y, X)

    results = model.fit()
    print('\nModel fit. Done')
    print(results.summary())

    # Create a new Document
    '''
    doc = Document()
    doc.add_heading('Regression Summary', 0)

    # Add the summary to the document
    # doc.add_paragraph(results)

    # Save the document
    filename = 'Multivariate_Regression_Summary.docx'
    doc.save(f'{output_path}{filename}')
    '''
    return results

# interpret_and_recommend_regression function here
def interpret_and_recommend_regression_with_MNLogit(results, predictors, outcome_categories):
    """
    Interpret the results of an ordinal logistic regression model and provide recommendations.

    Parameters:
    - results: Results object from an ordinal logistic regression model (e.g., model.fit()).
    - predictors: List of predictor variable names.
    - outcome_categories: List of ordered outcome category labels.

    Returns:
    - Interpretation, recommendations, and odds ratio changes based on the model results.
    """
    
    # Create a DataFrame to store the results
    result_df = pd.DataFrame({'Predictor': predictors})
    
    # Get the model coefficients, p-values, and confidence intervals
    coefficients = results.params
    p_values = results.pvalues
    conf_int = results.conf_int()
    conf_int['Odds Ratio'] = np.exp(conf_int[0]), np.exp(conf_int[1])
    
    # Add coefficients, odds ratios, and p-values to the result DataFrame
    result_df['Coefficient'] = coefficients
    result_df['Odds Ratio'] = np.exp(coefficients)
    result_df['95% CI (Lower)'] = conf_int[0]
    result_df['95% CI (Upper)'] = conf_int[1]
    result_df['P-value'] = p_values
    
    # Interpretation, recommendations, and odds ratio changes
    interpretation = []
    recommendations = []
    odds_ratio_changes = []

    for index, row in result_df.iterrows():
        predictor = row['Predictor']
        coef = row['Coefficient']
        p_value = row['P-value']

        if p_value < 0.05:  # Significant predictors
            if coef > 0:
                interpretation.append(f"{predictor}: Increase in odds by {np.exp(coef):.2f} (95% CI: {np.exp(conf_int.loc[predictor]['95% CI (Lower)']):.2f} - {np.exp(conf_int.loc[predictor]['95% CI (Upper)']):.2f}).")
                recommendations.append(f"Recommendation for {predictor}: Consider strategies that increase the likelihood of {outcome_categories[-1]}.")
                odds_ratio_changes.append(f"Odds Ratio Change for {predictor}: An increase in {predictor} is associated with a {np.exp(coef):.2f} times higher odds of {outcome_categories[-1]}.")
            else:
                interpretation.append(f"{predictor}: Decrease in odds by {1/np.exp(coef):.2f} (95% CI: {1/np.exp(conf_int.loc[predictor]['95% CI (Upper)']):.2f} - {1/np.exp(conf_int.loc[predictor]['95% CI (Lower)']):.2f}).")
                recommendations.append(f"Recommendation for {predictor}: Consider strategies that decrease the likelihood of {outcome_categories[-1]}.")
                odds_ratio_changes.append(f"Odds Ratio Change for {predictor}: A decrease in {predictor} is associated with a {1/np.exp(coef):.2f} times lower odds of {outcome_categories[-1]}.")
        else:
            interpretation.append(f"{predictor}: Not statistically significant.")
            recommendations.append(f"No specific recommendation for {predictor} due to lack of statistical significance.")
            odds_ratio_changes.append(f"No significant odds ratio change for {predictor} due to lack of statistical significance.")
    
    return interpretation, recommendations, odds_ratio_changes

