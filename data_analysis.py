from pingouin import cronbach_alpha
from factor_analyzer import FactorAnalyzer
from docx import Document
import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder


output_path = './'

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
        stats_col ={}
        freq = df[col_name].value_counts().sort_index()
        percent = (freq / nbr_data_points_cleaned) * 100
        percent = percent.round(1)  # Round to two decimal places
        stats_table[col_name] = pd.DataFrame({'Frequency': freq, 'Percent': percent})
        
        # Sort the DataFrame by 'Frequency' in descending order
        stats_table[col_name] = stats_table[col_name].sort_values(by='Frequency', ascending=False)
    
    # Save to Excel file
    output_file_name = f'{output_file_name}_Stats'
    save_to_excel(output_file_name, stats_table)

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
    data_to_extract = convert_values_to_numeric(data_to_extract)
    return data_to_extract

# Convert values of dataframe to numeric
def convert_values_to_numeric(df):
    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Encode each column in the DataFrame
    for column in df.columns:
        df[column] = label_encoder.fit_transform(df[column])
    return df

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
def compute_cronbach_alpha(selected_data):
    # Calculate Cronbach's alpha
    alpha = cronbach_alpha(selected_data)

    # Print the result
    print(f"Overall Cronbach's Alpha: {alpha}")

    return alpha

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

# Regression Analysis
def do_multivariate_regression_analysis(target_variable_data, independent_variable_data):
    # Add a constant term for the intercept
    independent_variable_data['intercept'] = 1

    # Perform multiple linear regression
    X = independent_variable_data
    Y = target_variable_data

    # Perform the regression
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

    return results.summary()


# Regression Analysis
def do_regression_analysis_2(df, dependent_variable_name, independent_cols_names):
    dependent_variable = dependent_variable_name
    #print(dependent_variable)
    independent_variables = independent_cols_names
    #print(independent_variables)

    # Add a constant term for the intercept
    df['intercept'] = 1

    # Perform multiple linear regression
    X = df[independent_variables + ['intercept']]
    Y = df[dependent_variable]

    # Perform the regression
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
    doc.save('Regression_Summary.docx')

    return results.summary()
