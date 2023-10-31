
import pandas as pd
import os

import math

from sklearn.preprocessing import LabelEncoder




def create_new_directory(dir_path):
    # Check if the directory exists
    if not os.path.exists(dir_path):
        # If it doesn't exist, create the directory
        os.makedirs(dir_path)
        print(f"Directory '{dir_path}' created.")
    else:
        for f in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, f))

def create_mapping_dict(text):
    # Split the input text into pairs of value and number
    pairs = [item.strip() for item in text.split(",")]

    # Create a dictionary from the pairs
    mapping_dict = {}
    for pair in pairs:
        value, number = pair.split("(")
        mapping_dict[value.strip()] = int(number.strip(")"))

    return mapping_dict

def get_value_from_text(mapping_dict, input_text):
    # Remove leading/trailing spaces and convert to title case for case-insensitive matching
    input_text = input_text.strip().lower()

    # Check if the input_text is in the mapping_dict
    return mapping_dict.get(input_text, None)

def apply_mapping_and_replace(mapping_dict, text):
    return get_value_from_text(mapping_dict, text)


def convert_categorical_columns(data):
    # Create a copy of the input DataFrame
    df = data.copy()

    # Initialize a dictionary to store mappings of original labels to numerical labels
    label_mappings = {}

    # Initialize the LabelEncoder
    label_encoder = LabelEncoder()

    # Iterate through each column in the DataFrame
    for column in df.columns:
        if df[column].dtype == 'object':  # Check if the column has categorical data
            # Use LabelEncoder to convert the categorical column to numerical
            df[column] = label_encoder.fit_transform(df[column])

            # Create a mapping of original labels to numerical labels
            label_mappings[column] = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))

    # Return both the transformed DataFrame and the label mappings
    return df, label_mappings

def get_back_original_label_from_numerical_label(label_mappings, column_name, numerical_label):

    # Access the label mapping for a specific column
    mapping_for_category = label_mappings[column_name]
    # Use the label mapping to get the original label from a numerical label
    original_label = mapping_for_category.get(numerical_label)    

    return original_label


def format_name(name):
    """Format the name by removing spaces and special characters."""
    name = str(name).strip()
    return name.replace(" ", "_").replace("(", "").replace(")", "")

def normalize_column_names(df):
    formatted_columns = {col: format_name(col) for col in df.columns}
    return df.rename(columns=formatted_columns)

def normalize_variable_names(variable_dict):
    # Create a new list to store the normalized variable names
    normalized_variables = []

    # Iterate through each variable in the variable_dict
    for var in variable_dict:
        var_name = var.get('Variable')  # Get the variable name from the dictionary
        normalized_name = format_name(var_name)  # Normalize the variable name
        var_copy = var.copy()  # Create a copy of the variable dictionary
        var_copy['Variable'] = normalized_name  # Update the 'Variable' key with the normalized name
        normalized_variables.append(var_copy)  # Add the updated variable to the new list

    return normalized_variables

# Define a function to normalize column names and string values
'''
def normalize_string(element):
    # If the input is a pandas Series
    if isinstance(element, pd.Series):
        if element.dtype == 'object':
            return element.str.strip().str.replace(' ', '_')
        else:
            return element  # Return the original column for non-string data
    # If the input is a string
    elif isinstance(element, str):
        return element.strip().replace(' ', '_')
    else:
        return element  # Return the original input for other data types

'''
# Define a function to normalize column names and string values
# Remove any leading and trailing spaces. Replace spaces with underscores
def normalize_dataframe(df):# Define a function to normalize a single string
    df = df.astype(str)

    def normalize_string(x):
        x = x.strip()
        if isinstance(x, str):
            if '+' in x:
                # Split the string by '+' and normalize each segment
                segments = [segment.strip().replace(' ', '_') for segment in x.split('+')]
                return '+'.join(segments)
            else :
                return x.strip().replace(' ', '_')

    column_to_clean = 'Variable'
    if column_to_clean in df.columns:
        df[column_to_clean] = df[column_to_clean].str.strip()

    column_to_clean = 'Related_Variables'
    if column_to_clean in df.columns:
        df[column_to_clean] = df[column_to_clean].map(normalize_string)

    # Automatically detect and convert columns with numerical data
    for column in df.columns:
        try:
            df[column] = pd.to_numeric(df[column])
        except ValueError:
            pass  # Ignore columns that cannot be converted to numerical

    return df

# Flatten a list
def flatten(lst):
    flat_list = []
    for item in lst:
        if isinstance(item, list):
            flat_list.extend(flatten(item))
        else:
            flat_list.append(item)
    return flat_list


def remove_duplicate_substrings(input_string):
    # Split the input string into substrings based on a space (' ') or any other delimiter
    substrings = input_string.split(' ')

    # Create a list to store unique substrings
    unique_substrings = []

    # Iterate through the substrings and add them to the unique_substrings list
    for substring in substrings:
        if substring not in unique_substrings:
            unique_substrings.append(substring)

    # Join the unique substrings back together into a single string
    result_string = ' '.join(unique_substrings)

    return result_string

def remove_duplicate_substrings_exclude(input_string, excluded_substring):
    # Split the input string into substrings based on a space (' ') or any other delimiter
    substrings = input_string.split(' ')

    # Create a list to store unique substrings
    unique_substrings = []

    # Iterate through the substrings and add them to the unique_substrings list
    for substring in substrings:
        # Check if the substring is the excluded substring (e.g., '+')
        if substring == excluded_substring:
            unique_substrings.append(substring)  # Add only one instance of the excluded substring
        elif substring not in unique_substrings:
            unique_substrings.append(substring)

    # Join the unique substrings back together into a single string
    result_string = ' '.join(unique_substrings)

    return result_string



# Rename columns for demographic data columns: to have shorter column names
def rename_columns_by_index(df, demographic_dict):
    demographic_cols_names = []  # Initialize a list to store independent column names
    for col in demographic_dict:
        col_index = col.get('column_index')
        new_column_name = col.get('Variable')
        # print(col_index, new_column_name)
        df.rename(columns={df.columns[col_index]: new_column_name}, inplace=True)
        demographic_cols_names.append(new_column_name)
    return df, demographic_cols_names

# Rename Independent columns according to Independent_Cols
def rename_variable_columns(df, variable_dict):
    variable_cols_names = []  # Initialize a list to store independent column names
    for col in variable_dict:
        col_name = col.get('Variable')
        col_index_from = col.get('column_index_from')
        number_questions = col.get('number_questions')
        # print('\n=========')
        # print(col_index_from, number_questions)
        # if math.isnan(float(col_index_from)) is not True or math.isnan(float(number_questions)) is not True:
        if type(col_index_from) == str or type(number_questions) == str:
            continue
        else:
            col_index_to = col_index_from + number_questions

            for i in range(col_index_from, col_index_to):
                old_col_name = df.columns[i]
                new_col_name = f'{col_name}_Q{i - col_index_from + 1}'
                df.rename(columns={old_col_name: new_col_name}, inplace=True)
                variable_cols_names.append(new_col_name)  # Append the new column name

    return df, variable_cols_names



def custom_describe(data):
    # Initialize a DataFrame to store the custom statistics
    custom_stats = pd.DataFrame()

    # Iterate through each column in the input DataFrame
    for column in data.columns:
        column_data = data[column]

        if pd.api.types.is_numeric_dtype(column_data):
            # For numerical columns, calculate statistics
            stats = {
                'Count': column_data.count(),
                'Mean': column_data.mean(),
                'Std': column_data.std(),
                'Min': column_data.min(),
                '25%': column_data.quantile(0.25),
                '50%': column_data.median(),
                '75%': column_data.quantile(0.75),
                'Max': column_data.max(),
            }
        else:
            # For non-numeric (categorical) columns, calculate value counts and unique values
            stats = {
                'Count': column_data.count(),
                'Unique': column_data.nunique(),
                'Top': column_data.mode().iloc[0],
                'Freq': column_data.value_counts().max(),
            }

        custom_stats[column] = stats

    return custom_stats



def comprehensive_describe(data):
    # Create a DataFrame to store the comprehensive description
    comprehensive_description = pd.DataFrame()

    # Get basic statistics for numerical columns
    numerical_stats = data.describe(include='number').T
    comprehensive_description = pd.concat([comprehensive_description, numerical_stats], axis=1)

    # Define a function to provide custom summaries for categorical columns
    def custom_categorical_summary(column):
        return pd.Series({
            'Count': column.count(),
            'Unique': column.nunique(),
            'Top': column.mode().iloc[0],
            'Freq': column.value_counts().max()
        })

    # Get custom summaries for categorical columns
    categorical_columns = data.select_dtypes(include='object')
    for column_name, column_data in categorical_columns.items():  # Use items() instead of iteritems()
        categorical_summary = custom_categorical_summary(column_data)
        comprehensive_description[column_name] = categorical_summary

    return comprehensive_description

