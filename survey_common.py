import pandas as pd


import numpy as np
import os

from helper_functions import *

output_path = './output/'
create_new_directory(output_path)



def read_model(filepath):
    # Load model from Excel sheets
    
    # Read Demographic Variables
    demographic_df = pd.read_excel(filepath, sheet_name='Demographic_Variables', header=0)
    demographic_df = normalize_dataframe(demographic_df)
    demographic_dict = demographic_df.to_dict(orient='records')
    print('\n======= Demographic Variables')
    print(demographic_dict)

    # Read Independent Variables
    independent_df = pd.read_excel(filepath, sheet_name='Independent_Variables', header=0)
    independent_df = normalize_dataframe(independent_df)
    independent_dict = independent_df.to_dict(orient='records')
    print('\n======= Independent Variables')
    print(independent_dict)

    # Read mediator Variables
    mediator_df = pd.read_excel(filepath, sheet_name='Mediator_Variables', header=0)
    mediator_df = normalize_dataframe(mediator_df)
    mediator_dict = mediator_df.to_dict(orient='records')
    print('\n======= Mediator Variables')
    print(mediator_dict)

    # Read Dependent Variables
    dependent_df = pd.read_excel(filepath, sheet_name='Dependent_Variables')
    dependent_df = normalize_dataframe(dependent_df)
    dependent_dict = dependent_df.to_dict(orient='records')
    print('\n======= Dependent Variables')
    print(dependent_dict)

    # Read Regression Relations
    relation_df = pd.read_excel(filepath, sheet_name='Relations')
    relation_df = normalize_dataframe(relation_df)
    relation_dict = relation_df.to_dict(orient='records')
    print('\n======= Relations')
    print(relation_dict)

    # Read variance/covariance relations
    # varcovar_df = pd.read_excel(filepath, sheet_name='VarCovar_Relations')
    # varcovar_df = normalize_dataframe(varcovar_df)
    # varcovar_dict = varcovar_df.to_dict(orient='records')
    varcovar_dict = []

    # Read Model parameters
    parameters_df = pd.read_excel(filepath, sheet_name='Parameters')
    parameters_df = normalize_dataframe(parameters_df)
    parameters_dict = parameters_df.to_dict(orient='records')
    print('\n======= Parameters')
    print(parameters_dict)

    return demographic_dict, independent_dict, mediator_dict, dependent_dict, relation_dict, varcovar_dict, parameters_dict


def pre_process_data(data, demographic_dict, independent_dict, dependent_dict):
    
    data = data.astype(str)

    # Rename Demographic columns to reduce the columns name
    data_normalized, demographic_cols_names = rename_columns_by_index(data, demographic_dict)

    # Rename Observation, mediator, Dependent columns to reduce the columns name
    data_normalized, independent_cols_names = rename_variable_columns(data_normalized, independent_dict)
    data_normalized, dependent_cols_names = rename_variable_columns(data_normalized, dependent_dict)

    # To convert the categorical columns to numerical
    data_normalized, label_mappings = convert_categorical_columns(data_normalized)

    data_normalized = data_normalized.astype(float)

    
    return data_normalized, label_mappings, demographic_cols_names, independent_cols_names, dependent_cols_names


