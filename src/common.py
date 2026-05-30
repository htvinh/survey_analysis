import pandas as pd
import numpy as np
import os
import re
from typing import Dict, List, Any, Tuple, Optional

from src.helpers import logger, normalize_dataframe, rename_columns_by_index, \
    rename_variable_columns, convert_categorical_columns


def convert_to_model_spec_dict(model_spec_text: str) -> Dict[str, Any]:
    """Parses a model specification text into a structured dictionary.

    Args:
        model_spec_text: The raw text containing variable and relation definitions.

    Returns:
        Dict[str, Any]: A dictionary with 'independent', 'dependent', and 'relations'.
    """
    lines = [line.strip() for line in model_spec_text.split('\n') if line.strip()]

    independent_vars = {}
    dependent_vars = {}
    relations = []

    # Patterns for =~ (measurement model) and ~ (structural model)
    independent_pattern = r'(\w+) =~ (.+)'
    dependent_pattern = r'(\w+) =~ (.+)'
    relation_pattern = r'(\w+) ~ (.+)'

    for line in lines:
        if re.match(independent_pattern, line):
            var, questions = re.match(independent_pattern, line).groups()
            independent_vars[var] = [q.strip() for q in questions.split('+')]
        elif re.match(dependent_pattern, line):
            var, questions = re.match(dependent_pattern, line).groups()
            dependent_vars[var] = [q.strip() for q in questions.split('+')]
        elif re.match(relation_pattern, line):
            relations.append(line.strip())

    return {
        'independent': independent_vars,
        'dependent': dependent_vars,
        'relations': relations
    }


def read_model(filepath: str) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict], List[Dict], List[Any], List[Dict]]:
    """Reads the model configuration from an Excel file with multiple sheets.

    Args:
        filepath: Path to the Excel model file.

    Returns:
        Tuple: Contains lists of dictionaries for demographic, independent, mediator, 
               dependent variables, relations, variance-covariance, and parameters.
    """
    logger.info(f"Reading model from {filepath}")
    
    try:
        excel_file = pd.ExcelFile(filepath)
    except Exception as e:
        logger.error(f"Failed to open model file {filepath}: {e}")
        raise

    def get_sheet_data(sheet_name: str, default: Any = None) -> List[Dict[str, Any]]:
        if sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            df = normalize_dataframe(df)
            return df.to_dict(orient='records')
        return default if default is not None else []

    demographic_dict = get_sheet_data('Demographic_Variables')
    independent_dict = get_sheet_data('Independent_Variables')
    mediator_dict = get_sheet_data('Mediator_Variables')
    dependent_dict = get_sheet_data('Dependent_Variables')
    relation_dict = get_sheet_data('Relations')
    parameters_dict = get_sheet_data('Parameters')
    
    # Placeholder for variance-covariance if needed in future
    varcovar_dict = []

    return (
        demographic_dict, independent_dict, mediator_dict, 
        dependent_dict, relation_dict, varcovar_dict, parameters_dict
    )


def pre_process_data(
    data: pd.DataFrame, 
    demographic_dict: List[Dict], 
    independent_dict: List[Dict], 
    dependent_dict: List[Dict]
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, str]], List[str], List[str], List[str]]:
    """Pre-processes raw survey data by renaming columns, converting types, and handling categorical data.

    Args:
        data: The raw input DataFrame.
        demographic_dict: Demographic variable mapping.
        independent_dict: Independent variable mapping.
        dependent_dict: Dependent variable mapping.

    Returns:
        Tuple: (normalized_df, label_mappings, demographic_cols, independent_cols, dependent_cols)
    """
    logger.info("Starting data pre-processing")
    
    # Ensure all data is treated as string initially for consistent renaming/normalization
    data = data.astype(str)
    
    data_normalized, demographic_cols_names = rename_columns_by_index(data, demographic_dict)
    data_normalized, independent_cols_names = rename_variable_columns(data_normalized, independent_dict)
    data_normalized, dependent_cols_names = rename_variable_columns(data_normalized, dependent_dict)
    
    data_normalized, label_mappings = convert_categorical_columns(data_normalized)
    
    # Convert back to float for analysis, errors='coerce' to handle any non-numeric leftovers
    for col in data_normalized.columns:
        data_normalized[col] = pd.to_numeric(data_normalized[col], errors='coerce')
        
    return (
        data_normalized, label_mappings, demographic_cols_names, 
        independent_cols_names, dependent_cols_names
    )
