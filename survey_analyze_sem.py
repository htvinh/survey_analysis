
import pandas as pd
import numpy as np
from semopy import Model
import semopy

import graphviz

import os
# import sys
# import re
import math


import matplotlib.pyplot as plt
import seaborn as sns


from helper_functions import *




output_path = './output/'
create_new_directory(output_path)


ATTENTION_SEMOPY_INTERPRETATION = f"""\n
    ATTENTION: \n
    Setting the estimate of the first construct to 1 in SEMopy establishes a baseline
    for comparison, simplifying the analysis by making one indicator a reference point 
    for evaluating relationships with other variables.
"""

def reformat_sem_model_to_display(sem_model_spec):
    # Split the input by lines and format it
    lines = sem_model_spec.split('\n')
    formatted_lines = []

    for line in lines:
        # Remove leading and trailing whitespaces
        line = line.strip()
        if line.startswith('#:') or line.startswith('#-'):
            line = line.replace('#:', '-')
            line = line.replace('#-', '- ')
        # Add '#' before ':' and '-' if not already present
        #if line and not line.startswith('#') and not line.startswith('-'):
        #    # line = '- ' + line
        formatted_lines.append(line)

    # Join the formatted lines and return the result
    formatted_text = '\n'.join(formatted_lines)

    return formatted_text

def pre_process_data(data, demographic_dict, observable_dict, latent_dict, dependent_dict):

    data = data.astype(str)

    # Normalize the names of columns: to replace spaces by '_'
    # data = normalize_column_names(data)
    data_normalized = normalize_dataframe(data)

    # To convert the categorical columns to numerical
    data_normalized, label_mappings = convert_categorical_columns(data_normalized)

    data_normalized = data_normalized.astype(float)

    # Rename Demographic columns to reduce the columns name
    data_normalized, demographic_cols_names = rename_columns_by_index(data_normalized, demographic_dict)

    # Rename Observation, Latent, Dependent columns to reduce the columns name
    data_normalized, observable_cols_names = rename_variable_columns(data_normalized, observable_dict)
    data_normalized, latent_cols_names = rename_variable_columns(data_normalized, latent_dict)
    data_normalized, dependent_cols_names = rename_variable_columns(data_normalized, dependent_dict)

    return data_normalized, label_mappings, demographic_cols_names, observable_cols_names, latent_cols_names, dependent_cols_names


def read_model_spec(filepath):
    # Load data from Excel sheets
    # Remove "Remarks" columns
    substring = 'remark'

    # Read Demographic Variables
    demographic_df = pd.read_excel(filepath, sheet_name='Demographic_Variables', header=0)
    demographic_df = normalize_dataframe(demographic_df)
    columns_to_exclude = [col for col in demographic_df.columns if substring in col.strip().lower()]
    demographic_df = demographic_df.drop(columns=columns_to_exclude)
    demographic_df['column_index'] = demographic_df['column_index']-1
    demographic_dict = demographic_df.to_dict(orient='records')

    # Read Observable Variables
    observable_df = pd.read_excel(filepath, sheet_name='Observable_Variables', header=0)
    observable_df = normalize_dataframe(observable_df)
    columns_to_exclude = [col for col in observable_df.columns if substring in col.strip().lower()]
    observable_df = observable_df.drop(columns=columns_to_exclude)
    observable_df['column_index_from'] = observable_df['column_index_from']-1
    observable_dict = observable_df.to_dict(orient='records')

    # Read Latent Variables
    latent_df = pd.read_excel(filepath, sheet_name='Latent_Variables')
    latent_df = normalize_dataframe(latent_df)
    columns_to_exclude = [col for col in latent_df.columns if substring in col.strip().lower()]
    latent_df = latent_df.drop(columns=columns_to_exclude)
    if math.isnan(float(latent_df['column_index_from'])) is not True:
        latent_df['column_index_from'] = latent_df['column_index_from']-1
    latent_dict = latent_df.to_dict(orient='records')

    # Read Dependent Variables
    dependent_df = pd.read_excel(filepath, sheet_name='Dependent_Variables')
    dependent_df = normalize_dataframe(dependent_df)
    columns_to_exclude = [col for col in dependent_df.columns if substring in col.strip().lower()]
    dependent_df = dependent_df.drop(columns=columns_to_exclude)
    if math.isnan(float(dependent_df['column_index_from'])) is not True:
        dependent_df['column_index_from'] = dependent_df['column_index_from']-1
    dependent_dict = dependent_df.to_dict(orient='records')

    # Read variance/covariance relations
    varcovar_df = pd.read_excel(filepath, sheet_name='VarCovar_Relations')
    varcovar_df = normalize_dataframe(varcovar_df)
    varcovar_dict = varcovar_df.to_dict(orient='records')

    # Read Model parameters
    parameters_df = pd.read_excel(filepath, sheet_name='Parameters')
    parameters_df = normalize_dataframe(parameters_df)
    columns_to_exclude = [col for col in parameters_df.columns if substring in col.strip().lower()]
    parameters_df = parameters_df.drop(columns=columns_to_exclude)
    parameters_dict = parameters_df.to_dict(orient='records')

    return demographic_dict, observable_dict, latent_dict, dependent_dict, varcovar_dict, parameters_dict


def create_construct(selected_variables):
    construct_dict = {}
    for col in selected_variables:
        construct = col.get('Variable')
        num_items = col.get('number_questions')
        if math.isnan(float(num_items)) is not True:
            items = [f"{construct}_Q{i+1}" for i in range(num_items)]
            construct_dict[construct] = ' + '.join(items)
    return construct_dict

def create_variable_specs(variable_dict):

    variable_df = pd.DataFrame(variable_dict)
    variable_spec = {row['Variable']: row['Related_Variables'] for _, row in variable_df.iterrows()}
    
    # Get the construct specs from create_construct
    construct_specs = create_construct(variable_dict)

    # For each latent variable in the construct specifications, update the existing spec
    for latent_var, related_vars in construct_specs.items():
        # If the latent_var exists in the latent_spec dictionary, update its related variables
        if latent_var in variable_spec:
            variable_spec[latent_var] = variable_spec[latent_var] + ' + ' + related_vars

    # Convert the updated specs back to SEMopy string format
    semopy_spec = "\n".join([f"{key} =~ {value}" for key, value in variable_spec.items()])

    return semopy_spec

def convert_list_to_semopy_spec(input_list):
    semopy_dict = {}
    
    for entry in input_list:
        variable = entry['Variable']
        num_items = entry['number_questions']
        questions = [f"{variable}_Q{i+1}" for i in range(num_items)]
        
        semopy_dict[variable] = ' + '.join(questions)
        
    return "\n".join([f"{key} =~ {value}" for key, value in semopy_dict.items()])


# Create SEM Model Spec Full 
def create_sem_model_spec(filepath):
    demographic_dict, observable_dict, latent_dict, \
    dependent_dict, varcovar_dict, parameters_dict = read_model_spec(filepath)

    observable_df = pd.DataFrame(observable_dict)
    latent_df = pd.DataFrame(latent_dict)
    dependent_df = pd.DataFrame(dependent_dict)
    varcovar_df = pd.DataFrame(varcovar_dict)
    
    observable_variable_list = "\n#-".join([f"{row['Variable']}" for _, row in observable_df.iterrows()])

    observable_spec= convert_list_to_semopy_spec(observable_dict)
    # print(observable_spec)

    latent_spec = create_variable_specs(latent_dict)

    dependent_spec = create_variable_specs(dependent_dict)

    structural_spec = "\n".join([f"{row['Variable']} ~ {row['Related_Variables']}" for _, row in dependent_df.iterrows()])

    # Generate variance and covariance specifications
    # Initialize empty lists for variances and covariances
    variance_specs = []
    covariance_specs = []
    # Iterate through each row in varcovar_df
    for idx, row in varcovar_df.iterrows():
        # Check if Variable_1 and Variable_2 are NaN before adding them to covariance_specs
        if not pd.isna(row['Variable_1']):
            if row['Variable_1'] == row['Variable_2']:
                variance_specs.append(f"{row['Variable_1']} =~ {row['Variable_2']}")
            elif row['Variable_1'] != row['Variable_2']:
                covariance_specs.append(f"{row['Variable_1']} ~~ {row['Variable_2']}")


    # Join the variance and covariance specs into strings
    variance_spec = "\n".join(variance_specs)
    covariance_spec = "\n".join(covariance_specs)

    # Combine all specs to form the final SEM spec
    sem_model_spec = f"""
    
    ### Observable / Measurement Variables
    {observable_spec} 

    ### Latent (Construct) Variables
    {latent_spec}
    
    ### Dependent (Construct) Variables
    {dependent_spec}

    ### Structural Model /Relations
    {structural_spec}

    """ 

    # Combine all specs to form the final SEM spec
    sem_model_spec_reduced = f"""
    
    ### Observable / Measurement Variables
    #: {observable_variable_list}

    ### Latent (Construct) Variables
    {latent_spec}
    
    ### Dependent (Construct) Variables
    {dependent_spec}

    ### Structural Model /Relations
    {structural_spec}

    """ 
    
    if len(variance_spec) > 0:
        sem_model_spec = f"""{sem_model_spec} 
        ### Variance Relations
        {variance_spec}
        """
        sem_model_spec_reduced = f"""{sem_model_spec_reduced} 
        ### Variance Relations
        {variance_spec}
        """
    if len(covariance_spec) > 0:
        sem_model_spec = f"""{sem_model_spec} 
        ### Co-Variance Relations
        {covariance_spec}
        """
        sem_model_spec_reduced = f"""{sem_model_spec_reduced} 
        ### Co-Variance Relations
        {covariance_spec}
        """

    print('\n===================')
    print(sem_model_spec)

    return  sem_model_spec, sem_model_spec_reduced, \
            demographic_dict, observable_dict, latent_dict, \
            dependent_dict, varcovar_dict, parameters_dict


def create_sem_model_spec_graph(sem_model_spec, observable_dict, latent_dict, dependent_dict, graph_name):
    # Initialize the Graphviz Digraph
    g = graphviz.Digraph('SEM', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')
    
    # Extract names from dictionaries
    latent_variable_names = [item['Variable'] for item in latent_dict]
    dependent_variable_names = [item['Variable'] for item in dependent_dict]
    observable_variable_names = [item['Variable'] for item in observable_dict]

    # observable_variable_names = sorted(observable_variable_names, reverse=True)
    # dependent_variable_names = sorted(dependent_variable_names, reverse=True)
    # latent_variable_names = sorted(latent_variable_names, reverse=True)


    # Create subgraphs for alignment
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in observable_variable_names:
            s.node(var_name, shape='box', fillcolor='#e6f2ff', style='filled')
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in latent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#cae6df', style='filled')
    with g.subgraph() as s:
        s.attr(rank='same') #rank='same')
        for var_name in dependent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#f5e6ca', style='filled')

    # Parse the sem_model_spec to extract relationships and nodes
    lines = sem_model_spec.strip().split("\n")
    for line in lines:
        # Observable and Measurement Variables
        if '=~' in line:
            lhs, rhs = line.split('=~', maxsplit=1)  
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                g.edge(var.strip(), lhs.strip(), dir='forward')
        
        # Covariances
        if '~~' in line:
            lhs, rhs = line.split('~~', maxsplit=1) 
            g.edge(lhs.strip(), rhs.strip(), dir='none', style='dashed')

    # Save to file
    # graph_name = 'sem_model_graph'
    g.attr(dpi='600', imagescale='true')
    g.render(f"{output_path}/{graph_name}")

    graph_path = f"{output_path}/{graph_name}.png"
    return graph_path


def conduct_sem_analysis(data, sem_model_spec, observable_dict, latent_dict, dependent_dict):
    #  Conduct SEM Analysis

    # Instantiate and fit the model
    sem_model = Model(sem_model_spec)
    sem_result = sem_model.fit(data)

    # Gather statistics
    sem_stats = semopy.calc_stats(sem_model)
    # print(sem_stats)

    # filename = 'SEM_Model_Stats'
    # excel_file_path = f'{output_path}{filename}.xlsx'
    # sem_stats.to_excel(excel_file_path, index=True)  

    # Retrieve the results using the inspect method
    sem_inspect = sem_model.inspect()
    print('\n===================================')
    print(sem_inspect)

    # Post process SEM results
    sem_inspect_enhanced, sem_inspect_filtered, graph_filtered_results, graph_fulll_results = post_process_sem_results(sem_model_spec, sem_inspect, observable_dict, latent_dict, dependent_dict)

    return sem_result, sem_stats, sem_inspect, sem_inspect_enhanced, sem_inspect_filtered, graph_filtered_results, graph_fulll_results

def post_process_sem_results(sem_model_spec, sem_inspect, observable_dict, latent_dict, dependent_dict):

    # Inhance the inspect table by inserting 2 new columns: significance and direction
    sem_inspect_enhanced = enhance_inspection(sem_inspect)
    print(sem_inspect_enhanced)

    filename = 'SEM_Results'
    excel_file_path = f'{output_path}{filename}.xlsx'
    sem_inspect_enhanced.to_excel(excel_file_path, index=True)  

    print('\n\n')
    sem_inspect_filtered_results = filter_inspect_table_from_spec(sem_inspect, sem_model_spec)
    # print(sem_inspect_filtered_results)

    graph_filtered_results = create_graph_for_sem_results_filtered(sem_model_spec, sem_inspect_filtered_results)

    graph_fulll_results = create_graph_for_sem_results_full(sem_model_spec, sem_inspect, observable_dict, latent_dict, dependent_dict)

    return sem_inspect_enhanced, sem_inspect_filtered_results, graph_filtered_results, graph_fulll_results

    

def filter_inspect_table_from_spec(sem_inspect, sem_model_spec):
    # Extract independent (latent) and dependent variables from sem_model_spec
    lines = sem_model_spec.strip().split("\n")
    independent_factors = set([line.split('=~')[0].strip() for line in lines if '=~' in line])
    dependent_variables = set([line.split('~')[0].strip() for line in lines if '~' in line])

    # Filter results_df to only include rows representing relationships between independent factors and dependent variables
    # Mask for the first condition (independent to dependent)
    mask1 = (sem_inspect['lval'].isin(independent_factors)) & (sem_inspect['rval'].isin(dependent_variables))

    # Mask for the second condition (dependent to independent)
    mask2 = (sem_inspect['lval'].isin(dependent_variables)) & (sem_inspect['rval'].isin(independent_factors))

    # Mask for the third condition (dependent to dependent)
    mask3 = (sem_inspect['lval'].isin(dependent_variables)) & (sem_inspect['rval'].isin(dependent_variables)) & (sem_inspect['lval'] != sem_inspect['rval'])

    # Combine the masks
    combined_mask = mask1 | mask2  | mask3

    # Apply the mask to the DataFrame
    sem_inspect_filtered = sem_inspect[combined_mask]


    return sem_inspect_filtered

# Insert 2 columns: 'significance' and 'relation'
def enhance_inspection(sem_inspection_df):
    """Enhances the given inspection DataFrame with 'Significance' and 'Relation' columns."""
    
    # Ensure p-value, estimate columns are numeric, and replace non-numeric values with NaN
    sem_inspection_df['p-value'] = pd.to_numeric(sem_inspection_df['p-value'], errors='coerce')
    sem_inspection_df['Estimate'] = pd.to_numeric(sem_inspection_df['Estimate'], errors='coerce')

    # Calculate significance and relation
    significance = ['Significant' if float(p) < 0.05 else "Not Significant" for p in sem_inspection_df['p-value']]
    relation = ["Positive" if float(e) > 0 else "Negative" if float(e) < 0 else "Neutral" for e in sem_inspection_df['Estimate']]
    
    # Add new columns
    sem_inspection_df['Significance'] = significance
    sem_inspection_df['Relation'] = relation
    
    return sem_inspection_df


def create_label(row):
    row_series = pd.Series(row)
    row_dict = row_series.to_dict()

    # Try converting to float or set to None if it fails
    try:
        estimate = float(row_dict.get('Estimate'))
    except (TypeError, ValueError):
        estimate = None

    try:
        std_err = float(row_dict.get('Std. Err'))
    except (TypeError, ValueError):
        std_err = None

    try:
        p_value = float(row_dict.get('p-value'))
    except (TypeError, ValueError):
        p_value = None

    try:
        z_value = float(row_dict.get('z-value'))
    except (TypeError, ValueError):
        z_value = None

    # Construct the label
    parts = []
    if estimate is not None and not pd.isna(estimate):
        parts.append(f"Est: {estimate:.2f}")
    if estimate is not None and not pd.isna(std_err):
        parts.append(f"Std. Err: {std_err:.2f}")
    if p_value is not None and not pd.isna(p_value):
        parts.append(f"p-val: {p_value:.3f}")
    #if z_value is not None and not pd.isna(z_value):
    #    parts.append(f"z-val: {z_value:.2f}")

    return ', '.join(parts)



def create_graph_for_sem_results_full(sem_model_spec, sem_inspect, observable_dict, latent_dict, dependent_dict):
    # Initialize the Graphviz Digraph
    g = graphviz.Digraph('SEM', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')
    
    # Extract names from dictionaries
    latent_variable_names = [item['Variable'] for item in latent_dict]
    dependent_variable_names = [item['Variable'] for item in dependent_dict]
    observable_variable_names = [item['Variable'] for item in observable_dict]

    # Create subgraphs for alignment
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in observable_variable_names:
            s.node(var_name, shape='box', fillcolor='#e6f2ff', style='filled')
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in latent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#cae6df', style='filled')
    with g.subgraph() as s:
        s.attr() #rank='same')
        for var_name in dependent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#f5e6ca', style='filled')

    # To keep track of nodes
    nodes_added = set()

    # Extract edge labels and values from sem_inspect dataframe
    edge_labels = {}
    edge_values = {}
    for _, row in sem_inspect.iterrows():
        label = create_label(row)
        edge_labels[(row['lval'], row['rval'])] = label
        edge_values[(row['lval'], row['rval'])] = row['Estimate']

    # Parse the sem_model_spec to extract relationships and nodes
    lines = sem_model_spec.strip().split("\n")
    for line in lines:
        # Observable and Measurement Variables
        if '=~' in line:
            lhs, rhs = line.split('=~', maxsplit=1)  
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                # Use edge_labels dictionary to get the label for the edge
                var = var.strip()
                lhs = lhs.strip()
                if var != lhs:
                    label = edge_labels.get((var, lhs), '')
                    value = edge_values.get((var, lhs), '')
                    # label_with_value = f"{label} (Est: {value:.2})"
                    label_with_value = f"{label}"
                g.edge(var, lhs, dir='forward', label=label_with_value)  # Add the label to the edge

        # Covariances
        elif '~~' in line:
            lhs, rhs = line.split('~~', maxsplit=1)
            label = "Cov"
            value = edge_values.get((lhs.strip(), rhs.strip()), '')
            label_with_value = f"Est: {value:.2f}"
            # label_with_value = ''
            g.edge(lhs.strip(), rhs.strip(), dir='none', style='dashed', label=label_with_value)

        # Variances (identity matrix)
        elif '=~' in line:
            var_name = line.split('=')[0].strip()
            label = "Var"
            value = edge_values.get((var_name, var_name), '')
            label_with_value = f"{label} (Est: {value:.2f})"
            # label_with_value = ''
            g.edge(var_name, var_name, dir='none', label=label_with_value)

    # Set the graph attributes for size and imagescale
    graph_name = 'sem_graph_results_full'
    g.attr(dpi='600', imagescale='true')
    save_path = f"{output_path}{graph_name}"
    g.render(save_path)

    graph_path = f"{output_path}{graph_name}.png"
    return graph_path


def create_graph_for_sem_results_filtered(sem_model_spec, sem_inspect):
    g = graphviz.Digraph('SEM', format='png', engine='dot')
    g.attr(rankdir='LR')
    g.attr(overlap='scale', splines='true')
    g.attr('edge', fontsize='12')
    # g.attr(size="10,15", ratio="fill")  # Enlarge image and maintain aspect ratio

    # To keep track of nodes
    nodes_added = set()
        
    # Parse the sem_model_spec to extract relationships and nodes
    lines = sem_model_spec.strip().split("\n")

    # Extract edge labels from sem_inspect dataframe
    edge_labels = {}
    for _, row in sem_inspect.iterrows():
        label = create_label(row)
        edge_labels[(row['lval'], row['rval'])] = label

    for line in lines:
            
        # Relationships
        if '~' in line and '=~' not in line:
            lhs, rhs = line.split('~', maxsplit=1)  
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                label = edge_labels.get((var.strip(), lhs.strip()), "")
                g.edge(var.strip(), lhs.strip(), label=label)
                nodes_added.add(var.strip())
                nodes_added.add(lhs.strip())
            
        # Covariances
        if '~~' in line:
            lhs, rhs = line.split('~~', maxsplit=1) 
            label = edge_labels.get((lhs.strip(), rhs.strip()), "")
            g.edge(lhs.strip(), rhs.strip(), dir='both', style='dashed', label=label)
            nodes_added.add(lhs.strip())
            nodes_added.add(rhs.strip())

    # Determine which variables are latent (those which are only on the left-hand side)
    latent_vars = set([line.split('~')[0].strip() for line in lines if '~' in line]) 
    latent_vars = sorted(latent_vars)
    measurement_vars = set([line.split('=~')[0].strip() for line in lines if '=~' in line])
    measurement_vars = sorted(measurement_vars)
    
    #latent_vars = set([line.split('~')[0].strip() for line in lines if '~' in line]) 
    # measurement_vars = set([line.split('=~')[0].strip() for line in lines if '=~' in line])

    # Style nodes
    for node in nodes_added:
        if node in latent_vars:
            g.node(node, shape='ellipse', fillcolor='#cae6df', style='filled')
        elif node in measurement_vars:
            g.node(node, shape='ellipse', fillcolor='#f5e6ca', style='filled')
        else:
            g.node(node, shape='box', fillcolor='#e6f2ff', style='filled')



    # Set the graph attributes for size and imagescale
    g.attr(dpi='600', imagescale='true')
    save_path = f"{output_path}sem_graph_results_filtered"
    g.render(save_path)

    graph_path = f"{output_path}/sem_graph_results_filtered.png"
    return graph_path

def extract_model_parameters(model_parameters_dict, parameter_name):
    for param_dict in model_parameters_dict:
        if param_dict['parameter'] == parameter_name:
            high_threshold = float(param_dict['high_threshold'])
            moderate_threshold = float(param_dict['moderate_threshold'])
            return high_threshold, moderate_threshold
    # Handle the case when the parameter name is not found
    return None, None

def interpret_sem_stats(sem_stats, parameters_dict):                                 
    # Default values for parameters
    threshold_high_chi2_pa_value, threshold_moderate_chi2_pa_value = 0.05, 0.05
    threshold_high_cfi, threshold_moderate_cfi = 0.9, 0.8
    threshold_high_rmsea, threshold_moderate_rmsea = 0.05, 0.08

    # Extract parameter values from Model
    parameter_name = 'chi2_p_value_threshold'
    threshold_high_chi2_pa_value, threshold_moderate_chi2_pa_value =  extract_model_parameters(parameters_dict, parameter_name)
    parameter_name = 'cfi_value_threshold'
    threshold_high_cfi, threshold_moderate_cfi =  extract_model_parameters(parameters_dict, parameter_name)
    parameter_name = 'rmsea_threshold'
    threshold_high_rmsea, threshold_moderate_rmsea =  extract_model_parameters(parameters_dict, parameter_name)

    # Transpose (flip rows and columns) to create a new DataFrame
    new_df = sem_stats.T.reset_index()

    # Rename the columns
    new_df.columns = ['Metric', 'Value']

    # Define a dictionary to map Metric values to 'Full Name' and 'Remark' values
    metric_mapping = {
        'DoF': {'Full Name': 'Degrees of Freedom', 'Remark': 'More flexibility with higher values'},
        'DoF Baseline': {'Full Name': 'Degrees of Freedom (Baseline Model)', 'Remark': 'Fewer paths, usually higher than DoF'},
        'chi2': {'Full Name': 'Chi-Square Test Statistic', 'Remark': 'Lower is better'},
        'chi2 p-value': {'Full Name': 'Chi-Square p-value', 'Remark': 'Higher (like > 0.05) is better'},
        'chi2 Baseline': {'Full Name': 'Chi-Square (Baseline Model)', 'Remark': 'Higher than regular chi2'},
        'CFI': {'Full Name': 'Comparative Fit Index', 'Remark': 'Close to 1 is great'},
        'GFI': {'Full Name': 'Goodness of Fit Index', 'Remark': 'Close to 1 is great'},
        'AGFI': {'Full Name': 'Adjusted Goodness of Fit Index', 'Remark': 'Close to 1 is great'},
        'NFI': {'Full Name': 'Normed Fit Index', 'Remark': 'Close to 1 is okay'},
        'TLI': {'Full Name': 'Tucker-Lewis Index', 'Remark': 'Close to 1 is great'},
        'RMSEA': {'Full Name': 'Root Mean Square Error of Approximation', 'Remark': 'Below 0.05 is great'},
        'AIC': {'Full Name': 'Akaike Information Criterion', 'Remark': 'Lower is better for comparing'},
        'BIC': {'Full Name': 'Bayesian Information Criterion (BIC)', 'Remark': 'Lower values indicate better fit.'},
        'LogLik': {'Full Name': 'Log Likelihood (LogLik)', 'Remark': 'Higher values indicate better fit.'}
    }

    # Add 'Full Name' and 'Remark' columns based on the mapping
    new_df['Full Name'] = new_df['Metric'].map(lambda x: metric_mapping[x]['Full Name'])
    new_df['Remark'] = new_df['Metric'].map(lambda x: metric_mapping[x]['Remark'])

    # Ensure 'Value' column is of type float
    new_df['Value'] = new_df['Value'].astype(float)

    print(new_df)
    
    filename = 'SEM_Model_Stats'
    excel_file_path = f'{output_path}{filename}.xlsx'
    new_df.to_excel(excel_file_path, index=True)  

    """Provide an overall judgment about the model fit."""
    # Extract primary fit indices
    chi2_p_value = sem_stats['chi2 p-value'].values[0]
    cfi = sem_stats['CFI'].values[0]
    rmsea = sem_stats['RMSEA'].values[0]
    
    # Check conditions for good fit
    if chi2_p_value > threshold_high_chi2_pa_value and cfi > threshold_high_cfi and rmsea < threshold_moderate_rmsea:
        overall_msg = f"The model likely has a good fit to the data. Because: chi2_p_value > {threshold_high_chi2_pa_value}, cfi > {threshold_high_cfi}, and rmsea < {threshold_moderate_rmsea}."
    elif chi2_p_value > threshold_moderate_chi2_pa_value and threshold_moderate_cfi < cfi < threshold_high_cfi and threshold_high_rmsea < rmsea < threshold_high_rmsea:
        overall_msg = f"The model has an acceptable fit to the data. Because: chi2_p_value > {threshold_high_chi2_pa_value}, {threshold_moderate_cfi} < cfi < {threshold_high_cfi}, and {threshold_moderate_rmsea} < rmsea < {threshold_high_rmsea}."
    else:
        reasons = []
        if chi2_p_value <= threshold_moderate_chi2_pa_value:
            reasons.append(f"chi2_p_value <= {threshold_moderate_chi2_pa_value}")
        elif cfi <= threshold_moderate_cfi:
            reasons.append(f"cfi <= {threshold_moderate_cfi}")
        elif rmsea >= threshold_high_rmsea:
            reasons.append(f"rmsea >= {threshold_moderate_rmsea}")
        else: 
            reasons.append(f"somthing wrong. Check chi2_p_value and rmsea.")
        reasons_str = ', or '.join(reasons)
        overall_msg = f"The model likely doesn't fit the data well. Because: {reasons_str}"


    return new_df, overall_msg


def interepret_sem_inspect(sem_inspect, dependent_dict, parameters_dict):
    # Ensure p-value, estimate columns are numeric, and replace non-numeric values with NaN
    sem_inspect = sem_inspect.copy()
    sem_inspect['Estimate'] = pd.to_numeric(sem_inspect['Estimate'], errors='coerce')
    sem_inspect['p-value'] = pd.to_numeric(sem_inspect['p-value'], errors='coerce')
    sem_inspect['z-value'] = pd.to_numeric(sem_inspect['z-value'], errors='coerce')
    sem_inspect['Std. Err'] = pd.to_numeric(sem_inspect['Std. Err'], errors='coerce')


    # Identify central constructs
    # Initialize the dictionary
    latent_dict = {}

    # Iterate through the dependent_dict
    for var in dependent_dict:
        dependent_variable_name = var.get('Variable')
        latent_variables_str = var.get('Related_Variables')
        latent_variables_df = [item.strip() for item in latent_variables_str.split('+')]
        
        # Create a nested dictionary with the dependent variable name and latent variables
        latent_dict[dependent_variable_name] = {
            'Name': dependent_variable_name,
            'Latent_Variables': latent_variables_df
        }   
    # print(latent_dict)

    interpretations = []
    interpretation = ATTENTION_SEMOPY_INTERPRETATION

    for key, value in latent_dict.items():
        dependent_variable_name = value['Name']
        latent_variables_df = value['Latent_Variables']
        # print(dependent_variable_name, latent_variables_df)
        
        # Interpret Role of the construct
        interpretation += f"### **{dependent_variable_name}**:\n"
        interpretation += f"\n**Role in the Model**:\n"
        interpretation += f"- `{dependent_variable_name}` plays a central role, being influenced by `{', '.join(latent_variables_df)}`. "
        interpretation += "Its centrality suggests that it's crucial for understanding the overall dynamics of the model.\n"
        # print(interpretation)

        # Interpret Influences on the construct
        interpretation += "\n**Influences**:\n"

        # Filter rows where the construct is the rval
        # print(sem_inspect)
        latent_rows = sem_inspect[sem_inspect['rval'] == dependent_variable_name]
        latent_rows = latent_rows.reset_index(drop=True)

        for index, latent_var in enumerate(latent_variables_df):
            row_index = index 
            latent_var_significant = latent_rows.at[row_index, 'Significance']
            latent_var_relation = latent_rows.at[row_index, 'Relation']
            latent_var_estimate = latent_rows.at[row_index, 'Estimate']
            latent_var_p_value = latent_rows.at[row_index, 'p-value']
            # print(row_index, latent_var_estimate)

            if row_index == 0: # Set as reference
                interpretation += f"\n  - `{latent_var}`, as the first construct, its Estimate is set to 1 (Est = 1), establishing a baseline for comparison."
            elif latent_var_significant == 'Significant':
                interpretation += f"\n  - `{latent_var}` has a statistically `{latent_var_significant}` `{latent_var_relation}` influence to `{dependent_variable_name}`, (Estimate: {latent_var_estimate:.3f}, p-value: {latent_var_p_value:.3f})."
            else: 
                interpretation += f"\n  - `{latent_var}` has a `{latent_var_relation}` influence to `{dependent_variable_name}`, but `NOT` statistically significant (Estimate: {latent_var_estimate:.3f}, p-value: {latent_var_p_value:.3f})."

        # Interpret Relations on the construct
        interpretation += '\n'
        interpretation += "\n**Relations**:\n"

        # Filter rows where the construct is the lval
        latent_rows = sem_inspect[sem_inspect['lval'] == dependent_variable_name]
        latent_rows = latent_rows.reset_index(drop=True)
        print('\n==================')
        # print(latent_rows)
        for index, latent_var in enumerate(latent_variables_df):
            row_index = index 
            latent_var_significant = latent_rows.at[row_index, 'Significance']
            latent_var_relation = latent_rows.at[row_index, 'Relation']
            latent_var_estimate = latent_rows.at[row_index, 'Estimate']
            latent_var_p_value = latent_rows.at[row_index, 'p-value']

            # if construc_significant == 'Significant':
            interpretation += f"  - `{dependent_variable_name}` has a statistically `{latent_var_significant}` {latent_var_relation.lower()} relationship with `{latent_var}` (Estimate: {latent_var_estimate:.3f}), suggesting "
            if latent_var_relation == 'Positive':
                interpretation += f"an increase in (1 unit) `{dependent_variable_name}` is associated with an increase (`{latent_var_estimate:.3f}`) in `{latent_var}`.\n"
            else:
                interpretation += f"an increase (1 unit) in `{dependent_variable_name}` is associated with a decrease (`{latent_var_estimate:.3f}`) in `{latent_var}`.\n"


    interpretations.append(interpretation)

    return interpretations


def check_if_analysis_with_moderator_required(demographic_dict):
    is_analysis_with_moderator_required = False
    # Check if there are Moderators in Model file
    moderators = []
    for col in demographic_dict:
            val = str(col.get('used_as_moderator'))
            if val.lower() == 'yes':    
                # moderators.append(col['used_as_moderator'])
                moderators.append(col.get('Variable'))
    if len(moderators):
        is_analysis_with_moderator_required = True
    return is_analysis_with_moderator_required, moderators

def conduct_sem_with_moderators(sem_model_spec, data_normalized, label_mappings, moderators):

    # Initialize a dictionary to store SEM results
    sem_results_full = []

    for moderator in moderators:
        moderator_values_unique = data_normalized[moderator].unique()

        sem_results = []
        for dem_val in moderator_values_unique: 
            column_name = moderator
            numerical_label = dem_val
            original_label = get_back_original_label_from_numerical_label(label_mappings, column_name, numerical_label)
             
            #print(f'\n====== Conduct SEM for  {moderator_value}   =========== ')
            # Extract the subset of data for the current moderator value
            subset_data = data_normalized[data_normalized[moderator] == dem_val]
            # Get the number of rows in the DataFrame
            num_rows = len(subset_data)
            # print('\n===========================')
            # print(moderator_value, num_rows)


            if num_rows >1:
                
                # Create the SEM model object
                sem_model = semopy.Model(sem_model_spec)

                # Fit the model using the subset of data
                sem_model.fit(subset_data)

                # Retrieve the results using the inspect method
                sem_inspect = sem_model.inspect()

                # Store the SEM results in the dictionary
                sem_inspect_filtered = filter_inspect_table_from_spec(sem_inspect, sem_model_spec)
                # sem_results[(demo_var_name, demo_value)] = sem_inspect_filtered
                sem_results.append([moderator, original_label, num_rows, sem_inspect_filtered])
                print(f'\n====== Conduct SEM for  {moderator}   Done \n')
        
            # print(sem_results)
            print('\n+++++++++++++++\n')


        sem_results_full.extend(sem_results)

    return sem_results_full


# Define a function to format a value to two decimal places
def format_as_two_decimal_places(number):
    if isinstance(number, (int, float)):
        return f'{number:.3f}'
    return number  # Return unchanged if it's not a number

def post_process_sem_with_moderator_resuls(sem_results_full, inspect_baseline, moderators):
    print('\n============ Post process SEM with Moderators =========\n')
    synthesis_table = inspect_baseline
    columns_to_remove = ['Std. Err', 'z-value', 'Significance', 'Relation']
    synthesis_table = synthesis_table.drop(columns=columns_to_remove)
    synthesis_table.rename(columns={'Estimate': 'Est. baselined'}, inplace=True)

    df_category_list = []
    for moderator in moderators:
        sub_category_name, category_number_data, category_data = get_data_by_category(sem_results_full, moderator)
        # print(sub_category_name, category_number_data, category_data)
        for idx in range(len(sub_category_name)):

            msg_to_display = (f'{moderator}:  {sub_category_name[idx]},  Number of data: {category_number_data[idx]}')

            df_category = category_data[idx]
            df_category.insert(loc=3, column='baseline', value=inspect_baseline['Estimate'].round(2))
            synthesis_table[f'Est category {idx+1}'] = df_category['Estimate']
            # Calculate the percentage difference between 'Estimate' and 'baseline'
            df_category['diff %'] = (((df_category['Estimate'] - df_category['baseline']) / abs(df_category['baseline'])) * 100).round(2)
            synthesis_table[f'diff % {idx+1}'] = (((df_category['Estimate'] - synthesis_table['Est. baselined']) / abs(synthesis_table['Est. baselined'])) * 100).round(2)

            # Reorder the columns to place 'diff' after 'Estimate'
            df_category = df_category[['lval', 'op', 'rval', 'baseline', 'Estimate', 'diff %', 'Std. Err', 'z-value', 'p-value']]
            columns_to_remove = ['Std. Err', 'z-value']
            df_category = df_category.drop(columns=columns_to_remove)
            columns_to_format = ['Estimate', 'p-value']
            df_category[columns_to_format] = df_category[columns_to_format].map(format_as_two_decimal_places)

            df_category_list.append([moderator, msg_to_display, df_category])

    # print(df_category_list)

    return df_category_list 


def get_value_by_key(data_list, desired_key):
    result_value = None

    for item in data_list:
        if item[0] == desired_key:
            result_value = item[1]
            break

    return result_value


def get_data_by_category(data_list, category_name):
    # print(data_list)
    # print(category_name)
    category_data = []
    sub_category_name = []
    sub_category_data_number = []
    for item in data_list:
        if item[0] == category_name:
            sub_category_name.append(item[1])
            sub_category_data_number.append(item[2])
            category_data.append(item[3])


    # print(category_data)
    return sub_category_name, sub_category_data_number, category_data


def create_synthesis_table(sem_results):
    # Initialize lists to store data
    lval_list = []
    relation_list = []
    rval_list = []
    max_estimate_list = []
    subkey_list = []

    # Iterate through SEM results
    key = ''
    for key, subkey, data_number, values in sem_results:
        df = pd.DataFrame(values)
        
        # Find the row with the maximum estimate for each relation
        max_estimate_rows = df.groupby(['lval', 'op', 'rval'])['Estimate'].idxmax()
        
        # Iterate through the rows with maximum estimates
        for idx in max_estimate_rows:
            max_estimate_row = df.loc[idx]
        
            # Extract values from the row
            lval = max_estimate_row['lval']
            relation = max_estimate_row['op']
            rval = max_estimate_row['rval']
            max_estimate = max_estimate_row['Estimate']
            
            # Append values to lists
            lval_list.append(lval)
            relation_list.append(relation)
            rval_list.append(rval)
            max_estimate_list.append(max_estimate)
            subkey_list.append(subkey)

    # Create the result DataFrame
    result_df = pd.DataFrame({
        'lval': lval_list,
        '~': relation_list,
        'rval': rval_list,
        'Estimate': max_estimate_list,
        f'{key}': subkey_list  # Assuming you want to store 'key' as a separate column
    })

    return result_df

def extract_names(variables):
        return [var['name'] for var in variables]
