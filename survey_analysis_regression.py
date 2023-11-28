# from docx import Document
import pandas as pd
import statsmodels.api as sm

import numpy as np
# import os

import re

import graphviz

from io import StringIO

from helper_functions import *
from survey_common import *


output_path = './output/'
create_new_directory(output_path)



def create_model_spec(independent_dict, dependent_dict, relation_dict, varcovar_dict):

    # Build the model spec
    model_spec = "### Independent Variables\n"
    model_spec += "\n".join([f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(d['number_questions'])]) for d in independent_dict])
    model_spec += "\n\n### Dependent Variables\n"
    model_spec += "\n".join([f"{d['Variable']} =~ " + " + ".join([f"{d['Variable']}_Q{i+1}" for i in range(d['number_questions'])]) for d in dependent_dict])
    model_spec += "\n\n### Relations\n"

    # Relations
    model_spec += "\n".join([f"{r['Variable']} ~ {r['Related_Variables']}" for r in relation_dict if r['Relation_Type'] == 'direct' or r['Relation_Type'] == 'both'])

    print('\n\n ========= Model Spec ===========')
    print(model_spec)

    # Constructing the model_spec_dict
    model_spec_dict = convert_to_model_spec_dict(model_spec)
    # print('\n\n ========= Model Spec Dict ===========')
    # print(model_spec_dict)
    
    return model_spec, model_spec_dict


def convert_to_model_spec_dict(model_spec_text):
    # Split the text into lines and filter out empty lines
    lines = [line.strip() for line in model_spec_text.split('\n') if line.strip()]

    # Initialize variables to store the parsed information
    independent_vars = {}
    dependent_vars = {}
    relations = []

    # Regular expressions for identifying parts of the spec
    independent_pattern = r'(\w+) =~ (.+)'
    dependent_pattern = r'(\w+) =~ (.+)'
    relation_pattern = r'(\w+) ~ (.+)'

    # Parse the spec text
    for line in lines:
        if re.match(independent_pattern, line):
            var, questions = re.match(independent_pattern, line).groups()
            independent_vars[var] = [q.strip() for q in questions.split('+')]
        elif re.match(dependent_pattern, line):
            var, questions = re.match(dependent_pattern, line).groups()
            dependent_vars[var] = [q.strip() for q in questions.split('+')]
        elif re.match(relation_pattern, line):
            relations.append(line.strip())

    # Combine parsed information into a dictionary
    model_spec_dict = {
        'independent': independent_vars,
        'dependent': dependent_vars,
        'relations': relations
    }

    # print(model_spec_dict)
    return model_spec_dict


def extract_indicators_from_model_spec(model_spec):

    # Regular expression to match indicators with 'Q' and a number at the end
    regex = re.compile(r'\b\w+_Q\d\b')

    # Use re.findall() to find all matching indicators
    indicators = re.findall(regex, model_spec)

    # Remove duplicates by converting the list to a set and back to a list
    indicators = list(set(indicators))

    # Sort the list for easier reading
    indicators.sort()

    return indicators


def extract_indicators_from_model_spec_2(model_spec):
    """ Extracts indicators from the model specification. """
    indicators = []
    lines = model_spec.strip().split("\n")
    for line in lines:
        if "=~" in line:
            _, rhs = line.split('=~', maxsplit=1)
            indicators.extend(rhs.strip().split('+'))
    return [indicator.strip() for indicator in indicators]


def create_model_spec_graph_full(model_spec, independent_dict, dependent_dict, graph_name):
    # Initialize the Graphviz Digraph
    g = graphviz.Digraph('Model Spec', format='png', engine='dot')
    g.attr(rankdir='RL', overlap='scale', splines='true', fontsize='12')  
    
    # Extract names from dictionaries
    dependent_variable_names = [item['Variable'] for item in dependent_dict]
    independent_variable_names = [item['Variable'] for item in independent_dict]

    indicator_variable_names = extract_indicators_from_model_spec(model_spec)

    # Create subgraphs for indicators
    with g.subgraph() as s:
        # s.attr(rank='same')
        for indicator_name in indicator_variable_names:
            s.node(indicator_name, shape='box', fillcolor='#e6f2ff', style='filled')

    # Create invisible edges between a node from one subgraph to a node from the next subgraph
    g.edge(indicator_variable_names[-1], independent_variable_names[0], style='invis')  # From indicators to observables


    # Create subgraphs for independent variables
    with g.subgraph() as s:
        # s.attr(rank='same')
        for var_name in independent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#cae6df', style='filled')

    # Create subgraphs for dependent variables
    with g.subgraph() as s:
        s.attr(rank='source')
        for var_name in dependent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#FFFF00', style='filled')

    # Parse the model specification to extract relationships and nodes
    lines = model_spec.strip().split("\n")
    for line in lines:
        # Relations
        if '~' in line and '=~' not in line:
            lhs, rhs = line.split('~', maxsplit=1)  
            lhs, rhs = lhs.strip(), rhs.strip()
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                g.edge(var.strip(), lhs, dir='forward', color='blue', penwidth='2')
        
        # Measurement Model
        if '=~' in line:
            lhs, rhs = line.split('=~', maxsplit=1)
            lhs, rhs = lhs.strip(), rhs.strip()
            for indicator in rhs.split('+'):
                g.edge(lhs, indicator.strip(), dir='forward')

    # Save to file
    # graph_name = 'sem_model_graph'
    g.attr(dpi='600', imagescale='true')
    g.render(f"{output_path}/{graph_name}")

    graph_path = f"{output_path}/{graph_name}.png"
    return graph_path



def create_model_spec_graph_short(model_spec, independent_dict, dependent_dict, graph_name):
    # Initialize the Graphviz Digraph
    g = graphviz.Digraph('Model Spec', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')  
    
    # Extract names from dictionaries
    dependent_variable_names = [item['Variable'] for item in dependent_dict]
    independent_variable_names = [item['Variable'] for item in independent_dict]

    # Create subgraphs for independent variables
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in independent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#cae6df', style='filled')

    # Create subgraphs for dependent variables
    with g.subgraph() as s:
        #s.attr(rank='max')
        for var_name in dependent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#FFFF00', style='filled')

    # Parse the model specification to extract relationships and nodes
    lines = model_spec.strip().split("\n")
    for line in lines:
        # Relations
        if '~' in line and '=~' not in line:
            lhs, rhs = line.split('~', maxsplit=1)  
            lhs, rhs = lhs.strip(), rhs.strip()
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                g.edge(var.strip(), lhs, dir='forward', color='blue', penwidth='2')

    # Save to file
    # graph_name = 'sem_model_graph'
    g.attr(dpi='600', imagescale='true')
    g.render(f"{output_path}/{graph_name}_short")

    graph_path = f"{output_path}/{graph_name}_short.png"
    return graph_path



def conduct_analysis(model_file_path, data_file_path):

    # Read model file
    demographic_dict, independent_dict, dependent_dict, relation_dict, varcovar_dict = read_model(model_file_path)

    # Create Model Spec (use Semopy format)
    model_spec = create_model_spec(demographic_dict, independent_dict, dependent_dict, relation_dict, varcovar_dict)

    # print('\n\n======== Model Spec ============')
    # print(model_spec)

    # Read data file


# Function to create composite scores
def create_composite_score(df, questions):
    df_temp = df.copy()
    return df_temp[questions].mean(axis=1)

# Function to run regression based on model spec relations
def run_regression(df, relation):
    dependent_var, independent_vars = relation.split('~')
    dependent_var = dependent_var.strip()
    independent_vars = independent_vars.strip().split('+')
    
    X = df[independent_vars].apply(lambda col: col)
    X = sm.add_constant(X)  # Add a constant to the model
    y = df[dependent_var.strip()]

    model = sm.OLS(y, X).fit()
    result = model.summary()

    return result

def conduct_regression_analysis(df, model_spec_dict):
    data = df.copy()
    # Create composite scores for each independent variable
    for construct, questions in model_spec_dict['independent'].items():
        data[construct] = create_composite_score(data, questions)

    # Create composite scores for each dependent variable
    for construct, questions in model_spec_dict['dependent'].items():
        data[construct] = create_composite_score(data, questions)

    # Running regression models based on relations in model_spec_dict
    results = []
    for relation in model_spec_dict['relations']:
        print(f"\n\n======= Regression Model: {relation}")
        result = run_regression(data, relation)
        results.append([relation, result])

        print(result)

    return results

# Function to run regression based on model spec relations
def run_regression_standardized(df, relation):
    dependent_var, independent_vars = relation.split('~')
    dependent_var = dependent_var.strip()
    independent_vars = independent_vars.strip().split('+')
    
    # Standardize the independent variables
    X = df[independent_vars].apply(lambda col: (col - col.mean()) / col.std())
    X = sm.add_constant(X)  # Add a constant to the model

    # Standardize the dependent variable
    y = df[dependent_var]
    y = (y - y.mean()) / y.std()

    model = sm.OLS(y, X).fit()
    result = model.summary()

    return result

def conduct_regression_analysis_standardized(data, model_spec_dict):
    # Create composite scores for each independent variable
    for construct, questions in model_spec_dict['independent'].items():
        data[construct] = create_composite_score(data, questions)

    # Create composite scores for each dependent variable
    for construct, questions in model_spec_dict['dependent'].items():
        data[construct] = create_composite_score(data, questions)

    # Running regression models based on relations in model_spec_dict
    results = []
    for relation in model_spec_dict['relations']:
        print(f"\n\n======= Regression Model: {relation}")
        result = run_regression_standardized(data, relation)
        results.append([relation, result])

        print(result)

    return results


def get_parameter_thresholds(parameter_name, parameters_dict):
    # Loop through the list of dictionaries and find the matching dictionary
    matching_dict = next((item for item in parameters_dict if item['parameter'] == parameter_name), None)

    # Check if a matching dictionary was found and retrieve the value
    if matching_dict:
        threshold_value_high = matching_dict['high_threshold']
        threshold_value_moderate = matching_dict['moderate_threshold']
        # print(f"The high threshold for {parameter_name} is: {efa_threshold_value}")
    else:
        print(f"No matching parameter found for {parameter_name}")

    return threshold_value_high, threshold_value_moderate


# Function to extract the information using regular expressions
def extract_regression_results(results):
    # Regular expressions to extract the needed information
    general_stats_patterns = {
        'Dependent Variable': r"Dep\. Variable:\s+([^\s]+)",
        'R-squared': r"R-squared:\s+([0-9.]+)",
        'Adjusted R-squared': r"Adj\.\s*R-squared:\s+([-0-9.]+)",
        'F-statistic': r"F-statistic:\s+([0-9.]+)",
        'Prob (F-statistic)': r"Prob \(F-statistic\):\s+([0-9.e-]+)",
        'Log-Likelihood': r"Log-Likelihood:\s+([-0-9.]+)",
        'AIC': r"AIC:\s+([0-9.]+)",
        'BIC': r"BIC:\s+([0-9.]+)"
    }
    results_txt = str(results)
    # Extracting general statistics
    general_stats = {key: re.search(pattern, results_txt).group(1) for key, pattern in general_stats_patterns.items() if re.search(pattern, results_txt)}
    # print('\nGeneral Stats Infos ------')
    # print(general_stats)

    # Regular expressions to capture the relevant statistics
    relevant_stat_patterns = {
        'Omnibus': r"Omnibus:\s+(\d+\.\d+)",
        'Prob(Omnibus)': r"Prob\(Omnibus\):\s+(\d+\.\d+)",
        'Durbin-Watson': r"Durbin-Watson:\s+(\d+\.\d+)",
        'Jarque-Bera (JB)': r"Jarque-Bera \(JB\):\s+(\d+\.\d+)",
        'Prob(JB)': r"Prob\(JB\):\s+(\d+\.\d+)",
        'Skew': r"Skew:\s+([-]?\d+\.\d+)",
        'Kurtosis': r"Kurtosis:\s+(\d+\.\d+)",
        'Cond. No.': r"Cond\. No\.\s+(\d+\.\d+)"
    }
    relevant_stats = {}
    for key, pattern in relevant_stat_patterns.items():
        match = re.search(pattern, results_txt)
        if match:
            relevant_stats[key] = float(match.group(1))
        else:
            relevant_stats[key] = None
    # print('\nRelevant Stats Infos ------')
    # print(relevant_stats)

    # Extracting variable infos
    summary_df = pd.read_html(StringIO(results.tables[1].as_html()), header=0, index_col=0)[0]
    # Create a dictionary to store variable information
    variable_info_dict = {}

    # Iterate over rows (variables) and store information in the dictionary
    for variable_name, variable_info in summary_df.iterrows():
        variable_dict = {
            "Estimated Coefficient": variable_info['coef'],
            "Standard Error": variable_info['std err'],
            "t-Value": variable_info['t'],
            "P-Value": variable_info['P>|t|']
        }
        variable_info_dict[variable_name] = variable_dict

    # Remove the 'const' key
    del variable_info_dict['const']
    # Print the variable information dictionary
    # print('\nVariable Infos ------')
    # print(variable_info_dict)

    return general_stats, relevant_stats, variable_info_dict


def create_ols_summary_table(general_stats):
    print(general_stats)
    # Extract key values
    dependent_variable = general_stats.get('Dependent Variable')
    r_squared = float(general_stats.get('R-squared'))
    r_squared_str = f"{r_squared:.1%}"
    adj_r_squared = float(general_stats.get('Adjusted R-squared'))
    adj_r_squared_str = f"{adj_r_squared:.1%}"
    f_statistic = float(general_stats.get('F-statistic'))
    f_statistic_str = "Highly Significant" if f_statistic < 0.05 else "Not Significant"
    prob_f = float(general_stats.get('Prob (F-statistic)'))
    prob_f_str = f"{prob_f:.3f}"

    # print(f'\n\n=== {dependent_variable}')
    # Assemble data for the DataFrame
    summary_data = {
        "Aspect of Model": [
            "What we're predicting", 
            "How well we can predict", 
            "Model Confidence", 
            "Model Fit", 
            "Statistical Significance"
        ],
        "Variable": [
            dependent_variable, 
            "R-squared", 
            "F-statistic", 
            "Adjusted R-squared", 
            "Prob (F-statistic)"
        ],
        "Value": [
            "-", 
            r_squared_str, 
            f_statistic_str, 
            adj_r_squared_str, 
            prob_f_str
        ],
        "Acceptable Threshold": [
            "-", 
            "> 50% is considered good", 
            "p < 0.05 indicates significance", 
            "Close to R-squared value", 
            "p < 0.05 indicates significance"
        ],
        "Details": [
            f"Predicts {dependent_variable}.", 
            f"The model explains {r_squared_str} of the variation in {dependent_variable}.", 
            f"The model is highly reliable in predicting {dependent_variable}.", 
            f"Adjusted for the number of predictors, still a strong fit ({adj_r_squared_str}).", 
            "Strong evidence that the model's predictions are not by chance."
        ]
    }

    # Create the DataFrame
    df_summary = pd.DataFrame(summary_data)

    return df_summary


def interpret_regression_results(regression_results, parameters_dict):

    parameter_name = 'r_squared'
    r_squared_high, r_squared_moderate = get_parameter_thresholds(parameter_name, parameters_dict)
    parameter_name = 'p_value_threshold'
    p_value_high, p_value_moderate = get_parameter_thresholds(parameter_name, parameters_dict)
    parameter_name = 'durbin_watson'
    durbin_watson_high, durbin_watson_moderate = get_parameter_thresholds(parameter_name, parameters_dict)


    interpretations = []
    predictors_dfs = []
    summary_tables = []

    idx = 0
    for result in regression_results:
        idx += 1
        interpretation = []
        relation = result[0]
        results = result[1]
        # results_text = str(results) 
        general_stats, relevant_stats, variable_info_dict = extract_regression_results(results)

        # 1. R-squared value
        r_squared = float(general_stats.get('R-squared'))
        msg = f"R_squared (Coefficient of Determination) of {r_squared:.3f} "
        msg += f"indicates that approximately {r_squared * 100:.1f}% of the variability in the target variable can be explained by the independent variables."
        interpretation.append(msg)
        # print(msg)

        # 2. Adjusted R^2
        # adjusted_r_squared = float(general_stats.get('Adjusted R-squared'))
        # msg = f" Adjusted R^2 {adjusted_r_squared:.3f} "
        # msg += 'is a more accurate measure when comparing models with different numbers of predictors.'
        # interpretation.append(msg)

        # 3. F-statistic and Prob (F-statistic)
        f_statistic = float(general_stats.get('F-statistic'))
        prob_f_statistic = float(general_stats.get('Prob (F-statistic)'))

        msg = f"F-statistic of ({f_statistic:.3f}) with its corresponding p-value Prob (F-statistic) ({prob_f_statistic:.2e}): "
        if f_statistic < p_value_moderate:
            msg += f"the Prob (F-statistic) is less than the significance level of {p_value_moderate}, indicating that the model as a whole is statistically significant."
        else:
            msg += f"the Prob (F-statistic) is greater than the significance level of {p_value_moderate}, indicating that the model as a whole is not statistically significant."        
        interpretation.append(msg)

        # 4. Durbin-Watson statistic
        durbin_watson = float(relevant_stats.get('Durbin-Watson'))
        # Define the lower and upper bounds for Durbin-Watson statistic
        # Interpret the Durbin-Watson statistic
        if durbin_watson < durbin_watson_moderate:
            msg = f"Durbin-Watson ({durbin_watson:.3f}) suggests positive autocorrelation."
        elif durbin_watson > durbin_watson_high:
            msg = f"Durbin-Watson ({durbin_watson:.3f}) suggests negative autocorrelation."
        else:
            msg = f"Durbin-Watson ({durbin_watson:.3f}) suggests that there is no autocorrelation."
        interpretation.append(msg)

        # 5. Individual Coefficients and Significant Predictors
        
        # Convert dictionary to DataFrame and reset index to get variable names as a column
        predictors_df = pd.DataFrame(variable_info_dict).T.reset_index()
        predictors_df.rename(columns={'index': 'Variable'}, inplace=True)

        # Exclude the 'const' variable
        predictors_df = predictors_df[predictors_df['Variable'] != 'const']

        # Format the 'Estimated Coefficient' and 'P-Value' columns
        predictors_df['Estimated Coefficient'] = predictors_df['Estimated Coefficient'].apply('{:.3f}'.format)
        predictors_df['P-Value'] = predictors_df['P-Value'].apply('{:.3f}'.format)

        # Splitting into significant and not significant predictors
        significant_predictors_df = predictors_df[predictors_df['P-Value'].astype(float) < p_value_moderate]
        not_significant_predictors_df = predictors_df[predictors_df['P-Value'].astype(float) >= p_value_moderate]

        # Constructing the message
        interpretation += ["\nIndividual Coefficients:"]

        # Significant Predictors
        if not significant_predictors_df.empty:
            interpretation.append('\nSignificant Predictors (p < {:.3f}):'.format(p_value_moderate))
            for _, row in significant_predictors_df.iterrows():
                line = f"- {row['Variable']}: Coefficient= {row['Estimated Coefficient']}, p-value= {row['P-Value']}"
                interpretation.append(line)
        else:
            interpretation.append('\nNo Significant Predictors (p < {:.3f}).'.format(p_value_moderate))

        # Not Significant Predictors
        if not not_significant_predictors_df.empty:
            interpretation.append('\nNot Significant Predictors (p >= {:.3f}):'.format(p_value_moderate))
            for _, row in not_significant_predictors_df.iterrows():
                line = f"- {row['Variable']}: Coefficient= {row['Estimated Coefficient']}, p-value= {row['P-Value']}"
                interpretation.append(line)
        #else:
        #    interpretation.append('\nAll Predictors are Significant (p < {:.3f}).'.format(p_value_moderate))

        # Joining the list elements into a single string
        interpretation_str = '\n'.join(interpretation)

        #print('\n\nInterpretation -----------')
        # print(interpretation)
        interpretations.append([relation, interpretation_str])

        predictors_dfs.append(predictors_df)

        # Create summary model interpretation table
        summary_table = create_ols_summary_table(general_stats)
        summary_tables.append(summary_table)

    return interpretations, predictors_dfs, summary_tables
        

def create_result_graph_short(model_spec, independent_dict, dependent_dict, predictors_dfs, graph_name):
    # Initialize the Graphviz Digraph
    g = graphviz.Digraph('Model Spec', format='png', engine='dot')
    g.attr(rankdir='LR', overlap='scale', splines='true', fontsize='12')  
    
    # Extract names from dictionaries
    dependent_variable_names = [item['Variable'] for item in dependent_dict]
    independent_variable_names = [item['Variable'] for item in independent_dict]

    # Create subgraphs for independent variables
    with g.subgraph() as s:
        s.attr(rank='same')
        for var_name in independent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#cae6df', style='filled')

    # Create subgraphs for dependent variables
    with g.subgraph() as s:
        #s.attr(rank='max')
        for var_name in dependent_variable_names:
            s.node(var_name, shape='ellipse', fillcolor='#FFFF00', style='filled')

    # Parse the model specification to extract relationships and nodes

    # Concatenate the list of DataFrames into one
    all_predictors_df = pd.concat(predictors_dfs, ignore_index=True)

    lines = model_spec.strip().split("\n")
    for line in lines:
        # Relations
        if '~' in line and '=~' not in line:
            lhs, rhs = line.split('~', maxsplit=1)  
            lhs, rhs = lhs.strip(), rhs.strip()
            rhs_vars = rhs.strip().split('+')
            for var in rhs_vars:
                var = var.strip()
                # Retrieve coefficient and p-value for the edge from predictors_dfs
                predictor_rows = all_predictors_df[all_predictors_df['Variable'] == var]
                if not predictor_rows.empty:
                    # Aggregate coefficients and p-values if there are multiple rows
                    coef = predictor_rows['Estimated Coefficient'].astype(float).mean()
                    p_val = predictor_rows['P-Value'].astype(float).mean()
                    edge_label = f"coef: {coef:.2f}, p_val: {p_val:.2f}"

                    # Set edge color based on p-value
                    edge_color = 'red' if p_val > 0.05 else 'blue'

                    g.edge(var, lhs, label=edge_label, dir='forward', color=edge_color, penwidth='2')




    # Save to file
    # graph_name = 'result_graph_short'
    g.attr(dpi='600', imagescale='true')
    g.render(f"{output_path}/{graph_name}")

    graph_path = f"{output_path}/{graph_name}.png"
    return graph_path


def conduct_regression_analysis_with_moderators(model_spec_dict, data_normalized, label_mappings, moderators, basedline_regression_results):
    # Initialize a dictionary to store Regression results
    results_full = []

    for moderator in moderators:
        moderator_values_unique = data_normalized[moderator].unique()

        moderator_results = []
        for dem_val in moderator_values_unique: 
            column_name = moderator
            numerical_label = dem_val
            original_label = get_back_original_label_from_numerical_label(label_mappings, column_name, numerical_label)
             
            # Extract the subset of data for the current moderator value
            subset_data = data_normalized[data_normalized[moderator] == dem_val]
            # Get the number of rows in the DataFrame
            num_rows = len(subset_data)
            # print('\n===========================')
            # print(moderator_value, num_rows)

            if num_rows >1: # If subdata not empty
                # Conduct Analysis
                sub_result_full = conduct_regression_analysis(subset_data, model_spec_dict)

                # Store the Sub results in the dictionary
                moderator_results.append([moderator, original_label, num_rows, sub_result_full])
                print(f'\n====== Regression Analysis for  {moderator} ({original_label}) Completed ====== \n')
        
            
        results_full.append(moderator_results)

        comparison_table = post_process_result_with_moderators(results_full, basedline_regression_results)
    
    return comparison_table

def create_table(df):
    # Create a DataFrame
    df = pd.DataFrame(df).transpose()
    # Keep only the "Estimated Coefficient" and "P-Value" columns
    df = df[['Estimated Coefficient', 'P-Value']]
    # Rename the columns
    df.columns = ['Baseline Coef', 'P-Value']

    return df


def post_process_result_with_moderators(results_full, baseline_regression_results):

    print('\n============ Post process Regression with Moderators =========\n')

    # Initialize an empty DataFrame for comparison table
    comparison_table = pd.DataFrame()

    # Dictionary to store tables
    tables = {}

    # Process baseline regression results
    for relation, summary in baseline_regression_results:
        _, _, coef_dict = extract_regression_results(summary)
        tables[relation] = create_table(coef_dict)
    
    # Process results with moderators
    for moderator_result in results_full:
        for result in moderator_result:
            moderator, label, num_rows, sub_result_full = result
            for var_relation, var_stats in sub_result_full:
                _, _, var_stat_ = extract_regression_results(var_stats)
                # Extract 'Estimated Coefficient' values into a list
                estimated_coefficients = [data['Estimated Coefficient'] for data in var_stat_.values()]
                # Add the estimated coefficients as a new column to an existing table
                if var_relation in tables:
                    name_tmp = f'{moderator}: {label}'
                    tables[var_relation][name_tmp] = estimated_coefficients

    # Create DataFrames for each relationship
    dataframes = {}
    for relationship, values in tables.items():
        df = pd.DataFrame(values)
        dataframes[relationship] = df
    
    return dataframes


def generate_regression_equations(model_spec):
    # Split the model specification into lines
    model_lines = model_spec.split('\n')

    # Initialize a dictionary to store coefficients
    coefficients = {}

    # Initialize a list to store equations
    all_equations = []

    # Regular expression pattern to extract variable names and coefficients
    variable_coefficient_pattern = r'([A-Za-z_][A-Za-z_0-9]*) ~ ([A-Za-z_][A-Za-z_0-9]*)(?: \+ ([A-Za-z_][A-Za-z_0-9]*))*'
    coefficient_pattern = r'([A-Za-z_][A-Za-z_0-9]*) =~ ([A-Za-z_][A-Za-z_0-9]*)(?: \+ ([A-Za-z_][A-Za-z_0-9]*))*'

    current_section = None  # Track the current section

    # Loop through each line in the model specification
    for line in model_lines:
        # Check if the line specifies a section header
        if "###" in line:
            current_section = line.strip("###").strip()  # Extract the section header
            continue

        # Check if the line specifies a coefficient
        coefficient_match = re.match(coefficient_pattern, line.strip())
        if coefficient_match:
            dependent_variable = coefficient_match.group(1)
            independent_variables = coefficient_match.group(2).split(' + ')
            coefficients[dependent_variable] = independent_variables

        # Check if the line specifies a relation
        relation_match = re.match(variable_coefficient_pattern, line.strip())
        if relation_match:
            dependent_variable = relation_match.group(1)
            independent_variables = relation_match.group(2).split(' + ')
            # Create the regression equation with coefficients
            equation = f"{dependent_variable} = "
            for i, independent_variable in enumerate(independent_variables):
                coefficient = coefficients.get(dependent_variable, [])[i]
                equation += f"{coefficient} * {independent_variable} + "
            equation = equation[:-2]  # Remove the trailing ' + '
            # Append the equation to the list
            all_equations.append(equation)

    # Combine all equations into a single text string
    equations_text = "\n".join(all_equations)

    return equations_text