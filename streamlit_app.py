import survey_analysis_regression as reg
import survey_analysis_sem as sem

from survey_quality_analysis import *
from survey_common import *


import streamlit as st



# Create a Streamlit app title and description
st.title("Survey Data Analysis")
st.write("This app helps you analyze survey data using Regression and SEM models.")
st.write('\n\n-----------\n\n\n\n')


st.subheader('2 Steps To Do:')
st.write('Step 1:  Download Data Model Sample file')
# To link to Data Model Sample Excel file
data_model_sample_url = 'https://docs.google.com/spreadsheets/d/1BkA6d7FMxBEBwO84NTgXEaGMOat9YAP3/edit?usp=sharing&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'- [Data Model Sample file download]({data_model_sample_url})')
st.write('- Customize this Excel file according to your model and questionnaire structure.')

st.write('Step 2: Upload the survey data (Excel file downloaded from Google Forms or others).')

# To link to Survey Data Sample Excel file
data_sample_url = 'https://docs.google.com/spreadsheets/d/1Bkp_SOZatsy1L69kiJSmHmBFvclIEagX/edit?usp=sharing&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'- [Survey Data Sample file to download]({data_sample_url})')

st.write('\n\n-----------')
st.write('Contact:     ho.tuong.vinh@gmail.com')
st.write('ATTENTION: I HAVE NO RESPONSIBILITY FOR THE OUTCOME OF THIS ANALYSIS. USE WITH CAUTION!')
st.write('\n\n-----------\n\n\n\n')

st.header("Analysis Workflow")

# Create a table for the workflow steps
workflow_table = """
| Step | Description |
|------|-------------|
| 1    | Upload Your Data Model and Survey Data |
| 2    | Descriptive Analysis |
| 3    | Cronbach's Alpha Analysis |
| 4    | Correlation Analysis |
| 5    | Regression Analysis |
| 6    | Standardized Regression Analysis |
| 7    | Regression Subgroup Analysis with Moderators |
| 8    | Structural Equation Modeling (SEM) Analysis if specified in Model file (SEM Structure Sheet) |
"""

# Display the workflow table using Markdown
st.markdown(workflow_table)

st.write('\n\n-----------\n\n\n\n')


def display_model_spec(model_spec):
    
    # Splitting the content at each section to print headers separately
    sections = model_spec.split("###")
    
    for section in sections:
        if section.strip() != '':
            header, content = section.split("\n", 1)
            st.markdown(f"## {header.strip()}")
            st.markdown(f"```\n{content.strip()}\n```")

def perform_descriptive_analysis(data, variable_dicts):

    for variable_dict in variable_dicts:
        variable_type = variable_dict['type']
        # st.header(f'{variable_type} Variables')
        
        for variable_info in variable_dict['variables']:
            column_index = variable_info['column_index']
            variable_name = variable_info['Variable']
            column_data = data.iloc[:, column_index]

            if column_data.dtype == 'object':
                # Perform analysis for categorical variables
                frequency_distribution = column_data.value_counts()
                total_count = len(column_data)
                
                st.write(f"{variable_name}:")
                
                # Create a table with counts and percentages
                data_table = pd.DataFrame({
                    'Count': frequency_distribution.values,
                    'Percentage': (frequency_distribution / total_count * 100).round(2)
                })
                st.write(data_table)
            else:
                # Perform analysis for numerical variables
                summary_stats = column_data.describe()
                st.write(f"{variable_name}")
                st.write(summary_stats)


st.header('Step 1: Upload Data Model and Survey Data')

# Upload DATA MODEL Excel file
model_file_path = st.file_uploader("Upload MODEL Excel file", type=["xlsx", "xls"])
# Upload Data
data_file_path = None
data_file_path = st.file_uploader("Upload DATA Excel file", type=["xlsx", "xls"])

st.write('\n\n-----------\n\n\n\n')

if model_file_path is not None and data_file_path is not None:
    if st.button("Start Analyzing ?"):
        st.write("\n\nI am working on analysis now ............ Enjoy !")

        st.write('\n\n-----------\n\n\n\n')
        st.header('\n\nStep 2:    Descriptive Analysis\n\n')

        # Read the model descritption
        model_description =pd.ExcelFile(model_file_path)
        # Display sheet by shee
        st.subheader('=========        Model Description      =========')
        sheet_names = model_description.sheet_names
        for sheet_name in sheet_names:
            sheet_df = pd.read_excel(model_description, sheet_name=sheet_name)
            st.write(sheet_name)
            st.write(sheet_df)

        # Convert Model Description into a dictionary
        demographic_dict, independent_dict, latent_dict, dependent_dict, \
            reg_relation_dict, sem_relation_dict, \
            varcovar_dict, parameters_dict = read_model(model_file_path)


        st.subheader('\n========== Data ===============')
        data_original = pd.read_excel(data_file_path)

        st.write(data_original)

        # To do Descriptive Analysis
        # Define your variable_dicts that include demographic variables
        variable_dicts = [
            {
                'type': 'Demographic',
                'variables': demographic_dict
            },
        ]
        st.subheader(f"Descriptive analysis for Demographic Variables")
        perform_descriptive_analysis(data_original, variable_dicts)


        # Pre-process data
        data_normalized, label_mappings, demographic_cols_names, \
            indepent_cols_names, dependent_cols_names = \
            pre_process_data(data_original, demographic_dict, independent_dict, dependent_dict)

        st.subheader('\nData Normalized')
        st.write(data_normalized)

        st.subheader('\nData Normalized Statistics')
        st.write(data_normalized.describe())

        st.write('\n\n-----------\n\n\n\n')
        st.header("Step 3:    Cronbach's Alpha Analysis")

        # Calculate Cronbach's alpha
        overall_alpha, alpha_table = compute_cronbach_alpha(data_normalized, independent_dict)
        st.write(f"Overall Cronbach's Alpha: {overall_alpha}")
        # Display Alpha for each Independent Variable
        st.write(f"Cronbach's Alpha for each independent variables:")
        for idx, col in enumerate(independent_dict):
            col_name = col.get('Variable')
            st.markdown(f'- **{col_name}**:  {alpha_table[idx]}')

        st.write('\n\n-----------\n\n\n\n')
        st.header("Step 4:    Correlation Analysis")

        # Compute Correclation Matrix
        correlation_matrix = compute_correlation(data_normalized, independent_dict)
        st.write('Corellation Analyis (Independent Variables) By Pearson Method')

        # Create a download button for the Excel file
        filename = 'Correlation_Matrix.xlsx'
        st.download_button(
            label="Download Correlation Matrix Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-cor'
        )
        st.write(correlation_matrix)

        st.subheader('\nCorrelation Heatmap')
        st.image(f'{output_path}correlation_heatmap.png')

        # Interpretation et recommendation 
        threshold_strong = 0.7  # Threshold for strong correlation
        threshold_moderate = 0.3    # Threshold for weak correlation
        corr_interpretation_df = interpret_correlation(correlation_matrix, threshold_strong,threshold_moderate)
        st.write(f"Correlation Interpretations with threshold_strong= {threshold_strong} and threshold_weak= {threshold_moderate}")
        st.write(corr_interpretation_df)

        del correlation_matrix

        st.write('\n\n-----------\n\n\n\n')
        st.header("Step 5:    Regression Analsyis")


        # Create model spec
        model_spec, model_spec_dict = reg.create_model_spec(independent_dict, dependent_dict, reg_relation_dict, varcovar_dict)

        # To create Model Spec Graph
        st.subheader('Model Specification')
        display_model_spec(model_spec)
        graph_name = 'model_spec_graph'
        graph_path_full = reg.create_model_spec_graph_full(model_spec, independent_dict, dependent_dict, graph_name)
        st.image(graph_path_full) #, caption='Model Specification Graph')
        graph_path_short = reg.create_model_spec_graph_short(model_spec, independent_dict, dependent_dict, graph_name)
        st.image(graph_path_short) #, caption='Model Specification Graph')

        regression_results = reg.conduct_regression_analysis(data_normalized, model_spec_dict)

        # Baseline results for later subgroup analysis with Moderators
        basedline_regression_results = regression_results

        # Interpret Regression Results
        interpretations, predictors_dfs, summary_tables = reg.interpret_regression_results(regression_results, parameters_dict)

        st.write('\n-------------\n')
        st.subheader('\n\nRegression Analsyis Results')
        idx = 0
        for interpret in interpretations:
            st.subheader(f'\nRegression Relation:  {idx+1}')
            st.text(interpret[0])
            st.write(regression_results[idx][1])
            st.subheader('\nInterpretation')
            st.code(interpret[1])
            st.subheader('Summary Table')
            st.write(summary_tables[idx])
            st.write("\n\n---")  # Adds a horizontal line for separation
            idx += 1
          
        # Create Result Graphs
        graph_name = f'result_graph_short'
        graph_path = reg.create_result_graph_short(model_spec, independent_dict, dependent_dict, predictors_dfs, graph_name)
        st.image(graph_path) 

        st.write('\n\n-----------\n\n\n\n')
        st.header("Step 6:    Standardized Regression Analsyis")

        # TO Make Standardized Regression Analysis
        regression_results = reg.conduct_regression_analysis_standardized(data_normalized, model_spec_dict)
        
        # Interpret Regression Results
        interpretations, predictors_dfs, summary_tables = reg.interpret_regression_results(regression_results, parameters_dict)

        st.subheader('\n\nInterpretion of Regression Analsyis Results')

        idx = 0
        for interpret in interpretations:
            st.subheader(f'\nRegression Relation:  {idx+1}')
            st.text(interpret[0])
            st.write(regression_results[idx][1])
            st.subheader('\nInterpretation')
            st.code(interpret[1])
            st.subheader('Summary Table')
            st.write(summary_tables[idx])
            st.write("\n\n---")  # Adds a horizontal line for separation
            idx += 1
            
        # Create Result Graphs
        graph_name = f'result_graph_short_standardized'
        graph_path = reg.create_result_graph_short(model_spec, independent_dict, dependent_dict, predictors_dfs, graph_name)
        st.image(graph_path) 

        st.write('\n\n-----------\n\n\n\n')
        st.header("Step 7:    Regression Subgroup Analsyis")

        # Conduct Regression Analysis with Moderators according to specific demographic variables
        # Extract moderators
        moderators = [var['Variable'] for var in demographic_dict if var['used_as_moderator'] == 'yes']
        if len(moderators) > 0:
            st.write('\n\nSubgroup Analysis using Demographic variables (as Moderators) and their values.')
            st.write('Extract subdata correspoding to each value of Moderators, and Apply the same Regression Analysis.')
            st.write('The Results with all groups (without the demographic data) are used as "baseline".')
            
            st.subheader('List of moderators')
            st.write(moderators)

            comparison_tables = reg.conduct_regression_analysis_with_moderators(model_spec_dict, data_normalized, label_mappings, moderators, basedline_regression_results)

            idx = 0
            st.subheader('Comparision Table - Subgroup Analysis')
            for relationship, df in comparison_tables.items():
                st.subheader(f'\nRelation: {idx+1}')
                st.write(f'{relationship}:\n')
                st.write(df)
                st.write("\n\n---")  # Adds a horizontal line for separation
                idx += 1
        
        #############################

        ### Check if SEM Analysis is required
        if len(sem_relation_dict) > 0:

            st.header("Step 8:    SEM Analsyis")

            # Create model spec
            sem_model_spec, sem_model_spec_dict = sem.create_model_spec(independent_dict, latent_dict, dependent_dict, sem_relation_dict, varcovar_dict)

            # To create Model Spec Graph
            st.title('Model Specification')
            display_model_spec(sem_model_spec)

            graph_name = 'model_spec_graph'
            graph_path_full = sem.create_model_spec_graph_full(sem_model_spec, independent_dict, latent_dict, dependent_dict)
            st.image(graph_path_full) #, caption='Model Specification Graph')
            graph_path_short = sem.create_model_spec_graph_short(sem_model_spec, independent_dict, latent_dict, dependent_dict)
            st.image(graph_path_short) #, caption='Model Specification Graph')

            sem_result, sem_stats, sem_inspect, sem_inspect_enhanced, sem_inspect_filtered, \
                graph_filtered_results, graph_fulll_results \
                = sem.conduct_sem_analysis(data_normalized, sem_model_spec, independent_dict, latent_dict, dependent_dict)

            st.subheader('\nSEM Model Statistics')
            st.write(sem_stats)
            stats_table, overall_msg = sem.interpret_sem_stats(sem_stats, parameters_dict)
            st.write(stats_table)
            st.write(overall_msg)

            st.subheader('\nSEM Model Results')
            st.write(sem_inspect_enhanced)

            interepretation_sem_inspect = sem.interepret_sem_inspect(sem_inspect_enhanced, dependent_dict, sem_relation_dict, parameters_dict)
            st.subheader('\nInterpret SEM Model Results')
            for interpretation in interepretation_sem_inspect:
                st.markdown(interpretation)
                st.write("---")  # Adds a horizontal line for separation

            st.subheader('SEM Results Graph')
            st.image(graph_fulll_results)

            st.image(graph_filtered_results)



        st.subheader('\n\nUsed Packages:')
        st.write('- scikit-learn and statsmodels for Regression Analysis')
        st.write('- semopy for SEM analysis, https://semopy.com/')
        st.write('- graphviz for Graphics')


        st.write('\n\n-----------\n\n\n\n')
        st.write('And more ... Only 1 minute to convert Youtube video to slides!')
        st.write('https://htvinh-youtube2slides-streamlit-app-k14x3w.streamlitapp.com/')

        st.write('\n\n-----------\n\n\n\n')
        st.header('\n\n========= The END ========')

