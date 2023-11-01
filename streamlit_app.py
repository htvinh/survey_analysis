
from PIL import Image
import random


import streamlit as st

# import matplotlib.pyplot as plt
# import seaborn as sns

from survey_quality_analysis import *

from survey_analyze_sem import *

# Increase the pixel limit for loading large images
Image.MAX_IMAGE_PIXELS = 1000000000  # This can be set to a higher value, or None to remove the limit

is_visualized = False # True False

st.set_page_config(layout="wide")

sem_logo_path = 'survey_sem_logo.png'
st.image(sem_logo_path, use_column_width=True)

st.title('SURVEY ANALYSIS WITH SEM')
st.write('SEM (Structural Equation Modeling) with Symopy (https://semopy.com/)')

st.subheader('2 Steps To Do:')
st.write('Step 1:')
st.write('Download Model Sample file')
# To link to Data Model Sample Excel file
model_sample_url = 'https://docs.google.com/spreadsheets/d/1AalALWVNmzILGB2Pc_-WJJ1S5-UIaQOg/edit?usp=drive_link&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'[Model Sample file download]({model_sample_url})')
st.write('Customize this Excel file according to your analysis model.')

st.write('Step 2:')
st.write('Upload your Model. Then, the tool will ask to upload the data.')
st.write('Upload the Data file.')

# To link to Survey Data Sample Excel file
st.header('Data Sample')
data_sample_url = 'https://docs.google.com/spreadsheets/d/1Aafpl9gu-HYWd-IybL6MasH64w9yzto5/edit?usp=sharing&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'[Data Sample file download]({data_sample_url})')

st.write('Contact: ho.tuong.vinh@gmail.com')
st.write('ATTENTION: I HAVE NO RESPONSIBILITY FOR THE OUTCOME OF THIS ANALYSIS. USE WITH CAUTION!')

def display_semopy_content(sem_model_spec_reduced):
    st.title("SEMopy Specification")
    
    # Splitting the content at each section to print headers separately
    sections = sem_model_spec_reduced.split("###")
    
    for section in sections:
        if section.strip() != '':
            header, content = section.split("\n", 1)
            st.markdown(f"### {header.strip()}")
            st.markdown(f"```\n{content.strip()}\n```")

def perform_descriptive_analysis(data, variable_dicts):

    for variable_dict in variable_dicts:
        variable_type = variable_dict['type']
        st.header(f'{variable_type} Variables')
        
        for variable_info in variable_dict['variables']:
            column_index = variable_info['column_index']
            variable_name = variable_info['Variable']
            column_data = data.iloc[:, column_index]

            if column_data.dtype == 'object':
                # Perform analysis for categorical variables
                frequency_distribution = column_data.value_counts()
                total_count = len(column_data)
                
                st.subheader(f"Descriptive analysis for {variable_name} (Categorical):")
                
                # Create a table with counts and percentages
                data_table = pd.DataFrame({
                    'Category': frequency_distribution.index,
                    'Count': frequency_distribution.values,
                    'Percentage': (frequency_distribution / total_count * 100).round(2)
                })
                st.write(data_table)
            else:
                # Perform analysis for numerical variables
                summary_stats = column_data.describe()
                st.subheader(f"Descriptive analysis for {variable_name} (Numerical):")
                st.write(summary_stats)


# Upload DATA MODEL Excel file
model_file_path = st.sidebar.file_uploader("Upload SEM MODEL Excel file", type=["xlsx", "xls"])
if model_file_path is not None:
    # Read the data into a dictionary
    data_model =pd.ExcelFile(model_file_path)
    # Display sheet by shee
    st.header('=========        Model Spefification      =========')
    sheet_names = data_model.sheet_names
    for sheet_name in sheet_names:
        sheet_df = pd.read_excel(data_model, sheet_name=sheet_name)
        st.subheader(sheet_name)
        st.write(sheet_df)


    # To create Model Spec Graph
    sem_model_spec, sem_model_spec_reduced, \
        demographic_dict, observable_dict, latent_dict, \
        dependent_dict, varcovar_dict, parameters_dict = create_sem_model_spec(model_file_path)
    
    
    st.subheader('Model Specification')
    # display_semopy_content(sem_model_spec)
    sem_model_spec_to_display = reformat_sem_model_to_display(sem_model_spec)
    st.code(sem_model_spec_to_display)

    st.write('Model Specification Graph')
    graph_name = 'model_spec_graph'
    graph_path = create_sem_model_spec_graph(sem_model_spec, observable_dict, latent_dict, dependent_dict, graph_name)
    st.image(graph_path) #, caption='Model Specification Graph')

    # Upload Data
    data_file_path = None
    data_file_path = st.sidebar.file_uploader("Upload DATA Excel file", type=["xlsx", "xls"])
    if data_file_path is not None:
        st.header('\n========== Data ===============')
        data_original = pd.read_excel(data_file_path)

        st.write(data_original)

        # Define your variable_dicts that include demographic, observable, and latent variables
        variable_dicts = [
            {
                'type': 'Demographic',
                'variables': demographic_dict
            },
        ]
        
        # Pre-process data
        data_normalized, label_mappings, demographic_cols_names, \
            observable_cols_names, latent_cols_names, \
            dependent_cols_names = \
            pre_process_data(data_original, demographic_dict, observable_dict, latent_dict, dependent_dict)

        st.subheader('\nData Normalized')
        st.write(data_normalized)

        st.subheader('\nData Normalized Statistics')
        st.write(data_normalized.describe())

        st.header("Testing the reliability of the scale using Cronbach's alpha for Observable Variables")
        # Calculate Cronbach's alpha
        overall_alpha, alpha_table = compute_cronbach_alpha(data_normalized, observable_dict)
        st.write(f"Overall Cronbach's Alpha: {overall_alpha}")
        # Display Alpha for each Independent Variable
        st.write(f"Cronbach's Alpha for each observale variables:")
        for idx, col in enumerate(observable_dict):
            col_name = col.get('Variable')
            st.markdown(f'- **{col_name}**:  {alpha_table[idx]}')


        # EFA Analysis
        efa_results = conduct_efa_analysis(data_normalized, observable_dict)
        st.header(f'Conduct EFA analysis')
        st.write('(factors automatically detected according to eigenvalues)')

        # Create a download button for the Excel file
        filename = 'EFA_Analysis.xlsx'
        st.download_button(
            label="Download EFA Analysis Results Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-efa'
        )

        st.write(efa_results)

        print('\nEFA analysis.   Done')

        # Interpret EFA Results
        threshold_high = 0.6  # Threshold for high loadings
        threshold_moderate = 0.3   # Threshold for low loadings
        efa_interpretation = interpret_efa_results(efa_results, threshold_high, threshold_moderate)
        print(efa_interpretation)
        st.subheader(f"EFA - Simple Interpretation with threshold_high= {threshold_high} and threshold_moderate= {threshold_moderate}")
        # Display the interpretations in Streamlit
        for interpretation in efa_interpretation:
            st.write(interpretation)


        # Compute Correclation Matrix
        correlation_matrix = compute_correlation(data_normalized, observable_dict)
        st.header('Corellation Analsyis By Pearson Method')

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

        print('\Correlation analysis.   Done')

        # Interpretation et recommendation 
        threshold_strong = 0.7  # Threshold for strong correlation
        threshold_moderate = 0.3    # Threshold for weak correlation
        corr_interpretation_df = interpret_correlation(correlation_matrix, threshold_strong,threshold_moderate)
        st.subheader(f"Correlation Interpretations with threshold_strong= {threshold_strong} and threshold_weak= {threshold_moderate}")
        st.write(corr_interpretation_df)

        del correlation_matrix

        # Conduct SEM Analysis
        st.header('\n========== SEM Analysis ===============')

        st.subheader('\nSEM Model Spec according to Semopy Format')
        sem_model_spec_to_display = reformat_sem_model_to_display(sem_model_spec)
        st.code(sem_model_spec_to_display)


        sem_result, sem_stats, sem_inspect, sem_inspect_enhanced, \
        sem_inspect_filtered, graph_filtered_results, \
        graph_fulll_results = conduct_sem_analysis(data_normalized, sem_model_spec, observable_dict, latent_dict, dependent_dict)
       

        st.subheader('\nSEM Stats')
        st.code(sem_result)
        # st.table(sem_stats)
        stats_interpret_df, overall_msg =  interpret_sem_stats(sem_stats, parameters_dict)
        st.subheader(overall_msg)
        st.write(stats_interpret_df)

        # Create a download button for the Excel file
        filename = 'SEM_Model_Stats.xlsx'
        st.download_button(
            label="Download SEM Model Statistics Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-sem_stats'
        )

        st.subheader('SEM Results')
        st.write(sem_inspect_enhanced)

        # Create a download button for the Excel file
        filename = 'SEM_Results.xlsx'
        st.download_button(
            label="Download SEM Results Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-sem_results'
        )

        st.subheader('SEM Results Full Graph')
        st.image(graph_fulll_results)
        
        st.subheader('\nInterepretations')
        
        interepretation_sem_inspect = interepret_sem_inspect(sem_inspect_enhanced, dependent_dict, parameters_dict)
        for interpretation in interepretation_sem_inspect:
            st.markdown(interpretation)
            st.write("---")  # Adds a horizontal line for separation
        

        # CONDUCT SEM with Moderator variables (Demographic variables)

        # Check if Multi Analyis with Demographic variables as Moderators is required
        is_analysis_with_moderator_required, moderators = check_if_analysis_with_moderator_required(demographic_dict)
        if is_analysis_with_moderator_required:
            # Base line SEM Inspect Results
            inspect_baseline = filter_inspect_table_from_spec(sem_inspect, sem_model_spec)

            st.header('\n\nSubgroup Analysis using Demographic variables (as Moderators) and their values.')
            st.write('Extract subdata correspoding to each value of Moderators, and Apply the same SEM Analysis.')
            st.write('The Results with all groups (without the demographic data) are used as "baseline".')
            
            st.subheader('List of moderators')
            st.write(moderators)

            sem_results_full = conduct_sem_with_moderators(sem_model_spec, data_normalized, label_mappings, moderators)
    
            post_process_results = post_process_sem_with_moderator_resuls(sem_results_full, inspect_baseline, moderators)
            for i, category_df in enumerate(post_process_results):
                st.subheader(f'{category_df[0]}')
                st.write(f'\n------')
                st.write(category_df[1])
                st.write(category_df[2])


        st.header('\n\n ================   The END  =================')

        st.subheader('Try:')
        st.header('Data Analysis with SEM')
        st.write('https://sem-analysis.streamlit.app')


        st.write('\n\n\n\n\n==============================================================================\n')
        st.write('And more ... Only 1 minute to convert Youtube video to slides!')
        st.write('https://htvinh-youtube2slides-streamlit-app-k14x3w.streamlitapp.com/')
        st.write('\n')

        st.write('See you next time ! == ho.tuong.vinh@gmail.com ==')
