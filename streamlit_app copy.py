from data_analysis import *

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

st.title('SURVEY DATA ANALYSIS')

st.write('Download these 2 sample files to prepare your files.')
st.write('Need only to adapt the Data Model according to your survey structure.')
st.write('For the survey data, only need to download as Excel from Google Forms.')

# To link to Data Model Sample Excel file
data_model_sample_url = 'https://docs.google.com/spreadsheets/d/19ymgAkEUgvux6z7ykWHnlmtfwu0h3iMa/edit?usp=drive_link&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'[Data Model Sample file download]({data_model_sample_url})')

# To link to Survey Data Sample Excel file
data_sample_url = 'https://docs.google.com/spreadsheets/d/1A-hKivtLFUJeOpLolfKy5Pd_rUDfsPY1/edit?usp=sharing&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'[Survey Data Sample file download]({data_sample_url})')
data_sample_url = 'https://docs.google.com/spreadsheets/d/1AATsrch7RkOoD-IEhEm-wBmb1n6kOsZd/edit?usp=sharing&ouid=103775647130982487748&rtpof=true&sd=true'
st.write(f'[Survey Data Sample (in English) file download]({data_sample_url})')


st.write('Contact: ho.tuong.vinh@gmail.com')
st.write('ATTENTION: I HAVE NO RESPONSIBILITY FOR THE OUTCOME OF THIS ANALYSIS. USE WITH CAUTION!')


is_visualized = True # True False
is_sem_analysis = False # True False

def visualize_stats_table(stats_table):
    for col_name, stats_df in stats_table.items():
        st.write(f'### Percentage Distribution (%) for {col_name}')
        fig, ax = plt.subplots(figsize=(3, 2))
        sns.barplot(x=stats_df.index, y=stats_df['Percent'], ax=ax, )

        # Set the fontsize for title, labels, and ticks
        title_fontsize = 7
        label_fontsize = 6
        ticks_fontsize = 5

        # Annotate each bar with the respective frequency value
        for i, v in enumerate(stats_df['Percent']):
            ax.text(i, v/2, str(v), ha='center', va='center', fontsize=ticks_fontsize)
        
        plt.title(f'Percentage Distribution for {col_name}', fontsize=title_fontsize)
        plt.xlabel(col_name, fontsize=label_fontsize)
        plt.ylabel('Percentage (%)', fontsize=label_fontsize)
        
        # Set the fontsize for x-ticks and y-ticks
        plt.xticks(rotation=30, ha='right', fontsize=ticks_fontsize)
        plt.yticks(fontsize=ticks_fontsize)

        st.pyplot(fig)

# Upload DATA MODEL Excel file
data_model_name = st.sidebar.file_uploader("Upload DATA MODEL Excel file", type=["xlsx", "xls"])
if data_model_name is not None:
    # Read the data into a dictionary
    data_model = pd.read_excel(data_model_name)
    # Display
    st.header('Data Model')
    # st.dataframe(data_model)
    
    demographic_cols = pd.read_excel(data_model_name, sheet_name='Demographic', header=0)
    origin_demographic_cols = demographic_cols.copy() # To display only
    # Convert column index to 0-Python based, supposing that the user begin the column index from 0
    demographic_cols['col_index'] = demographic_cols['col_index']-1
    demographic_cols = demographic_cols.to_dict(orient='records')

    independent_cols = pd.read_excel(data_model_name, sheet_name='Independent')
    origin_independent_cols = independent_cols.copy() # To display only
    independent_cols['col_index_from'] = independent_cols['col_index_from']-1
    independent_cols = independent_cols.to_dict(orient='records')

    target_cols = pd.read_excel(data_model_name, sheet_name='Dependent')
    origin_target_cols = target_cols.copy() # To display only
    target_cols['col_index_from'] = target_cols['col_index_from']-1
    target_cols = target_cols.to_dict(orient='records')
    
    st.write('Demographic Variables/Columns')
    st.dataframe(origin_demographic_cols)
    print('Demographic Variables/Columns')
    print(origin_demographic_cols)

    st.write('Independent Variables/Columns')
    st.dataframe(origin_independent_cols)
    print('Independent Variables/Columns')
    print(origin_independent_cols)

    st.write('Dependent/Target Variables/Columns')
    st.dataframe(origin_target_cols)
    print('Dependent/Target Variables/Columns')
    print(origin_target_cols)

    data_file_name = st.sidebar.file_uploader("Upload SURVEY DATA Excel file", type=["xlsx", "xls"])
    if data_file_name is not None:
        # Read the data into a dictionary
        data = pd.read_excel(data_file_name)
        st.header('Survey data')
        st.write(data)

        data = pd.DataFrame(data)

        # Get the number of columns and rows. nbr_rows = data points
        nbr_columns = data.shape[1]
        nbr_rows = data.shape[0]
        total_num_data_points = nbr_rows
        st.write(f"Number of rows: {nbr_rows} and columns: {nbr_columns}")

        # Rename columns for demographic data columns: to have shorter column names
        data, demographic_cols_names = rename_demographic_columns_by_index(data, demographic_cols)
        # print('\nDemographic Column Names')
        # print(demographic_cols_names)
        data_points_removed_1 = total_num_data_points - data.shape[0]

        # Rename Independent columns according to Independent_Cols
        total_num_data_points = data.shape[0]
        data, independent_cols_names = rename_independent_columns(data, independent_cols)
        data_points_removed = total_num_data_points - data.shape[0]
        # print('\nIndependent Column Names')
        # print(independent_cols_names)


        # Rename Dependent/Target columns according to Dependent_Cols
        total_num_data_points = data.shape[0]
        data, target_cols_names = rename_target_columns(data, target_cols)
        data_points_removed = total_num_data_points - data.shape[0]
        # print('\nTarget Column Names')
        # print(target_cols_names)

        # print(data.columns)

        # Remove data points (rows) with missing data related to demographic data columns
        total_num_data_points = data.shape[0]
        data = remove_rows_with_missing_data_related_to_selected_cols(data, demographic_cols_names)
        data_points_removed_1 = total_num_data_points - data.shape[0]

        # Remove data points (rows) with missing data related to independent data columns
        total_num_data_points = data.shape[0]
        data = remove_rows_with_missing_data_related_to_selected_cols(data, independent_cols_names)
        data_points_removed_2 = total_num_data_points - data.shape[0]

        # Remove data points (rows) with missing data related to Target data columns
        total_num_data_points = data.shape[0]
        data = remove_rows_with_missing_data_related_to_selected_cols(data, target_cols_names)
        data_points_removed_3 = total_num_data_points - data.shape[0]

        total_removed = data_points_removed_3 + data_points_removed_2 + data_points_removed_1
        st.write(f'Number of data points removed because of missing data:   {total_removed}')

        # To check if the mapping between Data Model and Survey Data is correct (in terms of Column Index)
        for index, column_name in enumerate(data.columns):
            print(f"Column {index}: {column_name}")

        # Create a statistics table for Demographic Columns
        selected_cols_names = demographic_cols_names
        output_file_name = 'Demographic'
        demographic_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        
        st.header('Statistics of Demographic Columns')
        # Create a download button for the Excel file
        filename = 'Demographic_Stats.xlsx'
        st.download_button(
            label="Download Statistics of Demographic Columns Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-dem'
        )
        # st.write(demographic_stats_table)
        with st.expander("To display"):
            st.write(demographic_stats_table)
        # Visualize
        if is_visualized is True:
            visualize_stats_table(demographic_stats_table)

        print('\nCreate a statistics table for Demographic Columns!  Done')

        # Create statistic table for independent_cols
        selected_cols_names = independent_cols_names
        output_file_name = 'Independent'
        independent_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        st.header('Statistics of Independent Columns')

        # Create a download button for the Excel file
        filename = 'Independent_Stats.xlsx'
        st.download_button(
            label="Download Statistics of Independent Columns Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-ind'
        )
        with st.expander("To display"):
            st.write(independent_stats_table)
        # st.write(independent_stats_table)
        # Visualize
        if is_visualized is True:
            visualize_stats_table(independent_stats_table)
        print('\nCreate a statistics table for Independent Columns!  Done')

        # Create statistic table for target_cols
        selected_cols_names = target_cols_names
        output_file_name = 'Target'
        target_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        st.header('Statistics of Target Columns')

        # Create a download button for the Excel file
        filename = 'Target_Stats.xlsx'
        st.download_button(
            label="Download Statistics of Target Columns Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-taget'
        )
        with st.expander("To display"):
            st.write(target_stats_table)
        # st.write(target_stats_table)
        if is_visualized is True:
            visualize_stats_table(target_stats_table)
        print('\nCreate a statistics table for Target Columns!  Done')

        # Extract data for Indendent Columns
        selected_cols_names = independent_cols_names
        independent_data = extract_selected_colums_data(data, selected_cols_names)
        # st.write(independent_data)
        print('\nExtract Independent Data.   Done\n')

        # Extract data for Dependent Columns
        selected_cols_names = target_cols_names
        # st.write(selected_cols_names)
        target_data = extract_selected_colums_data(data, selected_cols_names)
        # st.write(target_data)

        st.header("Testing the reliability of the scale using Cronbach's alpha for independent columns")
        # Calculate Cronbach's alpha
        overall_alpha, alpha_table = compute_cronbach_alpha(independent_data, independent_cols)
        # Print the result
        print(alpha_table)
        st.write(f"Overall Cronbach's Alpha: {overall_alpha}")
        # Display Alpha for each Independent Variable
        st.write(f"Cronbach's Alpha for each independent variables:")
        for idx, col in enumerate(independent_cols):
            col_name = col.get('name')
            st.markdown(f'- **{col_name}**:  {alpha_table[idx]}')


        # EFA Analysis
        selected_data = independent_data
        num_factors = 5
        efa_results = do_efa_analysis(selected_data, num_factors)
        st.header('EFA analysis')

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
        # loadings_df = do_efa_analysis(selected_data, num_factors)
        efa_interpretation_df = interpret_based_on_loadings(efa_results)
        print(efa_interpretation_df)
        st.write("### Simple Interpretation")
        for index, row in efa_interpretation_df.iterrows():
            st.write(f"**{index+1}. {row['Factor']}**: {row['Interpretation']}")

        # Recommendation
        recommendation_df = recommendations_based_on_loadings(efa_results, threshold=0.5)
        st.write("### Recommendations")
        for index, row in recommendation_df.iterrows():
            st.write(f"**{index+1}. {row['Factor']}**: {row['Recommendation']}")

        # Compute Correclation Matrix
        correlation_table = compute_correlation(selected_data)
        st.header('Corellation Analsyis By Pearson Method')

        # Create a download button for the Excel file
        filename = 'Correlation_Table.xlsx'
        st.download_button(
            label="Download Correlation Matrix Excel File",
            file_name = filename,
            data=open(f'./output/{filename}', 'rb').read(),
            key='excel-download-button-cor'
        )
        st.write(correlation_table)
        print('\Correlation analysis.   Done')

        # Interpretation et recommendation 
        corr_interpretation_df, corr_recommendation_df = interpret_and_recommend_correlation(correlation_table, threshold=0.5)
        st.write("### Correlation Interpretations")
        for index, row in corr_interpretation_df.iterrows():
            st.write(f"**{index+1}. {row['Variables']} (Correlation: {row['Correlation']:.2f})**: {row['Interpretation']}")

        st.write("### Recommendations Based on Correlations")
        for index, row in corr_recommendation_df.iterrows():
            st.write(f"**{index+1}. {row['Variables']}**: {row['Recommendation']}")

        # Make Multivarate Regression Analysis
        st.header('Multivariate Regression Analysis')
        for idx, target_col in enumerate(target_cols_names):
            st.write('\n========================================')
            st.write(f'Regression Analysis for Target {target_cols_names[idx]}')
            target_data = extract_selected_colums_data(data, [target_cols_names[idx]])
            variable_data = independent_data
            output_filename = f'Multivariate_Regression_Summary_{idx+1}.docx'
            regression_results = do_multivariate_regression_analysis_with_OLS(target_data, variable_data, output_filename)
            summary_result = regression_results.summary() 

            # Create a download button for the Excel file
            filename = output_filename
            st.download_button(
                label="Download Multivariate Regression Analysis Report Docx File",
                file_name = filename,
                data=open(f'./output/{filename}', 'rb').read(),
                key=f'excel-download-button-reg-{idx+1}'
            )
            st.write(summary_result)

            # Interpretation and recommendations
            reg_interpretation, reg_recommendation = interpret_and_recommend_regression_with_OLS(regression_results)
            # Display in Streamlit
            st.write("### Regression Interpretations")
            st.write(reg_interpretation)

            # Display recommendations as a list for better readability
            # st.write("### Detailed Recommendations Based on Regression")
            # recommendations_list = reg_recommendation.split('. ')
            #for i, rec in enumerate(recommendations_list):
            #    if rec:  # Check if the recommendation string is not empty
            #        st.write(f"{i+1}. {rec}.")

        # Testing hypothesis: if a factor/independent variable has effect on the dependent variable
        st.header('Testing if a factor/independent variable has effect on the dependent variable.')
        st.write('The F-test (Linear Regression) helps you determine if the factor you are studying has a significant impact on the dependent variable you are measuring.')
        for idx, target_col in enumerate(target_cols_names):
            st.write('\n========================================')
            st.write(f'Testing Effect hypothesis for Target {target_cols_names[idx]}')
            target_data = extract_selected_colums_data(data, [target_cols_names[idx]])
            variable_data = independent_data
            testing_results = test_if_factor_has_effect_on_target(target_data, variable_data, independent_cols)
            st.write('\nTesting Results:\n', testing_results)


        # CONDUCT SEM analysis
        st.header('Conduct SEM (Structural Equation Modeling)')
        target_data = extract_selected_colums_data(data, selected_cols_names)
        sem_results = conduct_sem_analysis(independent_cols, target_col, independent_data, target_data)
        st.write('SEM Results')
        st.write(sem_results)
        st.write('\n\nInterpretation')
        sem_results = interpret_sem_results(sem_results)
        st.write(sem_results)




    st.write('\n\n\n\n\n==============================================================================\n')
    st.write('And more ... Only 1 minute to convert Youtube video to slides!')
    st.write('https://htvinh-youtube2slides-streamlit-app-k14x3w.streamlitapp.com/')
    st.write('\n')

    st.write('See you next time ! == ho.tuong.vinh@gmail.com ==')



