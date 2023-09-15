from data_analysis import *

import streamlit as st

st.title('SURVEY DATA ANALYSIS')

# Upload DATA MODEL Excel file
data_model_name = st.sidebar.file_uploader("Upload DATA MODEL Excel file", type=["xlsx", "xls"])
if data_model_name is not None:
    # Read the data into a dictionary
    data_model = pd.read_excel(data_model_name)
    # Display
    st.header('Data Model')
    # st.dataframe(data_model)
    
    demographic_cols = pd.read_excel(data_model_name, sheet_name='Demographic', header=0)
    demographic_cols = demographic_cols.to_dict(orient='records')
    independent_cols = pd.read_excel(data_model_name, sheet_name='Independent')
    independent_cols = independent_cols.to_dict(orient='records')
    target_cols = pd.read_excel(data_model_name, sheet_name='Dependent')
    target_cols = target_cols.to_dict(orient='records')
    
    st.write('Demographic Variables/Columns')
    st.dataframe(demographic_cols)
    print('Demographic Variables/Columns')
    print(demographic_cols)

    st.write('Independent Variables/Columns')
    st.dataframe(independent_cols)
    print('Independent Variables/Columns')
    print(independent_cols)

    st.write('Dependent/Target Variables/Columns')
    st.dataframe(target_cols)
    print('Dependent/Target Variables/Columns')
    print(target_cols)

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
        # st.header('Survey Data')
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

        # Create a statistics table for Demographic Columns
        selected_cols_names = demographic_cols_names
        output_file_name = 'Demographic'
        demographic_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        
        st.header('Statistics of Demographic Columns')
        st.write(demographic_stats_table)
        print('\nCreate a statistics table for Demographic Columns!  Done')

        # Create statistic table for independent_cols
        selected_cols_names = independent_cols_names
        output_file_name = 'Independent'
        independent_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        st.header('Statistics of Independent Columns')
        st.write(independent_stats_table)
        print('\nCreate a statistics table for Independent Columns!  Done')

        # Create statistic table for target_cols
        selected_cols_names = target_cols_names
        output_file_name = 'Target'
        target_stats_table = compute_selected_cols_statistics(data, selected_cols_names, output_file_name)
        st.header('Statistics of Target Columns')
        st.write(target_stats_table)
        print('\nCreate a statistics table for Target Columns!  Done')

        # Extract data for Indendent Columns
        selected_cols_names = independent_cols_names
        independent_data = extract_selected_colums_data(data, selected_cols_names)
        print(independent_data)
        print('\nExtract Independent Data.   Done\n')

        st.header("Testing the reliability of the scale using Cronbach's alpha for independent columns")
        # Calculate Cronbach's alpha
        alpha = compute_cronbach_alpha(independent_data)
        # Print the result
        st.write(f"Overall Cronbach's Alpha: {alpha}")

        # EFA Analysis
        selected_data = independent_data
        num_factors = 5
        efa_results = do_efa_analysis(selected_data, num_factors)
        st.header('EFA analysis')
        st.write(efa_results)
        print('\nEFA analysis.   Done')

        # Compute Correclation Matrix
        correlation_table = compute_correlation(selected_data)
        st.header('Corellation Analsyis By Pearson Method')
        st.write(correlation_table)
        print('\Correlation analysis.   Done')

        # Make Multivarate Regression Analysis
        st.header('Multivarate Regression Analysis')
        target_data = extract_selected_colums_data(data, [target_cols_names[0]])
        variable_data = independent_data
        regression_results = do_multivariate_regression_analysis(target_data, variable_data)
        st.write(regression_results)





