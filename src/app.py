import sys
import os
import pandas as pd
import streamlit as st
from PIL import Image
from typing import List, Dict, Any

import src.regression as reg
import src.sem as sem
from src.quality import (
    compute_cronbach_alpha, compute_correlation, interpret_correlation,
    compute_cr_ave, run_harman_single_factor_test, check_discriminant_validity
)
from src.common import read_model, pre_process_data
from src.helpers import logger, get_output_path
from src.reporting import generate_markdown_report, convert_markdown_to_docx, get_markdown_download_link, get_docx_download_link

Image.MAX_IMAGE_PIXELS = None

def display_model_spec(model_spec: str):
    sections = model_spec.split("###")
    for section in sections:
        if section.strip():
            lines = section.strip().split("\n", 1)
            st.markdown(f"### {lines[0].strip()}")
            if len(lines) > 1:
                st.code(lines[1].strip())

def perform_descriptive_analysis(data: pd.DataFrame, demographic_dict: List[Dict]):
    st.header('Descriptive Analysis')
    for var_info in demographic_dict:
        var_name = var_info.get('Variable')
        idx = var_info.get('column_index')
        if idx is not None and 0 <= int(idx) < len(data.columns):
            col_data = data.iloc[:, int(idx)]
            st.subheader(f"Variable: {var_name}")
            if col_data.dtype == 'object' or col_data.nunique() < 10:
                summary_df = pd.DataFrame({'Count': col_data.value_counts(), 'Percentage (%)': (col_data.value_counts(normalize=True) * 100).round(2)})
                st.write(summary_df)
            else:
                st.write(col_data.describe())

def cleanup_output():
    output_path = get_output_path()
    if os.path.exists(output_path):
        for f in os.listdir(output_path):
            file_path = os.path.join(output_path, f)
            if os.path.isfile(file_path):
                os.remove(file_path)

def main():
    st.set_page_config(page_title="SDA: Survey Data Analysis", layout="wide")
    st.title("SDA: Survey Data Analysis")

    model_file = st.file_uploader("Upload MODEL Excel file", type=["xlsx", "xls"])
    data_file = st.file_uploader("Upload DATA Excel file", type=["xlsx", "xls"])

    if model_file and data_file:
        if st.button("Run Full Analysis", type="primary"):
            try:
                output_path = get_output_path()
                dem_dict, indep_dict, med_dict, dep_dict, rel_dict, _, param_dict = read_model(model_file)
                data_raw = pd.read_excel(data_file)
                data_norm, label_map, _, _, _ = pre_process_data(data_raw, dem_dict, indep_dict, dep_dict)

                perform_descriptive_analysis(data_raw, dem_dict)
                
                # SEM Analysis for Metrics
                sem_m_spec, _ = sem.create_model_spec(indep_dict, med_dict, dep_dict, rel_dict, [])
                _, sem_stats, _, sem_enh, _, g_short, g_full = sem.conduct_sem_analysis(data_norm, sem_m_spec, indep_dict, med_dict, dep_dict)
                
                all_constructs = {d['Variable'] for d in indep_dict} | {d['Variable'] for d in dep_dict}
                loadings_df = sem_enh[sem_enh['rval'].isin(all_constructs)]
                
                construct_metrics = {}
                for construct in all_constructs:
                    c_loadings = pd.to_numeric(loadings_df[loadings_df['rval'] == construct]['Est. Std'], errors='coerce').dropna().tolist()
                    cr, ave = compute_cr_ave(c_loadings)
                    construct_metrics[construct] = {'CR': cr, 'AVE': ave}

                overall_a, a_table = compute_cronbach_alpha(data_norm, indep_dict)
                
                # Validity
                ave_data = [[c, m['AVE'], "Adequate" if m['AVE'] > 0.5 else "Low"] for c, m in construct_metrics.items()]
                
                # MGA
                moderators = [v['Variable'] for v in dem_dict if str(v.get('used_as_moderator')).lower() == 'yes']
                mga_markdown = ""
                if moderators:
                    st.header("Step 11: Multi-Group Moderation Analysis")
                    mga_markdown = reg.generate_dynamic_multi_group_analysis(data_norm, pd.DataFrame(rel_dict), pd.DataFrame(dem_dict))
                    st.markdown(mga_markdown)

                # Report
                report_md = generate_markdown_report(
                    demographic_dict=dem_dict, data_raw=data_raw, overall_alpha=overall_a,
                    reliability_df=pd.DataFrame(), validity_ave_df=pd.DataFrame(ave_data),
                    discriminant_validity_results=[], harman_msg="", correlation_matrix=pd.DataFrame(),
                    corr_interpretations=[], recommended_drops=[], heatmap_path="", reg_m_spec="",
                    reg_interp=[], reg_summ_tables=[], reg_vif_tables=[], reg_result_graph_path="",
                    std_reg_interpretations=[], std_reg_result_graph_path="", mga_markdown=mga_markdown,
                    sem_model_spec=sem_m_spec, sem_stats_table=sem_stats, sem_overall_msg="",
                    sem_interps=[], sem_graph_full_path=g_full, sem_graph_short_path=g_short
                )
                
                st.markdown(get_markdown_download_link(report_md), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                cleanup_output()

if __name__ == "__main__":
    main()
