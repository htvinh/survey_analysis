import sys
import os

# Add the project root to sys.path to allow absolute imports from 'src'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import streamlit as st
from PIL import Image
from io import BytesIO
from typing import List, Dict, Any, Optional

import src.regression as reg
import src.sem as sem
from src.quality import (
    compute_cronbach_alpha, compute_correlation, interpret_correlation,
    compute_cr_ave, run_harman_single_factor_test, check_discriminant_validity
)
from src.common import read_model, pre_process_data
from src.helpers import logger, get_output_path
from src.reporting import generate_markdown_report, convert_markdown_to_docx, get_markdown_download_link, get_docx_download_link

# Global Configuration
Image.MAX_IMAGE_PIXELS = None

def display_model_spec(model_spec: str):
    """Displays the model specification in a clean format using Streamlit."""
    sections = model_spec.split("###")
    for section in sections:
        if section.strip():
            lines = section.strip().split("\n", 1)
            header = lines[0]
            content = lines[1] if len(lines) > 1 else ""
            st.markdown(f"### {header.strip()}")
            if content:
                st.code(content.strip())

def perform_descriptive_analysis(data: pd.DataFrame, demographic_dict: List[Dict]):
    """Generates and displays descriptive statistics for demographic variables."""
    st.header('Descriptive Analysis')
    for var_info in demographic_dict:
        var_name = var_info.get('Variable')
        idx = var_info.get('column_index')
        
        if idx is not None and 0 <= int(idx) < len(data.columns):
            col_data = data.iloc[:, int(idx)]
            st.subheader(f"Variable: {var_name}")
            
            if col_data.dtype == 'object' or col_data.nunique() < 10:
                counts = col_data.value_counts()
                percent = (col_data.value_counts(normalize=True) * 100).round(2)
                summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percent})
                st.write(summary_df)
            else:
                st.write(col_data.describe())

def cleanup_output():
    """Removes the output directory and its contents."""
    output_path = get_output_path()
    if os.path.exists(output_path):
        for f in os.listdir(output_path):
            file_path = os.path.join(output_path, f)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="SDA: Survey Data Analysis", layout="wide")

    # Main Content
    st.title("SDA: Survey Data Analysis")
    st.markdown("""
    Analyze survey data using **Regression** and **Structural Equation Modeling (SEM). Publication-Grade Analysis**.
    """)

    with st.expander("Analysis Workflow Summary"):
        workflow_table = """
| Step | Phase | Description |
|------|-------|-------------|
| 1 | **Data Prep** | Cleaning, Missing Data & Pre-processing |
| 2 | **Descriptive** | Sample Profile & Distribution Checks |
| 3 | **Reliability** | Cronbach's Alpha & Composite Reliability (CR) |
| 4 | **Validity** | AVE & Discriminant Validity (Fornell-Larcker) |
| 5 | **Bias Check** | Common Method Bias (Harman's Single-Factor Test) |
| 6 | **Correlation** | Item-level & Construct-level Analysis |
| 7 | **OLS Diagnostics**| Multi-collinearity (VIF, Tolerance) & Residuals |
| 8 | **OLS Results** | Path Estimates (Standardized β) & R² |
| 9 | **CFA** | Measurement Model: Factor Loadings & Fit Indices |
| 10| **CB-SEM** | Structural Model: Direct & Indirect latent effects |
| 11| **Subgroup** | Multi-Group Moderation Analysis |
| 12| **Final Report** | Comprehensive Markdown & DOCX Documentation |
"""
        st.markdown(workflow_table)
        st.info("**Methodological Note:** Hypothesized relationships are first examined using OLS regression for interpretable diagnostics, subsequently validated using CB-SEM to account for measurement error.")

    with st.expander("Model and Data Preparation"):
        st.markdown("""
        ### 1. Conceptual Framework: The OLS-CFA-SEM Defensive Sequence
        To satisfy diverse reviewer perspectives (traditional vs. SEM-oriented), this application follows a defensible sequence:
        - **OLS Path Modeling:** Hypothesized relationships are first examined using OLS regression on composite scores to provide robust diagnostic statistics (VIF, R², residuals).
        - **Confirmatory Factor Analysis (CFA):** Validates the measurement model by assessing factor loadings and construct reliability/validity before structural estimation.
        - **Covariance-Based SEM (CB-SEM):** Validates findings by simultaneously estimating all paths while accounting for measurement error across latent constructs.

        ### 2. Data Preparation Standards
        - **Consistency:** Use consistent scales (e.g., 5-point Likert) across indicators to ensure interpretability of effect sizes.
        - **Standard 5-Point Mapping:** The pipeline expects numerical data. Standard anchors are:
          | Value | English Anchor | Vietnamese Anchor |
          | :--- | :--- | :--- |
          | **1** | Strongly Disagree | Rất không đồng ý |
          | **2** | Disagree | Không đồng ý |
          | **3** | Neutral | Trung lập / Bình thường |
          | **4** | Agree | Đồng ý |
          | **5** | Strongly Agree | Rất đồng ý |
        - **Structure:** Indicator items must be in **adjacent, continuous columns**. The Model Excel file uses **1-based indexing** (A=1).
        - **Quality:** All response items must be **numerical**. Categorical demographics are automatically encoded.

        ### 3. Automated Validation Pipeline
        1. **Reliability:** Calculates **Cronbach's Alpha** and **Composite Reliability (CR)**. Both should ideally exceed 0.70.
        2. **Validity:** Evaluates **Convergent Validity (AVE > 0.50)** and **Discriminant Validity** using the Fornell-Larcker criterion.
        3. **Diagnostics:** Checks for **Common Method Bias** via Harman's Test and **Multi-collinearity** via VIF (< 5 or < 10).
        """)

    with st.expander("Advanced Interpretation Guide"):
        st.markdown("""
        ### 1. Reliability Assessment (Alpha vs. CR)
        - **Cronbach's Alpha:** Assumes all items contribute equally (tau-equivalence).
        - **Composite Reliability (CR):** Preferred in SEM as it accounts for varying factor loadings.
        - **Threshold:** > 0.70 is standard; > 0.60 is acceptable for exploratory research.

        ### 2. Validity (AVE & Discriminant)
        - **Convergent Validity (AVE):** Measures the variance captured by a construct vs. measurement error. Goal: **AVE > 0.50**.
        - **Discriminant Validity (Fornell-Larcker):** Ensures constructs are truly distinct. The **Sqrt(AVE)** of a construct should be greater than its correlations with any other construct.

        ### 3. Common Method Bias (Harman's Test)
        - If a single factor explains **> 50%** of total variance, your results may be biased by the survey instrument itself rather than true relationships.

        ### 4. The 'Significance vs. Fit' Paradox
        - Significant paths (p < 0.05) with poor model fit (RMSEA > 0.08) often indicate **Indicator Inflation**.
        - **Fix:** Prune items with high redundancy (r > 0.80) to improve model parsimony and lower standard errors.

        ### 5. Regression Diagnostics (VIF)
        - **VIF > 5.0:** Indicates high multi-collinearity, which can make path estimates unstable. Consider consolidating highly correlated independent variables.
        """)

    with st.expander("🏛️ Model Configuration Blueprint"):
        st.markdown("""
        `model_sample_1.xlsx` serves as the **central configuration blueprint (metadata schema)** for your entire data analysis pipeline. It externalizes the model rules, allowing you to scale your research without changing the underlying code.

        ### 1. Sheet Breakdown
        - **Demographic_Variables**: Maps background traits (Gender, Age). The `used_as_moderator` flag designates variables for subgroup/multi-group analysis.
        - **Independent_Variables**: Indexes exogenous latent constructs (drivers). It specifies the start column and number of questions (indicators) for each construct.
        - **Mediator_Variables**: (Current Status: Optional) Defines intermediate variables for automated mediation and indirect effect analysis.
        - **Dependent_Variables**: Indexes endogenous constructs (outcomes) like `Behavioral_Intention` or `Actual_Use`.
        - **Relations**: The core SEM engine. It defines the structural path equations (e.g., `BI ~ PE + EE + SI`).
        - **Parameters**: The quality control center. It defines mathematical benchmarks for validation.

        ### 2. Global Quality Control Benchmarks
        | Parameter | Threshold (High/Mod) | Contextual Meaning |
        | --- | --- | --- |
        | `efa_threshold` | 0.60 / 0.40 | **Factor Loadings:** Variance shared with assigned construct. |
        | `correlation_threshold` | 0.70 / 0.40 | **Collinearity Gate:** High values flag extreme redundancy. |
        | `cfi_value_threshold` | 0.90 / 0.80 | **Model Fit Adequacy:** Below 0.80 indicates structural errors. |
        | `rmsea_threshold` | 0.05 / 0.08 | **Error Margin:** Ideally < 0.05. > 0.08 signals poor fit. |
        | `r_squared` | 0.80 / 0.65 | **Variance Explained:** Measures predictive power. |

        ### 3. Scaling Recommendation
        - **To add a new question:** Simply update the `number_questions` cell in the Excel file. The system will automatically adjust the mapping.
        - **To add a moderator:** Set `used_as_moderator` to `yes` in the Demographic sheet to trigger automated subgroup regressions.
        """)

    # Sidebar for Info and Links
    with st.sidebar:
        if os.path.exists("sda_logo.png"):
            st.image("sda_logo.png", width='stretch')
        st.header("Resources")
        st.markdown("[Model Sample File](https://github.com/htvinh/dataset/blob/main/survey_analysis_samples/model_sample_1.xlsx)")
        st.markdown("[Survey Data Sample](https://github.com/htvinh/dataset/blob/main/survey_analysis_samples/survey_data_sample_1.xlsx)")
        st.divider()
        st.info("Contact: ho.tuong.vinh@gmail.com")
        st.warning("Disclaimer: Use results with caution. No liability for analysis outcomes.")

    # Step 1: File Upload
    st.header('Step 1: Upload Model and Data')
    col1, col2 = st.columns(2)
    with col1:
        model_file = st.file_uploader("Upload MODEL Excel file", type=["xlsx", "xls"])
    with col2:
        data_file = st.file_uploader("Upload DATA Excel file", type=["xlsx", "xls"])

    if model_file and data_file:
        if st.button("Run Full Analysis", type="primary"):
            try:
                output_path = get_output_path()
                st.success("Analysis started...")

                # 1. Read Model
                dem_dict, indep_dict, med_dict, dep_dict, rel_dict, _, param_dict = read_model(model_file)
                
                # 2. Load and Pre-process Data (Step 1: Data Prep)
                data_raw = pd.read_excel(data_file)
                data_norm, label_map, dem_cols, indep_cols, dep_cols = pre_process_data(
                    data_raw, dem_dict, indep_dict, dep_dict
                )

                # 3. Descriptive Analysis (Step 2: Descriptive)
                perform_descriptive_analysis(data_raw, dem_dict)
                st.divider()

                # --- BACKGROUND SEM CALCULATION FOR METRICS ---
                # We need loadings for CR/AVE which come from Step 9 (CFA)
                sem_m_spec, _ = sem.create_model_spec(indep_dict, med_dict, dep_dict, rel_dict, [])
                _, sem_stats, _, sem_enh, sem_filt, g_short, g_full = sem.conduct_sem_analysis(
                    data_norm, sem_m_spec, indep_dict, med_dict, dep_dict
                )
                
                # Extract factor loadings. Based on logs, indicators (e.g., Var_Q1) are lval, and construct is rval.
                # All measurement relations have the construct name in 'rval'.
                # Filter for rows where rval is one of our constructs.
                all_constructs = {d['Variable'] for d in indep_dict} | {d['Variable'] for d in dep_dict}
                loadings_df = sem_enh[sem_enh['rval'].isin(all_constructs)]
                
                construct_metrics = {}
                for construct in all_constructs:
                    # Convert to numeric, coerce errors to NaN and drop them
                    c_loadings = pd.to_numeric(loadings_df[loadings_df['rval'] == construct]['Est. Std'], errors='coerce').dropna().tolist()
                    cr, ave = compute_cr_ave(c_loadings)
                    construct_metrics[construct] = {'CR': cr, 'AVE': ave}

                # 4. Reliability Analysis (Step 3: Reliability)
                st.header("Step 3: Reliability Analysis")
                overall_a, a_table = compute_cronbach_alpha(data_norm, indep_dict)
                st.write(f"**Overall Cronbach's Alpha:** {overall_a}")
                
                rel_data = []
                for idx, col in enumerate(indep_dict):
                    name = col.get('Variable')
                    cr_val = construct_metrics.get(name, {}).get('CR', 0.0)
                    rel_data.append([name, a_table[idx][0], f"{cr_val:.3f}", a_table[idx][1]])
                
                rel_df = pd.DataFrame(rel_data, columns=['Construct', 'Alpha', 'CR', 'Evaluation'])
                st.table(rel_df)
                st.divider()

                # 5. Validity Assessment (Step 4: Validity)
                st.header("Step 4: Validity Assessment")
                st.subheader("Convergent Validity (AVE)")
                ave_data = [[c, f"{m['AVE']:.3f}", "Adequate (>0.5)" if m['AVE'] > 0.5 else "Low"] 
                             for c, m in construct_metrics.items()]
                st.table(pd.DataFrame(ave_data, columns=['Construct', 'AVE', 'Result']))
                
                st.subheader("Discriminant Validity (Fornell-Larcker)")
                # Need construct correlation matrix
                latent_names = list(construct_metrics.keys())
                construct_corrs = sem_enh[(sem_enh['op'] == '~~') & (sem_enh['lval'].isin(latent_names)) & (sem_enh['rval'].isin(latent_names)) & (sem_enh['lval'] != sem_enh['rval'])]
                
                # Build construct corr matrix
                c_corr_matrix = pd.DataFrame(0.0, index=latent_names, columns=latent_names)
                for name in latent_names:
                    c_corr_matrix.loc[name, name] = 1.0
                    
                for _, row in construct_corrs.iterrows():
                    val = float(row['Est. Std']) if pd.notnull(row['Est. Std']) else 0.0
                    c_corr_matrix.loc[row['lval'], row['rval']] = val
                    c_corr_matrix.loc[row['rval'], row['lval']] = val
                
                dv_results = check_discriminant_validity({c: m['AVE'] for c, m in construct_metrics.items()}, c_corr_matrix)
                for res in dv_results:
                    st.info(res)
                st.divider()

                # 6. Bias Check (Step 5: Bias Check)
                st.header("Step 5: Common Method Bias Assessment")
                all_indicators = []
                for d in indep_dict:
                    for i in range(int(d['number_questions'])):
                        all_indicators.append(f"{d['Variable']}_Q{i+1}")
                
                var_exp, harman_msg = run_harman_single_factor_test(data_norm, all_indicators)
                st.write(harman_msg)
                st.divider()

                # 7. Correlation Analysis (Step 6: Correlation)
                st.header("Step 6: Correlation Analysis")
                corr_matrix = compute_correlation(data_norm, indep_dict)
                heatmap_path = f'{output_path}correlation_heatmap.png'
                st.image(heatmap_path, caption="Correlation Heatmap")
                
                corr_interp, rec_drops = interpret_correlation(corr_matrix, 0.7, 0.3)
                for msg in corr_interp:
                    st.info(msg)
                
                if rec_drops:
                    st.warning(f"**Optimization Tip:** To improve model fit (RMSEA), consider dropping these redundant items: {', '.join(rec_drops)}")
                st.divider()

                # 8. OLS Regression Analysis (Step 7 & 8)
                st.header("Step 7 & 8: OLS Regression & Diagnostics")
                reg_m_spec, reg_m_spec_dict = reg.create_model_spec(indep_dict, dep_dict, rel_dict, [])
                
                # Composite scores for OLS
                ols_data = data_norm.copy()
                for construct, questions in {**reg_m_spec_dict['independent'], **reg_m_spec_dict['dependent']}.items():
                    ols_data[construct] = reg.create_composite_score(ols_data, questions)

                # OLS Path results
                reg_results = reg.conduct_regression_analysis(data_norm, reg_m_spec_dict)
                reg_interp, reg_coef_dfs, reg_summ_tables = reg.interpret_regression_results(reg_results, param_dict)
                
                reg_vif_tables = []
                for i, (rel, msg) in enumerate(reg_interp):
                    with st.expander(f"OLS Path: {rel}"):
                        st.subheader("Diagnostics (VIF/Tolerance)")
                        vif_df = reg.calculate_collinearity_diagnostics(ols_data, rel)
                        reg_vif_tables.append(vif_df)
                        st.table(vif_df)
                        
                        st.subheader("Results")
                        st.text(reg_results[i][1]) # Summary text
                        st.markdown(f"**Interpretation:**\n{msg}")
                        st.table(reg_summ_tables[i])
                
                reg_result_graph_path = reg.create_result_graph_short(reg_m_spec, indep_dict, dep_dict, reg_coef_dfs, "reg_results_short")
                st.image(reg_result_graph_path, caption="OLS Path Estimates (Standardized β)")
                st.divider()

                # 7. Standardized Regression
                st.header("Step 5: Standardized Regression")
                std_reg_results = reg.conduct_regression_analysis_standardized(data_norm, reg_m_spec_dict)
                std_reg_interp, std_reg_coef_dfs, _ = reg.interpret_regression_results(std_reg_results, param_dict)
                
                for i, (rel, msg) in enumerate(std_reg_interp):
                    with st.expander(f"Standardized Relationship: {rel}"):
                        st.markdown(msg)
                
                std_reg_result_graph_path = reg.create_result_graph_short(reg_m_spec, indep_dict, dep_dict, std_reg_coef_dfs, "reg_std_results_short")
                st.image(std_reg_result_graph_path, caption="Standardized Path Estimates (β)")
                st.divider()

                # 9. SEM Analysis (Step 9 & 10)
                st.header("Step 9 & 10: CFA & Structural Equation Modeling")
                display_model_spec(sem_m_spec)
                
                st.subheader("Global Fit Indices")
                stats_table, overall_msg = sem.interpret_sem_stats(sem_stats, param_dict)
                st.dataframe(stats_table)
                st.success(overall_msg)
                
                st.subheader("Path Analysis (Latent Variable Model)")
                sem_interps = sem.interepret_sem_inspect(sem_enh, dep_dict, rel_dict, param_dict)
                for si in sem_interps:
                    st.markdown(si)
                
                st.image(g_full, caption="SEM Results (Full Model)")
                st.image(g_short, caption="SEM Results (Simplified Paths)")
                st.divider()

                # 10. Subgroup Analysis (Step 11)
                moderators = [v['Variable'] for v in dem_dict if v.get('used_as_moderator') == 'yes']
                mod_tables = {}
                if moderators:
                    st.header("Step 11: Multi-Group Moderation Analysis")
                    mod_tables = reg.conduct_regression_analysis_with_moderators(
                        reg_m_spec_dict, data_norm, label_map, moderators, reg_results
                    )
                    mod_observations = reg.interpret_moderator_results(mod_tables)
                    
                    for rel, df in mod_tables.items():
                        st.subheader(f"Cohort Comparison: {rel}")
                        st.dataframe(df)
                        
                        if rel in mod_observations:
                            st.markdown("**Observations:**")
                            for obs in mod_observations[rel]:
                                st.info(obs)
                st.divider()

                # 11. Final Report (Step 12)
                st.header("Step 12: Generate Final Report")
                
                # Generate the Markdown report content (with table)
                report_md = generate_markdown_report(
                    demographic_dict=dem_dict,
                    data_raw=data_raw,
                    overall_alpha=overall_a,
                    reliability_df=rel_df,
                    validity_ave_df=pd.DataFrame(ave_data, columns=['Construct', 'AVE', 'Result']),
                    discriminant_validity_results=dv_results,
                    harman_msg=harman_msg,
                    correlation_matrix=corr_matrix,
                    corr_interpretations=corr_interp,
                    recommended_drops=rec_drops,
                    heatmap_path=heatmap_path,
                    reg_m_spec=reg_m_spec,
                    reg_interp=reg_interp,
                    reg_summ_tables=reg_summ_tables,
                    reg_vif_tables=reg_vif_tables,
                    reg_result_graph_path=reg_result_graph_path,
                    std_reg_interpretations=std_reg_interp,
                    std_reg_result_graph_path=std_reg_result_graph_path,
                    moderator_results=mod_tables,
                    moderator_observations=mod_observations,
                    sem_model_spec=sem_m_spec,
                    sem_stats_table=stats_table,
                    sem_overall_msg=overall_msg,
                    sem_interps=sem_interps,
                    sem_graph_full_path=g_full,
                    sem_graph_short_path=g_short,
                    include_corr_table=True
                )
                
                # Generate the DOCX report content (without table)
                report_md_no_table = generate_markdown_report(
                    demographic_dict=dem_dict,
                    data_raw=data_raw,
                    overall_alpha=overall_a,
                    reliability_df=rel_df,
                    validity_ave_df=pd.DataFrame(ave_data, columns=['Construct', 'AVE', 'Result']),
                    discriminant_validity_results=dv_results,
                    harman_msg=harman_msg,
                    correlation_matrix=corr_matrix,
                    corr_interpretations=corr_interp,
                    recommended_drops=rec_drops,
                    heatmap_path=heatmap_path,
                    reg_m_spec=reg_m_spec,
                    reg_interp=reg_interp,
                    reg_summ_tables=reg_summ_tables,
                    reg_vif_tables=reg_vif_tables,
                    reg_result_graph_path=reg_result_graph_path,
                    std_reg_interpretations=std_reg_interp,
                    std_reg_result_graph_path=std_reg_result_graph_path,
                    moderator_results=mod_tables,
                    moderator_observations=mod_observations,
                    sem_model_spec=sem_m_spec,
                    sem_stats_table=stats_table,
                    sem_overall_msg=overall_msg,
                    sem_interps=sem_interps,
                    sem_graph_full_path=g_full,
                    sem_graph_short_path=g_short,
                    include_corr_table=False
                )
                
                # Generate DOCX content
                report_docx = convert_markdown_to_docx(report_md_no_table)
                
                # Provide download links
                st.markdown(get_markdown_download_link(report_md), unsafe_allow_html=True)
                st.markdown(get_docx_download_link(report_docx), unsafe_allow_html=True)

                st.balloons()

            except Exception as e:
                logger.exception("Analysis failed")
                st.error(f"An error occurred during analysis: {e}")
            
            finally:
                cleanup_output()

if __name__ == "__main__":
    main()
