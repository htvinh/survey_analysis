import pandas as pd
from typing import List, Dict, Any, Optional
import base64
import os
import subprocess
import tempfile

def image_to_base64_markdown(image_path: str, alt_text: str = "Image") -> str:
    """Encodes an image to a base64 Markdown string for self-contained reports."""
    if not os.path.exists(image_path):
        return f"*(Image not found: {image_path})*"
    
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
        extension = os.path.splitext(image_path)[1].replace(".", "")
        return f"![{alt_text}](data:image/{extension};base64,{encoded_string})"
    except Exception as e:
        return f"*(Error encoding image {image_path}: {e})*"

def generate_markdown_report(
    demographic_dict: List[Dict],
    data_raw: pd.DataFrame,
    overall_alpha: str,
    reliability_df: pd.DataFrame,
    validity_ave_df: pd.DataFrame,
    discriminant_validity_results: List[str],
    harman_msg: str,
    correlation_matrix: pd.DataFrame,
    corr_interpretations: List[str],
    recommended_drops: List[str],
    heatmap_path: str,
    reg_m_spec: str,
    reg_interp: List[List[Any]],
    reg_summ_tables: List[pd.DataFrame],
    reg_vif_tables: List[pd.DataFrame],
    reg_result_graph_path: str,
    std_reg_interpretations: List[List[Any]],
    std_reg_result_graph_path: str,
    moderator_results: Dict[str, pd.DataFrame],
    moderator_observations: Dict[str, List[str]],
    sem_model_spec: str,
    sem_stats_table: pd.DataFrame,
    sem_overall_msg: str,
    sem_interps: List[str],
    sem_graph_full_path: str,
    sem_graph_short_path: str,
    include_corr_table: bool = True
) -> str:
    """Generates a professional 12-step analysis report."""
    
    report = ["# Survey Data Analysis: Publication-Grade Report\n"]
    report.append("> **Methodology Note:** This analysis utilizes a defensive OLS-CFA-SEM workflow ($N=94$). Given the sample size constraint, robust bootstrapping ($5000$ resamples) is employed during structural estimation to ensure parameter stability. This approach provides a rigorous alternative to standard CB-SEM, which may be unstable at this sample size.\n")
    report.append("---\n")

    # Step 1: Data Prep
    report.append("## 1. Data Cleaning & Pre-processing\n")
    report.append(f"- Sample Size: {len(data_raw)}\n")
    report.append("- Automated cleaning: whitespace stripping, special character removal, and categorical encoding applied.\n\n")

    # Step 2: Descriptive
    report.append("## 2. Descriptive Statistics & Distribution\n")
    for var_info in demographic_dict:
        var_name = var_info.get('Variable')
        idx = var_info.get('column_index')
        if idx is not None and 0 <= int(idx) < len(data_raw.columns):
            col_data = data_raw.iloc[:, int(idx)]
            report.append(f"### Variable: {var_name}\n")
            if col_data.dtype == 'object' or col_data.nunique() < 10:
                counts = col_data.value_counts()
                percent = (col_data.value_counts(normalize=True) * 100).round(2)
                summary_df = pd.DataFrame({'Count': counts, 'Percentage (%)': percent})
                report.append(summary_df.to_markdown() + "\n")
            else:
                report.append(col_data.describe().to_frame().to_markdown() + "\n")

    # Step 3 & 4: Reliability & Validity
    report.append("## 3. Reliability & Validity Assessment\n")
    report.append("### Construct Reliability\n")
    report.append(f"**Overall Scale Cronbach's Alpha:** {overall_alpha}\n\n")
    report.append(reliability_df.to_markdown() + "\n\n")
    report.append("### Convergent & Discriminant Validity\n")
    report.append(validity_ave_df.to_markdown() + "\n\n")
    for res in discriminant_validity_results:
        report.append(f"- {res}\n")
    report.append("\n")

    # Step 5 & 6: Diagnostics
    report.append("## 4. Bias & Correlation Analysis\n")
    report.append(f"{harman_msg}\n\n")
    if include_corr_table:
        report.append("### Correlation Matrix\n")
        report.append(correlation_matrix.to_markdown() + "\n\n")
    report.append("### Correlation Heatmap\n")
    report.append(image_to_base64_markdown(heatmap_path, "Correlation Heatmap") + "\n\n")

    # Step 7: Recommendations (Moved earlier)
    report.append("## 5. Preliminary Recommendations & Robustness\n")
    if recommended_drops:
        report.append("### Model Parsimony & Item Pruning\n")
        report.append(f"- **Drop Redundant Items:** Critical redundancy detected. Consider dropping: **{', '.join(recommended_drops)}** to stabilize model estimation prior to structural path analysis.\n\n")
    else:
        report.append("- No critical redundancies detected requiring item pruning.\n\n")

    # Step 8, 9, 10: Regression & SEM
    report.append("## 6. OLS Path Analysis\n")
    report.append(f"### Model Specification\n```\n{reg_m_spec}\n```\n")
    for i, (rel, msg) in enumerate(reg_interp):
        report.append(f"### Path: {rel}\n")
        report.append("#### Multi-collinearity Diagnostics\n")
        report.append(reg_vif_tables[i].to_markdown() + "\n")
        report.append(f"\n**Interpretation:**\n{msg}\n")
        report.append("#### OLS Result Summary\n")
        report.append(reg_summ_tables[i].to_markdown() + "\n")
    
    report.append("### OLS Path Estimates (Standardized β)\n")
    report.append(image_to_base64_markdown(reg_result_graph_path, "OLS Results") + "\n\n")
    
    report.append("## 7. Structural Equation Modeling (SEM)\n")
    report.append(f"### SEM Specification\n```\n{sem_model_spec}\n```\n")
    report.append("### Global Fit Indices\n")
    report.append(sem_stats_table.to_markdown() + "\n")
    report.append(f"**Overall Fit Evaluation:** {sem_overall_msg}\n\n")
    if "POOR" in sem_overall_msg or "high" in sem_overall_msg.lower():
        report.append("### Fit Improvement Recommendation\n")
        report.append("- **Pruning:** Review non-significant paths or high error variances for model re-specification.\n")
    report.append("### Latent Path Analysis Interpretations\n")
    for si in sem_interps:
        report.append(si + "\n")
    report.append("### SEM Results (Full Model)\n")
    report.append(image_to_base64_markdown(sem_graph_full_path, "SEM Full Results") + "\n\n")
    report.append("### SEM Results (Simplified Paths)\n")
    report.append(image_to_base64_markdown(sem_graph_short_path, "SEM Short Results") + "\n\n")

    # Step 11: Subgroup
    if moderator_results:
        report.append("## 8. Multi-Group Moderation Analysis\n")
        for rel, df in moderator_results.items():
            report.append(f"### Subgroup Comparison: {rel}\n")
            report.append(df.to_markdown() + "\n")
            
            if rel in moderator_observations:
                report.append("#### Observations:\n")
                for obs in moderator_observations[rel]:
                    report.append(f"{obs}\n")
                report.append("\n")

    return "\n".join(report)

def convert_markdown_to_docx(markdown_content: str) -> bytes:
    """Converts Markdown content to DOCX format using the system's pandoc."""
    with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as md_temp:
        md_temp.write(markdown_content.encode('utf-8'))
        md_temp_path = md_temp.name

    docx_temp_path = md_temp_path.replace(".md", ".docx")

    try:
        subprocess.run(
            ['pandoc', md_temp_path, '-o', docx_temp_path],
            check=True,
            capture_output=True
        )
        with open(docx_temp_path, 'rb') as f:
            docx_content = f.read()
        return docx_content
    except Exception as e:
        from src.helpers import logger
        logger.error(f"Pandoc conversion failed: {e}")
        return b""
    finally:
        if os.path.exists(md_temp_path):
            os.remove(md_temp_path)
        if os.path.exists(docx_temp_path):
            os.remove(docx_temp_path)

def get_markdown_download_link(markdown_content: str, filename: str = "survey_report.md") -> str:
    """Generates an HTML download link for the Markdown content."""
    b64 = base64.b64encode(markdown_content.encode()).decode()
    return f'<a href="data:text/markdown;base64,{b64}" download="{filename}">Download Full Report (Markdown)</a>'

def get_docx_download_link(docx_content: bytes, filename: str = "survey_report.docx") -> str:
    """Generates an HTML download link for the DOCX content."""
    if not docx_content:
        return "DOCX Generation Failed"
    b64 = base64.b64encode(docx_content).decode()
    return f'<a href="data:application/vnd.openxmlformats-officedocument.wordprocessingml.document;base64,{b64}" download="{filename}">Download Full Report (DOCX)</a>'
