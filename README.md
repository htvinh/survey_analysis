# Survey Data Analysis Tool

Turn survey responses into research-ready insights in minutes.

Survey Data Analysis (SDA) is an AI-powered platform that guides you through the entire survey research process—from raw data to publication-grade results. Upload your dataset and model specification, and SDA automatically performs data screening, descriptive analysis, reliability and validity assessment, bias detection, Regression Analysis, Confirmatory Factor Analysis (CFA), and Structural Equation Modeling (SEM).

Designed for researchers, students, consultants, and organizations, SDA helps validate theoretical models, test hypotheses, evaluate measurement quality, and uncover meaningful relationships within survey data. The result is a comprehensive report with statistical findings, visualizations, model diagnostics, and actionable insights—without requiring advanced statistical software expertise.


## Features
- **Data Pipeline:** Automated cleaning, pre-processing, and categorical encoding.
- **Methodological Rigor:** Implements a multi-step analytical workflow:
    - **Reliability:** Cronbach's Alpha and Composite Reliability (CR).
    - **Validity:** Convergent (AVE) and Discriminant Validity (Fornell-Larcker).
    - **Bias Check:** Harman's Single-Factor Test.
    - **Regression:** OLS path analysis with collinearity diagnostics (VIF/Tolerance).
    - **SEM:** Confirmatory Factor Analysis (CFA) and Structural Equation Modeling (CB-SEM).
    - **Moderation Analysis:** Multi-Group Moderation Analysis with automated cohort comparison and **qualitative observational insights**.
- **Reporting:** Automated publication-grade reports generated in both **Markdown** and **DOCX** formats.

## Usage
Run the application using the provided script:
```bash
./run_app.sh
```

## Configuration
The analysis is driven by a central Excel model configuration file. Please refer to the `model_sample_1.xlsx` for the required schema structure.
