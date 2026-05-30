# Survey Data Analysis Tool

A comprehensive application designed to analyze survey data using a defensive sequence of Regression (OLS) and Structural Equation Modeling (SEM).

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
