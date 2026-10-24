# Actuarial Risk Modeling Portfolio

Welcome to my actuarial modeling portfolio. This repository contains two projects that tell a complete story about how we analyze risk in the insurance industry.

Instead of just showing two random scripts, I built these to show the *process* of modeling. We start with a hypothesis, test it, and when the data tells us we're wrong, we pivot to a better solution.

## The Story: From Prediction to Segmentation

### Part 1: The Hypothesis (Python Regression)
**Folder:** `2_Python_Regression/`

I started by asking a simple question: *Can we predict a smoker's BMI (morbidity risk) just by knowing their Age and how many cigarettes they smoke per day?*

I built a Multiple Linear Regression model in Python to find out. I went through the whole process: checking assumptions, fixing normality with log transformations, and checking for multicollinearity.

**The Honest Truth:** The model failed.
The data showed that Age and Smoking Volume *do not* significantly predict BMI. The p-values were too high, and the error rate was around 30%. 

**Why this matters:** This isn't a failure; it's a finding. It proves that BMI is driven by complex factors (diet, genetics, lifestyle) that a simple formula can't capture.

### Part 2: The Solution (R Clustering)
**Folder:** `1_R_Clustering/`

Because the regression told us we can't *predict* BMI with a formula, I needed a different way to assess risk. Instead of trying to force a prediction, I let the data speak for itself.

I used K-Means Clustering in R to group smokers based on their actual observed risk profiles. This approach successfully segmented the population into two distinct, defensible underwriting tiers:
1.  **Standard Risk:** Older smokers, lower smoking volume, lower BMI.
2.  **Substandard Risk:** Younger smokers, higher smoking volume, clinically obese BMI.

## How to Read This Repo

I recommend reading the projects in this order to understand the full logic:

1.  **Start here:** [2_Python_Regression/README.md](2_Python_Regression/README.md)
    *   Read this to see the statistical proof of why a simple predictive formula doesn't work for this problem.
2.  **Then go here:** [1_R_Clustering/README.md](1_R_Clustering/README.md)
    *   Read this to see how we solved the problem using unsupervised learning to create real-world underwriting classes.

## Technical Stack
-   **Python:** `pandas`, `statsmodels`, `scipy`, `seaborn` (for the regression analysis).
-   **R:** `tidyverse`, `cluster`, `factoextra` (for the clustering analysis).
-   **Data:** Africa Health Lab "Smoking Health Dataset".
