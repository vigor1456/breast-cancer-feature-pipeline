# Breast Cancer Feature Selection & Classification Pipeline

This project implements a complete machine learning pipeline to classify breast cancer tumors as malignant or benign using multiple models and interpretability techniques.

## Objective

Build a robust and interpretable classification model for breast cancer diagnosis based on tumor measurements, while identifying the most predictive features using multiple techniques.

## Workflow

1. **Data loading & preprocessing**
2. **Exploratory data analysis (EDA)**  
   - Structural summary
   - Correlation with malignancy
   - Redundancy filtering based on correlation threshold
3. **Feature selection & reduction**
4. **Model training**
   - Logistic Regression (with hyperparameter tuning)
   - Random Forest
5. **Feature importance analysis**
   - Logistic regression coefficients
   - Permutation importance (model-agnostic)
   - Random forest feature importance
   - Permutation importance on random forest
6. **Cross-method feature comparison**
7. **Visualization**
   - Top features from each method
   - Heatmaps and pairplots for feature relationships

## Models Used

- Logistic Regression (with `GridSearchCV`)
- Random Forest Classifier

## Feature Ranking Techniques

- Absolute coefficients (logistic regression)
- Permutation importance (logreg + RF)
- Tree-based impurity (Random Forest)
- Consensus voting across methods

## Key Techniques

- Feature de-correlation via pairwise correlation thresholding
- Model explainability
- Modular pipeline design (every step is reusable and testable)
- Logging instead of print statements for clean output
