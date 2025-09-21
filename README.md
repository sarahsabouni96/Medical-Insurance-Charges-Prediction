# Medical-Insurance-Charges-Prediction
End-to-end project predicting medical insurance charges. Includes data exploration, preprocessing, PyTorch linear regression training, evaluation with R² and residual plots, and comparison with scikit-learn.

## Project Overview
This project aims to predict **medical insurance charges** using demographic and health-related features.  
We implement a **linear regression model in PyTorch**, evaluate its performance, and compare it with scikit-learn's implementation.

---

## Dataset
- **Source:** [Kaggle: Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mosapabdelghany/medical-insurance-cost-dataset?select=insurance.csv)  
- **Columns / Features:**
  - `age`: Age of the individual
  - `sex`: Male or Female
  - `bmi`: Body mass index
  - `children`: Number of children/dependents
  - `smoker`: Whether the person smokes (yes/no)
  - `region`: Geographic region (northeast, northwest, southeast, southwest)
  - `charges`: Insurance charges (target)

---

## Project Goals
- Predict insurance charges (`charges`) based on independent features.
- Understand the relationship between features and target.
- Build a linear regression model using PyTorch and evaluate its performance.
- Compare PyTorch implementation with scikit-learn’s LinearRegression.

---

## Methodology

1. **Data Exploration & Visualization**
   - Scatter plots to explore relationships (e.g., age vs charges, BMI vs charges).
   - Checking categorical values for `sex`, `smoker`, and `region`.

2. **Data Preprocessing**
   - Convert categorical features (`sex`, `smoker`) to numerical (0/1).  
   - One-hot encode `region` to avoid dummy variable trap.  
   - Standardize features and target for PyTorch model.

3. **PyTorch Linear Regression**
   - Initialize weights and bias tensors with gradients enabled.
   - Implement training loop with **Mean Squared Error (MSE) loss**.
   - Update weights using **gradient descent**.

4. **Model Evaluation**
   - Rescale predictions to original target units.
   - Compute **R² score** to measure variance explained by the model.
   - Visualize **actual vs predicted charges**.
   - Create **residual plot** to analyze prediction errors.

5. **Comparison**
   - Fit the same dataset using scikit-learn LinearRegression.
   - Compare coefficients, intercept, and predictions.

---

## Results

- **R² score (PyTorch model):** ~0.75 → Model explains ~75% of the variance in insurance charges.
- **Key plots included:**
  - Actual vs Predicted Charges
  - Residual Plot
- **Insights:**
  - `smoker` status strongly influences insurance charges.
  - Some non-linear patterns exist in residuals, suggesting potential for more complex models.

---

## Conclusion

This project demonstrates the ability to:
- Handle **real-world datasets** with categorical and numerical features.
- Build and train a **linear regression model in PyTorch** from scratch.
- Evaluate model performance using metrics and visualizations.
- Compare a custom PyTorch implementation with **scikit-learn baseline**.

---

## Files in the Repository
- `insurance.csv` → Dataset  
- `PyTorch_LinearRegression.ipynb` → Notebook with full PyTorch implementation  
- `plots/` → Optional folder for saving visualizations  

---

## Future Improvements
- Explore **non-linear models** (e.g., polynomial regression, neural networks).  
- Feature engineering: interaction terms like `age * smoker`.  
- Hyperparameter tuning (learning rate, epochs).  
- Cross-validation to improve model reliability.

---

