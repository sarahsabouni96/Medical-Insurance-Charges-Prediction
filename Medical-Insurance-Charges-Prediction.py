#%% IMPORTS
import numpy as np
import pandas as pd
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#%% DATA IMPORT
df_file = "C:/Users/Lenovo/Desktop/Pytroch_course/first_project/archive/insurance.csv"
df = pd.read_csv(df_file)
print(df.head())

#%% VISUALIZE RELATIONSHIPS
sns.scatterplot(x='age', y='charges', data=df)
sns.regplot(x='age', y='charges', data=df, scatter=False)
plt.show()

sns.scatterplot(x='bmi', y='charges', data=df)
sns.regplot(x='bmi', y='charges', data=df, scatter=False)
plt.show()

#%% CHECK CATEGORICAL VALUES
print("Sex categories:", df['sex'].unique())
print("Smoker categories:", df['smoker'].unique())
print("Region categories:", df['region'].unique())

#%% CONVERT CATEGORICAL TO NUMERIC
df['sex'] = df['sex'].map({'female': 0, 'male': 1})
df['smoker'] = df['smoker'].map({'no': 0, 'yes': 1})
df = pd.get_dummies(df, columns=['region'], drop_first=True)  # one-hot encode regions

# Ensure all boolean dummies are integer
for col in df.select_dtypes('bool').columns:
    df[col] = df[col].astype(int)

print(df.head())

#%% PREPARE FEATURES AND TARGET
X_np = df.drop('charges', axis=1).values.astype(np.float32)
y_np = df['charges'].values.astype(np.float32).reshape(-1, 1)

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X_np)
X = torch.from_numpy(X_scaled)

# Standardize target
y_mean = y_np.mean()
y_std = y_np.std()
y_scaled = (y_np - y_mean) / y_std
y_true = torch.from_numpy(y_scaled)

print("X shape:", X.shape, "y_true shape:", y_true.shape)

#%% TRAIN PYTORCH LINEAR REGRESSION
num_features = X.shape[1]
w = torch.rand(num_features, 1, requires_grad=True)
b = torch.rand(1, requires_grad=True)

num_epochs = 1000
learning_rate = 0.01

for epoch in range(num_epochs):
    y_pred = X @ w + b
    loss = torch.mean((y_pred - y_true) ** 2)
    loss.backward()
    
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
    
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

#%% PREDICTIONS AFTER TRAINING
with torch.no_grad():
    y_pred_scaled = X @ w + b
    y_pred_original = y_pred_scaled * y_std + y_mean  # rescale to original charges

print("First 5 predicted charges:", y_pred_original[:5].numpy().flatten())
print("First 5 actual charges:", y_np[:5].flatten())

#%% MODEL PARAMETERS
print("Weights:", w.detach().numpy().flatten())
print("Bias:", b.item())

#%% PLOT PREDICTIONS VS ACTUAL (AGE)
plt.scatter(df['age'], y_np, label='Actual')
plt.scatter(df['age'], y_pred_original.numpy(), label='Predicted', color='red', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()

#%% SCikit-learn Linear Regression (for comparison)
reg = LinearRegression().fit(X_np, y_np)
print("Coefficients:", reg.coef_.flatten())
print("Intercept:", reg.intercept_)

y_sklearn_pred = reg.predict(X_np)
plt.scatter(df['age'], y_np, label='Actual')
plt.scatter(df['age'], y_sklearn_pred, label='Sklearn Predicted', color='green', alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Charges')
plt.legend()
plt.show()

#%% ACTUAL VS PREDICTED CHARGES (ALL FEATURES)
plt.figure(figsize=(8,6))
plt.scatter(y_np, y_pred_original.numpy(), alpha=0.6)
plt.plot([y_np.min(), y_np.max()], [y_np.min(), y_np.max()], 'r--')  # perfect prediction line
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("PyTorch Linear Regression: Actual vs Predicted Charges")
plt.show()

#%% R² SCORE
r2 = r2_score(y_np, y_pred_original.numpy())
print("R² score (PyTorch model):", r2)

#%% RESIDUAL PLOT
residuals = y_np - y_pred_original.numpy()
plt.figure(figsize=(8,6))
plt.scatter(y_pred_original.numpy(), residuals, alpha=0.6)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Charges")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot for PyTorch Linear Regression")
plt.show()
