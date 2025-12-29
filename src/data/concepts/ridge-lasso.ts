import { ConceptNode } from '../types';

export const ridgeLasso: ConceptNode = {
  id: 'ridge-lasso',
  title: 'Ridge / Lasso Regression',
  description: 'Regularized regression techniques that prevent overfitting through penalty terms',
  color: "bg-gradient-to-br from-purple-500 to-pink-600",
  overview: `Ridge and Lasso regression are regularized versions of linear regression that add penalty terms to the cost function to prevent overfitting and improve generalization. These methods are particularly valuable when dealing with high-dimensional data, multicollinearity, or when the number of features approaches or exceeds the number of observations.

Ridge Regression (L2 Regularization) adds a penalty proportional to the sum of squared coefficients:

$$\\text{Cost} = \\text{RSS} + \\lambda \\sum_{j=1}^p \\beta_j^2 = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda ||\\boldsymbol{\\beta}||_2^2$$

Lasso Regression (L1 Regularization) adds a penalty proportional to the sum of absolute coefficients:

$$\\text{Cost} = \\text{RSS} + \\lambda \\sum_{j=1}^p |\\beta_j| = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda ||\\boldsymbol{\\beta}||_1$$

where $\\lambda$ (lambda) is the regularization parameter that controls the strength of the penalty. Larger $\\lambda$ values increase regularization, shrinking coefficients more aggressively.

Elastic Net combines both penalties:

$$\\text{Cost} = \\text{RSS} + \\lambda_1 ||\\boldsymbol{\\beta}||_1 + \\lambda_2 ||\\boldsymbol{\\beta}||_2^2$$

This provides a balance between Ridge's coefficient shrinkage and Lasso's feature selection.`,
  
  howItWorks: `The Goal

The primary goal of Ridge and Lasso regression is to prevent overfitting by constraining the magnitude of model coefficients. This is achieved by adding a regularization term to the cost function that penalizes large coefficients. The key difference between Ridge and Lasso lies in how they shrink coefficients:

- Ridge (L2): Shrinks coefficients smoothly toward zero but rarely sets them exactly to zero

- Lasso (L1): Can set coefficients exactly to zero, effectively performing feature selection

Cost Functions

Ridge Regression Cost Function:

$$L_{\\text{Ridge}} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p \\beta_j^2$$

In matrix form:

$$L_{\\text{Ridge}} = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) + \\lambda \\boldsymbol{\\beta}^T\\boldsymbol{\\beta}$$

Lasso Regression Cost Function:

$$L_{\\text{Lasso}} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p |\\beta_j|$$

The absolute value in Lasso makes the optimization problem non-differentiable, requiring specialized algorithms like coordinate descent or least angle regression (LARS).`,
  
  applications: [
    'High-dimensional genomics data with thousands of features',
    'Financial modeling with many correlated predictors',
    'Image processing with pixel-level features',
    'Text analysis with high-dimensional word features',
    'Feature selection in datasets with many irrelevant features (Lasso)',
    'Stable predictions when features are highly correlated (Ridge)'
  ],
  
  advantages: [
    'Prevents overfitting, especially in high-dimensional settings',
    'Handles multicollinearity better than standard linear regression',
    'Lasso performs automatic feature selection by setting coefficients to zero',
    'Ridge provides stable solutions even when $p > n$'
  ],
  
  limitations: [
    'Requires careful hyperparameter tuning via cross-validation',
    'Ridge doesn\'t perform feature selection (all features retained)',
    'Lasso may arbitrarily select one feature from correlated groups',
    'Requires feature scaling for proper regularization'
  ],
  
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

# Generate sample data: predicting property value from many features
# Simulating high-dimensional scenario with some irrelevant features
np.random.seed(42)
n_samples = 200
n_features = 50  # More features than would be ideal for n_samples

# Create feature matrix with some correlation
X = np.random.randn(n_samples, n_features)
# Make some features correlated
X[:, 1] = X[:, 0] + 0.1 * np.random.randn(n_samples)  # Feature 1 correlated with 0
X[:, 2] = X[:, 0] + 0.1 * np.random.randn(n_samples)  # Feature 2 correlated with 0

# True relationship: only first 5 features matter
true_coef = np.zeros(n_features)
true_coef[:5] = [100, 50, 30, 20, 10]  # Only first 5 features are relevant
y = X @ true_coef + np.random.randn(n_samples) * 10

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for regularization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define range of lambda (alpha in sklearn) values
alphas = np.logspace(-4, 2, 50)  # From 0.0001 to 100

# Ridge Regression
ridge_scores = []
ridge_coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    scores = cross_val_score(ridge, X_train_scaled, y_train, cv=5, scoring='r2')
    ridge_scores.append(scores.mean())
    ridge.fit(X_train_scaled, y_train)
    ridge_coefs.append(ridge.coef_)

best_ridge_alpha = alphas[np.argmax(ridge_scores)]
ridge_best = Ridge(alpha=best_ridge_alpha)
ridge_best.fit(X_train_scaled, y_train)
ridge_pred = ridge_best.predict(X_test_scaled)

# Lasso Regression
lasso_scores = []
lasso_coefs = []

for alpha in alphas:
    lasso = Lasso(alpha=alpha, max_iter=2000)
    scores = cross_val_score(lasso, X_train_scaled, y_train, cv=5, scoring='r2')
    lasso_scores.append(scores.mean())
    lasso.fit(X_train_scaled, y_train)
    lasso_coefs.append(lasso.coef_)

best_lasso_alpha = alphas[np.argmax(lasso_scores)]
lasso_best = Lasso(alpha=best_lasso_alpha, max_iter=2000)
lasso_best.fit(X_train_scaled, y_train)
lasso_pred = lasso_best.predict(X_test_scaled)

# Compare with OLS (no regularization)
from sklearn.linear_model import LinearRegression
ols = LinearRegression()
ols.fit(X_train_scaled, y_train)
ols_pred = ols.predict(X_test_scaled)

# Print results
print("=== Model Comparison ===")
print(f"\\nRidge (α={best_ridge_alpha:.4f}):")
print(f"  Test R²: {r2_score(y_test, ridge_pred):.4f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, ridge_pred)):.2f} €")
print(f"  Non-zero coefficients: {np.sum(np.abs(ridge_best.coef_) > 1e-6)}/{n_features}")

print(f"\\nLasso (α={best_lasso_alpha:.4f}):")
print(f"  Test R²: {r2_score(y_test, lasso_pred):.4f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, lasso_pred)):.2f} €")
print(f"  Non-zero coefficients: {np.sum(np.abs(lasso_best.coef_) > 1e-6)}/{n_features}")

print(f"\\nOLS (no regularization):")
print(f"  Test R²: {r2_score(y_test, ols_pred):.4f}")
print(f"  Test RMSE: {np.sqrt(mean_squared_error(y_test, ols_pred)):.2f} €")
print(f"  Non-zero coefficients: {n_features}/{n_features}")

# Visualization
plt.figure(figsize=(10, 6))
plt.semilogx(alphas, ridge_scores, label='Ridge', linewidth=2)
plt.semilogx(alphas, lasso_scores, label='Lasso', linewidth=2)
plt.axvline(best_ridge_alpha, color='blue', linestyle='--', alpha=0.5)
plt.axvline(best_lasso_alpha, color='orange', linestyle='--', alpha=0.5)
plt.xlabel('Regularization Parameter (α)')
plt.ylabel('Cross-Validation R² Score')
plt.title('Regularization Parameter Selection')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Example prediction
new_property = np.random.randn(1, n_features)
new_property_scaled = scaler.transform(new_property)
predicted_value_ridge = ridge_best.predict(new_property_scaled)
predicted_value_lasso = lasso_best.predict(new_property_scaled)
print(f"\\nExample prediction for new property:")
print(f"  Ridge: {predicted_value_ridge[0]:.2f} €")
print(f"  Lasso: {predicted_value_lasso[0]:.2f} €")
`,
  children: []
};
