import { ConceptNode } from '../../../../types';

export const ridgeLasso: ConceptNode = {
  id: 'ridge-lasso',
  title: 'Ridge / Lasso Regression',
  description: 'Regularized regression techniques that prevent overfitting through penalty terms',
  color: "bg-gradient-to-br from-teal-400 to-cyan-500",
  overview: `In many practical applications, fitting a linear regression model to high-dimensional data presents a fundamental challenge: when the number of features $p$ is large relative to the number of observations $n$, or when features exhibit substantial multicollinearity, standard Ordinary Least Squares regression produces unstable and poorly generalizing models. Ridge and Lasso regression address this critical problem through the introduction of regularization—penalty terms that constrain the magnitude of fitted coefficients, thereby trading a small amount of bias for substantial reductions in variance.

Ridge regression, also known as L2 regularization, introduces a penalty term proportional to the sum of squared coefficients into the cost function:

$$\\text{Cost} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p \\beta_j^2 = \\text{RSS} + \\lambda ||\\boldsymbol{\\beta}||_2^2$$

Lasso regression, by contrast, employs L1 regularization, which penalizes the sum of absolute coefficient values:

$$\\text{Cost} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p |\\beta_j| = \\text{RSS} + \\lambda ||\\boldsymbol{\\beta}||_1$$

In both formulations, $\\lambda$ (lambda) serves as the regularization parameter, a non-negative scalar that controls the strength of the penalty. A value of $\\lambda = 0$ recovers standard linear regression, while larger values increasingly penalize non-zero coefficients. The critical insight is that these two approaches, though similar in structure, produce markedly different behaviors: Ridge shrinks coefficients toward zero smoothly but never sets them exactly to zero, whereas Lasso can drive coefficients exactly to zero, thereby performing automatic feature selection.

A third approach, Elastic Net, combines both penalty terms to leverage the strengths of each method:

$$\\text{Cost} = \\text{RSS} + \\lambda_1 ||\\boldsymbol{\\beta}||_1 + \\lambda_2 ||\\boldsymbol{\\beta}||_2^2$$

This hybrid formulation provides the feature selection capability of Lasso while maintaining the stability of Ridge regression.`,
  
  howItWorks: `The fundamental motivation for regularized regression emerges from the bias-variance tradeoff. Standard linear regression minimizes prediction error on the training data, but this approach often leads to overfitting: models with large coefficients that fit noise rather than genuine patterns. When features are correlated—a condition known as multicollinearity—the design matrix $\\mathbf{X}^T\\mathbf{X}$ becomes ill-conditioned, making the OLS solution numerically unstable and highly sensitive to small perturbations in the data. Regularization mitigates these problems by explicitly constraining the size of coefficients, thus introducing a controlled amount of bias in exchange for substantially reduced variance.

Ridge regression modifies the objective function by introducing an L2 penalty that grows with the squared magnitude of coefficients:

$$L_{\\text{Ridge}} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p \\beta_j^2$$

In matrix form, this becomes:

$$L_{\\text{Ridge}} = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) + \\lambda \\boldsymbol{\\beta}^T\\boldsymbol{\\beta}$$

Despite the addition of the penalty term, Ridge has a remarkably clean closed-form solution. Taking the gradient with respect to $\\boldsymbol{\\beta}$ and setting it to zero yields:

$$\\hat{\\boldsymbol{\\beta}}_{\\text{Ridge}} = (\\mathbf{X}^T\\mathbf{X} + \\lambda \\mathbf{I})^{-1}\\mathbf{X}^T\\mathbf{y}$$

The addition of $\\lambda \\mathbf{I}$ to the design matrix Gram product has a profound effect: it improves numerical conditioning (making $\\mathbf{X}^T\\mathbf{X} + \\lambda \\mathbf{I}$ invertible even when $\\mathbf{X}^T\\mathbf{X}$ is singular), and it introduces shrinkage—all coefficients are pulled toward zero. Notably, Ridge regression never sets coefficients exactly to zero; instead, it reduces them proportionally based on their magnitude and the regularization strength.

Lasso regression takes a different approach by penalizing the sum of absolute coefficient values:

$$L_{\\text{Lasso}} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 + \\lambda \\sum_{j=1}^p |\\beta_j|$$

This seemingly small change from squared coefficients to absolute values creates a dramatic difference in behavior. The absolute value function is piecewise linear and non-differentiable at zero, which means there is no closed-form solution for the Lasso estimator. Instead, specialized optimization algorithms such as coordinate descent or least angle regression (LARS) must be employed to solve the problem iteratively. However, this computational cost is rewarded with a remarkable property: the L1 penalty tends to produce sparse solutions where many coefficients are exactly zero. This occurs because the L1 penalty creates a "corner" in the constraint region at the origin, and the optimal solution often aligns with this corner, forcing some coefficients to zero exactly.

This sparsity property makes Lasso invaluable for feature selection: by identifying which coefficients the algorithm sets to zero, we automatically discover which features are irrelevant to predicting the target. In contrast to Ridge, which retains all features but shrinks their coefficients, Lasso explicitly discards irrelevant features, producing a more interpretable and parsimonious model.

The choice between Ridge and Lasso depends on the problem structure. When all features are believed to contribute meaningfully to prediction—perhaps in slightly different magnitudes due to multicollinearity—Ridge is the appropriate choice. When a sparse feature subset is suspected and interpretability is paramount, Lasso excels. In practice, Elastic Net provides a compromise, combining both penalties with two regularization parameters ($\\lambda_1$ and $\\lambda_2$) that can be tuned via cross-validation to balance the desirable properties of both methods.

A critical practical consideration is the selection of the regularization parameter $\\lambda$. Since larger values induce more regularization (moving coefficients closer to zero and, in Lasso, producing more zeros), we must identify the value that optimizes generalization performance. The standard approach is $k$-fold cross-validation: partition the training data into $k$ folds, fit models with different $\\lambda$ values on $k-1$ folds, evaluate performance on the held-out fold, and repeat. The $\\lambda$ value that achieves the best average test performance is selected. Additionally, features must be standardized before regularization, since the penalty term treats all coefficients equally regardless of the underlying feature scale; without standardization, features with larger natural magnitudes would be penalized less severely.`,
  
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
