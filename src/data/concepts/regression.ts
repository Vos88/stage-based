import { ConceptNode } from '../types';

export const polynomialRegression: ConceptNode = {
  id: 'polynomial-regression',
  title: 'Polynomial Regression',
  description: 'Fitting polynomial relationships between variables to capture non-linear patterns',
  color: "bg-gradient-to-br from-teal-500 to-cyan-600",
  overview: `
  
Polynomial regression extendslinear regression by modeling non-linear relationships between variables. While linear regression assumes a straight-line relationship between the predictor $x$ and the response $y$, polynomial regression can capture curves, bends, and more complex patterns. This flexibility makes it particularly useful for modeling real-world phenomena such as growth rates, physical laws, or economic trends.

The key idea is to transform the original features into polynomial features, allowing us to fit curves instead of just lines. Despite its ability to model non-linear relationships, polynomial regression remains alinear model in terms of its parameters, which means we can still use familiar techniques like Ordinary Least Squares (OLS) to estimate the coefficients.

The polynomial regression model of degree $d$ is given by:

$$ y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + \\cdots + \\beta_d x^d + \\varepsilon $$

Here, $ (\\beta_0, \\beta_1, \\ldots, \\beta_d) $ are the coefficients to be estimated, and $ \\varepsilon $ represents the error term.

To express polynomial regression in matrix form, we construct apolynomial feature matrix} $ \\mathbf{X}_{\\text{poly}} $:
$$
\\mathbf{X}_{\\text{poly}} =
\\begin{bmatrix}
1 & x_1 & x_1^2 & \\cdots & x_1^d \\\\
1 & x_2 & x_2^2 & \\cdots & x_2^d \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\
1 & x_n & x_n^2 & \\cdots & x_n^d
\\end{bmatrix}
$$
Each row of $\\mathbf{X}_{\\text{poly}}$ corresponds to an observation, and each column represents a polynomial term ($x^0, x^1, x^2, \\ldots, x^d$ for example). This transformation allows us to treat polynomial regression as a special case of multiple linear regression.

Polynomial regression uses the same cost function as linear regression: the Residual Sum of Squares (RSS):
$$
\\text{RSS} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^n \\left(y_i - (\\beta_0 + \\beta_1 x_i + \\beta_2 x_i^2 + \\cdots + \\beta_d x_i^d)\\right)^2
$$
In matrix form, the RSS is:
$$
\\text{RSS} = (\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})
$$
Since polynomial regression is linear in the parameters, we can use the OLS solution to estimate the coefficients:
$$
\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}_{\\text{poly}}^T\\mathbf{X}_{\\text{poly}})^{-1}\\mathbf{X}_{\\text{poly}}^T\\mathbf{y}
$$

Polynomial regression uses the same evaluation metrics as linear regression, such as $R^2$, RMSE, and MAE. However, selecting the optimal degree $d$ requires careful consideration:

1. Use cross-validation to compare models with different degrees.
2. Split the data into training and validation sets.
3. Fit models with varying degrees and evaluate their performance on the validation set.
4. Choose the degree that yields the best validation performance.

Lower-degree polynomials tend to have high bias but low variance, while higher-degree polynomials have low bias but high variance. The goal is to find a degree that balances these competing concerns.
1. Plotting the fitted curve against the data helps visualize whether the model is overfitting (e.g., a wiggly curve that follows noise) or underfitting (e.g., a straight line through a curved pattern).
2. Feature scaling is often helpful, especially for higher-degree polynomials, to improve numerical stability.
3. Regularization techniques, such as Ridge or Lasso regression, can be applied to polynomial features to mitigate overfitting.
`,
  
  applications: [
    'Modeling growth curves in biology and economics',
    'Predicting energy consumption with non-linear patterns',
    'Forecasting sales with seasonal acceleration effects',
    'Analyzing dose-response relationships in pharmacology',
    'Modeling physical processes with known polynomial relationships',
    'Curve fitting in scientific data analysis'
  ],
  
  advantages: [
    'Captures non-linear relationships that linear regression cannot',
    'Still uses linear regression machinery (OLS), so computationally efficient',
    'Interpretable: coefficients show contribution of each polynomial term',
    'Flexible: can model various curve shapes by adjusting degree'
  ],
  
  limitations: [
    'High risk of overfitting, especially with high degrees and small datasets',
    'Sensitive to outliers, which can dramatically affect curve shape',
    'Poor extrapolation: polynomial curves diverge rapidly outside training range',
    'Multicollinearity: polynomial features are highly correlated (e.g., $x$ and $x^2$)'
  ],
  
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Generate sample data: energy consumption vs. temperature (non-linear relationship)
np.random.seed(42)
n_samples = 150

# Temperature in Celsius
temperature = np.random.uniform(-10, 35, n_samples)

# Energy consumption follows quadratic relationship (heating/cooling costs)
# Higher energy use at extreme temperatures
true_consumption = 2000 + 50 * (temperature - 12.5)**2
noise = np.random.normal(0, 200, n_samples)
energy_consumption = true_consumption + noise  # in kWh

# Prepare data
X = temperature.reshape(-1, 1)
y = energy_consumption

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Try different polynomial degrees
degrees = [1, 2, 3, 5, 10]
results = {}

for degree in degrees:
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_train_poly, y_train)
    
    # Make predictions
    y_train_pred = model.predict(X_train_poly)
    y_test_pred = model.predict(X_test_poly)
    
    # Calculate metrics
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_poly, y_train, cv=5, scoring='r2')
    
    results[degree] = {
        'model': model,
        'poly_features': poly_features,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\\n=== Degree {degree} ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f} kWh")
    print(f"CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Select best model (degree 2 in this case)
best_degree = 2
best_model = results[best_degree]['model']
best_poly = results[best_degree]['poly_features']

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, alpha=0.6, label='Training data', s=30)
plt.scatter(X_test, y_test, alpha=0.6, label='Test data', color='orange', s=30)

# Plot fitted curve for best degree
temp_range = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
temp_poly = best_poly.transform(temp_range)
pred = best_model.predict(temp_poly)
plt.plot(temp_range, pred, 'r-', label=f'Degree {best_degree}', linewidth=2)

plt.xlabel('Temperature (°C)')
plt.ylabel('Energy Consumption (kWh)')
plt.title('Polynomial Regression: Energy vs. Temperature')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Example prediction
new_temp = np.array([[25]])  # 25°C
new_temp_poly = best_poly.transform(new_temp)
predicted_energy = best_model.predict(new_temp_poly)
print(f"\\nPredicted energy consumption at 25°C: {predicted_energy[0]:.2f} kWh")
`,
  children: []
};

export const linearRegression: ConceptNode = {
  id: 'linear-regression',
  title: 'Linear Regression',
  color: "bg-gradient-to-br from-teal-500 to-cyan-600",
  description: 'A fundamental machine learning method for modeling relationships between variables by fitting a linear equation to observed data.',
  overview:
   `Linear regression is one of the most fundamental and widely-used machine learning algorithms. It models the relationship between a dependent variable (what we want to predict) and one or more independent variables (features) by fitting a linear equation to the observed data.

Simple Linear Regression models the relationship between a single predictor $x$ and a response variable $y$:

$$y = \\beta_0 + \\beta_1 x + \\varepsilon$$

where $\\beta_0$ is the intercept, $\\beta_1$ is the slope, and $\\varepsilon$ represents the error term.

Multiple Linear Regression extends this to multiple predictors:

$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_p x_p + \\varepsilon$$

In matrix notation, with $n$ observations and $p$ features:

$$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\varepsilon}$$

where $\\mathbf{y}$ is an $n \\times 1$ vector of responses, $\\mathbf{X}$ is an $n \\times (p+1)$ design matrix (with a column of ones for the intercept), $\\boldsymbol{\\beta}$ is a $(p+1) \\times 1$ vector of coefficients, and $\\boldsymbol{\\varepsilon}$ is the error vector.`,
  
  howItWorks: `The Goal

The primary goal of linear regression is to find the best-fitting line (or hyperplane in multiple dimensions) that minimizes the difference between predicted and actual values. We want to estimate the coefficients $\\boldsymbol{\\beta}$ that make our predictions as accurate as possible.

Cost Functions

The most common cost function is the Mean Squared Error (MSE), which measures the average squared difference between predicted and actual values:

$$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$

where $\\hat{y}_i = \\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_p x_{ip}$ is the predicted value.

Equivalently, we minimize the Residual Sum of Squares (RSS) or Sum of Squared Errors (SSE):

$$\\text{RSS} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^n \\left(y_i - (\\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_p x_{ip})\\right)^2$$

In matrix form:

$$\\text{RSS} = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})$$

Minimization of Error

To find the optimal coefficients, we take the derivative of the cost function with respect to $\\boldsymbol{\\beta}$ and set it to zero. This gives us the Normal Equations:

$$\\frac{\\partial \\text{RSS}}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) = 0$$

Solving for $\\boldsymbol{\\beta}$, we get the Ordinary Least Squares (OLS) solution:

$$\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}$$

For simple linear regression, the closed-form solutions are:

$$\\beta_1 = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^n (x_i - \\bar{x})^2} = \\frac{\\text{Cov}(x, y)}{\\text{Var}(x)}$$

$$\\beta_0 = \\bar{y} - \\beta_1 \\bar{x}$$

where $\\bar{x}$ and $\\bar{y}$ are the sample means.

Model Evaluation

After fitting the model, we evaluate its performance using several metrics:

- R-squared ($R^2$) measures the proportion of variance explained by the model:

$$R^2 = 1 - \\frac{\\text{RSS}}{\\text{TSS}} = 1 - \\frac{\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^n (y_i - \\bar{y})^2}$$

where TSS is the Total Sum of Squares. $R^2$ ranges from 0 to 1, with 1 indicating a perfect fit.

- Adjusted R-squared accounts for the number of predictors:

$$R^2_{\\text{adj}} = 1 - \\frac{(1-R^2)(n-1)}{n-p-1}$$

- Root Mean Squared Error (RMSE):

$$\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2} = \\sqrt{\\text{MSE}}$$

- Mean Absolute Error (MAE):

$$\\text{MAE} = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|$$

These metrics help us understand how well our model fits the data and how accurate our predictions are.`,
  applications: [
    'Predicting house prices based on size, location, and features',
    'Forecasting sales based on advertising spend and other factors',
    'Estimating medical outcomes from patient characteristics',
    'Analyzing relationships between variables in scientific research',
    'Demand forecasting in economics and business',
    'Quality control and process optimization'
  ],
  advantages: [
    'Simple and interpretable: coefficients show the relationship between features and target',
    'Fast and computationally efficient, especially with closed-form OLS solution',
    'No hyperparameters to tune (unlike many other ML methods)',
    'Provides statistical inference (confidence intervals, p-values)'
  ],
  limitations: [
    'Assumes a linear relationship between features and target variable',
    'Sensitive to outliers, which can significantly affect the fitted line',
    'Assumes features are independent (multicollinearity can be problematic)',
    'Requires homoscedasticity (constant variance of errors)'
  ],
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Generate sample data: house prices based on size and age
np.random.seed(42)
n_samples = 100

# Features: house size (m²) and age (years)
house_size = np.random.uniform(50, 200, n_samples)  # 50-200 m²
house_age = np.random.uniform(0, 30, n_samples)

# True relationship: price = 2000 * size - 1500 * age + 150000 + noise (in euros)
true_price = 2000 * house_size - 1500 * house_age + 150000
noise = np.random.normal(0, 20000, n_samples)
price = true_price + noise

# Prepare data for multiple linear regression
X = np.column_stack([house_size, house_age])
y = price

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Print model coefficients
print("=== Multiple Linear Regression Results ===")
print(f"Intercept (β₀): {model.intercept_:.2f}")
print(f"Coefficient for size (β₁): {model.coef_[0]:.2f}")
print(f"Coefficient for age (β₂): {model.coef_[1]:.2f}")
print()

# Evaluation metrics
print("=== Training Set Performance ===")
print(f"R² Score: {r2_score(y_train, y_train_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_train_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_train, y_train_pred):.2f}")
print()

print("=== Test Set Performance ===")
print(f"R² Score: {r2_score(y_test, y_test_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_test_pred)):.2f}")
print(f"MAE: {mean_absolute_error(y_test, y_test_pred):.2f}")
print()

# Example prediction
new_house = np.array([[120, 5]])  # 120 m², 5 years old
predicted_price = model.predict(new_house)
print(f"Predicted price for 120 m², 5-year-old house: {predicted_price[0]:,.2f} €")

# Visualization (for simple case with one feature)
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, 0], y_train, alpha=0.6, label='Training data')
plt.scatter(X_test[:, 0], y_test, alpha=0.6, label='Test data', color='orange')

# Plot regression line (using only size feature for visualization)
size_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
# For visualization, use average age
avg_age = X[:, 1].mean()
pred_line = model.predict(np.column_stack([size_range, np.full(100, avg_age)]))
plt.plot(size_range, pred_line, 'r-', linewidth=2, label='Regression line')

plt.xlabel('House Size (m²)')
plt.ylabel('Price (€)')
plt.title('Linear Regression: House Price Prediction')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
`,
  children: [],
};

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

The absolute value in Lasso makes the optimization problem non-differentiable, requiring specialized algorithms like coordinate descent or least angle regression (LARS).

Minimization of Error

Ridge Regression Solution:

Taking the derivative and setting to zero gives us the Ridge solution:

$$\\frac{\\partial L_{\\text{Ridge}}}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) + 2\\lambda\\boldsymbol{\\beta} = 0$$

Solving for $\\boldsymbol{\\beta}$:

$$\\hat{\\boldsymbol{\\beta}}_{\\text{Ridge}} = (\\mathbf{X}^T\\mathbf{X} + \\lambda\\mathbf{I})^{-1}\\mathbf{X}^T\\mathbf{y}$$

The addition of $\\lambda\\mathbf{I}$ to $\\mathbf{X}^T\\mathbf{X}$ ensures the matrix is always invertible, even when $p > n$ or when features are perfectly correlated.

Lasso Regression Solution:

Lasso has no closed-form solution due to the absolute value term. It's solved using iterative algorithms:

1. Coordinate Descent: Updates one coefficient at a time

2. Least Angle Regression (LARS): Efficiently computes the entire regularization path

3. Proximal Gradient Methods: Uses soft-thresholding operator

The soft-thresholding operator for Lasso is:

$$\\beta_j = \\text{sign}(z_j) \\max(|z_j| - \\lambda, 0)$$

where $z_j$ is the unregularized coefficient estimate.

Regularization Parameter Selection

The regularization parameter $\\lambda$ is crucial and must be selected via cross-validation:

- $\\lambda = 0$: No regularization, equivalent to ordinary least squares

- Small $\\lambda$: Light regularization, coefficients slightly shrunk

- Large $\\lambda$: Strong regularization, coefficients heavily shrunk (Ridge) or set to zero (Lasso)

- $\\lambda \\to \\infty$: All coefficients approach zero (Ridge) or become exactly zero (Lasso)

Cross-Validation Process:

1. Define a grid of $\\lambda$ values

2. For each $\\lambda$, perform k-fold cross-validation

3. Select $\\lambda$ with best cross-validation performance

4. Refit model on full training set with selected $\\lambda$

Model Evaluation

Ridge and Lasso use the same evaluation metrics as linear regression (see Linear Regression for R², RMSE, MAE). Additional considerations:

Regularization Path: Plotting coefficients vs. $\\lambda$ shows how features are selected/shrunk. Lasso's path shows clear feature selection (coefficients hitting zero), while Ridge shows gradual shrinkage.

Feature Selection (Lasso): Lasso automatically performs feature selection by setting irrelevant coefficients to zero. This is particularly useful for high-dimensional problems.

Multicollinearity Handling: Ridge regression handles multicollinearity well by shrinking correlated features together, while Lasso tends to select one feature from a correlated group.`,
  
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

export const decisionTreeRegressor: ConceptNode = {
  id: 'decision-tree-regressor',
  title: 'Decision Tree Regressor',
  description: 'Non-parametric tree-based model that partitions feature space into regions',
  color: "bg-gradient-to-br from-orange-500 to-red-600",
  overview: `Decision tree regression is a non-parametric method that recursively partitions the feature space into rectangular regions and predicts a constant value (typically the mean) within each region. Unlike linear models that assume a global linear relationship, decision trees can capture complex, non-linear, and interaction effects without requiring feature transformations.

The model builds a tree structure where:

- Internal nodes represent decision rules based on feature values (e.g., "temperature < 20°C")

- Branches represent outcomes of decisions (e.g., "yes" or "no")

- Leaf nodes contain the predicted value (mean of target values in that region)

Each path from root to leaf represents a series of conditions that define a region in feature space. The prediction for a new observation is the average target value of training samples in the corresponding leaf node.`,
  
  howItWorks: `The Goal

The primary goal of decision tree regression is to partition the feature space into regions where the target variable is relatively homogeneous, then predict the average value within each region. This creates a piecewise constant function that can approximate complex non-linear relationships.

Splitting Criterion

Decision trees use a splitting criterion to determine the best feature and threshold at each node. For regression, the most common criterion is Mean Squared Error (MSE) reduction:

At each node, we find the split that minimizes:

$$\\text{MSE}_{\\text{split}} = \\frac{n_{\\text{left}}}{n} \\text{MSE}_{\\text{left}} + \\frac{n_{\\text{right}}}{n} \\text{MSE}_{\\text{right}}$$

where:

$$\\text{MSE}_{\\text{left}} = \\frac{1}{n_{\\text{left}}} \\sum_{i \\in \\text{left}} (y_i - \\bar{y}_{\\text{left}})^2$$

$$\\text{MSE}_{\\text{right}} = \\frac{1}{n_{\\text{right}}} \\sum_{i \\in \\text{right}} (y_i - \\bar{y}_{\\text{right}})^2$$

The split that maximizes the reduction in MSE is chosen:

$$\\Delta \\text{MSE} = \\text{MSE}_{\\text{parent}} - \\text{MSE}_{\\text{split}}$$

Alternative criteria include:

- Mean Absolute Error (MAE): More robust to outliers

- Friedman's MSE: Adjusts for the number of samples in each child

Tree Construction Algorithm

The tree is built recursively using a greedy algorithm:

1. Start with all training data at the root node

2. For each node:

   - If stopping criterion met → create leaf node with mean of target values

   - Otherwise:

     - Try all possible splits (feature + threshold combinations)

     - Select split with maximum MSE reduction

     - Create left and right child nodes

     - Recursively apply to child nodes

3. Stopping criteria (prevent overfitting):

   - Maximum tree depth

   - Minimum samples per leaf

   - Minimum samples to split

   - Minimum MSE reduction threshold

Prediction

For a new observation $\\mathbf{x}$:

1. Start at root node

2. Follow path based on feature conditions

3. Reach leaf node

4. Predict: $\\hat{y} = \\bar{y}_{\\text{leaf}} = \\frac{1}{n_{\\text{leaf}}} \\sum_{i \\in \\text{leaf}} y_i$

Cost Functions

The training process minimizes the total MSE across all leaf nodes:

$$L = \\sum_{m=1}^{M} \\sum_{i \\in R_m} (y_i - \\bar{y}_m)^2$$

where $M$ is the number of leaf nodes and $R_m$ is the $m$-th region (leaf).

However, the actual optimization is done greedily at each split, not globally.

Regularization and Pruning

Decision trees are prone to overfitting. Common regularization techniques:

Pre-pruning (Early Stopping):

- max_depth: Maximum depth of tree

- min_samples_split: Minimum samples required to split

- min_samples_leaf: Minimum samples in leaf nodes

- min_impurity_decrease: Minimum decrease in impurity to split

Post-pruning:

- Build full tree, then prune branches that don't improve validation performance

- Uses cost-complexity pruning: $\\text{Cost} = \\text{MSE} + \\alpha |T|$ where $|T|$ is number of leaves

Model Evaluation

Decision trees use standard regression metrics (see Linear Regression for R², RMSE, MAE). Additional considerations:

Bias-Variance Trade-off:

- Deep trees: Low bias, high variance (overfitting)

- Shallow trees: High bias, low variance (underfitting)

Feature Importance: Decision trees provide feature importance scores based on how much each feature reduces MSE across all splits:

$$\\text{Importance}_j = \\frac{1}{N} \\sum_{t} p(t) \\Delta_j(t)$$

where $p(t)$ is the proportion of samples reaching node $t$ and $\\Delta_j(t)$ is the MSE reduction from feature $j$ at node $t$.

  Interpretability: Decision trees are highly interpretable - you can visualize the entire decision process as a flowchart.`,
  
  applications: [
    'Real estate valuation with categorical and numerical features',
    'Medical diagnosis scoring with interpretable rules',
    'Financial risk assessment requiring explainable decisions',
    'Customer segmentation and targeting',
    'Quality control in manufacturing',
    'Environmental modeling with non-linear relationships'
  ],
  
  advantages: [
    'Non-parametric: no assumptions about data distribution',
    'Handles non-linear relationships and interactions automatically',
    'Works with both numerical and categorical features',
    'Highly interpretable: can visualize decision process'
  ],
  
  limitations: [
    'Prone to overfitting, especially with deep trees',
    'Unstable: small data changes can create very different trees',
    'Biased toward features with many levels or high cardinality',
    'Cannot extrapolate beyond training data range'
  ],
  
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Generate sample data: predicting apartment rent based on features
np.random.seed(42)
n_samples = 300

# Features: size (m²), distance to city center (km), floor number, has_parking (0/1)
size = np.random.uniform(30, 120, n_samples)
distance = np.random.uniform(0.5, 15, n_samples)
floor = np.random.randint(0, 10, n_samples)
has_parking = np.random.randint(0, 2, n_samples)

# Non-linear relationship with interactions
base_rent = 500 + 15 * size - 20 * distance + 10 * floor + 100 * has_parking
# Interaction: larger apartments further from center have different pricing
interaction = 2 * size * (distance > 5)
# Non-linearity: premium for mid-range sizes
size_premium = 50 * np.sin((size - 75) / 20)
rent = base_rent + interaction + size_premium + np.random.randn(n_samples) * 50

# Prepare data
X = np.column_stack([size, distance, floor, has_parking])
y = rent

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Try different tree depths
depths = [1, 3, 5, 10, 20, None]  # None = no limit
results = {}

for depth in depths:
    tree = DecisionTreeRegressor(max_depth=depth, random_state=42)
    tree.fit(X_train, y_train)
    
    y_train_pred = tree.predict(X_train)
    y_test_pred = tree.predict(X_test)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(tree, X_train, y_train, cv=5, scoring='r2')
    
    results[depth] = {
        'tree': tree,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    depth_str = 'Unlimited' if depth is None else str(depth)
    print(f"\\n=== Max Depth: {depth_str} ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f} €")
    print(f"CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"Number of leaves: {tree.tree_.n_leaves}")

# Select best model (depth=5 in this case)
best_depth = 5
best_tree = results[best_depth]['tree']

# Feature importance
feature_names = ['Size (m²)', 'Distance (km)', 'Floor', 'Has Parking']
importances = best_tree.feature_importances_

print("\\n=== Feature Importance ===")
for name, importance in zip(feature_names, importances):
    print(f"{name}: {importance:.4f}")

# Visualization
plt.figure(figsize=(10, 6))
y_test_pred = best_tree.predict(X_test)
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Rent (€)')
plt.ylabel('Predicted Rent (€)')
plt.title('Decision Tree: Predictions vs. Actual Values')
plt.grid(True, alpha=0.3)
plt.show()

# Example prediction
new_apartment = np.array([[65, 3.5, 4, 1]])  # 65 m², 3.5 km, 4th floor, has parking
predicted_rent = best_tree.predict(new_apartment)
print(f"\\nExample prediction for new apartment:")
print(f"  Size: 65 m², Distance: 3.5 km, Floor: 4, Parking: Yes")
print(f"  Predicted rent: {predicted_rent[0]:.2f} €")
`,
  children: []
};

export const knnRegressor: ConceptNode = {
  id: 'knn-regressor',
  title: 'K-Nearest Neighbors Regressor',
  description: 'Instance-based non-parametric regression using local averaging',
  color: "bg-gradient-to-br from-amber-500 to-orange-600",
  overview: `K-Nearest Neighbors (KNN) regression is a non-parametric, instance-based learning method that makes predictions by averaging the target values of the k most similar training examples. Unlike parametric methods that learn a fixed model, KNN is a "lazy learner" that stores all training data and computes predictions on-the-fly.

The fundamental assumption is that similar inputs should have similar outputs. KNN finds the k training examples closest to a query point in feature space and predicts their average target value. The method is highly flexible and can adapt to complex, non-linear relationships without making strong assumptions about the underlying function.`,
  
  howItWorks: `The Goal

The primary goal of KNN regression is to make predictions based on local patterns in the training data. For each new observation, we find its k nearest neighbors in feature space and use their target values to make a prediction. This creates a locally adaptive, non-parametric regression surface that can capture complex relationships.

Distance Metrics

KNN requires a distance metric to measure similarity between observations. Common choices:

- Euclidean Distance (most common):

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{m=1}^p (x_{im} - x_{jm})^2} = ||\\mathbf{x}_i - \\mathbf{x}_j||_2$$

- Manhattan Distance (L1 norm):

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{m=1}^p |x_{im} - x_{jm}| = ||\\mathbf{x}_i - \\mathbf{x}_j||_1$$

- Minkowski Distance (generalization):

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\left(\\sum_{m=1}^p |x_{im} - x_{jm}|^q\\right)^{1/q}$$

where $q=2$ gives Euclidean and $q=1$ gives Manhattan.

- Weighted Distances: Some implementations use inverse distance weighting, giving more influence to closer neighbors.

Prediction

For a query point $\\mathbf{x}_0$, KNN regression:

1. Find k nearest neighbors: $\\mathcal{N}_k(\\mathbf{x}_0) = \\{\\mathbf{x}_{(1)}, \\mathbf{x}_{(2)}, \\ldots, \\mathbf{x}_{(k)}\\}$

2. Predict using uniform averaging:

$$\\hat{y}_0 = \\frac{1}{k} \\sum_{i=1}^k y_{(i)}$$

where $y_{(i)}$ is the target value of the $i$-th nearest neighbor.

Weighted KNN (distance-weighted):

$$\\hat{y}_0 = \\frac{\\sum_{i=1}^k w_i y_{(i)}}{\\sum_{i=1}^k w_i}$$

where weights are typically:

$$w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_{(i)})^p}$$

with $p$ controlling the influence of distance (common values: 1 or 2).

Cost Functions

KNN doesn't have an explicit cost function to minimize during training (it's a lazy learner). However, we can think of the prediction error:

$$\\text{MSE} = \\frac{1}{n} \\sum_{i=1}^n \\left(y_i - \\frac{1}{k} \\sum_{j \\in \\mathcal{N}_k(\\mathbf{x}_i)} y_j\\right)^2$$

The "training" process is simply storing the training data. The actual computation happens during prediction.

Hyperparameter Selection

The main hyperparameter is k (number of neighbors), which controls the bias-variance trade-off:

- Small k (k=1): Low bias, high variance (overfitting, sensitive to noise)

- Large k: High bias, low variance (underfitting, smooth predictions)

- Optimal k: Balances bias and variance, typically found via cross-validation

Rule of thumb: $k = \\sqrt{n}$ where $n$ is the number of training samples, but cross-validation is preferred.

Model Evaluation

KNN uses standard regression metrics (see Linear Regression for R², RMSE, MAE). Important considerations:

Computational Complexity:

- Training: $O(1)$ - just store data

- Prediction: $O(n \\cdot p)$ - must compute distances to all training points

- With optimizations (KD-trees, Ball trees): $O(\\log n \\cdot p)$ average case

Curse of Dimensionality: As dimensionality increases, all points become approximately equidistant, making KNN ineffective. This is why feature selection and dimensionality reduction are crucial for high-dimensional data.

Local vs. Global: KNN is excellent at capturing local patterns but may struggle with global trends or when the relationship varies significantly across feature space.`,
  
  applications: [
    'Recommendation systems based on user/item similarity',
    'Image processing with pixel-level predictions',
    'Collaborative filtering in e-commerce',
    'Spatial interpolation in geostatistics',
    'Time series imputation using similar historical patterns',
    'Anomaly detection by comparing to nearest neighbors'
  ],
  
  advantages: [
    'Simple and intuitive: easy to understand and implement',
    'Non-parametric: no assumptions about data distribution or relationship form',
    'Adapts to local patterns: can model complex non-linear relationships',
    'No training phase: just store data, predictions computed on-demand'
  ],
  
  limitations: [
    'Computationally expensive for prediction: must compute distances to all training points',
    'Sensitive to irrelevant features: all features contribute equally to distance',
    'Poor performance in high dimensions (curse of dimensionality)',
    'Requires feature scaling: distance metrics are sensitive to feature scales'
  ],
  
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split, cross_val_score

# Generate sample data: predicting bicycle rental demand
# Based on temperature, humidity, wind speed, and day of week
np.random.seed(42)
n_samples = 400

# Features
temperature = np.random.uniform(5, 35, n_samples)  # Celsius
humidity = np.random.uniform(30, 90, n_samples)  # Percentage
wind_speed = np.random.uniform(0, 25, n_samples)  # km/h
day_of_week = np.random.randint(0, 7, n_samples)  # 0=Monday, 6=Sunday

# Non-linear relationship: demand peaks at moderate temperatures
# Lower on weekends, decreases with wind and high humidity
base_demand = 100
temp_effect = 50 * np.exp(-((temperature - 20)**2) / 100)  # Peak at 20°C
weekend_penalty = -30 * (day_of_week >= 5)  # Lower on weekends
wind_penalty = -2 * wind_speed
humidity_penalty = -0.5 * (humidity - 50)

demand = base_demand + temp_effect + weekend_penalty + wind_penalty + humidity_penalty
demand += np.random.randn(n_samples) * 10
demand = np.maximum(demand, 0)  # No negative demand

# Prepare data
X = np.column_stack([temperature, humidity, wind_speed, day_of_week])
y = demand

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for distance-based methods)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different k values
k_values = [1, 3, 5, 10, 20, 50, 100]
results = {}

for k in k_values:
    knn = KNeighborsRegressor(n_neighbors=k, weights='uniform')
    knn.fit(X_train_scaled, y_train)
    
    y_train_pred = knn.predict(X_train_scaled)
    y_test_pred = knn.predict(X_test_scaled)
    
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    
    # Cross-validation
    cv_scores = cross_val_score(knn, X_train_scaled, y_train, cv=5, scoring='r2')
    
    results[k] = {
        'knn': knn,
        'train_r2': train_r2,
        'test_r2': test_r2,
        'test_rmse': test_rmse,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std()
    }
    
    print(f"\\n=== k = {k} ===")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.2f} rentals")
    print(f"CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Select best k
best_k = k_values[np.argmax([results[k]['cv_mean'] for k in k_values])]
best_knn = results[best_k]['knn']

print(f"\\nBest k: {best_k}")

# Compare uniform vs. distance-weighted
knn_uniform = KNeighborsRegressor(n_neighbors=best_k, weights='uniform')
knn_distance = KNeighborsRegressor(n_neighbors=best_k, weights='distance')

knn_uniform.fit(X_train_scaled, y_train)
knn_distance.fit(X_train_scaled, y_train)

pred_uniform = knn_uniform.predict(X_test_scaled)
pred_distance = knn_distance.predict(X_test_scaled)

print(f"\\nUniform weights - Test R²: {r2_score(y_test, pred_uniform):.4f}")
print(f"Distance weights - Test R²: {r2_score(y_test, pred_distance):.4f}")

# Visualization
plt.figure(figsize=(10, 6))
y_test_pred = best_knn.predict(X_test_scaled)
plt.scatter(y_test, y_test_pred, alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('Actual Demand (rentals)')
plt.ylabel('Predicted Demand (rentals)')
plt.title('KNN: Predictions vs. Actual Values')
plt.grid(True, alpha=0.3)
plt.show()

# Example prediction
new_day = np.array([[22, 60, 8, 3]])  # 22°C, 60% humidity, 8 km/h wind, Thursday
new_day_scaled = scaler.transform(new_day)
predicted_demand = best_knn.predict(new_day_scaled)
print(f"\\nExample prediction:")
print(f"  Temperature: 22°C, Humidity: 60%, Wind: 8 km/h, Day: Thursday")
print(f"  Predicted demand: {predicted_demand[0]:.2f} rentals")
`,
  children: []
};

