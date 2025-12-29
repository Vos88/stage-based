import { ConceptNode } from '../types';

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
