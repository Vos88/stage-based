import { ConceptNode } from '../../../../types';

export const linearRegression: ConceptNode = {
  id: 'linear-regression',
  title: 'Linear Regression',
  color: "bg-gradient-to-br from-teal-400 to-emerald-500",
  description: 'A fundamental machine learning method for modeling relationships between variables by fitting a linear equation to observed data.',
  overview:
   `Linear regression stands as one of the most fundamental and widely-used machine learning algorithms, serving as the foundation for much of statistical learning theory and practice. The core objective is elegantly simple: to model the relationship between a dependent variable (the target we wish to predict) and one or more independent variables (the features we observe) by fitting a linear equation to empirical data.

In the simplest setting, we begin with a single predictor variable. The relationship between a predictor $x$ and a response variable $y$ is modeled as:

$$y = \\beta_0 + \\beta_1 x + \\varepsilon$$

Here, $\\beta_0$ represents the intercept (the predicted value when $x = 0$), $\\beta_1$ represents the slope (the expected change in $y$ for each unit increase in $x$), and $\\varepsilon$ denotes the error term, capturing all the variability in $y$ that cannot be explained by $x$ alone. This simple linear model provides immediate interpretability: the slope coefficient directly quantifies the strength and direction of the relationship between predictor and response.

In practice, real-world problems rarely involve a single predictor. Multiple linear regression extends the framework to accommodate $p$ distinct features, modeling the conditional expectation of $y$ as a linear combination of all predictors:

$$y = \\beta_0 + \\beta_1 x_1 + \\beta_2 x_2 + \\cdots + \\beta_p x_p + \\varepsilon$$

To facilitate computational and theoretical analysis, we adopt matrix notation. With $n$ observations and $p$ features, we express the model compactly as:

$$\\mathbf{y} = \\mathbf{X}\\boldsymbol{\\beta} + \\boldsymbol{\\varepsilon}$$

where $\\mathbf{y}$ is an $n \\times 1$ vector of response values, $\\mathbf{X}$ is an $n \\times (p+1)$ design matrix whose first column contains ones (representing the intercept) and whose remaining columns contain the observed feature values, $\\boldsymbol{\\beta}$ is a $(p+1) \\times 1$ vector of coefficients to be estimated, and $\\boldsymbol{\\varepsilon}$ is an $n \\times 1$ vector of errors. This formulation unifies simple and multiple regression under a single mathematical framework and enables efficient computation of parameter estimates.`,
  
  howItWorks: `The fundamental challenge in linear regression is to determine the coefficient vector $\\boldsymbol{\\beta}$ that yields predictions closest to the observed data. We seek the best-fitting line (or hyperplane in higher dimensions) in a precise mathematical sense: the line that minimizes a measure of the overall prediction error.

The most widely adopted measure of prediction error is the Mean Squared Error (MSE), which quantifies the average squared deviation between predicted and observed values:

$$\\text{MSE} = \\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2$$

where $\\hat{y}_i = \\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_p x_{ip}$ denotes the predicted value for the $i$-th observation. The squaring of errors serves two purposes: it penalizes large deviations more heavily than small ones, and it makes the function smooth and differentiable, enabling calculus-based optimization.

We can equivalently minimize the Residual Sum of Squares (RSS), which omits the $1/n$ scaling:

$$\\text{RSS} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = \\sum_{i=1}^n \\left(y_i - (\\beta_0 + \\beta_1 x_{i1} + \\cdots + \\beta_p x_{ip})\\right)^2$$

In compact matrix form, RSS becomes:

$$\\text{RSS} = (\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta})$$

To find the coefficient vector that minimizes RSS, we employ the calculus technique of setting the gradient equal to zero. Taking the partial derivative with respect to $\\boldsymbol{\\beta}$ and equating it to zero yields the Normal Equations:

$$\\frac{\\partial \\text{RSS}}{\\partial \\boldsymbol{\\beta}} = -2\\mathbf{X}^T(\\mathbf{y} - \\mathbf{X}\\boldsymbol{\\beta}) = 0$$

Rearranging and solving for $\\boldsymbol{\\beta}$ produces the Ordinary Least Squares (OLS) estimator, a closed-form solution that can be computed directly:

$$\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{y}$$

This remarkable result—that we can solve for the optimal coefficients analytically rather than through iterative optimization—is one of the key advantages of linear regression. For simple linear regression with a single feature, the closed-form formulas become more explicit. The slope coefficient is given by the ratio of the covariance between $x$ and $y$ to the variance of $x$:

$$\\beta_1 = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^n (x_i - \\bar{x})^2} = \\frac{\\text{Cov}(x, y)}{\\text{Var}(x)}$$

And the intercept is determined by the requirement that the fitted line passes through the point $(\\bar{x}, \\bar{y})$:

$$\\beta_0 = \\bar{y} - \\beta_1 \\bar{x}$$

where $\\bar{x}$ and $\\bar{y}$ denote the sample means. These formulas reveal an important principle: the estimated slope depends on the strength of the linear association between variables, while the intercept is determined by the marginal mean of the response.

Once we have fitted the model to training data, we must evaluate its predictive performance. Several complementary metrics serve this purpose, each offering different insights. The coefficient of determination, commonly denoted $R^2$, measures the proportion of variance in the response that is explained by the linear model:

$$R^2 = 1 - \\frac{\\text{RSS}}{\\text{TSS}} = 1 - \\frac{\\sum_{i=1}^n (y_i - \\hat{y}_i)^2}{\\sum_{i=1}^n (y_i - \\bar{y})^2}$$

where TSS denotes the Total Sum of Squares, a measure of the total variance in the response variable. The $R^2$ statistic ranges from 0 to 1: a value of 0 indicates that the model explains none of the variance (predictions equal the mean), while 1 indicates perfect prediction. In practice, $R^2$ values between 0.5 and 0.9 are typical for real-world problems.

A subtle weakness of $R^2$ is that it increases monotonically as we add more predictors to the model, even if those predictors are irrelevant. Adjusted $R^2$ addresses this by penalizing model complexity:

$$R^2_{\\text{adj}} = 1 - \\frac{(1-R^2)(n-1)}{n-p-1}$$

This adjustment favors more parsimonious models and provides a fairer basis for comparing models with different numbers of features.

Beyond $R^2$, we often report the Root Mean Squared Error (RMSE), which measures the typical magnitude of prediction errors in the original units of the target variable:

$$\\text{RMSE} = \\sqrt{\\frac{1}{n}\\sum_{i=1}^n (y_i - \\hat{y}_i)^2} = \\sqrt{\\text{MSE}}$$

RMSE is particularly useful for practitioners because it is interpretable in the same units as the response variable. A related metric is the Mean Absolute Error (MAE), which computes the average absolute deviation:

$$\\text{MAE} = \\frac{1}{n}\\sum_{i=1}^n |y_i - \\hat{y}_i|$$

Unlike RMSE, MAE is less sensitive to outliers, making it valuable when extreme errors occur in the data. Together, these metrics provide a comprehensive picture of model performance: $R^2$ and adjusted $R^2$ capture overall explanatory power, while RMSE and MAE quantify the magnitude of typical errors.`,
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
