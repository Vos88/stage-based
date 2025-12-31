import { ConceptNode } from '../../../../types';

export const polynomialRegression: ConceptNode = {
  id: 'polynomial-regression',
  title: 'Polynomial Regression',
  description: 'Fitting polynomial relationships between variables to capture non-linear patterns',
  color: "bg-gradient-to-br from-emerald-400 to-green-500",
  overview: `Many real-world phenomena exhibit non-linear behavior that cannot be adequately captured by a simple straight-line model. Consider the energy consumption of a building as a function of temperature: consumption is high at both extreme cold and extreme heat (due to heating and cooling requirements), creating a U-shaped curve. Linear regression, which assumes a constant change in the response for each unit change in the predictor, would fail to capture this essential feature of the underlying relationship.

Polynomial regression addresses this limitation by extending the linear regression framework to model curved relationships. The fundamental insight is deceptively simple: we can include higher-order polynomial terms of the original predictor as additional features in our regression model. Despite its ability to capture non-linear patterns in the data, polynomial regression remains linear with respect to its parameters, allowing us to leverage the efficient and well-understood Ordinary Least Squares (OLS) machinery from linear regression.

A polynomial regression model of degree $d$ models the conditional expectation of the response $y$ as a polynomial function of the predictor $x$:

$$y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + \\cdots + \\beta_d x^d + \\varepsilon$$

Here, $\\beta_0$ represents the intercept, $\\beta_1, \\beta_2, \\ldots, \\beta_d$ are the regression coefficients corresponding to each polynomial term, and $\\varepsilon$ denotes the random error term with mean zero. The degree $d$ controls the flexibility of the fitted curve: a degree-1 polynomial is simply linear regression, while higher degrees allow the curve to bend and follow more intricate patterns in the data.

The key innovation enabling polynomial regression is the feature transformation: we can reframe the problem as multiple linear regression on the transformed feature space. We construct a polynomial feature matrix $\\mathbf{X}_{\\text{poly}}$ where each row represents an observation and each column represents a polynomial term:

$$\\mathbf{X}_{\\text{poly}} = 
\\begin{bmatrix} 1 & x_1 & x_1^2 & \\cdots & x_1^d \\\\ 
1 & x_2 & x_2^2 & \\cdots & x_2^d \\\\
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
1 & x_n & x_n^2 & \\cdots & x_n^d 
\\end{bmatrix}$$

The first column consists of ones (the intercept term), while subsequent columns contain increasing powers of the original predictor variable. This transformation reduces polynomial regression to the multiple linear regression problem we have already solved.

The objective in polynomial regression is to minimize the Residual Sum of Squares (RSS), which measures the total squared distance between observed values and predictions:

$$\\text{RSS} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = (\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})$$

The Ordinary Least Squares estimator solves the normal equations by setting the gradient of RSS to zero, yielding the closed-form solution:

$$\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}_{\\text{poly}}^T\\mathbf{X}_{\\text{poly}})^{-1}\\mathbf{X}_{\\text{poly}}^T\\mathbf{y}$$

This solution is computationally efficient when $\\mathbf{X}_{\\text{poly}}^T\\mathbf{X}_{\\text{poly}}$ is well-conditioned. In practice, one must check that this matrix is invertible and not numerically ill-conditioned.

Choosing the appropriate polynomial degree $d$ is one of the most critical decisions in polynomial regression. This choice represents a fundamental tradeoff in statistical learning between bias and variance. A polynomial of degree 1 (linear regression) introduces substantial bias because it assumes a rigid linear relationship, but it exhibits low variance across different samples. Conversely, a degree-10 polynomial can fit the training data almost perfectly, minimizing bias, but at the cost of high variance: small perturbations in the training data lead to substantially different fitted curves, and the model may capture noise rather than genuine patterns.

To select an appropriate degree, practitioners typically employ several complementary strategies. Cross-validation is the most robust approach: partition the training data into $k$ folds, train models with different polynomial degrees on $k-1$ folds, evaluate on the held-out fold, and repeat for all folds. The average test error across folds provides an unbiased estimate of generalization performance. Visualization of the fitted polynomial against the data offers intuitive insight into whether the curve captures meaningful patterns or merely follows noise. Finally, information criteria such as AIC (Akaike Information Criterion) or BIC (Bayesian Information Criterion) penalize model complexity, automatically balancing fit quality against the number of parameters.

In practice, several nuances merit attention. Feature scaling becomes increasingly important for higher-degree polynomials: the values of $x^{10}$ can be many orders of magnitude larger than $x$, creating numerical instability in the normal equations and coefficient estimates. Centering and scaling the original predictor variable mitigates this issue. Additionally, multicollinearity—high correlation among predictor variables—increases naturally with polynomial degree, since $x^{k+1}$ is necessarily highly correlated with $x^k$. Regularization techniques such as Ridge regression or Lasso regression can address both multicollinearity and overfitting by constraining the magnitude of coefficients.

A particularly important pitfall in polynomial regression is extrapolation. Polynomial curves exhibit wild behavior outside the range of the training data, diverging rapidly as $x$ moves far from the observed values. One should never use a polynomial regression model to make predictions far beyond the observed data range. Outliers pose another challenge: a single extreme observation can substantially distort the polynomial fit, especially for high-degree polynomials. Robust regression or careful outlier detection may be necessary.
`,

  applications: [
    'Predicting energy consumption with non-linear patterns',
    'Forecasting sales with seasonal acceleration effects',
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
