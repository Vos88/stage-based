import { ConceptNode } from '../types';

export const polynomialRegression: ConceptNode = {
  id: 'polynomial-regression',
  title: 'Polynomial Regression',
  description: 'Fitting polynomial relationships between variables to capture non-linear patterns',
  color: "bg-gradient-to-br from-teal-500 to-cyan-600",
  overview: `Polynomial regression extends linear regression by modeling non-linear relationships between variables. While linear regression assumes a straight-line relationship between the predictor $x$ and the response $y$, polynomial regression can capture curves, bends, and more complex patterns. This flexibility makes it particularly useful for modeling real-world phenomena such as growth rates, physical laws, economic trends, and dose-response relationships.

The key idea is to transform the original features into polynomial features, allowing us to fit curves instead of just lines. Despite its ability to model non-linear relationships, polynomial regression remains linear in terms of its parameters, which means we can still use familiar techniques like Ordinary Least Squares (OLS) to estimate the coefficients efficiently.

A polynomial regression model of degree $d$ is given by:

$$y = \\beta_0 + \\beta_1 x + \\beta_2 x^2 + \\beta_3 x^3 + \\cdots + \\beta_d x^d + \\varepsilon$$

where $(\\beta_0, \\beta_1, \\ldots, \\beta_d)$ are the coefficients to be estimated, and $\\varepsilon$ represents the error term. To express polynomial regression in matrix form, we construct a polynomial feature matrix $\\mathbf{X}_{\\text{poly}}$:

$$\\mathbf{X}_{\\text{poly}} = 
\\begin{bmatrix} 1 & x_1 & x_1^2 & \\cdots & x_1^d \\\\\ 
1 & x_2 & x_2^2 & \\cdots & x_2^d \\\\\ 
\\vdots & \\vdots & \\vdots & \\ddots & \\vdots \\\\ 
1 & x_n & x_n^2 & \\cdots & x_n^d 
\\end{bmatrix}$$

Each row corresponds to an observation, and each column represents a polynomial term ($x^0, x^1, x^2, \\ldots, x^d$). This transformation allows us to treat polynomial regression as a special case of multiple linear regression.

The cost function is the Residual Sum of Squares (RSS):

$$\\text{RSS} = \\sum_{i=1}^n (y_i - \\hat{y}_i)^2 = (\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})^T(\\mathbf{y} - \\mathbf{X}_{\\text{poly}}\\boldsymbol{\\beta})$$

The Ordinary Least Squares (OLS) solution minimizes this cost:

$$\\hat{\\boldsymbol{\\beta}} = (\\mathbf{X}_{\\text{poly}}^T\\mathbf{X}_{\\text{poly}})^{-1}\\mathbf{X}_{\\text{poly}}^T\\mathbf{y}$$

Selecting the optimal degree $d$ is critical and requires careful consideration:

1. Cross-validation: Compare models with different degrees using k-fold cross-validation
2. Visualization: Plot the fitted curve against the data to assess fit quality
3. Bias-Variance Tradeoff: 
   1. Lower-degree polynomials: high bias, low variance (underfitting)
   2. Higher-degree polynomials: low bias, high variance (overfitting)
4. Information Criteria: Use AIC or BIC to balance model complexity and fit

Some closing practical considerations:

1. Feature scaling is helpful, especially for higher-degree polynomials, to improve numerical stability
2. Regularization techniques (Ridge or Lasso) can be applied to polynomial features to mitigate overfitting
3. Watch for overfitting: a curve that follows noise rather than true patterns is a sign of excessive model complexity
4. Multicollinearity increases with polynomial degree (e.g., $x$ and $x^2$ are inherently correlated), which can be addressed with regularization
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
