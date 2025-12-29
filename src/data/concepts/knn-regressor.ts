import { ConceptNode } from '../types';

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

with $p$ controlling the influence of distance (common values: 1 or 2).`,
  
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
