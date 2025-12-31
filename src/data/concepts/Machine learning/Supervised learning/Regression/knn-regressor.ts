import { ConceptNode } from '../../../../types';

export const knnRegressor: ConceptNode = {
  id: 'knn-regressor',
  title: 'K-Nearest Neighbors Regressor',
  description: 'Instance-based non-parametric regression using local averaging',
  color: "bg-gradient-to-br from-lime-400 to-emerald-500",
  overview: `K-Nearest Neighbors (KNN) regression represents a fundamentally different paradigm from parametric regression methods. Rather than learning a fixed functional relationship through parameter estimation, KNN is an instance-based learning algorithm that retains the entire training dataset and makes predictions by exploiting local patterns in the data. This approach is sometimes called a "lazy learner" because it defers computation until prediction time: no fitting or parameter estimation occurs during training, only during inference.

The core principle underlying KNN is elegantly simple: observations that are similar in feature space should have similar target values. Formally, similarity is quantified through a distance metric in the feature space. When presented with a new, unobserved data point, KNN identifies the $k$ training examples that lie closest to this point and generates a prediction by averaging their target values. The choice of $k$ determines the locality of the prediction: small values of $k$ yield predictions based on very nearby neighbors (high variability, low bias), while large values smooth predictions over broader neighborhoods (low variability, high bias). This locality-based approach enables KNN to capture complex, non-linear relationships and adaptive local structure without imposing any parametric assumptions on the underlying functional form.`,
  
  howItWorks: `The prediction mechanism in KNN operates through two conceptually distinct phases. First, the algorithm identifies the neighborhood: given a query point $\\mathbf{x}_0$, it computes distances to all training examples and selects the $k$ points with smallest distances. Second, it generates a prediction by aggregating the target values of these neighbors. This locality principle is the method's defining characteristic: predictions depend only on nearby observations, allowing the model to adapt flexibly to local data structure.

Central to the KNN framework is the concept of distance in the feature space. Formally, a distance metric quantifies dissimilarity between pairs of observations. The Euclidean distance is the most widely used choice, computing the straight-line distance in the $p$-dimensional feature space:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{m=1}^p (x_{im} - x_{jm})^2} = ||\\mathbf{x}_i - \\mathbf{x}_j||_2$$

This metric is intuitive and computationally efficient, but it treats all features symmetrically and is sensitive to feature scaling: a feature with large variance naturally dominates distance calculations. Alternative distance metrics accommodate different assumptions about feature relationships. The Manhattan distance (also called city-block or L1 distance) sums absolute differences:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{m=1}^p |x_{im} - x_{jm}| = ||\\mathbf{x}_i - \\mathbf{x}_j||_1$$

Manhattan distance is more robust to outliers than Euclidean distance and may be preferable for data with categorical or ordinal features. The Minkowski distance generalizes both as a family:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\left(\\sum_{m=1}^p |x_{im} - x_{jm}|^q\\right)^{1/q}$$

where $q=2$ recovers Euclidean distance and $q=1$ recovers Manhattan distance. By varying $q$, practitioners can tune the metric's behavior. Other specialized metrics—cosine similarity for text data, Hamming distance for categorical data—exist for specific problem domains, but Euclidean and Manhattan distances remain standard choices.

Given a query point $\\mathbf{x}_0$, the prediction phase proceeds as follows. The algorithm first identifies the set of $k$ nearest neighbors $\\mathcal{N}_k(\\mathbf{x}_0) = \\{\\mathbf{x}_{(1)}, \\mathbf{x}_{(2)}, \\ldots, \\mathbf{x}_{(k)}\\}$, where indices denote ordering by distance (closest first). The simplest prediction rule uses uniform averaging:

$$\\hat{y}_0 = \\frac{1}{k} \\sum_{i=1}^k y_{(i)}$$

where $y_{(i)}$ is the target value of the $i$-th nearest neighbor. This approach treats all selected neighbors equally regardless of their distance to the query point. In practice, this assumption is often suboptimal: intuitively, closer neighbors should exert greater influence on the prediction than distant neighbors.

Distance-weighted KNN addresses this limitation by assigning weights inversely proportional to distance:

$$\\hat{y}_0 = \\frac{\\sum_{i=1}^k w_i y_{(i)}}{\\sum_{i=1}^k w_i}$$

where typical weight functions include:

$$w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_{(i)})^p}$$

The exponent $p$ controls distance sensitivity: larger $p$ values give much more influence to the closest neighbors, while $p=1$ provides a moderate weighting scheme. Distance-weighted KNN often outperforms uniform averaging because it appropriately captures the intuition that very close neighbors are more informative than merely nearby neighbors.

The choice of $k$ is critical and directly governs the bias-variance tradeoff. When $k=1$, predictions are determined by the single nearest neighbor, yielding a model with zero training error but extreme variance—small perturbations in training data or test query locations cause predictions to change discontinuously. Conversely, as $k$ increases toward $n$ (the training set size), predictions increasingly reflect global data structure and become smoother but potentially more biased. Optimal $k$ selection typically requires cross-validation: partition training data into folds, fit KNN models with different $k$ values on $k-1$ folds, evaluate on the held-out fold, and select the $k$ maximizing generalization performance.

A critical practical consideration is feature scaling. Since distance metrics treat all features equally, features with larger natural magnitudes dominate distance calculations. Standardizing features—subtracting the mean and dividing by standard deviation—ensures each feature contributes proportionally to distance. Without standardization, a feature measured in thousands can obscure subtle patterns in features measured in single digits, leading to severely degraded performance.

The KNN framework also highlights a subtle but important assumption: that high-dimensional Euclidean distance meaningfully quantifies similarity. However, in very high-dimensional spaces, distances between points become increasingly uniform, a phenomenon known as the curse of dimensionality. Most points in high dimensions are approximately equidistant from any given query point, making the nearest neighbor distinction less informative. This suggests KNN performs best in moderate-dimensional settings and may require feature selection or dimensionality reduction techniques in high-dimensional problems.`,
  
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
