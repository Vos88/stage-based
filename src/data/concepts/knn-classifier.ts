import { ConceptNode } from '../types';

export const knn: ConceptNode = {
  id: 'knn',
  title: 'K-Nearest Neighbors (KNN)',
  description: 'Instance-based classification using majority vote of nearest neighbors',
  color: "bg-gradient-to-br from-yellow-500 to-orange-600",
  overview: `K-Nearest Neighbors (KNN) classification is a non-parametric, instance-based method that classifies observations based on the class labels of their $k$ nearest neighbors in feature space. Rather than learning a fixed, global model during training, KNN defers computation to prediction time—a "lazy learner" that simply retains all training data and applies local voting rules on-the-fly. This locality-based approach enables KNN to capture complex, adaptive decision boundaries without assuming any parametric form.

The core principle is intuitive: observations that are similar in feature space should share similar class labels. When presented with a new observation, KNN identifies the $k$ training examples that lie closest to this point according to a distance metric and assigns the most common class among these neighbors as the prediction. The choice of $k$ directly governs the bias-variance tradeoff: small $k$ yields adaptive, high-variance predictions based on very local information, while large $k$ produces smooth, high-bias predictions that reflect broader neighborhoods. This simple yet flexible framework enables KNN to handle complex, non-linear decision boundaries and adapt to local data structure.`,
  
  howItWorks: `The classification mechanism in KNN operates through distance-based neighborhood identification followed by majority-class voting. For a query point $\\mathbf{x}_0$, the algorithm computes distances to all training examples and identifies the $k$ closest neighbors. A distance metric quantifies dissimilarity in feature space; the Euclidean distance is standard, computing the straight-line distance in the $p$-dimensional space:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{m=1}^p (x_{im} - x_{jm})^2}$$

The Manhattan distance offers an alternative that is more robust to outliers:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{m=1}^p |x_{im} - x_{jm}|$$

A critical practical consideration is that distance metrics treat all features equally and are sensitive to feature scaling. Features with large natural magnitudes dominate distance calculations, potentially obscuring patterns in smaller-scale features. Standardizing features—subtracting the mean and dividing by standard deviation—ensures each feature contributes proportionally.

Given the $k$ nearest neighbors $\\mathcal{N}_k(\\mathbf{x}_0) = \\{\\mathbf{x}_{(1)}, \\mathbf{x}_{(2)}, \\ldots, \\mathbf{x}_{(k)}\\}$ ordered by distance, the basic KNN classification rule uses uniform majority voting:

$$\\hat{y} = \\arg\\max_{c} \\sum_{i=1}^k \\mathbb{1}(y_{(i)} = c)$$

This simple approach counts votes equally among all neighbors. However, this can be suboptimal when neighbors have substantially different distances to the query point. Weighted KNN assigns voting weights inversely proportional to distance, giving closer neighbors greater influence:

$$\\hat{y} = \\arg\\max_{c} \\sum_{i=1}^k w_i \\mathbb{1}(y_{(i)} = c)$$

where typical weight functions are:

$$w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_{(i)})^p}$$

The exponent $p$ controls distance sensitivity: larger values strongly favor close neighbors, while smaller values produce more balanced weights. Weighted KNN often outperforms uniform voting because it appropriately captures the intuition that very close neighbors are more informative.

The choice of $k$ is critical and directly controls the bias-variance tradeoff. When $k=1$, predictions depend solely on the single nearest neighbor, producing zero training error but extreme prediction variance—small changes in test data cause predictions to change discontinuously. As $k$ increases toward $n$ (the training set size), predictions increasingly reflect global class structure and become smoother but potentially more biased. Optimal $k$ selection requires cross-validation: partition training data into folds, fit KNN models with different $k$ values, evaluate on held-out folds, and select the $k$ maximizing generalization performance.

A subtle yet important limitation emerges in high-dimensional spaces: the curse of dimensionality. In very high dimensions, distances between most point pairs become approximately uniform, making the notion of "nearest neighbors" less informative. Most points are nearly equidistant from any given query point, weakening the local structure that KNN exploits. This suggests KNN performs best in moderate-dimensional settings and may require feature selection or dimensionality reduction for high-dimensional problems.`,
  
  applications: [
    'Recommendation systems based on user/item similarity',
    'Pattern recognition in images',
    'Outlier detection by comparing to neighbors',
    'Medical diagnosis using similar patient cases'
  ],
  
  advantages: [
    'Simple and intuitive algorithm',
    'No assumptions about data distribution',
    'Naturally handles multi-class problems',
    'Adapts to local patterns in data'
  ],
  
  limitations: [
    'Computationally expensive for prediction',
    'Sensitive to irrelevant features',
    'Poor performance with high-dimensional data',
    'Requires feature scaling for proper distance calculation'
  ],
  
  codeExample: `python
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score

# Generate sample data: wine quality classification
np.random.seed(42)
n_samples = 300

# Features: alcohol content (%), acidity, sugar content (g/L), pH
alcohol = np.random.uniform(9, 15, n_samples)
acidity = np.random.uniform(3, 8, n_samples)
sugar = np.random.uniform(1, 20, n_samples)
ph = np.random.uniform(2.8, 4.0, n_samples)

# Quality: high alcohol, moderate acidity, balanced sugar → good quality
quality_score = 0.3 * alcohol - 0.2 * acidity + 0.1 * sugar - 2 * abs(ph - 3.5)
quality = (quality_score > np.median(quality_score)).astype(int)  # 0=low, 1=high

# Prepare data
X = np.column_stack([alcohol, acidity, sugar, ph])
y = quality

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for distance-based methods)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different k values
k_values = [1, 3, 5, 10, 20]
best_k = 1
best_score = 0

for k in k_values:
    model = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    mean_score = scores.mean()
    if mean_score > best_score:
        best_score = mean_score
        best_k = k
    print(f"k={k}: CV Accuracy = {mean_score:.4f}")

# Train best model
model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

# Predictions
y_test_pred = model.predict(X_test_scaled)

# Print results
print(f"\\n=== KNN Results (k={best_k}) ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Low Quality', 'High Quality']))

# Example prediction
new_wine = np.array([[12.5, 5.0, 8, 3.4]])  # 12.5% alcohol, moderate features
new_wine_scaled = scaler.transform(new_wine)
prob = model.predict_proba(new_wine_scaled)[0]
prediction = model.predict(new_wine_scaled)[0]
print(f"\\nExample prediction for new wine:")
print(f"  Probability: Low={prob[0]:.4f}, High={prob[1]:.4f}")
print(f"  Predicted: {'High Quality' if prediction == 1 else 'Low Quality'}")`,
  
  children: []
};
