import { ConceptNode } from '../types';

export const svm: ConceptNode = {
  id: 'svm',
  title: 'Support Vector Machine (SVM)',
  description: 'Maximum margin classifier that finds optimal separating hyperplane',
  color: "bg-gradient-to-br from-red-500 to-pink-600",
  overview: `Support Vector Machines find the optimal hyperplane that maximally separates different classes in feature space. The key insight is to maximize the margin (distance) between the decision boundary and the nearest training examples (support vectors).

For linearly separable data, the optimal hyperplane is:

$$\\mathbf{w}^T\\mathbf{x} + b = 0$$

where $\\mathbf{w}$ is the weight vector and $b$ is the bias. The margin is $\\frac{2}{||\\mathbf{w}||}$, so maximizing margin is equivalent to minimizing $||\\mathbf{w}||^2$.

For non-linearly separable data, we use the kernel trick to map data to higher dimensions where it becomes linearly separable.`,
  
  howItWorks: `Support Vector Machines are grounded in a fundamental geometric principle: the optimal decision boundary between classes is the hyperplane that maximizes the margin (distance) to the nearest training examples. This margin-maximization principle is theoretically motivated by statistical learning theory, which shows that larger margins improve generalization performance.

For linearly separable binary classification with labels $y_i \\in \\{-1, +1\\}$, the decision boundary is a hyperplane:

$$\\mathbf{w}^T\\mathbf{x} + b = 0$$

where $\\mathbf{w}$ is the normal vector to the hyperplane and $b$ is the bias term. The signed distance from point $\\mathbf{x}_i$ to this hyperplane is $\\frac{y_i(\\mathbf{w}^T\\mathbf{x}_i + b)}{\\|\\mathbf{w}\\|}$. For correct classification, this quantity is positive; the margin is the minimum signed distance across all training points, which equals $\\frac{2}{\\|\\mathbf{w}\\|}$ (derived from the constraint that correctly classified points satisfy $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$).

Maximizing the margin is equivalent to minimizing $\\|\\mathbf{w}\\|^2$. The hard-margin SVM optimization problem is:

$$\\min_{\\mathbf{w}, b} \\frac{1}{2}\\|\\mathbf{w}\\|^2$$

subject to the constraint that all training points are correctly classified with margin at least 1: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$ for all $i$.

In practice, training data is rarely linearly separable. The soft-margin SVM relaxes the hard constraints by introducing slack variables $\\xi_i \\geq 0$ that quantify the amount of constraint violation for each point:

$$\\min_{\\mathbf{w}, b, \\boldsymbol{\\xi}} \\frac{1}{2}\\|\\mathbf{w}\\|^2 + C\\sum_{i=1}^n \\xi_i$$

subject to: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1 - \\xi_i$ and $\\xi_i \\geq 0$.

The parameter $C > 0$ controls the regularization tradeoff: larger $C$ penalizes training errors heavily, seeking a small margin that classifies most points correctly, while smaller $C$ tolerates more errors in exchange for larger margins and better generalization.

Real-world data often exhibits non-linear decision boundaries that cannot be separated by any hyperplane. The kernel trick addresses this by implicitly mapping data to a higher-dimensional space where linear separation becomes possible. A kernel function computes inner products in this high-dimensional space without explicitly constructing the mapping:

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\phi(\\mathbf{x}_i)^T\\phi(\\mathbf{x}_j)$$

where $\\phi$ is the implicit feature map. The optimization problem remains identical, but uses kernel evaluations instead of inner products. Common kernels include the linear kernel $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\mathbf{x}_i^T\\mathbf{x}_j$ (no transformation), the polynomial kernel $K(\\mathbf{x}_i, \\mathbf{x}_j) = (\\gamma \\mathbf{x}_i^T\\mathbf{x}_j + r)^d$ (captures polynomial interactions), and the Radial Basis Function (RBF) kernel $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma \\|\\mathbf{x}_i - \\mathbf{x}_j\\|^2)$ (highly flexible, infinite-dimensional feature space).

A remarkable property of SVM is sparsity: the optimal solution depends only on a subset of training examples—the support vectors—which are those points on or near the margin. Points far from the decision boundary do not affect the final model, making SVM memory-efficient and interpretable in terms of which training examples matter.

A practical consideration is feature scaling. SVMs are distance-sensitive, and features with large magnitudes dominate kernel computations. Standardizing features ensures each contributes equally. Additionally, hyperparameter tuning is essential: the regularization parameter $C$ and kernel parameters (e.g., $\\gamma$ for RBF) significantly impact performance and must be selected via cross-validation.`,

  applications: [
    'Text classification with high-dimensional word features',
    'Image recognition with pixel-level features',
    'Gene classification in bioinformatics',
    'Handwriting recognition'
  ],
  
  advantages: [
    'Effective in high-dimensional spaces',
    'Memory efficient (only stores support vectors)',
    'Versatile with different kernel functions',
    'Robust to overfitting with appropriate regularization'
  ],
  
  limitations: [
    'Poor performance on large datasets (slow training)',
    'Sensitive to feature scaling',
    'No probabilistic output (requires Platt scaling)',
    'Difficult to interpret with non-linear kernels'
  ],
  
  codeExample: `python
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Generate sample data: image classification (cats vs. dogs)
# Simulating pixel features (simplified)
np.random.seed(42)
n_samples = 200

# Features: pixel intensity statistics (mean, std, skewness, kurtosis)
# Cats tend to have different texture patterns than dogs
mean_intensity = np.random.uniform(50, 200, n_samples)
std_intensity = np.random.uniform(10, 50, n_samples)
skewness = np.random.uniform(-1, 1, n_samples)
kurtosis = np.random.uniform(-1, 3, n_samples)

# Create labels: 0=cat, 1=dog
# Dogs have higher mean intensity and different texture
labels = ((mean_intensity > 125) & (std_intensity > 30)).astype(int)
labels = np.where(np.random.rand(n_samples) < 0.15, 1 - labels, labels)

# Prepare data
X = np.column_stack([mean_intensity, std_intensity, skewness, kurtosis])
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model with RBF kernel
model = SVC(kernel='rbf', random_state=42, probability=True)
model.fit(X_train_scaled, y_train)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)

# Print results
print("=== RBF SVM Results ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Cat', 'Dog']))

# Example prediction
new_image = np.array([[140, 35, 0.2, 1.5]])  # High mean, high std
new_image_scaled = scaler.transform(new_image)
prob = model.predict_proba(new_image_scaled)[0]
prediction = model.predict(new_image_scaled)[0]
print(f"\\nExample prediction for new image:")
print(f"  Probability: Cat={prob[0]:.4f}, Dog={prob[1]:.4f}")
print(f"  Predicted: {'Dog' if prediction == 1 else 'Cat'}")`,
  
  children: []
};
