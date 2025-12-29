import { ConceptNode } from '../types';

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
