import { ConceptNode } from '../types';

export const decisionTreeRegressor: ConceptNode = {
  id: 'decision-tree-regressor',
  title: 'Decision Tree Regressor',
  description: 'Non-parametric tree-based model that partitions feature space into regions',
  color: "bg-gradient-to-br from-emerald-400 to-teal-500",
  overview: `Decision tree regression represents a fundamentally different approach to predictive modeling compared to the parametric methods of linear and polynomial regression. Rather than assuming a global functional form relating predictors to the response, decision trees recursively partition the feature space into rectangular regions and predict a constant value within each region. This non-parametric methodology enables the model to capture complex non-linear relationships, high-order interactions, and threshold effects without requiring explicit feature transformations or knowledge of the underlying relationship structure.

The tree structure consists of three types of nodes: internal nodes, which contain decision rules (e.g., "feature $x_1 < 25$?"), branches, which represent the outcomes of these decisions, and leaf nodes, which contain the predicted values (typically the mean of target values for all training samples that reached that leaf). Each path from the root node to a leaf represents a series of conditions that partitions a region of the feature space. When predicting for a new observation, the model traverses the tree from root to leaf by evaluating the feature conditions at each internal node, ultimately arriving at a leaf node whose associated constant serves as the prediction. This transparent decision pathway is a key strength of tree-based models: practitioners can understand precisely how the model arrived at a particular prediction, making trees ideal for applications where interpretability is paramount.`,
  
  howItWorks: `The fundamental principle underlying decision tree construction is recursive partitioning: we successively divide the feature space into smaller and smaller regions with the goal of creating regions where the target variable exhibits low variance. Within each region, we predict the mean of observed target values, creating a piecewise constant function that approximates the underlying relationship.

The core algorithmic question at each node is: which feature and threshold should we use to split the data? To answer this, decision trees employ a splitting criterion that quantifies the homogeneity of the target variable in the resulting regions. For regression tasks, Mean Squared Error (MSE) reduction serves as the standard criterion. At each node, we evaluate all possible splits (all features paired with all unique thresholds) and select the split that minimizes the weighted average MSE of the resulting child nodes.

Specifically, if a split divides the data into left and right subsets, the MSE after the split is:

$$\\text{MSE}_{\\text{split}} = \\frac{n_{\\text{left}}}{n} \\text{MSE}_{\\text{left}} + \\frac{n_{\\text{right}}}{n} \\text{MSE}_{\\text{right}}$$

where the MSE within each child is computed as:

$$\\text{MSE}_{\\text{left}} = \\frac{1}{n_{\\text{left}}} \\sum_{i \\in \\text{left}} (y_i - \\bar{y}_{\\text{left}})^2$$

$$\\text{MSE}_{\\text{right}} = \\frac{1}{n_{\\text{right}}} \\sum_{i \\in \\text{right}} (y_i - \\bar{y}_{\\text{right}})^2$$

The quality of a split is measured by the reduction in MSE compared to the parent node:

$$\\Delta \\text{MSE} = \\text{MSE}_{\\text{parent}} - \\text{MSE}_{\\text{split}}$$

The split that maximizes this reduction is selected because it produces the greatest decrease in overall prediction error. Alternative splitting criteria exist for specialized purposes: Mean Absolute Error (MAE) is more robust to outliers than MSE, and Friedman's MSE variant adjusts the criterion to account for imbalances in the sizes of child nodes, reducing bias toward features that naturally produce unbalanced splits.

The tree is constructed recursively using a greedy algorithm. We begin with all training data at the root node. At each node, we evaluate whether to split further or create a leaf node (stopping criterion). If stopping criteria are not met, the algorithm searches through all possible splits, selects the one with maximum MSE reduction, partitions the data accordingly, and recursively applies the same process to the left and right child nodes. This greedy approach is computationally efficient, though it does not guarantee a globally optimal tree.

Stopping criteria prevent the tree from growing indefinitely and overfitting the training data. Common stopping rules include maximum tree depth (limiting the number of levels), minimum samples per leaf (ensuring each leaf contains a minimum number of observations), minimum samples to split (avoiding splits on tiny subsets), and minimum MSE reduction threshold (avoiding splits that yield negligible improvements). These hyperparameters control the complexity of the resulting model and directly influence the bias-variance tradeoff: shallow trees introduce bias but exhibit low variance and generalize well, while deep trees minimize training error but often overfit and perform poorly on new data.

For prediction, when a new observation $\\mathbf{x}$ arrives, we traverse the tree from the root by evaluating the feature conditions at each internal node, following the appropriate branch at each step until reaching a leaf node. The prediction is then the mean of all training target values in that leaf:

$$\\hat{y} = \\bar{y}_{\\text{leaf}} = \\frac{1}{n_{\\text{leaf}}} \\sum_{i \\in \\text{leaf}} y_i$$

The training objective, viewed globally, is to minimize the total squared error across all leaf nodes:

$$L = \\sum_{m=1}^{M} \\sum_{i \\in R_m} (y_i - \\bar{y}_m)^2$$

where $M$ is the number of leaf nodes and $R_m$ represents the $m$-th region. However, the actual optimization process is greedy: we optimize each split locally without considering global optimality.

Decision trees are notorious for overfitting, particularly when grown to full depth. Two complementary approaches address this problem. Pre-pruning (or early stopping) prevents overfitting during tree construction by enforcing hyperparameters such as maximum depth, minimum samples to split, and minimum samples in each leaf. Alternatively, post-pruning grows a full tree on training data, then removes branches that fail to improve performance on a separate validation set. Cost-complexity pruning, a systematic post-pruning approach, introduces a complexity penalty:

$$\\text{Cost}(T) = \\text{MSE}(T) + \\alpha |T|$$

where $|T|$ is the number of leaf nodes and $\\alpha$ controls the trade-off between training fit and tree complexity. By varying $\\alpha$, we obtain a sequence of candidate trees, and we select the tree that optimizes validation performance.

Decision trees provide an interpretable ranking of feature importance based on their collective contribution to reducing MSE across all splits in the tree:

$$\\text{Importance}_j = \\frac{1}{N} \\sum_{t} p(t) \\Delta_j(t)$$

where $p(t)$ is the proportion of samples reaching node $t$ (the relative frequency of that node in predictions), and $\\Delta_j(t)$ is the MSE reduction from splitting on feature $j$ at node $t$. Features that consistently appear in splits and produce substantial MSE reductions receive high importance scores. This ranking guides feature selection and identifies the most influential predictors in the model.`,
  
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
