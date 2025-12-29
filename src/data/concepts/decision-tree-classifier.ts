import { ConceptNode } from '../types';

export const decisionTreeClassifier: ConceptNode = {
  id: 'decision-tree-classifier',
  title: 'Decision Tree Classifier',
  description: 'Tree-based classification model that partitions feature space using decision rules',
  color: "bg-gradient-to-br from-orange-500 to-amber-600",
  overview: `Decision tree classification represents a non-parametric approach to learning discrete decision boundaries through recursive partitioning of the feature space. Unlike linear classifiers that assume a fixed functional form, decision trees adaptively construct a series of hierarchical decision rules by examining feature values. The resulting model is a tree structure where each internal node represents a decision rule (e.g., "feature $x_1 < 25$?"), branches represent outcomes of these decisions, and leaf nodes contain class label predictions. 

The intuition is appealingly straightforward: we recursively partition the feature space into increasingly homogeneous regions until each region is dominated by a single class. When presented with a new observation, we traverse the tree from root to leaf by answering the feature conditions at each internal node, ultimately arriving at a leaf whose associated class label serves as the prediction. This transparent, interpretable decision pathway makes decision trees particularly valuable in applications where stakeholders must understand the reasoning behind predictions.`,
  
  howItWorks: `The fundamental principle underlying decision tree classification is recursive partitioning: we successively divide the feature space into smaller, increasingly pure regions until each region is dominated by a single class. The core algorithmic question at each node is therefore: which feature and threshold should we use to split the data to maximize class purity in the resulting regions?

To answer this question, decision trees employ splitting criteria that quantify the impurity (heterogeneity) of class distributions. Gini impurity is the most commonly used criterion in classification settings. It measures the probability of incorrectly classifying a randomly selected element if we classified it according to the class distribution of the node:

$$\\text{Gini}(t) = 1 - \\sum_{k=1}^K p_k^2(t)$$

where $p_k(t)$ denotes the proportion of class $k$ samples at node $t$, and $K$ is the number of classes. Gini ranges from 0 (pure node, all samples belong to one class) to values approaching 1 (maximally impure, samples distributed equally across classes). An alternative impurity measure is entropy, which captures information-theoretic uncertainty:

$$\\text{Entropy}(t) = -\\sum_{k=1}^K p_k(t) \\log_2 p_k(t)$$

When we partition a node's data through a split, we create child nodes with (ideally) lower impurity than the parent. The quality of a split is quantified by the reduction in impurity:

$$\\text{Information Gain} = \\text{Impurity}(\\text{parent}) - \\sum_{\\text{child}} \\frac{n_{\\text{child}}}{n_{\\text{parent}}} \\text{Impurity}(\\text{child})$$

The split that maximizes information gain is selected at each node. This greedy approach is computationally efficient, though it does not guarantee globally optimal trees.

The tree is constructed recursively starting from all training data at the root node. At each node, the algorithm evaluates all possible splits (all features paired with all unique thresholds) and selects the one with maximum information gain. This creates two child nodes, and the process repeats recursively on each child until a stopping criterion is met. Common stopping criteria include maximum tree depth (limiting complexity), minimum samples per node (ensuring sufficient data for reliability), and minimum information gain threshold (avoiding negligible improvements).

For prediction, when a new observation $\\mathbf{x}$ arrives, we traverse from root to leaf by evaluating feature conditions, ultimately reaching a leaf node. The predicted class is the most frequent class among training samples in that leaf:

$$\\hat{y} = \\arg\\max_k n_k(\\text{leaf})$$

where $n_k(\\text{leaf})$ is the count of class $k$ samples in the leaf.

A natural byproduct of tree construction is feature importance: a measure of each feature's collective contribution to reducing impurity across all splits. Features that frequently appear in splits and produce substantial impurity reductions receive high importance scores. This ranking provides valuable insight into which predictors most strongly influence the model's predictions.

Decision trees are notorious for overfitting, particularly when grown to full depth. The model's high flexibility enables zero training error, but leads to poor generalization. Regularization through depth constraints and minimum sample requirements (pre-pruning) prevents overfitting during construction. Alternatively, post-pruning grows a full tree then removes branches that fail to improve validation performance, using cost-complexity analysis to identify optimal pruning levels.`,
  
  applications: [
    'Medical diagnosis with interpretable decision rules',
    'Credit approval requiring explainable decisions',
    'Customer segmentation based on behavior patterns',
    'Rule extraction for expert systems'
  ],
  
  advantages: [
    'Highly interpretable decision process',
    'Handles both numerical and categorical features',
    'No need for feature scaling',
    'Can capture non-linear relationships and interactions'
  ],
  
  limitations: [
    'Prone to overfitting, especially with deep trees',
    'Unstable: small data changes create very different trees',
    'Biased toward features with many levels',
    'Cannot extrapolate beyond training data range'
  ],
  
  codeExample: `python
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sample data: customer churn prediction
np.random.seed(42)
n_samples = 300

# Features: contract length (months), monthly fee (€), support calls, satisfaction score
contract_length = np.random.uniform(1, 24, n_samples)
monthly_fee = np.random.uniform(20, 80, n_samples)
support_calls = np.random.poisson(2, n_samples)
satisfaction = np.random.uniform(1, 10, n_samples)

# Churn logic: higher fee, more calls, lower satisfaction → churn
churn_score = -0.1 * contract_length + 0.05 * monthly_fee + 0.3 * support_calls - 0.2 * satisfaction
churn = (churn_score > np.median(churn_score)).astype(int)

# Prepare data
X = np.column_stack([contract_length, monthly_fee, support_calls, satisfaction])
y = churn

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_test_pred = model.predict(X_test)

# Print results
print("=== Decision Tree Classifier Results ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['No Churn', 'Churn']))

# Example prediction
new_customer = np.array([[12, 50, 3, 6]])  # 12 months, €50, 3 calls, satisfaction 6
prediction = model.predict(new_customer)[0]
prob = model.predict_proba(new_customer)[0]
print(f"\\nExample prediction for new customer:")
print(f"  Probability: No Churn={prob[0]:.4f}, Churn={prob[1]:.4f}")
print(f"  Predicted: {'Churn' if prediction == 1 else 'No Churn'}")`,
  
  children: []
};
