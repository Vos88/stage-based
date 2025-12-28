import { ConceptNode } from '../types';

export const logisticRegression: ConceptNode = {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  description: 'Statistical method for binary and multiclass classification using the logistic function',
  color: "bg-gradient-to-br from-indigo-500 to-blue-600",
  overview: `Logistic regression is a fundamental classification algorithm that models the probability of class membership using the logistic (sigmoid) function. Unlike linear regression which predicts continuous values, logistic regression predicts probabilities and classifies observations based on these probabilities.

For binary classification, the model predicts the probability $P(y=1|x)$ using:

$$P(y=1|x) = \\frac{1}{1 + e^{-(\\beta_0 + \\beta_1 x_1 + \\cdots + \\beta_p x_p)}} = \\sigma(\\mathbf{x}^T\\boldsymbol{\\beta})$$

where $\\sigma(z) = \\frac{1}{1 + e^{-z}}$ is the logistic (sigmoid) function that transforms the linear combination into a probability between 0 and 1.

The log-odds (logit) form is:

$$\\log\\left(\\frac{P(y=1|x)}{1-P(y=1|x)}\\right) = \\beta_0 + \\beta_1 x_1 + \\cdots + \\beta_p x_p = \\mathbf{x}^T\\boldsymbol{\\beta}$$

For multiclass classification, we use the softmax function to extend logistic regression to multiple classes.`,
  
  howItWorks: `The Goal

The primary goal of logistic regression is to model the probability of class membership and make classification decisions based on these probabilities. We want to estimate the coefficients $\\boldsymbol{\\beta}$ that best separate the classes in feature space.

Cost Functions

Logistic regression uses the log-likelihood (or equivalently, negative log-likelihood) as the cost function. For binary classification with $n$ observations:

$$L(\\boldsymbol{\\beta}) = -\\sum_{i=1}^n \\left[y_i \\log(p_i) + (1-y_i) \\log(1-p_i)\\right]$$

where $p_i = P(y_i=1|x_i) = \\sigma(\\mathbf{x}_i^T\\boldsymbol{\\beta})$ is the predicted probability for observation $i$.

This is also called the cross-entropy loss or logistic loss. The negative sign makes it a minimization problem.

Minimization of Error

Unlike linear regression, logistic regression has no closed-form solution. We use iterative optimization methods:

Gradient Descent: Updates coefficients using the gradient of the log-likelihood:

$$\\frac{\\partial L}{\\partial \\beta_j} = -\\sum_{i=1}^n (y_i - p_i)x_{ij}$$

$$\\beta_j^{(t+1)} = \\beta_j^{(t)} - \\alpha \\frac{\\partial L}{\\partial \\beta_j}$$

where $\\alpha$ is the learning rate.

Newton-Raphson Method (also called Iteratively Reweighted Least Squares): Uses second-order information for faster convergence:

$$\\boldsymbol{\\beta}^{(t+1)} = \\boldsymbol{\\beta}^{(t)} - (\\mathbf{H}^{(t)})^{-1} \\nabla L^{(t)}$$

where $\\mathbf{H}$ is the Hessian matrix of second derivatives.

Decision Boundary

The decision boundary occurs where $P(y=1|x) = 0.5$, which corresponds to:

$$\\mathbf{x}^T\\boldsymbol{\\beta} = 0$$

This is a linear decision boundary (hyperplane in multiple dimensions). Observations with $\\mathbf{x}^T\\boldsymbol{\\beta} > 0$ are classified as class 1, and those with $\\mathbf{x}^T\\boldsymbol{\\beta} < 0$ are classified as class 0.

Model Evaluation

Logistic regression uses classification-specific metrics (see evaluation section for details on accuracy, precision, recall, F1-score, and ROC-AUC). Key considerations:

Probability Calibration: Logistic regression outputs well-calibrated probabilities, meaning predicted probabilities accurately reflect true class probabilities.

Coefficient Interpretation: Coefficients represent the change in log-odds per unit change in the feature. The odds ratio is $e^{\\beta_j}$, representing the multiplicative change in odds.`,
  
  applications: [
    'Medical diagnosis: predicting disease presence from patient symptoms and test results',
    'Credit scoring: assessing loan default risk based on financial history',
    'Marketing: predicting customer response to campaigns',
    'Quality control: classifying products as defective or acceptable'
  ],
  
  advantages: [
    'Provides probabilistic output, not just class predictions',
    'No hyperparameters to tune (unlike many ML methods)',
    'Less prone to overfitting than complex models',
    'Fast and computationally efficient for training and prediction'
  ],
  
  limitations: [
    'Assumes linear relationship between features and log-odds',
    'Sensitive to outliers, which can distort the decision boundary',
    'Requires large sample sizes for stable coefficient estimates',
    'Cannot capture non-linear decision boundaries without feature engineering'
  ],
  
  codeExample: 
`python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate sample data: loan default prediction
np.random.seed(42)
n_samples = 500

# Features: income (€/month), debt ratio, credit history (years), employment status (0=unemployed, 1=employed)
income = np.random.uniform(1500, 6000, n_samples)
debt_ratio = np.random.uniform(0.1, 0.8, n_samples)
credit_history = np.random.uniform(0, 20, n_samples)
employment = np.random.randint(0, 2, n_samples)

# True relationship: higher income, lower debt, longer credit history, employment reduce default risk
log_odds = -3 + 0.001 * income - 2 * debt_ratio + 0.1 * credit_history + 1.5 * employment
prob_default = 1 / (1 + np.exp(-log_odds))
default = (np.random.rand(n_samples) < prob_default).astype(int)

# Prepare data
X = np.column_stack([income, debt_ratio, credit_history, employment])
y = default

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Standardize features (important for logistic regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)
y_test_proba = model.predict_proba(X_test_scaled)[:, 1]

# Print results
print("=== Logistic Regression Results ===")
print(f"Intercept: {model.intercept_[0]:.4f}")
print("\\n=== Test Set Performance ===")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_proba):.4f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_test_pred)
print(f"\\nConfusion Matrix:")
print(f"  True Negatives: {cm[0,0]}, False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}, True Positives: {cm[1,1]}")

# Visualization
plt.figure(figsize=(8, 6))
plt.hist(y_test_proba[y_test == 0], bins=20, alpha=0.6, label='No Default', color='green')
plt.hist(y_test_proba[y_test == 1], bins=20, alpha=0.6, label='Default', color='red')
plt.xlabel('Predicted Probability of Default')
plt.ylabel('Frequency')
plt.title('Probability Distribution by Class')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Example prediction
new_applicant = np.array([[3500, 0.4, 5, 1]])  # €3500/month, 40% debt, 5 years credit, employed
new_applicant_scaled = scaler.transform(new_applicant)
prob = model.predict_proba(new_applicant_scaled)[0, 1]
prediction = model.predict(new_applicant_scaled)[0]
print(f"\\nExample prediction for new applicant:")
print(f"  Probability of default: {prob:.4f}")
print(f"  Predicted class: {'Default' if prediction == 1 else 'No Default'}")
`,
  children: []
};

export const naiveBayes: ConceptNode = {
  id: 'naive-bayes',
  title: 'Naive Bayes',
  description: 'Probabilistic classifier based on Bayes theorem with feature independence assumption',
  color: "bg-gradient-to-br from-green-500 to-emerald-600",
  overview: `Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class. Despite this simplifying assumption, Naive Bayes often performs remarkably well in practice, especially for text classification and high-dimensional problems.

Bayes' theorem states:

$$P(y|x_1, \\ldots, x_p) = \\frac{P(x_1, \\ldots, x_p|y) P(y)}{P(x_1, \\ldots, x_p)}$$

The naive assumption allows us to factor the likelihood:

$$P(x_1, \\ldots, x_p|y) = \\prod_{j=1}^p P(x_j|y)$$

This simplifies the posterior probability to:

$$P(y|x_1, \\ldots, x_p) \\propto P(y) \\prod_{j=1}^p P(x_j|y)$$

We classify by choosing the class with the highest posterior probability.`,
  
  howItWorks: `The Goal

The primary goal of Naive Bayes is to estimate class probabilities using Bayes' theorem, making the simplifying assumption that features are independent given the class. This allows efficient probability estimation even with many features.

Probability Estimation

For each class $k$ and feature $j$, we estimate:

Prior probability: $P(y=k)$ from class frequencies in training data

Likelihood: $P(x_j|y=k)$ depends on feature type:

- Gaussian Naive Bayes: For continuous features, assumes normal distribution:

  $$P(x_j|y=k) = \\frac{1}{\\sqrt{2\\pi\\sigma_{jk}^2}} \\exp\\left(-\\frac{(x_j - \\mu_{jk})^2}{2\\sigma_{jk}^2}\\right)$$

  where $\\mu_{jk}$ and $\\sigma_{jk}^2$ are estimated from training data.

- Multinomial Naive Bayes: For count data (e.g., word counts in text):

  $$P(x_j|y=k) = \\frac{N_{jk} + \\alpha}{N_k + \\alpha p}$$

  where $N_{jk}$ is count of feature $j$ in class $k$, $N_k$ is total count in class $k$, and $\\alpha$ is smoothing parameter.

- Bernoulli Naive Bayes: For binary features:

  $$P(x_j=1|y=k) = \\frac{N_{jk} + \\alpha}{N_k + 2\\alpha}$$

Prediction

For a new observation $\\mathbf{x}$, we compute the posterior probability for each class:

$$P(y=k|\\mathbf{x}) \\propto P(y=k) \\prod_{j=1}^p P(x_j|y=k)$$

We predict the class with the highest posterior probability:

$$\\hat{y} = \\arg\\max_k P(y=k|\\mathbf{x})$$

In practice, we work with log-probabilities to avoid numerical underflow:

$$\\log P(y=k|\\mathbf{x}) = \\log P(y=k) + \\sum_{j=1}^p \\log P(x_j|y=k)$$

Model Evaluation

Naive Bayes uses standard classification metrics (see evaluation section). Key considerations:

Calibration: Naive Bayes probabilities are often poorly calibrated but can be improved with Platt scaling or isotonic regression.

Feature Independence: The independence assumption is rarely true in practice, but the method often works well regardless, especially when features are conditionally independent or when the decision boundary is approximately linear.`,
  
  applications: [
    'Spam email filtering based on word frequencies',
    'Text classification: sentiment analysis, topic classification',
    'Medical diagnosis with multiple symptoms',
    'Document categorization in information retrieval'
  ],
  
  advantages: [
    'Fast training and prediction, even with many features',
    'Works well with small datasets',
    'Handles multiple classes naturally',
    'Simple to implement and interpret'
  ],
  
  limitations: [
    'Strong independence assumption rarely holds in practice',
    'Can be outperformed by more sophisticated methods',
    'Sensitive to skewed data distributions',
    'Poor probability calibration without post-processing'
  ],
  
  codeExample: 
`python
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Generate sample data: email classification (spam vs. ham)
np.random.seed(42)
n_samples = 400

# Features: word count ratios (normalized)
# Spam emails have higher "urgent", "free", "click" words
# Ham emails have higher "meeting", "project", "team" words
urgent_ratio = np.random.beta(2, 5, n_samples)
free_ratio = np.random.beta(2, 5, n_samples)
click_ratio = np.random.beta(2, 5, n_samples)
meeting_ratio = np.random.beta(5, 2, n_samples)
project_ratio = np.random.beta(5, 2, n_samples)

# Create labels: spam (1) or ham (0)
# Higher urgent/free/click → spam, higher meeting/project → ham
spam_score = urgent_ratio + free_ratio + click_ratio - meeting_ratio - project_ratio
labels = (spam_score > np.median(spam_score)).astype(int)

# Add some noise
labels = np.where(np.random.rand(n_samples) < 0.1, 1 - labels, labels)

# Prepare data
X = np.column_stack([urgent_ratio, free_ratio, click_ratio, meeting_ratio, project_ratio])
y = labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Train model
model = GaussianNB()
model.fit(X_train, y_train)

# Predictions
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
y_test_proba = model.predict_proba(X_test)

# Print results
print("=== Naive Bayes Results ===")
print(f"Test Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print("\\nClassification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Ham', 'Spam']))

# Example prediction
new_email = np.array([[0.3, 0.25, 0.2, 0.1, 0.05]])  # High urgent/free/click words
prob = model.predict_proba(new_email)[0]
prediction = model.predict(new_email)[0]
print(f"\\nExample prediction for new email:")
print(f"  Probability: Ham={prob[0]:.4f}, Spam={prob[1]:.4f}")
print(f"  Predicted class: {'Spam' if prediction == 1 else 'Ham'}")
`,
  children: []
};

export const decisionTreeClassifier: ConceptNode = {
  id: 'decision-tree-classifier',
  title: 'Decision Tree Classifier',
  description: 'Tree-based classification model that partitions feature space using decision rules',
  color: "bg-gradient-to-br from-orange-500 to-amber-600",
  overview: `Decision tree classification recursively partitions the feature space into regions and assigns a class label to each region. The model builds a tree structure where internal nodes represent decision rules based on feature values, branches represent outcomes, and leaf nodes contain class predictions.

Each path from root to leaf represents a series of conditions that define a region in feature space. The prediction for a new observation is the majority class of training samples in the corresponding leaf node.`,
  
  howItWorks: `The Goal

The primary goal of decision tree classification is to partition the feature space into regions where one class dominates, then predict the majority class within each region. This creates a piecewise constant decision boundary.

Splitting Criterion

Decision trees use impurity measures to determine the best split. Common criteria:

Gini Impurity:
$$Gini(t) = 1 - \\sum_{k=1}^K p_k^2(t)$$
where $p_k(t)$ is the proportion of class $k$ samples in node $t$.

Entropy:
$$Entropy(t) = -\\sum_{k=1}^K p_k(t) \\log_2 p_k(t)$$

Information Gain: The reduction in impurity from splitting:
$$IG = Impurity(parent) - \\sum_{child} \\frac{n_{child}}{n_{parent}} Impurity(child)$$

The split that maximizes information gain is chosen.

Tree Construction

The tree is built recursively:

1. Start with all training data at root

2. For each node, try all possible splits (feature + threshold)

3. Select split with maximum information gain

4. Create child nodes and recurse

5. Stop when stopping criteria met (max depth, min samples, etc.)

Prediction

For a new observation, follow the tree from root to leaf based on feature conditions, then predict the majority class in that leaf.

Model Evaluation

Decision trees use standard classification metrics. Key considerations:

Feature Importance: Based on total information gain from each feature across all splits.

Interpretability: Highly interpretable - can visualize the entire decision process as a flowchart.`,
  
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
  
  codeExample: 
`python
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
print(f"  Predicted: {'Churn' if prediction == 1 else 'No Churn'}")
`,
  children: []
};

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
  
  howItWorks: `The Goal

The primary goal of SVM is to find the optimal separating hyperplane that maximizes the margin between classes while minimizing classification errors. This leads to good generalization performance.

Optimization Problem

For hard-margin SVM (linearly separable):

$$\\min_{\\mathbf{w}, b} \\frac{1}{2}||\\mathbf{w}||^2$$

subject to: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1$ for all $i$

For soft-margin SVM (non-separable):

$$\\min_{\\mathbf{w}, b, \\xi} \\frac{1}{2}||\\mathbf{w}||^2 + C\\sum_{i=1}^n \\xi_i$$

subject to: $y_i(\\mathbf{w}^T\\mathbf{x}_i + b) \\geq 1 - \\xi_i$, $\\xi_i \\geq 0$

where $C$ controls the trade-off between margin size and classification errors, and $\\xi_i$ are slack variables.

Kernel Trick

For non-linear decision boundaries, we map features to higher dimensions using a kernel function:

$$K(\\mathbf{x}_i, \\mathbf{x}_j) = \\phi(\\mathbf{x}_i)^T\\phi(\\mathbf{x}_j)$$

Common kernels:

- Linear: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\mathbf{x}_i^T\\mathbf{x}_j$

- Polynomial: $K(\\mathbf{x}_i, \\mathbf{x}_j) = (\\gamma \\mathbf{x}_i^T\\mathbf{x}_j + r)^d$

- RBF: $K(\\mathbf{x}_i, \\mathbf{x}_j) = \\exp(-\\gamma ||\\mathbf{x}_i - \\mathbf{x}_j||^2)$

Support Vectors

Only training examples on or within the margin (support vectors) affect the decision boundary. This makes SVM memory efficient.

Model Evaluation

SVM uses standard classification metrics. Key considerations:

Hyperparameter Tuning: $C$ (regularization) and kernel parameters must be tuned via cross-validation.

Scalability: Training time scales poorly with dataset size, making it unsuitable for very large datasets.`,
  
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
  
  codeExample: 
`python
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
print(f"  Predicted: {'Dog' if prediction == 1 else 'Cat'}")
`,
  children: []
};

export const knn: ConceptNode = {
  id: 'knn',
  title: 'K-Nearest Neighbors (KNN)',
  description: 'Instance-based classification using majority vote of nearest neighbors',
  color: "bg-gradient-to-br from-yellow-500 to-orange-600",
  overview: `K-Nearest Neighbors is a non-parametric, instance-based classification method that classifies observations based on the majority class among their k nearest neighbors in feature space. It's a "lazy learner" that stores all training data and computes predictions on-the-fly.

The fundamental assumption is that similar inputs should have similar outputs. KNN finds the k training examples closest to a query point and assigns the most common class among them.`,
  
  howItWorks: `The Goal

The primary goal of KNN classification is to make predictions based on local patterns in the training data. For each new observation, we find its k nearest neighbors and use their class labels to make a prediction.

Distance Metrics

KNN requires a distance metric. Common choices:

- Euclidean Distance:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sqrt{\\sum_{m=1}^p (x_{im} - x_{jm})^2}$$

- Manhattan Distance:

$$d(\\mathbf{x}_i, \\mathbf{x}_j) = \\sum_{m=1}^p |x_{im} - x_{jm}|$$

Prediction

For a query point $\\mathbf{x}_0$:

1. Find k nearest neighbors: $\\mathcal{N}_k(\\mathbf{x}_0)$

2. Predict majority class:

$$\\hat{y} = \\arg\\max_{c} \\sum_{i \\in \\mathcal{N}_k(\\mathbf{x}_0)} \\mathbb{1}(y_i = c)$$

Weighted KNN: Give more weight to closer neighbors:
$$\\hat{y} = \\arg\\max_{c} \\sum_{i \\in \\mathcal{N}_k(\\mathbf{x}_0)} w_i \\mathbb{1}(y_i = c)$$
where $w_i = \\frac{1}{d(\\mathbf{x}_0, \\mathbf{x}_i)^p}$.

Hyperparameter Selection

The main hyperparameter is k, controlling bias-variance trade-off:

- Small k: Low bias, high variance (overfitting)

- Large k: High bias, low variance (underfitting)

- Optimal k: Found via cross-validation

Model Evaluation

KNN uses standard classification metrics. Key considerations:

Computational Complexity: Prediction requires computing distances to all training points, making it slow for large datasets.

Curse of Dimensionality: Performance degrades in high dimensions as all points become approximately equidistant.`,
  
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
  
  codeExample: 
`python
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
print(f"  Predicted: {'High Quality' if prediction == 1 else 'Low Quality'}")
`,
  children: []
};
