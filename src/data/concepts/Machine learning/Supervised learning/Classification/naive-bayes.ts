import { ConceptNode } from '../../../../types';

export const naiveBayes: ConceptNode = {
  id: 'naive-bayes',
  title: 'Naive Bayes',
  description: 'Probabilistic classifier based on Bayes theorem with feature independence assumption',
  color: "bg-gradient-to-br from-fuchsia-400 to-violet-500",
  overview: `Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem with the "naive" assumption that features are conditionally independent given the class. Despite this simplifying assumption, Naive Bayes often performs remarkably well in practice, especially for text classification and high-dimensional problems.

Bayes' theorem states:

$$P(y|x_1, \\ldots, x_p) = \\frac{P(x_1, \\ldots, x_p|y) P(y)}{P(x_1, \\ldots, x_p)}$$

The naive assumption allows us to factor the likelihood:

$$P(x_1, \\ldots, x_p|y) = \\prod_{j=1}^p P(x_j|y)$$

This simplifies the posterior probability to:

$$P(y|x_1, \\ldots, x_p) \\propto P(y) \\prod_{j=1}^p P(x_j|y)$$

We classify by choosing the class with the highest posterior probability.`,
  
  howItWorks: `Naive Bayes is fundamentally a probabilistic classifier grounded in Bayes' theorem, which provides a principled framework for computing class probabilities from feature observations. The method's surprising effectiveness despite its strong simplifying assumptions makes it both theoretically interesting and practically valuable.

Bayes' theorem relates the posterior probability (class given features) to the likelihood and prior:

$$P(y|\\mathbf{x}_1, \\ldots, x_p) = \\frac{P(\\mathbf{x}_1, \\ldots, x_p|y) P(y)}{P(\\mathbf{x}_1, \\ldots, x_p)}$$

Computing the joint likelihood $P(\\mathbf{x}_1, \\ldots, x_p|y)$ directly is intractable in high dimensions due to the exponential number of feature value combinations. The "naive" assumption—that features are conditionally independent given the class—dramatically simplifies this. Under this assumption:

$$P(\\mathbf{x}_1, \\ldots, x_p|y) = \\prod_{j=1}^p P(x_j|y)$$

This factorization reduces the posterior to:

$$P(y|\\mathbf{x}_1, \\ldots, x_p) \\propto P(y) \\prod_{j=1}^p P(x_j|y)$$

Notably, the denominator $P(\\mathbf{x}_1, \\ldots, x_p)$ is constant across classes and can be omitted for classification since we only need to identify the class with highest posterior probability.

Naive Bayes requires estimating the prior class probability $P(y=k)$ and the likelihood $P(x_j|y=k)$ for each feature-class combination. The prior is simply the empirical class frequency in training data: $P(y=k) = n_k / n$, where $n_k$ is the count of class $k$ samples.

The likelihood estimation depends on feature type. For continuous features, Gaussian Naive Bayes assumes each feature follows a normal distribution within each class:

$$P(x_j|y=k) = \\frac{1}{\\sqrt{2\\pi\\sigma_{jk}^2}} \\exp\\left(-\\frac{(x_j - \\mu_{jk})^2}{2\\sigma_{jk}^2}\\right)$$

where $\\mu_{jk}$ and $\\sigma_{jk}^2$ are estimated as sample mean and variance of feature $j$ in class $k$.

For count data (e.g., word frequencies in text classification), Multinomial Naive Bayes models feature counts with smoothing to handle unseen feature values:

$$P(x_j|y=k) = \\frac{N_{jk} + \\alpha}{N_k + \\alpha p}$$

Here, $N_{jk}$ is the total count of feature $j$ in class $k$, $N_k$ is the total count across all features in class $k$, $p$ is the number of features, and $\\alpha$ is a smoothing parameter (typically 1) that prevents zero probabilities for unobserved feature-class combinations.

For binary features, Bernoulli Naive Bayes assumes each feature is present or absent:

$$P(x_j=1|y=k) = \\frac{N_{jk} + \\alpha}{N_k + 2\\alpha}$$

Given a new observation $\\mathbf{x}$, classification proceeds by computing the posterior probability for each class and selecting the maximum. In practice, we work with log-probabilities to avoid numerical underflow when multiplying many small probability values:

$$\\log P(y=k|\\mathbf{x}) = \\log P(y=k) + \\sum_{j=1}^p \\log P(x_j|y=k)$$

The predicted class is then:

$$\\hat{y} = \\arg\\max_k \\log P(y=k|\\mathbf{x})$$

Despite the strong independence assumption—which is rarely true in real data—Naive Bayes often achieves excellent generalization. This paradox arises because classification only requires ranking probabilities correctly, not estimating them accurately. When features are conditionally independent or when decision boundaries are approximately linear, Naive Bayes is nearly optimal. Even when independence is violated, the method frequently produces reasonable classifications because the errors partially cancel.

One important caveat is that Naive Bayes probabilities are often poorly calibrated: predicted probabilities may not accurately reflect true posterior probabilities. This occurs because the model makes strong assumptions that are violated in practice. When well-calibrated probabilities are needed, post-processing techniques such as Platt scaling or isotonic regression can be applied to transform raw Naive Bayes outputs.`,
  
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
  
  codeExample: `python
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
print(f"  Predicted class: {'Spam' if prediction == 1 else 'Ham'}")`,
  
  children: []
};
