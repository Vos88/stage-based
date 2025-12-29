import { ConceptNode } from '../types';

export const logisticRegression: ConceptNode = {
  id: 'logistic-regression',
  title: 'Logistic Regression',
  description: 'Probabilistic linear classifier for binary and multiclass outcomes using the logistic/softmax link.',
  color: "bg-gradient-to-br from-purple-400 to-violet-500",
  overview: `Logistic regression models the conditional probability of categorical outcomes as a function of predictors via a logit (binary) or softmax (multiclass) link. For binary labels $y \\in \\{0,1\\}$ and feature vector $x \\in R^p$ the canonical form is:

$$P(y=1\\mid x) = \\sigma(x^T\\beta)\\quad\\text{where }\\sigma(z)=\\frac{1}{1+e^{-z}}.$$ 

Equivalently, the log-odds (logit) is linear in x:

$$\operatorname{logit}\\,P(y=1\\mid x)=\\log\\frac{P(y=1\\mid x)}{1-P(y=1\\mid x)}=x^T\\beta.$$ 

For K>2 classes the model generalises via the softmax mapping:

$$P(y=k\\mid x)=\\frac{e^{x^T\\beta^{(k)}}}{\\sum_{l=1}^K e^{x^T\\beta^{(l)}}},\\qquad k=1,\\dots,K,$$

with an identifiability constraint (for example, $(\\beta^{(K)}=0))$.`,
  
  howItWorks: `Logistic regression models class probabilities through a probabilistic framework that combines a linear predictor with a sigmoid link function. This ensures predictions remain valid probabilities bounded in $[0, 1]$.

For binary classification with labels $y \\in \\{0, 1\\}$ and feature vector $\\mathbf{x} \\in \\mathbb{R}^p$, logistic regression specifies the conditional probability of class 1 via the sigmoid (logistic) function:

$$P(y=1|\\mathbf{x}) = \\sigma(\\mathbf{x}^T\\boldsymbol{\\beta}) \\quad\\text{where }\\sigma(z) = \\frac{1}{1+e^{-z}}$$

This formulation has a remarkable interpretation: the log-odds (logit) is linear in the features:

$$\\operatorname{logit}[P(y=1|\\mathbf{x})] = \\log\\frac{P(y=1|\\mathbf{x})}{1-P(y=1|\\mathbf{x})} = \\mathbf{x}^T\\boldsymbol{\\beta}$$

For multiclass problems with $K > 2$ classes, the model generalizes through the softmax mapping, which assigns probabilities to each class in a way that sums to one:

$$P(y=k|\\mathbf{x}) = \\frac{e^{\\mathbf{x}^T\\boldsymbol{\\beta}^{(k)}}}{\\sum_{l=1}^K e^{\\mathbf{x}^T\\boldsymbol{\\beta}^{(l)}}}, \\quad k = 1, \\ldots, K$$

with an identifiability constraint (e.g., $\\boldsymbol{\\beta}^{(K)} = \\mathbf{0}$).

Given $n$ i.i.d. observations $\\{(\\mathbf{x}_i, y_i)\\}_{i=1}^n$, we estimate coefficients by minimizing the cross-entropy loss (negative log-likelihood):

$$L(\\boldsymbol{\\beta}) = -\\sum_{i=1}^n\\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big], \\quad p_i = \\sigma(\\mathbf{x}_i^T\\boldsymbol{\\beta})$$

This convex objective has a unique global minimum, ensuring reliable optimization. The gradient of the loss, computed in vectorized form, is elegant. Define $\\mathbf{X} \\in \\mathbb{R}^{n \\times (p+1)}$ as the design matrix with a column of ones for the intercept, $\\mathbf{p} = \\sigma(\\mathbf{X}\\boldsymbol{\\beta})$ as the vector of predicted probabilities, and $\\mathbf{y}$ as the label vector. The gradient becomes:

$$\\nabla L(\\boldsymbol{\\beta}) = \\mathbf{X}^T(\\mathbf{p} - \\mathbf{y})$$

The Hessian matrix, which encodes curvature, is:

$$\\mathbf{H}(\\boldsymbol{\\beta}) = \\mathbf{X}^T\\mathbf{WX}, \\quad \\mathbf{W} = \\operatorname{diag}(p_i(1-p_i))$$

Gradient descent provides a straightforward optimization approach. At each iteration, we update coefficients in the direction that decreases loss:

$$\\boldsymbol{\\beta}^{(t+1)} = \\boldsymbol{\\beta}^{(t)} - \\alpha \\mathbf{X}^T(\\mathbf{p}^{(t)} - \\mathbf{y})$$

where $\\alpha > 0$ is the step size (learning rate). This method is simple but may require many iterations to converge.

Newton–Raphson and Iteratively Reweighted Least Squares (IRLS) employ second-order information to accelerate convergence. The Newton update leverages the Hessian:

$$\\boldsymbol{\\beta}^{(t+1)} = \\boldsymbol{\\beta}^{(t)} - (\\mathbf{X}^T\\mathbf{W}^{(t)}\\mathbf{X})^{-1}\\mathbf{X}^T(\\mathbf{p}^{(t)} - \\mathbf{y})$$

This iteration is equivalent to fitting a weighted least squares problem at each step, hence the name IRLS. With appropriate step sizes, Newton methods typically converge in far fewer iterations than gradient descent, though each iteration requires solving a linear system, which is computationally more expensive.

Regularization improves numerical stability and generalization, especially when features are correlated. L2 regularization (Ridge) adds a penalty proportional to the squared magnitude of coefficients:

$$L_\\lambda(\\boldsymbol{\\beta}) = L(\\boldsymbol{\\beta}) + \\frac{\\lambda}{2}\\|\\boldsymbol{\\beta}\\|_2^2$$

This modification adjusts both gradient and Hessian by adding $\\lambda\\boldsymbol{\\beta}$ and $\\lambda\\mathbf{I}$, respectively, improving conditioning and shrinking coefficients toward zero.

The decision boundary—the set where $P(y=1|\\mathbf{x}) = 0.5$—occurs at $\\mathbf{x}^T\\boldsymbol{\\beta} = 0$, a hyperplane. Regularization shrinks coefficient magnitudes, thus increasing the margin between classes in an L2 sense and improving generalization.`,
  
  applications: [
    'Binary and multiclass classification in medicine, finance, and marketing',
    'Baseline model for comparative evaluation and probability calibration',
    'Interpretable models when feature effects on log-odds are required'
  ],
  
  advantages: [
    'Convex optimisation with a unique global minimum (for L2-regularised loss)',
    'Probabilistic outputs that are interpretable and calibratable',
    'Computationally efficient and easy to regularise'
  ],
  
  limitations: [
    'Linear in features with respect to the log-odds; requires feature engineering for non-linear separations',
    'Parameter estimates can be unstable for small n or collinear features',
    'Requires careful regularisation/hyperparameter selection for generalisation'
  ],
  
  codeExample: `# Implementing logistic regression (vectorised gradient descent)
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score

# Synthetic binary dataset
np.random.seed(0)
X = np.vstack([np.random.normal([-2,0], 1, (200,2)), np.random.normal([2,0],1,(200,2))])
y = np.hstack([np.zeros(200), np.ones(200)])

# Add intercept
Xb = np.hstack([np.ones((X.shape[0],1)), X])

# Standardise non-intercept columns
scaler = StandardScaler()
Xb[:,1:] = scaler.fit_transform(Xb[:,1:])

def sigmoid(z):
    return 1/(1+np.exp(-z))

# Gradient descent
beta = np.zeros(Xb.shape[1])
alpha = 0.1
for t in range(1000):
    p = sigmoid(Xb.dot(beta))
    grad = Xb.T.dot(p - y)
    beta -= alpha * grad / Xb.shape[0]

# Evaluate
probs = sigmoid(Xb.dot(beta))
print('GD accuracy:', accuracy_score(y, probs > 0.5))

# scikit-learn reference (lbfgs)
clf = LogisticRegression(penalty='l2', solver='lbfgs')
clf.fit(X, y)
print('sklearn accuracy:', accuracy_score(y, clf.predict(X)))
print('ROC AUC (sklearn):', roc_auc_score(y, clf.predict_proba(X)[:,1]))

Visualization guidance:
- Plot 2D decision boundaries by evaluating p(x) on a grid and contouring the 0.5 level set.
- Show evolution of parameter norm and training loss for different learning rates.
- Display confusion matrix and ROC curve for varying thresholds.

Suggested further reading:
- Bishop, C. M., "Pattern Recognition and Machine Learning" — chapter on discriminative models.
- Hastie, Tibshirani, Friedman, "The Elements of Statistical Learning" — logistic regression and GLMs.`,
  
  children: []
};
