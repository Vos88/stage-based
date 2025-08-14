export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  color: string;
  children?: ConceptNode[];
  overview?: string;
  howItWorks?: string;
  applications?: string[];
  advantages?: string[];
  limitations?: string[];
  codeExample?: string;
}

export const aiConceptTree: ConceptNode = {
  id: 'root',
  title: 'AI',
  description: 'Artificial Intelligence - The simulation of human intelligence in machines',
  color: "bg-gradient-to-br from-purple-600 to-blue-600",
  children: [
    {
      id: 'symbolic-ai',
      title: 'Symbolic AI',
      description: 'AI based on symbolic representation and logical reasoning',
      color: "bg-gradient-to-br from-indigo-500 to-purple-600",
      overview: 'Symbolic AI represents knowledge using symbols and manipulates them according to logical rules to solve problems and make decisions.',
      howItWorks: 'Uses formal logic, knowledge graphs, and rule-based systems to represent and process information symbolically.',
      applications: ['Expert systems', 'Knowledge graphs', 'Automated theorem proving', 'Semantic web technologies'],
      advantages: ['Interpretable reasoning', 'Explicit knowledge representation', 'Logical consistency'],
      limitations: ['Difficulty handling uncertainty', 'Brittle to exceptions', 'Limited learning capabilities'],
      children: []
    },
    {
      id: 'machine-learning',
      title: 'Machine Learning',
      description: 'Algorithms that improve automatically through experience',
      color: "bg-gradient-to-br from-blue-500 to-cyan-600",
      children: [
        {
          id: 'supervised-learning',
          title: 'Supervised Learning',
          description: 'Learning with labeled training data',
          color: "bg-gradient-to-br from-green-500 to-blue-600",
          children: [
            {
              id: 'regression',
              title: 'Regression',
              description: 'Predicting continuous numerical values',
              color: "bg-gradient-to-br from-emerald-500 to-teal-600",
              children: [
                { 
                  id: 'polynomial-regression', 
                  title: 'Polynomial Regression', 
                  description: 'Fitting polynomial relationships between variables',
                  color: "bg-gradient-to-br from-teal-500 to-cyan-600",
                  overview: 'Polynomial regression extends linear regression by modeling the relationship between variables using polynomial functions.',
                  howItWorks: 'Fits curves by adding polynomial terms (x², x³, etc.) to capture non-linear relationships in data.',
                  applications: ['Curve fitting', 'Growth modeling', 'Price prediction with non-linear trends'],
                  advantages: ['Captures non-linear relationships', 'Flexible model complexity', 'Interpretable coefficients'],
                  limitations: ['Risk of overfitting', 'Sensitive to outliers', 'Extrapolation issues'],
                  children: []
                },
                {
                  id: 'linear-regression', 
                  title: 'Linear Regression',
                  color: "bg-gradient-to-br from-teal-500 to-cyan-600",
                  description: 'Modeling the relationship between one independent variable and one dependent variable by fitting a straight line to observed data.',
                  overview: 'Simple linear regression models the dependent variable $y$ as a linear function of a single predictor $x$: $y = \\beta_0 + \\beta_1 x + \\varepsilon$, where $\\beta_0$ is the intercept, $\\beta_1$ is the slope, and $\\varepsilon$ is a random error term. The model assumes a linear association and additive noise.',
                  howItWorks: 'Given $n$ observations $(x_i, y_i)$, the parameters $\\beta_0$ and $\\beta_1$ are estimated by minimizing the residual sum of squares (RSS): $$\\text{RSS}(\\beta_0, \\beta_1) = \\sum_{i=1}^n \\left(y_i - \\beta_0 - \\beta_1 x_i\\right)^2.$$ The closed-form Ordinary Least Squares (OLS) solutions are $$\\beta_1 = \\frac{\\sum_{i=1}^n (x_i - \\bar{x})(y_i - \\bar{y})}{\\sum_{i=1}^n (x_i - \\bar{x})^2}, \\quad \\beta_0 = \\bar{y} - \\beta_1 \\bar{x}.$$ The fitted model is $\\hat{y} = \\beta_0 + \\beta_1 x$, which, under Gauss–Markov assumptions, is the Best Linear Unbiased Estimator (BLUE).',
                  applications: [
                    'Forecasting continuous outcomes from a single predictor',
                    'Calibrating measurement systems',
                    'Economic demand estimation with one explanatory variable',
                    'Analyzing experimental dose-response relationships'
                  ],
                  advantages: [
                    'Closed-form solution; computationally efficient',
                    'Interpretable coefficients (slope and intercept)',
                    'Well-established statistical properties under standard assumptions'
                  ],
                  limitations: [
                    'Assumes linearity between predictor and response',
                    'Sensitive to influential outliers',
                    'Assumes homoskedastic and uncorrelated residuals',
                    'Cannot capture nonlinear patterns without transformation'
                  ],
                  codeExample: 
                `python
                import numpy as np

                # Example data
                x = np.array([1, 2, 3, 4, 5], dtype=float)
                y = np.array([2, 3, 5, 7, 11], dtype=float)

                # Means
                x_mean = np.mean(x)
                y_mean = np.mean(y)

                # OLS estimators
                beta1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean)**2)
                beta0 = y_mean - beta1 * x_mean

                print(f"β₀ (intercept): {beta0:.4f}")
                print(f"β₁ (slope): {beta1:.4f}")

                # Prediction
                x_new = 6
                y_pred = beta0 + beta1 * x_new
                print(f"Predicted y for x={x_new}: {y_pred:.4f}")
                `,
                children: []
                },
                { 
                  id: 'ridge-lasso', 
                  title: 'Ridge / Lasso Regression', 
                  description: 'Regularized regression techniques',
                  color: "bg-gradient-to-br from-purple-500 to-pink-600",
                  overview: 'Regularized regression methods that add penalty terms to prevent overfitting and improve model generalization.',
                  howItWorks: 'Ridge adds L2 penalty (sum of squared coefficients), Lasso adds L1 penalty (sum of absolute coefficients).',
                  applications: ['High-dimensional data analysis', 'Feature selection', 'Genomics', 'Finance modeling'],
                  advantages: ['Prevents overfitting', 'Handles multicollinearity', 'Lasso performs feature selection'],
                  limitations: ['Requires hyperparameter tuning', 'May overshrink coefficients', 'Less interpretable'],
                  children: []
                },
                { 
                  id: 'decision-tree-regressor', 
                  title: 'Decision Tree Regressor', 
                  description: 'Tree-based regression model',
                  color: "bg-gradient-to-br from-orange-500 to-red-600",
                  overview: 'Decision trees for regression split data into regions and predict the average value within each region.',
                  howItWorks: 'Recursively splits data based on feature values to minimize prediction error within each leaf node.',
                  applications: ['Real estate valuation', 'Medical diagnosis scoring', 'Financial risk assessment'],
                  advantages: ['Non-parametric', 'Handles non-linear relationships', 'Easy to interpret'],
                  limitations: ['Prone to overfitting', 'Unstable to small data changes', 'Biased toward features with many levels'],
                  children: []
                },
                { 
                  id: 'knn-regressor', 
                  title: 'K-Nearest Neighbors Regressor', 
                  description: 'Non-parametric regression method',
                  color: "bg-gradient-to-br from-amber-500 to-orange-600",
                  overview: 'KNN regression predicts values by averaging the target values of the k nearest neighbors in feature space.',
                  howItWorks: 'For each prediction, finds k most similar data points and returns their average target value.',
                  applications: ['Recommendation systems', 'Image processing', 'Collaborative filtering'],
                  advantages: ['Simple and intuitive', 'No assumptions about data distribution', 'Adapts to local patterns'],
                  limitations: ['Computationally expensive', 'Sensitive to irrelevant features', 'Poor performance in high dimensions'],
                  children: []
                }
              ]
            },
            {
              id: 'classification',
              title: 'Classification',
              description: 'Predicting discrete class labels',
              color: "bg-gradient-to-br from-violet-500 to-purple-600",
              children: [
                { 
                  id: 'logistic-regression', 
                  title: 'Logistic Regression', 
                  description: 'Statistical method for binary and multiclass classification',
                  color: "bg-gradient-to-br from-indigo-500 to-blue-600",
                  overview: 'Logistic regression uses the logistic function to model the probability of class membership.',
                  howItWorks: 'Applies logistic function to linear combination of features to output probabilities between 0 and 1.',
                  applications: ['Medical diagnosis', 'Marketing response prediction', 'Quality control', 'Credit scoring'],
                  advantages: ['Probabilistic output', 'No tuning of hyperparameters', 'Less prone to overfitting', 'Fast and efficient'],
                  limitations: ['Assumes linear relationship between features and log-odds', 'Sensitive to outliers', 'Requires large sample sizes'],
                  children: []
                },
                {
                  id: 'naive-bayes', 
                  title: 'Naive Bayes', 
                  description: 'Probabilistic classifier based on Bayes theorem',
                  color: "bg-gradient-to-br from-green-500 to-emerald-600",
                  overview: 'Naive Bayes applies Bayes theorem with the naive assumption of feature independence.',
                  howItWorks: 'Calculates posterior probabilities using prior probabilities and likelihood, assuming features are independent.',
                  applications: ['Spam filtering', 'Text classification', 'Sentiment analysis', 'Medical diagnosis'],
                  advantages: ['Fast and simple', 'Works well with small datasets', 'Handles multiple classes naturally'],
                  limitations: ['Strong independence assumption', 'Can be outperformed by more sophisticated methods', 'Sensitive to skewed data'],
                  children: []
                },
                { 
                  id: 'decision-tree-classifier', 
                  title: 'Decision Tree Classifier', 
                  description: 'Tree-based classification model',
                  color: "bg-gradient-to-br from-orange-500 to-amber-600",
                  overview: 'Decision trees classify data by creating a tree-like model of decisions based on feature values.',
                  howItWorks: 'Recursively splits data using feature thresholds to maximize information gain or minimize impurity.',
                  applications: ['Medical diagnosis', 'Credit approval', 'Customer segmentation', 'Rule extraction'],
                  advantages: ['Highly interpretable', 'Handles both numerical and categorical data', 'No need for feature scaling'],
                  limitations: ['Prone to overfitting', 'Unstable', 'Biased toward features with more levels'],
                  children: []
                },
                { 
                  id: 'svm', 
                  title: 'Support Vector Machine (SVM)', 
                  description: 'Maximum margin classifier',
                  color: "bg-gradient-to-br from-red-500 to-pink-600",
                  overview: 'SVM finds the optimal hyperplane that maximally separates different classes in feature space.',
                  howItWorks: 'Identifies support vectors and constructs decision boundary with maximum margin between classes.',
                  applications: ['Text classification', 'Image recognition', 'Gene classification', 'Handwriting recognition'],
                  advantages: ['Effective in high dimensions', 'Memory efficient', 'Versatile with different kernel functions'],
                  limitations: ['Poor performance on large datasets', 'Sensitive to feature scaling', 'No probabilistic output'],
                  children: []
                },
                { 
                  id: 'knn', 
                  title: 'K-Nearest Neighbors (KNN)', 
                  description: 'Classification based on nearest neighbors',
                  color: "bg-gradient-to-br from-yellow-500 to-orange-600",
                  overview: 'KNN classifies data points based on the majority class among their k nearest neighbors.',
                  howItWorks: 'For each prediction, finds k most similar training examples and assigns the most common class.',
                  applications: ['Recommendation systems', 'Pattern recognition', 'Outlier detection', 'Image classification'],
                  advantages: ['Simple and intuitive', 'No assumptions about data', 'Naturally handles multi-class problems'],
                  limitations: ['Computationally expensive', 'Sensitive to irrelevant features', 'Poor performance with high-dimensional data'],
                  children: []
                }
              ]
            }
          ]
        }, 
        {
          id: 'unsupervised-learning',
          title: 'Unsupervised Learning',
          description: 'Learning patterns from unlabeled data',
          color: "bg-gradient-to-br from-cyan-500 to-blue-600",
          children: [
            {
              id: 'clustering',
              title: 'Clustering',
              description: 'Grouping similar data points together',
              color: "bg-gradient-to-br from-emerald-500 to-green-600",
              children: [
                { 
                  id: 'k-means', 
                  title: 'K-Means', 
                  description: 'Partitioning data into k clusters',
                  color: "bg-gradient-to-br from-blue-500 to-indigo-600",
                  overview: 'K-means clustering partitions data into k clusters by minimizing within-cluster sum of squares.',
                  howItWorks: 'Iteratively assigns points to nearest centroid and updates centroids until convergence.',
                  applications: ['Customer segmentation', 'Image segmentation', 'Market research', 'Data compression'],
                  advantages: ['Simple and fast', 'Works well with globular clusters', 'Guaranteed convergence'],
                  limitations: ['Requires specifying k', 'Sensitive to initialization', 'Assumes spherical clusters'],
                  children: []
                },
                { 
                  id: 'gmm', 
                  title: 'Gaussian Mixture Models (GMM)', 
                  description: 'Probabilistic clustering with Gaussian distributions',
                  color: "bg-gradient-to-br from-purple-500 to-violet-600",
                  overview: 'GMM assumes data comes from a mixture of Gaussian distributions and estimates their parameters.',
                  howItWorks: 'Uses Expectation-Maximization algorithm to estimate mixture components and assignment probabilities.',
                  applications: ['Speech recognition', 'Computer vision', 'Density estimation', 'Anomaly detection'],
                  advantages: ['Probabilistic output', 'Flexible cluster shapes', 'Handles overlapping clusters'],
                  limitations: ['Computationally intensive', 'Sensitive to initialization', 'Requires choosing number of components'],
                  children: []
                },
                { 
                  id: 'dbscan', 
                  title: 'DBSCAN', 
                  description: 'Density-based clustering algorithm',
                  color: "bg-gradient-to-br from-teal-500 to-green-600",
                  overview: 'DBSCAN groups together points in high-density areas and marks points in low-density areas as outliers.',
                  howItWorks: 'Identifies core points with sufficient neighbors and expands clusters by connecting density-reachable points.',
                  applications: ['Anomaly detection', 'Image processing', 'Social network analysis', 'Spatial data analysis'],
                  advantages: ['Finds arbitrary shaped clusters', 'Automatically determines outliers', 'Robust to noise'],
                  limitations: ['Sensitive to hyperparameters', 'Struggles with varying densities', 'Memory intensive'],
                  children: []
                }
              ] 
            },
            {
              id: 'feature-extraction',
              title: 'Feature Extraction',
              description: 'Reducing dimensionality while preserving important information',
              color: "bg-gradient-to-br from-pink-500 to-rose-600",
              children: [
                { 
                  id: 'pca', 
                  title: 'Principal Component Analysis (PCA)', 
                  description: 'Linear dimensionality reduction technique',
                  color: "bg-gradient-to-br from-indigo-500 to-purple-600",
                  overview: 'PCA reduces dimensionality by projecting data onto principal components that capture maximum variance.',
                  howItWorks: 'Computes eigenvectors of covariance matrix and projects data onto top eigenvectors.',
                  applications: ['Data visualization', 'Feature reduction', 'Data compression', 'Noise reduction'],
                  advantages: ['Reduces overfitting', 'Removes correlation', 'Computational efficiency'],
                  limitations: ['Linear transformation only', 'Components may not be interpretable', 'Sensitive to scaling'],
                  children: []
                },
                { 
                  id: 't-sne', 
                  title: 't-SNE', 
                  description: 'Non-linear dimensionality reduction technique for visualization',
                  color: "bg-gradient-to-br from-cyan-500 to-blue-600",
                  overview: 't-SNE preserves local structure by modeling pairwise similarities in high and low dimensions.',
                  howItWorks: 'Minimizes divergence between probability distributions of pairwise similarities in original and reduced space.',
                  applications: ['Data visualization', 'Exploratory data analysis', 'Cluster visualization', 'Image analysis'],
                  advantages: ['Preserves local structure', 'Reveals clusters', 'Non-linear mapping'],
                  limitations: ['Computationally expensive', 'Non-deterministic', 'Hyperparameter sensitive'],
                  children: []
                }
              ]
            }
          ]
        }, 
        {
          id: 'reinforcement-learning',
          title: 'Reinforcement Learning',
          description: 'Learning through interaction with environment via rewards and penalties',
          color: "bg-gradient-to-br from-orange-500 to-red-600",
          children: [
            { 
              id: 'q-learning', 
              title: 'Q-Learning', 
              description: 'Model-free reinforcement learning algorithm',
              color: "bg-gradient-to-br from-amber-500 to-orange-600",
              overview: 'Q-learning learns optimal action-value function without requiring a model of the environment.',
              howItWorks: 'Updates Q-values using Bellman equation based on rewards received from actions in states.',
              applications: ['Game playing', 'Robot navigation', 'Trading strategies', 'Resource allocation'],
              advantages: ['Model-free', 'Guaranteed convergence', 'Off-policy learning'],
              limitations: ['Requires discrete state/action spaces', 'Slow convergence', 'Memory intensive for large state spaces'],
              children: []
            },
            { 
              id: 'dqn', 
              title: 'Deep Q-Networks (DQN)', 
              description: 'Deep learning approach to Q-learning',
              color: "bg-gradient-to-br from-red-500 to-pink-600",
              overview: 'DQN uses deep neural networks to approximate Q-values for high-dimensional state spaces.',
              howItWorks: 'Combines Q-learning with deep networks, using experience replay and target networks for stability.',
              applications: ['Video game AI', 'Robotics', 'Autonomous vehicles', 'Strategic planning'],
              advantages: ['Handles high-dimensional states', 'End-to-end learning', 'Scales to complex problems'],
              limitations: ['Sample inefficient', 'Unstable training', 'Requires careful hyperparameter tuning'],
              children: []
            },
            { 
              id: 'policy-gradient', 
              title: 'Policy Gradient Methods', 
              description: 'Directly optimizing policy parameters',
              color: "bg-gradient-to-br from-violet-500 to-purple-600",
              overview: 'Policy gradient methods directly optimize policy parameters using gradient ascent on expected returns.',
              howItWorks: 'Estimates gradient of expected return with respect to policy parameters and updates parameters accordingly.',
              applications: ['Continuous control', 'Natural language generation', 'Multi-agent systems', 'Portfolio optimization'],
              advantages: ['Handles continuous actions', 'Direct policy optimization', 'Stochastic policies'],
              limitations: ['High variance gradients', 'Sample inefficient', 'Local optima'],
              children: []
            }
          ]
        },
        {
          id: 'neural-networks',
          title: 'Neural Networks',
          description: 'Computing systems inspired by biological neural networks',
          color: "bg-gradient-to-br from-violet-500 to-indigo-600",
          children: [
            { 
              id: 'mlp', 
              title: 'Multi-layer Perceptron', 
              description: 'Feedforward artificial neural network',
              color: "bg-gradient-to-br from-blue-500 to-purple-600",
              overview: 'MLPs consist of multiple layers of neurons with non-linear activation functions for learning complex patterns.',
              howItWorks: 'Forward propagates input through hidden layers and uses backpropagation to update weights.',
              applications: ['Pattern recognition', 'Function approximation', 'Classification', 'Regression'],
              advantages: ['Universal function approximator', 'Non-linear modeling', 'Flexible architecture'],
              limitations: ['Requires large amounts of data', 'Prone to overfitting', 'Black box nature'],
              children: []
            },
            { 
              id: 'cnn', 
              title: 'Convolutional Neural Network', 
              description: 'Deep learning architecture for processing grid-like data',
              color: "bg-gradient-to-br from-green-500 to-teal-600",
              overview: 'CNNs use convolutional layers to detect local features and pooling layers to reduce dimensionality.',
              howItWorks: 'Applies learnable filters across input to detect features, followed by pooling for translation invariance.',
              applications: ['Image classification', 'Object detection', 'Medical imaging', 'Computer vision'],
              advantages: ['Translation invariant', 'Parameter sharing', 'Hierarchical feature learning'],
              limitations: ['Requires large datasets', 'Computationally intensive', 'Not suitable for non-grid data'],
              children: []
            },
            { 
              id: 'transformers', 
              title: 'Transformers', 
              description: 'Attention-based architecture for sequence modeling and generation',
              color: "bg-gradient-to-br from-pink-500 to-red-600",
              overview: 'Transformers use self-attention mechanisms to process sequences in parallel and capture long-range dependencies.',
              howItWorks: 'Attention mechanism computes weighted representations based on similarity between sequence elements.',
              applications: ['Machine translation', 'Text generation', 'Question answering', 'Code generation'],
              advantages: ['Parallel processing', 'Long-range dependencies', 'Transfer learning'],
              limitations: ['Memory intensive', 'Requires large datasets', 'Quadratic complexity with sequence length'],
              children: [
                { 
                  id: 'nlp', 
                  title: 'Natural Language Processing (NLP)', 
                  description: 'AI for understanding and generating human language',
                  color: "bg-gradient-to-br from-emerald-500 to-cyan-600",
                  overview: 'NLP combines computational linguistics with machine learning to process and understand human language.',
                  howItWorks: 'Uses various techniques from tokenization to deep learning for language understanding and generation.',
                  applications: ['Chatbots', 'Machine translation', 'Sentiment analysis', 'Document summarization'],
                  advantages: ['Versatile applications', 'Improving rapidly', 'Transfer learning'],
                  limitations: ['Context understanding', 'Ambiguity handling', 'Cultural and linguistic biases'],
                  children: []
                }
              ]
            },
            { 
              id: 'rnn', 
              title: 'Recurrent Neural Network (RNN)', 
              description: 'Neural network architecture for sequential data using temporal dependencies',
              color: "bg-gradient-to-br from-amber-500 to-yellow-600",
              overview: 'RNNs process sequential data by maintaining hidden states that capture information from previous time steps.',
              howItWorks: 'Uses recurrent connections to maintain memory of previous inputs while processing sequences.',
              applications: ['Time series prediction', 'Speech recognition', 'Language modeling', 'Sequence generation'],
              advantages: ['Handles variable-length sequences', 'Memory of past information', 'Parameter sharing across time'],
              limitations: ['Vanishing gradient problem', 'Sequential processing', 'Difficulty with long sequences'],
              children: []
            },
            { 
              id: 'gan', 
              title: 'Generative Adversarial Network (GAN)', 
              description: 'Neural network architecture for generating realistic data through adversarial training',
              color: "bg-gradient-to-br from-purple-500 to-pink-600",
              overview: 'GANs consist of generator and discriminator networks competing against each other to generate realistic data.',
              howItWorks: 'Generator creates fake data while discriminator learns to distinguish real from fake data.',
              applications: ['Image generation', 'Data augmentation', 'Style transfer', 'Super resolution'],
              advantages: ['High-quality generation', 'Unsupervised learning', 'Flexible data types'],
              limitations: ['Training instability', 'Mode collapse', 'Difficult to evaluate'],
              children: []
            },
            { 
              id: 'diffusion', 
              title: 'Diffusion Models', 
              description: 'Generative models that learn to reverse a noise process to synthesize data',
              color: "bg-gradient-to-br from-indigo-500 to-blue-600",
              overview: 'Diffusion models learn to denoise data by reversing a gradual noise addition process.',
              howItWorks: 'Trains neural network to predict noise added at each step of forward diffusion process.',
              applications: ['Image generation', 'Audio synthesis', 'Video generation', 'Molecular design'],
              advantages: ['High-quality samples', 'Stable training', 'Controllable generation'],
              limitations: ['Slow sampling', 'Computationally expensive', 'Many sampling steps required'],
              children: []
            },
            { 
              id: 'autoencoders', 
              title: 'Autoencoders', 
              description: 'Neural networks for feature learning',
              color: "bg-gradient-to-br from-teal-500 to-green-600",
              overview: 'Autoencoders learn efficient data representations by encoding input to lower dimension and reconstructing it.',
              howItWorks: 'Encoder compresses input to latent representation, decoder reconstructs original input from latent code.',
              applications: ['Dimensionality reduction', 'Anomaly detection', 'Denoising', 'Data compression'],
              advantages: ['Unsupervised learning', 'Learns meaningful representations', 'Flexible architecture'],
              limitations: ['May lose important information', 'Requires careful architecture design', 'Prone to overfitting'],
              children: []
            }
          ]
        },
        {
          id: 'ensemble-learning',
          title: 'Ensemble Learning',
          description: 'Combining multiple models to improve performance and robustness',
          color: "bg-gradient-to-br from-rose-500 to-pink-600",
          children: [
            {
              id: 'bagging',
              title: 'Bagging',
              description: 'Bootstrap aggregation to reduce variance',
              color: "bg-gradient-to-br from-green-500 to-emerald-600",
              children: [
                { 
                  id: 'random-forest', 
                  title: 'Random Forest', 
                  description: 'Ensemble of decision trees using bagging',
                  color: "bg-gradient-to-br from-emerald-500 to-teal-600",
                  overview: 'Random Forest combines multiple decision trees trained on random subsets of data and features.',
                  howItWorks: 'Trains multiple trees on bootstrap samples with random feature selection and averages predictions.',
                  applications: ['Feature importance', 'Classification', 'Regression', 'Bioinformatics'],
                  advantages: ['Reduces overfitting', 'Handles missing values', 'Provides feature importance'],
                  limitations: ['Less interpretable than single tree', 'Can overfit with very noisy data', 'Memory intensive'],
                  children: []
                }
              ]
            },
            {
              id: 'boosting',
              title: 'Boosting',
              description: 'Sequentially combining weak learners to reduce bias',
              color: "bg-gradient-to-br from-orange-500 to-red-600",
              children: [
                { 
                  id: 'gradient-boosting', 
                  title: 'Gradient Boosting', 
                  description: 'Boosting method for regression and classification',
                  color: "bg-gradient-to-br from-amber-500 to-orange-600",
                  overview: 'Gradient boosting builds models sequentially, with each model correcting errors of previous models.',
                  howItWorks: 'Fits new models to residual errors of ensemble, using gradient descent to minimize loss function.',
                  applications: ['Tabular data prediction', 'Ranking', 'Regression', 'Feature selection'],
                  advantages: ['High predictive accuracy', 'Handles different data types', 'Built-in feature selection'],
                  limitations: ['Prone to overfitting', 'Sensitive to hyperparameters', 'Sequential training'],
                  children: [
                    { 
                      id: 'xgboost', 
                      title: 'XGBoost', 
                      description: 'Optimized gradient boosting library for performance and scalability',
                      color: "bg-gradient-to-br from-yellow-500 to-orange-600",
                      overview: 'XGBoost is an optimized gradient boosting framework designed for speed and performance.',
                      howItWorks: 'Uses second-order gradients, regularization, and optimized data structures for efficient training.',
                      applications: ['Machine learning competitions', 'Click-through rate prediction', 'Risk modeling', 'Ranking systems'],
                      advantages: ['State-of-the-art performance', 'Fast training', 'Built-in regularization'],
                      limitations: ['Many hyperparameters', 'Memory intensive', 'Requires feature engineering'],
                      children: []
                    }
                  ] 
                },
                {
                  id: 'adaboost', 
                  title: 'AdaBoost', 
                  description: 'Adaptive boosting method using weighted weak learners',
                  color: "bg-gradient-to-br from-red-500 to-pink-600",
                  overview: 'AdaBoost adaptively adjusts weights of training examples based on previous classifier errors.',
                  howItWorks: 'Sequentially trains weak learners on weighted datasets, increasing weights of misclassified examples.',
                  applications: ['Face detection', 'Object recognition', 'Text classification', 'Medical diagnosis'],
                  advantages: ['Simple and effective', 'Automatic feature selection', 'Good generalization'],
                  limitations: ['Sensitive to noise and outliers', 'Can overfit', 'Performance depends on weak learner choice'],
                  children: []
                }
              ]
            },
            { 
              id: 'stacking', 
              title: 'Stacking', 
              description: 'Combining multiple models using a meta-learner',
              color: "bg-gradient-to-br from-violet-500 to-purple-600",
              overview: 'Stacking trains a meta-model to optimally combine predictions from multiple base models.',
              howItWorks: 'Base models make predictions, then meta-learner is trained on these predictions to make final prediction.',
              applications: ['Machine learning competitions', 'Complex prediction tasks', 'Model combination'],
              advantages: ['Can improve upon best individual model', 'Flexible combination strategy', 'Leverages model diversity'],
              limitations: ['Increased complexity', 'Risk of overfitting', 'Computationally expensive'],
              children: []
            },
            { 
              id: 'voting', 
              title: 'Voting', 
              description: 'Combining predictions by majority',
              color: "bg-gradient-to-br from-indigo-500 to-blue-600",
              overview: 'Voting ensembles combine multiple models by taking majority vote (hard) or average (soft) of predictions.',
              howItWorks: 'Each model makes prediction, final prediction is determined by majority vote or weighted average.',
              applications: ['Classification tasks', 'Model combination', 'Reducing prediction variance'],
              advantages: ['Simple and effective', 'Reduces overfitting', 'Improves robustness'],
              limitations: ['All models weighted equally', 'May not be optimal', 'Requires diverse models'],
              children: []
            },
            { 
              id: 'averaging', 
              title: 'Averaging', 
              description: 'Combining predictions by averaging outputs from multiple models, typically used in regression',
              color: "bg-gradient-to-br from-cyan-500 to-teal-600",
              overview: 'Model averaging combines predictions from multiple models by computing their weighted or simple average.',
              howItWorks: 'Trains multiple models independently and combines their predictions through averaging.',
              applications: ['Regression tasks', 'Time series forecasting', 'Risk modeling', 'Ensemble learning'],
              advantages: ['Reduces variance', 'Simple to implement', 'Often improves accuracy'],
              limitations: ['May not capture model interactions', 'Equal weighting may be suboptimal', 'Requires model diversity'],
              children: []
            }
          ]
        }
      ]
    }
  ]
};