export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  category: string;
  color: string;
  children?: ConceptNode[];
  content?: {
    overview?: string;
    howItWorks?: string;
    applications?: string[];
    advantages?: string[];
    limitations?: string[];
    codeExample?: string;
    codeLanguage?: string;
    links?: { title: string; url: string }[];
    image?: string;
    keyPoints?: string[];
  };
}

export const aiConceptTree: ConceptNode = {
  id: 'root',
  title: 'AI',
  description: 'Artificial Intelligence - The simulation of human intelligence in machines',
  category: 'ai-root',
  color: '258 96% 67%',
  children: [
    {
      id: 'symbolic-ai',
      title: 'Symbolic AI',
      description: 'AI based on symbolic representation and logical reasoning',
      category: 'symbolic',
      color: '280 85% 65%',
      content: {
        overview: 'Symbolic AI represents knowledge using symbols and manipulates them according to logical rules to solve problems and make decisions.',
        howItWorks: 'Uses formal logic, knowledge graphs, and rule-based systems to represent and process information symbolically.',
        applications: ['Expert systems', 'Knowledge graphs', 'Automated theorem proving', 'Semantic web technologies'],
        advantages: ['Interpretable reasoning', 'Explicit knowledge representation', 'Logical consistency'],
        limitations: ['Difficulty handling uncertainty', 'Brittle to exceptions', 'Limited learning capabilities']
      },
      children: []
    },
    {
      id: 'machine-learning',
      title: 'Machine Learning',
      description: 'Algorithms that improve automatically through experience',
      category: 'ml-core',
      color: '215 100% 65%',
      children: [
        {
          id: 'supervised-learning',
          title: 'Supervised Learning',
          description: 'Learning with labeled training data',
          category: 'supervised',
          color: '25 95% 60%',
          children: [
            {
              id: 'regression',
              title: 'Regression',
              description: 'Predicting continuous numerical values',
              category: 'supervised',
              color: '25 95% 60%',
              children: [
                { 
                  id: 'polynomial-regression', 
                  title: 'Polynomial Regression', 
                  description: 'Fitting polynomial relationships between variables',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Polynomial regression extends linear regression by modeling the relationship between variables using polynomial functions.',
                    howItWorks: 'Fits curves by adding polynomial terms (x², x³, etc.) to capture non-linear relationships in data.',
                    applications: ['Curve fitting', 'Growth modeling', 'Price prediction with non-linear trends'],
                    advantages: ['Captures non-linear relationships', 'Flexible model complexity', 'Interpretable coefficients'],
                    limitations: ['Risk of overfitting', 'Sensitive to outliers', 'Extrapolation issues']
                  },
                  children: [
                    {
                      id: 'linear-regression', 
                      title: 'Linear Regression', 
                      description: 'Fitting linear relationships between variables',
                      category: 'algorithms',
                      color: '195 85% 60%',
                      content: {
                        overview: 'Linear regression models the relationship between a dependent variable and independent variables using a linear equation.',
                        howItWorks: 'Finds the best-fitting line through data points using least squares optimization.',
                        applications: ['Sales forecasting', 'Risk assessment', 'Economic modeling', 'Scientific research'],
                        advantages: ['Simple and interpretable', 'Fast computation', 'Well-understood theory'],
                        limitations: ['Assumes linear relationships', 'Sensitive to outliers', 'May underfit complex data']
                      },
                      children: [
                        { 
                          id: 'ridge-lasso', 
                          title: 'Ridge / Lasso Regression', 
                          description: 'Regularized regression techniques',
                          category: 'algorithms',
                          color: '195 85% 60%',
                          content: {
                            overview: 'Regularized regression methods that add penalty terms to prevent overfitting and improve model generalization.',
                            howItWorks: 'Ridge adds L2 penalty (sum of squared coefficients), Lasso adds L1 penalty (sum of absolute coefficients).',
                            applications: ['High-dimensional data analysis', 'Feature selection', 'Genomics', 'Finance modeling'],
                            advantages: ['Prevents overfitting', 'Handles multicollinearity', 'Lasso performs feature selection'],
                            limitations: ['Requires hyperparameter tuning', 'May overshrink coefficients', 'Less interpretable']
                          }
                        }
                      ]
                    }
                  ] 
                },
                { 
                  id: 'decision-tree-regressor', 
                  title: 'Decision Tree Regressor', 
                  description: 'Tree-based regression model',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Decision trees for regression split data into regions and predict the average value within each region.',
                    howItWorks: 'Recursively splits data based on feature values to minimize prediction error within each leaf node.',
                    applications: ['Real estate valuation', 'Medical diagnosis scoring', 'Financial risk assessment'],
                    advantages: ['Non-parametric', 'Handles non-linear relationships', 'Easy to interpret'],
                    limitations: ['Prone to overfitting', 'Unstable to small data changes', 'Biased toward features with many levels']
                  }
                },
                { 
                  id: 'knn-regressor', 
                  title: 'K-Nearest Neighbors Regressor', 
                  description: 'Non-parametric regression method',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'KNN regression predicts values by averaging the target values of the k nearest neighbors in feature space.',
                    howItWorks: 'For each prediction, finds k most similar data points and returns their average target value.',
                    applications: ['Recommendation systems', 'Image processing', 'Collaborative filtering'],
                    advantages: ['Simple and intuitive', 'No assumptions about data distribution', 'Adapts to local patterns'],
                    limitations: ['Computationally expensive', 'Sensitive to irrelevant features', 'Poor performance in high dimensions']
                  }
                }
              ]
            },
            {
              id: 'classification',
              title: 'Classification',
              description: 'Predicting discrete class labels',
              category: 'supervised',
              color: '25 95% 60%',
              children: [
                { 
                  id: 'logistic-regression', 
                  title: 'Logistic Regression', 
                  description: 'Statistical method for binary and multiclass classification',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Logistic regression uses the logistic function to model the probability of class membership.',
                    howItWorks: 'Applies logistic function to linear combination of features to output probabilities between 0 and 1.',
                    applications: ['Medical diagnosis', 'Marketing response prediction', 'Quality control', 'Credit scoring'],
                    advantages: ['Probabilistic output', 'No tuning of hyperparameters', 'Less prone to overfitting', 'Fast and efficient'],
                    limitations: ['Assumes linear relationship between features and log-odds', 'Sensitive to outliers', 'Requires large sample sizes']
                  }
                },
                { 
                  id: 'naive-bayes', 
                  title: 'Naive Bayes', 
                  description: 'Probabilistic classifier based on Bayes theorem',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Naive Bayes applies Bayes theorem with the naive assumption of feature independence.',
                    howItWorks: 'Calculates posterior probabilities using prior probabilities and likelihood, assuming features are independent.',
                    applications: ['Spam filtering', 'Text classification', 'Sentiment analysis', 'Medical diagnosis'],
                    advantages: ['Fast and simple', 'Works well with small datasets', 'Handles multiple classes naturally'],
                    limitations: ['Strong independence assumption', 'Can be outperformed by more sophisticated methods', 'Sensitive to skewed data']
                  }
                },
                { 
                  id: 'decision-tree-classifier', 
                  title: 'Decision Tree Classifier', 
                  description: 'Tree-based classification model',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Decision trees classify data by creating a tree-like model of decisions based on feature values.',
                    howItWorks: 'Recursively splits data using feature thresholds to maximize information gain or minimize impurity.',
                    applications: ['Medical diagnosis', 'Credit approval', 'Customer segmentation', 'Rule extraction'],
                    advantages: ['Highly interpretable', 'Handles both numerical and categorical data', 'No need for feature scaling'],
                    limitations: ['Prone to overfitting', 'Unstable', 'Biased toward features with more levels']
                  }
                },
                { 
                  id: 'svm', 
                  title: 'Support Vector Machine (SVM)', 
                  description: 'Maximum margin classifier',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'SVM finds the optimal hyperplane that maximally separates different classes in feature space.',
                    howItWorks: 'Identifies support vectors and constructs decision boundary with maximum margin between classes.',
                    applications: ['Text classification', 'Image recognition', 'Gene classification', 'Handwriting recognition'],
                    advantages: ['Effective in high dimensions', 'Memory efficient', 'Versatile with different kernel functions'],
                    limitations: ['Poor performance on large datasets', 'Sensitive to feature scaling', 'No probabilistic output']
                  }
                },
                { 
                  id: 'knn', 
                  title: 'K-Nearest Neighbors (KNN)', 
                  description: 'Classification based on nearest neighbors',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'KNN classifies data points based on the majority class among their k nearest neighbors.',
                    howItWorks: 'For each prediction, finds k most similar training examples and assigns the most common class.',
                    applications: ['Recommendation systems', 'Pattern recognition', 'Outlier detection', 'Image classification'],
                    advantages: ['Simple and intuitive', 'No assumptions about data', 'Naturally handles multi-class problems'],
                    limitations: ['Computationally expensive', 'Sensitive to irrelevant features', 'Poor performance with high-dimensional data']
                  }
                }
              ]
            }
          ]
        }, 
        {
          id: 'unsupervised-learning',
          title: 'Unsupervised Learning',
          description: 'Learning patterns from unlabeled data',
          category: 'unsupervised',
          color: '315 85% 70%',
          children: [
            {
              id: 'clustering',
              title: 'Clustering',
              description: 'Grouping similar data points together',
              category: 'unsupervised',
              color: '315 85% 70%',
              children: [
                { 
                  id: 'k-means', 
                  title: 'K-Means', 
                  description: 'Partitioning data into k clusters',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'K-means clustering partitions data into k clusters by minimizing within-cluster sum of squares.',
                    howItWorks: 'Iteratively assigns points to nearest centroid and updates centroids until convergence.',
                    applications: ['Customer segmentation', 'Image segmentation', 'Market research', 'Data compression'],
                    advantages: ['Simple and fast', 'Works well with globular clusters', 'Guaranteed convergence'],
                    limitations: ['Requires specifying k', 'Sensitive to initialization', 'Assumes spherical clusters']
                  }
                },
                { 
                  id: 'gmm', 
                  title: 'Gaussian Mixture Models (GMM)', 
                  description: 'Probabilistic clustering with Gaussian distributions',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'GMM assumes data comes from a mixture of Gaussian distributions and estimates their parameters.',
                    howItWorks: 'Uses Expectation-Maximization algorithm to estimate mixture components and assignment probabilities.',
                    applications: ['Speech recognition', 'Computer vision', 'Density estimation', 'Anomaly detection'],
                    advantages: ['Probabilistic output', 'Flexible cluster shapes', 'Handles overlapping clusters'],
                    limitations: ['Computationally intensive', 'Sensitive to initialization', 'Requires choosing number of components']
                  }
                },
                { 
                  id: 'dbscan', 
                  title: 'DBSCAN', 
                  description: 'Density-based clustering algorithm',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'DBSCAN groups together points in high-density areas and marks points in low-density areas as outliers.',
                    howItWorks: 'Identifies core points with sufficient neighbors and expands clusters by connecting density-reachable points.',
                    applications: ['Anomaly detection', 'Image processing', 'Social network analysis', 'Spatial data analysis'],
                    advantages: ['Finds arbitrary shaped clusters', 'Automatically determines outliers', 'Robust to noise'],
                    limitations: ['Sensitive to hyperparameters', 'Struggles with varying densities', 'Memory intensive']
                  }
                }
              ] 
            },
            {
              id: 'feature-extraction',
              title: 'Feature Extraction',
              description: 'Reducing dimensionality while preserving important information',
              category: 'unsupervised',
              color: '315 85% 70%',
              children: [
                { 
                  id: 'pca', 
                  title: 'Principal Component Analysis (PCA)', 
                  description: 'Linear dimensionality reduction technique',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'PCA reduces dimensionality by projecting data onto principal components that capture maximum variance.',
                    howItWorks: 'Computes eigenvectors of covariance matrix and projects data onto top eigenvectors.',
                    applications: ['Data visualization', 'Feature reduction', 'Data compression', 'Noise reduction'],
                    advantages: ['Reduces overfitting', 'Removes correlation', 'Computational efficiency'],
                    limitations: ['Linear transformation only', 'Components may not be interpretable', 'Sensitive to scaling']
                  }
                },
                { 
                  id: 't-sne', 
                  title: 't-SNE', 
                  description: 'Non-linear dimensionality reduction technique for visualization',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 't-SNE preserves local structure by modeling pairwise similarities in high and low dimensions.',
                    howItWorks: 'Minimizes divergence between probability distributions of pairwise similarities in original and reduced space.',
                    applications: ['Data visualization', 'Exploratory data analysis', 'Cluster visualization', 'Image analysis'],
                    advantages: ['Preserves local structure', 'Reveals clusters', 'Non-linear mapping'],
                    limitations: ['Computationally expensive', 'Non-deterministic', 'Hyperparameter sensitive']
                  }
                }
              ]
            }
          ]
        }, 
        {
          id: 'reinforcement-learning',
          title: 'Reinforcement Learning',
          description: 'Learning through interaction with environment via rewards and penalties',
          category: 'reinforcement',
          color: '285 85% 65%',
          children: [
            { 
              id: 'q-learning', 
              title: 'Q-Learning', 
              description: 'Model-free reinforcement learning algorithm',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'Q-learning learns optimal action-value function without requiring a model of the environment.',
                howItWorks: 'Updates Q-values using Bellman equation based on rewards received from actions in states.',
                applications: ['Game playing', 'Robot navigation', 'Trading strategies', 'Resource allocation'],
                advantages: ['Model-free', 'Guaranteed convergence', 'Off-policy learning'],
                limitations: ['Requires discrete state/action spaces', 'Slow convergence', 'Memory intensive for large state spaces']
              }
            },
            { 
              id: 'dqn', 
              title: 'Deep Q-Networks (DQN)', 
              description: 'Deep learning approach to Q-learning',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'DQN uses deep neural networks to approximate Q-values for high-dimensional state spaces.',
                howItWorks: 'Combines Q-learning with deep networks, using experience replay and target networks for stability.',
                applications: ['Video game AI', 'Robotics', 'Autonomous vehicles', 'Strategic planning'],
                advantages: ['Handles high-dimensional states', 'End-to-end learning', 'Scales to complex problems'],
                limitations: ['Sample inefficient', 'Unstable training', 'Requires careful hyperparameter tuning']
              }
            },
            { 
              id: 'policy-gradient', 
              title: 'Policy Gradient Methods', 
              description: 'Directly optimizing policy parameters',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'Policy gradient methods directly optimize policy parameters using gradient ascent on expected returns.',
                howItWorks: 'Estimates gradient of expected return with respect to policy parameters and updates parameters accordingly.',
                applications: ['Continuous control', 'Natural language generation', 'Multi-agent systems', 'Portfolio optimization'],
                advantages: ['Handles continuous actions', 'Direct policy optimization', 'Stochastic policies'],
                limitations: ['High variance gradients', 'Sample inefficient', 'Local optima']
              }
            }
          ]
        },
        {
          id: 'neural-networks',
          title: 'Neural Networks',
          description: 'Computing systems inspired by biological neural networks',
          category: 'deep-learning',
          color: '210 100% 55%',
          children: [
            { 
              id: 'mlp', 
              title: 'Multi-layer Perceptron', 
              description: 'Feedforward artificial neural network',
              category: 'deep-learning',
              color: '210 100% 55%',
              content: {
                overview: 'MLPs consist of multiple layers of neurons with non-linear activation functions for learning complex patterns.',
                howItWorks: 'Forward propagates input through hidden layers and uses backpropagation to update weights.',
                applications: ['Pattern recognition', 'Function approximation', 'Classification', 'Regression'],
                advantages: ['Universal function approximator', 'Non-linear modeling', 'Flexible architecture'],
                limitations: ['Requires large amounts of data', 'Prone to overfitting', 'Black box nature']
              }
            },
            { 
              id: 'cnn', 
              title: 'Convolutional Neural Network', 
              description: 'Deep learning architecture for processing grid-like data',
              category: 'computer-vision',
              color: '45 95% 65%',
              content: {
                overview: 'CNNs use convolutional layers to detect local features and pooling layers to reduce dimensionality.',
                howItWorks: 'Applies learnable filters across input to detect features, followed by pooling for translation invariance.',
                applications: ['Image classification', 'Object detection', 'Medical imaging', 'Computer vision'],
                advantages: ['Translation invariant', 'Parameter sharing', 'Hierarchical feature learning'],
                limitations: ['Requires large datasets', 'Computationally intensive', 'Not suitable for non-grid data']
              }
            },
            { 
              id: 'transformers', 
              title: 'Transformers', 
              description: 'Attention-based architecture for sequence modeling and generation',
              category: 'nlp',
              color: '165 85% 55%',
              content: {
                overview: 'Transformers use self-attention mechanisms to process sequences in parallel and capture long-range dependencies.',
                howItWorks: 'Attention mechanism computes weighted representations based on similarity between sequence elements.',
                applications: ['Machine translation', 'Text generation', 'Question answering', 'Code generation'],
                advantages: ['Parallel processing', 'Long-range dependencies', 'Transfer learning'],
                limitations: ['Memory intensive', 'Requires large datasets', 'Quadratic complexity with sequence length']
              },
              children: [
                { 
                  id: 'nlp', 
                  title: 'Natural Language Processing (NLP)', 
                  description: 'AI for understanding and generating human language',
                  category: 'nlp',
                  color: '165 85% 55%',
                  content: {
                    overview: 'NLP combines computational linguistics with machine learning to process and understand human language.',
                    howItWorks: 'Uses various techniques from tokenization to deep learning for language understanding and generation.',
                    applications: ['Chatbots', 'Machine translation', 'Sentiment analysis', 'Document summarization'],
                    advantages: ['Versatile applications', 'Improving rapidly', 'Transfer learning'],
                    limitations: ['Context understanding', 'Ambiguity handling', 'Cultural and linguistic biases']
                  }
                }
              ]
            },
            { 
              id: 'rnn', 
              title: 'Recurrent Neural Network (RNN)', 
              description: 'Neural network architecture for sequential data using temporal dependencies',
              category: 'nlp',
              color: '165 85% 55%',
              content: {
                overview: 'RNNs process sequential data by maintaining hidden states that capture information from previous time steps.',
                howItWorks: 'Uses recurrent connections to maintain memory of previous inputs while processing sequences.',
                applications: ['Time series prediction', 'Speech recognition', 'Language modeling', 'Sequence generation'],
                advantages: ['Handles variable-length sequences', 'Memory of past information', 'Parameter sharing across time'],
                limitations: ['Vanishing gradient problem', 'Sequential processing', 'Difficulty with long sequences']
              }
            },
            { 
              id: 'gan', 
              title: 'Generative Adversarial Network (GAN)', 
              description: 'Neural network architecture for generating realistic data through adversarial training',
              category: 'deep-learning',
              color: '210 100% 55%',
              content: {
                overview: 'GANs consist of generator and discriminator networks competing against each other to generate realistic data.',
                howItWorks: 'Generator creates fake data while discriminator learns to distinguish real from fake data.',
                applications: ['Image generation', 'Data augmentation', 'Style transfer', 'Super resolution'],
                advantages: ['High-quality generation', 'Unsupervised learning', 'Flexible data types'],
                limitations: ['Training instability', 'Mode collapse', 'Difficult to evaluate']
              }
            },
            { 
              id: 'diffusion', 
              title: 'Diffusion Models', 
              description: 'Generative models that learn to reverse a noise process to synthesize data',
              category: 'deep-learning',
              color: '210 100% 55%',
              content: {
                overview: 'Diffusion models learn to denoise data by reversing a gradual noise addition process.',
                howItWorks: 'Trains neural network to predict noise added at each step of forward diffusion process.',
                applications: ['Image generation', 'Audio synthesis', 'Video generation', 'Molecular design'],
                advantages: ['High-quality samples', 'Stable training', 'Controllable generation'],
                limitations: ['Slow sampling', 'Computationally expensive', 'Many sampling steps required']
              }
            },
            { 
              id: 'autoencoders', 
              title: 'Autoencoders', 
              description: 'Neural networks for feature learning',
              category: 'deep-learning',
              color: '210 100% 55%',
              content: {
                overview: 'Autoencoders learn efficient data representations by encoding input to lower dimension and reconstructing it.',
                howItWorks: 'Encoder compresses input to latent representation, decoder reconstructs original input from latent code.',
                applications: ['Dimensionality reduction', 'Anomaly detection', 'Denoising', 'Data compression'],
                advantages: ['Unsupervised learning', 'Learns meaningful representations', 'Flexible architecture'],
                limitations: ['May lose important information', 'Requires careful architecture design', 'Prone to overfitting']
              }
            }
          ]
        },
        {
          id: 'ensemble-learning',
          title: 'Ensemble Learning',
          description: 'Combining multiple models to improve performance and robustness',
          category: 'ensemble',
          color: '120 85% 55%',
          children: [
            {
              id: 'bagging',
              title: 'Bagging',
              description: 'Bootstrap aggregation to reduce variance',
              category: 'ensemble',
              color: '120 85% 55%',
              children: [
                { 
                  id: 'random-forest', 
                  title: 'Random Forest', 
                  description: 'Ensemble of decision trees using bagging',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Random Forest combines multiple decision trees trained on random subsets of data and features.',
                    howItWorks: 'Trains multiple trees on bootstrap samples with random feature selection and averages predictions.',
                    applications: ['Feature importance', 'Classification', 'Regression', 'Bioinformatics'],
                    advantages: ['Reduces overfitting', 'Handles missing values', 'Provides feature importance'],
                    limitations: ['Less interpretable than single tree', 'Can overfit with very noisy data', 'Memory intensive']
                  }
                }
              ]
            },
            {
              id: 'boosting',
              title: 'Boosting',
              description: 'Sequentially combining weak learners to reduce bias',
              category: 'ensemble',
              color: '120 85% 55%',
              children: [
                { 
                  id: 'gradient-boosting', 
                  title: 'Gradient Boosting', 
                  description: 'Boosting method for regression and classification',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'Gradient boosting builds models sequentially, with each model correcting errors of previous models.',
                    howItWorks: 'Fits new models to residual errors of ensemble, using gradient descent to minimize loss function.',
                    applications: ['Tabular data prediction', 'Ranking', 'Regression', 'Feature selection'],
                    advantages: ['High predictive accuracy', 'Handles different data types', 'Built-in feature selection'],
                    limitations: ['Prone to overfitting', 'Sensitive to hyperparameters', 'Sequential training']
                  },
                  children: [
                    { 
                      id: 'xgboost', 
                      title: 'XGBoost', 
                      description: 'Optimized gradient boosting library for performance and scalability',
                      category: 'algorithms',
                      color: '195 85% 60%',
                      content: {
                        overview: 'XGBoost is an optimized gradient boosting framework designed for speed and performance.',
                        howItWorks: 'Uses second-order gradients, regularization, and optimized data structures for efficient training.',
                        applications: ['Machine learning competitions', 'Click-through rate prediction', 'Risk modeling', 'Ranking systems'],
                        advantages: ['State-of-the-art performance', 'Fast training', 'Built-in regularization'],
                        limitations: ['Many hyperparameters', 'Memory intensive', 'Requires feature engineering']
                      }
                    }
                  ] 
                },
                {
                  id: 'adaboost', 
                  title: 'AdaBoost', 
                  description: 'Adaptive boosting method using weighted weak learners',
                  category: 'algorithms',
                  color: '195 85% 60%',
                  content: {
                    overview: 'AdaBoost adaptively adjusts weights of training examples based on previous classifier errors.',
                    howItWorks: 'Sequentially trains weak learners on weighted datasets, increasing weights of misclassified examples.',
                    applications: ['Face detection', 'Object recognition', 'Text classification', 'Medical diagnosis'],
                    advantages: ['Simple and effective', 'Automatic feature selection', 'Good generalization'],
                    limitations: ['Sensitive to noise and outliers', 'Can overfit', 'Performance depends on weak learner choice']
                  }
                }
              ]
            },
            { 
              id: 'stacking', 
              title: 'Stacking', 
              description: 'Combining multiple models using a meta-learner',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'Stacking trains a meta-model to optimally combine predictions from multiple base models.',
                howItWorks: 'Base models make predictions, then meta-learner is trained on these predictions to make final prediction.',
                applications: ['Machine learning competitions', 'Complex prediction tasks', 'Model combination'],
                advantages: ['Can improve upon best individual model', 'Flexible combination strategy', 'Leverages model diversity'],
                limitations: ['Increased complexity', 'Risk of overfitting', 'Computationally expensive']
              }
            },
            { 
              id: 'voting', 
              title: 'Voting', 
              description: 'Combining predictions by majority',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'Voting ensembles combine multiple models by taking majority vote (hard) or average (soft) of predictions.',
                howItWorks: 'Each model makes prediction, final prediction is determined by majority vote or weighted average.',
                applications: ['Classification tasks', 'Model combination', 'Reducing prediction variance'],
                advantages: ['Simple and effective', 'Reduces overfitting', 'Improves robustness'],
                limitations: ['All models weighted equally', 'May not be optimal', 'Requires diverse models']
              }
            },
            { 
              id: 'averaging', 
              title: 'Averaging', 
              description: 'Combining predictions by averaging outputs from multiple models, typically used in regression',
              category: 'algorithms',
              color: '195 85% 60%',
              content: {
                overview: 'Model averaging combines predictions from multiple models by computing their weighted or simple average.',
                howItWorks: 'Trains multiple models independently and combines their predictions through averaging.',
                applications: ['Regression tasks', 'Time series forecasting', 'Risk modeling', 'Ensemble learning'],
                advantages: ['Reduces variance', 'Simple to implement', 'Often improves accuracy'],
                limitations: ['May not capture model interactions', 'Equal weighting may be suboptimal', 'Requires model diversity']
              }
            }
          ]
        }
      ]
    }
  ]
};