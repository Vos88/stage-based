import { ConceptNode } from '../types';

export const randomForest: ConceptNode = {
  id: 'random-forest',
  title: 'Random Forest',
  description: 'Ensemble of decision trees using bagging',
  color: "bg-gradient-to-br from-red-400 to-rose-500",
  overview: 'Random Forest combines multiple decision trees trained on random subsets of data and features.',
  howItWorks: 'Trains multiple trees on bootstrap samples with random feature selection and averages predictions.',
  applications: ['Feature importance', 'Classification', 'Regression', 'Bioinformatics'],
  advantages: ['Reduces overfitting', 'Handles missing values', 'Provides feature importance'],
  limitations: ['Less interpretable than single tree', 'Can overfit with very noisy data', 'Memory intensive'],
  children: []
};

export const bagging: ConceptNode = {
  id: 'bagging',
  title: 'Bagging',
  description: 'Bootstrap aggregation to reduce variance',
  color: "bg-gradient-to-br from-orange-500 to-red-600",
  children: [randomForest]
};

export const xgboost: ConceptNode = {
  id: 'xgboost',
  title: 'XGBoost',
  description: 'Optimized gradient boosting library for performance and scalability',
  color: "bg-gradient-to-br from-rose-400 to-red-500",
  overview: 'XGBoost is an optimized gradient boosting framework designed for speed and performance.',
  howItWorks: 'Uses second-order gradients, regularization, and optimized data structures for efficient training.',
  applications: ['Machine learning competitions', 'Click-through rate prediction', 'Risk modeling', 'Ranking systems'],
  advantages: ['State-of-the-art performance', 'Fast training', 'Built-in regularization'],
  limitations: ['Many hyperparameters', 'Memory intensive', 'Requires feature engineering'],
  children: []
};

export const gradientBoosting: ConceptNode = {
  id: 'gradient-boosting',
  title: 'Gradient Boosting',
  description: 'Boosting method for regression and classification',
  color: "bg-gradient-to-br from-red-400 to-orange-500",
  overview: 'Gradient boosting builds models sequentially, with each model correcting errors of previous models.',
  howItWorks: 'Fits new models to residual errors of ensemble, using gradient descent to minimize loss function.',
  applications: ['Tabular data prediction', 'Ranking', 'Regression', 'Feature selection'],
  advantages: ['High predictive accuracy', 'Handles different data types', 'Built-in feature selection'],
  limitations: ['Prone to overfitting', 'Sensitive to hyperparameters', 'Sequential training'],
  children: [xgboost]
};

export const adaboost: ConceptNode = {
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
};

export const boosting: ConceptNode = {
  id: 'boosting',
  title: 'Boosting',
  description: 'Sequentially combining weak learners to reduce bias',
  color: "bg-gradient-to-br from-red-500 to-rose-600",
  children: [gradientBoosting, adaboost]
};

export const stacking: ConceptNode = {
  id: 'stacking',
  title: 'Stacking',
  description: 'Combining multiple models using a meta-learner',
  color: "bg-gradient-to-br from-pink-400 to-rose-500",
  overview: 'Stacking trains a meta-model to optimally combine predictions from multiple base models.',
  howItWorks: 'Base models make predictions, then meta-learner is trained on these predictions to make final prediction.',
  applications: ['Machine learning competitions', 'Complex prediction tasks', 'Model combination'],
  advantages: ['Can improve upon best individual model', 'Flexible combination strategy', 'Leverages model diversity'],
  limitations: ['Increased complexity', 'Risk of overfitting', 'Computationally expensive'],
  children: []
};

export const voting: ConceptNode = {
  id: 'voting',
  title: 'Voting',
  description: 'Combining predictions by majority',
  color: "bg-gradient-to-br from-rose-400 to-pink-500",
  overview: 'Voting ensembles combine multiple models by taking majority vote (hard) or average (soft) of predictions.',
  howItWorks: 'Each model makes prediction, final prediction is determined by majority vote or weighted average.',
  applications: ['Classification tasks', 'Model combination', 'Reducing prediction variance'],
  advantages: ['Simple and effective', 'Reduces overfitting', 'Improves robustness'],
  limitations: ['All models weighted equally', 'May not be optimal', 'Requires diverse models'],
  children: []
};

export const averaging: ConceptNode = {
  id: 'averaging',
  title: 'Averaging',
  description: 'Combining predictions by averaging outputs from multiple models, typically used in regression',
  color: "bg-gradient-to-br from-red-500 to-orange-600",
  overview: 'Model averaging combines predictions from multiple models by computing their weighted or simple average.',
  howItWorks: 'Trains multiple models independently and combines their predictions through averaging.',
  applications: ['Regression tasks', 'Time series forecasting', 'Risk modeling', 'Ensemble learning'],
  advantages: ['Reduces variance', 'Simple to implement', 'Often improves accuracy'],
  limitations: ['May not capture model interactions', 'Equal weighting may be suboptimal', 'Requires model diversity'],
  children: []
};

