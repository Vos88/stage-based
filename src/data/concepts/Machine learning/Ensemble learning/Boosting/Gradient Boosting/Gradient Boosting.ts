import { ConceptNode } from '../../../../../types'
import { xgboost } from './XGBoost'

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
