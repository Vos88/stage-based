import { ConceptNode } from '../../../../types';

export const xgboost: ConceptNode = {
  id: 'xgboost',
  title: 'XGBoost',
  description: 'Optimized gradient boosting library for performance and scalability',
  color: "bg-gradient-to-br from-rose-500 to-red-600",
  overview: 'XGBoost is an optimized gradient boosting framework designed for speed and performance.',
  howItWorks: 'Uses second-order gradients, regularization, and optimized data structures for efficient training.',
  applications: ['Machine learning competitions', 'Click-through rate prediction', 'Risk modeling', 'Ranking systems'],
  advantages: ['State-of-the-art performance', 'Fast training', 'Built-in regularization'],
  limitations: ['Many hyperparameters', 'Memory intensive', 'Requires feature engineering'],
  children: []
};
