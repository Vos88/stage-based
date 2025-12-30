import { ConceptNode } from '../../../../types';

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
