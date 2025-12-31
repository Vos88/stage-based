import { ConceptNode } from '../../../../types';

export const adaboost: ConceptNode = {
  id: 'adaboost',
  title: 'AdaBoost',
  description: 'Adaptive boosting method using weighted weak learners',
  color: "bg-gradient-to-br from-red-500 to-rose-500",
  overview: 'AdaBoost adaptively adjusts weights of training examples based on previous classifier errors.',
  howItWorks: 'Sequentially trains weak learners on weighted datasets, increasing weights of misclassified examples.',
  applications: ['Face detection', 'Object recognition', 'Text classification', 'Medical diagnosis'],
  advantages: ['Simple and effective', 'Automatic feature selection', 'Good generalization'],
  limitations: ['Sensitive to noise and outliers', 'Can overfit', 'Performance depends on weak learner choice'],
  children: []
};
