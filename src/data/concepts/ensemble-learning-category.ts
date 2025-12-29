import { ConceptNode } from '../types';
import { bagging, boosting, stacking, voting, averaging } from './ensemble-learning';

export const ensembleLearning: ConceptNode = {
  id: 'ensemble-learning',
  title: 'Ensemble Learning',
  description: 'Combining multiple models to improve performance and robustness',
  color: "bg-gradient-to-br from-red-500 to-rose-600",
  children: [
    bagging,
    boosting,
    stacking,
    voting,
    averaging
  ]
};

