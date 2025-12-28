import { ConceptNode } from '../types';
import { bagging, boosting, stacking, voting, averaging } from './ensemble-learning';

export const ensembleLearning: ConceptNode = {
  id: 'ensemble-learning',
  title: 'Ensemble Learning',
  description: 'Combining multiple models to improve performance and robustness',
  color: "bg-gradient-to-br from-rose-500 to-pink-600",
  children: [
    bagging,
    boosting,
    stacking,
    voting,
    averaging
  ]
};

