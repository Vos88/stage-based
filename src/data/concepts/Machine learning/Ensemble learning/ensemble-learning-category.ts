import { ConceptNode } from '../../../types';
import { bagging } from './Bagging/bagging-category'
import { boosting } from './Boosting/boosting-category'
import { stacking } from './Stacking'
import { voting } from './Voting'
import { averaging } from './Averaging';

export const ensembleLearning: ConceptNode = {
  id: 'ensemble-learning',
  title: 'Ensemble Learning',
  description: 'Combining multiple models to improve performance and robustness',
  color: "bg-gradient-to-br from-amber-500 to-orange-600",
  children: [
    bagging,
    boosting,
    stacking,
    voting,
    averaging
  ]
};
