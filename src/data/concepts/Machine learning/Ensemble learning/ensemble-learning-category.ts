import { ConceptNode } from '../../../types';
import { bagging } from './Bagging/Bagging'
import { boosting} from './Boosting/Boosting'
import { stacking } from './Stacking'
import { voting } from './Voting'
import { averaging } from './Averaging';

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
