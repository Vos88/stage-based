import { ConceptNode } from '../../../../types';
import { gradientBoosting } from './Gradient Boosting/gradient-boosting-category';
import { adaboost } from './Adaboost';

export const boosting: ConceptNode = {
  id: 'boosting',
  title: 'Boosting',
  description: 'Sequentially combining weak learners to reduce bias',
  color: "bg-gradient-to-br from-red-500 to-rose-600",
  children: [gradientBoosting, adaboost]
};
