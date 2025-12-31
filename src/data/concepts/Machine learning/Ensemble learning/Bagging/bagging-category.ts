import { ConceptNode } from '../../../../types';
import { randomForest } from './Random Forest'

export const bagging: ConceptNode = {
  id: 'bagging',
  title: 'Bagging',
  description: 'Building multiple models independently and combining their predictions to reduce variance and improve robustness',
  color: "bg-gradient-to-br from-orange-500 to-red-600",
  children: [
    randomForest,
  ]
};
