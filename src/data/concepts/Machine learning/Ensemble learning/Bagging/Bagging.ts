import { ConceptNode } from '../../../../types'
import { randomForest } from './Random Forest'

export const bagging: ConceptNode = {
  id: 'bagging',
  title: 'Bagging',
  description: 'Bootstrap aggregation to reduce variance',
  color: "bg-gradient-to-br from-orange-500 to-red-600",
  children: [randomForest]
};
