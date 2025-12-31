import { ConceptNode } from '../../../../types';
import { xgboost } from './XGBoost';
import { adaboost } from './Adaboost';

export const boosting: ConceptNode = {
  id: 'boosting',
  title: 'Boosting',
  description: 'Sequentially training models to correct errors of previous models, improving accuracy and performance',
  color: "bg-gradient-to-br from-orange-700 to-red-800",
  children: [
    adaboost,
    xgboost
  ]
};
