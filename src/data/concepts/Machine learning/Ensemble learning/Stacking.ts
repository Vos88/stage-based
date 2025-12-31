import { ConceptNode } from '../../../types';

export const stacking: ConceptNode = {
  id: 'stacking',
  title: 'Stacking',
  description: 'Combining multiple models using a meta-learner',
  color: "bg-gradient-to-br from-yellow-500 to-amber-600",
  overview: 'Stacking trains a meta-model to optimally combine predictions from multiple base models.',
  howItWorks: 'Base models make predictions, then meta-learner is trained on these predictions to make final prediction.',
  applications: ['Machine learning competitions', 'Complex prediction tasks', 'Model combination'],
  advantages: ['Can improve upon best individual model', 'Flexible combination strategy', 'Leverages model diversity'],
  limitations: ['Increased complexity', 'Risk of overfitting', 'Computationally expensive'],
  children: []
};
