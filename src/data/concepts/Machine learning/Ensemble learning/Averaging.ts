import { ConceptNode } from '../../../types';

export const averaging: ConceptNode = {
  id: 'averaging',
  title: 'Averaging',
  description: 'Combining predictions by averaging outputs from multiple models, typically used in regression',
  color: "bg-gradient-to-br from-red-400 to-rose-500",
  overview: 'Model averaging combines predictions from multiple models by computing their weighted or simple average.',
  howItWorks: 'Trains multiple models independently and combines their predictions through averaging.',
  applications: ['Regression tasks', 'Time series forecasting', 'Risk modeling', 'Ensemble learning'],
  advantages: ['Reduces variance', 'Simple to implement', 'Often improves accuracy'],
  limitations: ['May not capture model interactions', 'Equal weighting may be suboptimal', 'Requires model diversity'],
  children: []
};
