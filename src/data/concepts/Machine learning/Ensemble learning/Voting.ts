import { ConceptNode } from '../../../types';

export const voting: ConceptNode = {
  id: 'voting',
  title: 'Voting',
  description: 'Combining predictions by majority',
  color: "bg-gradient-to-br from-rose-400 to-orange-500",
  overview: 'Voting ensembles combine multiple models by taking majority vote (hard) or average (soft) of predictions.',
  howItWorks: 'Each model makes prediction, final prediction is determined by majority vote or weighted average.',
  applications: ['Classification tasks', 'Model combination', 'Reducing prediction variance'],
  advantages: ['Simple and effective', 'Reduces overfitting', 'Improves robustness'],
  limitations: ['All models weighted equally', 'May not be optimal', 'Requires diverse models'],
  children: []
};
