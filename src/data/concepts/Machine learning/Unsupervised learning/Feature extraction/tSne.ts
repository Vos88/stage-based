import { ConceptNode } from '../../../../types';

export const tSne: ConceptNode = {
  id: 't-sne',
  title: 't-SNE',
  description: 'Non-linear dimensionality reduction technique for visualization',
  color: "bg-gradient-to-br from-cyan-400 to-blue-500",
  overview: 't-SNE preserves local structure by modeling pairwise similarities in high and low dimensions.',
  howItWorks: 'Minimizes divergence between probability distributions of pairwise similarities in original and reduced space.',
  applications: ['Data visualization', 'Exploratory data analysis', 'Cluster visualization', 'Image analysis'],
  advantages: ['Preserves local structure', 'Reveals clusters', 'Non-linear mapping'],
  limitations: ['Computationally expensive', 'Non-deterministic', 'Hyperparameter sensitive'],
  children: []
};
