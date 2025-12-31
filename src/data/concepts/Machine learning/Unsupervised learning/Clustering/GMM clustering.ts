import { ConceptNode } from '../../../../types';

export const gmm: ConceptNode = {
  id: 'gmm',
  title: 'Gaussian Mixture Models (GMM)',
  description: 'Probabilistic clustering with Gaussian distributions',
  color: "bg-gradient-to-br from-sky-400 to-cyan-500",
  overview: 'GMM assumes data comes from a mixture of Gaussian distributions and estimates their parameters.',
  howItWorks: 'Uses Expectation-Maximization algorithm to estimate mixture components and assignment probabilities.',
  applications: ['Speech recognition', 'Computer vision', 'Density estimation', 'Anomaly detection'],
  advantages: ['Probabilistic output', 'Flexible cluster shapes', 'Handles overlapping clusters'],
  limitations: ['Computationally intensive', 'Sensitive to initialization', 'Requires choosing number of components'],
  children: []
};
