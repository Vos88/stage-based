import { ConceptNode } from '../../types';
import { supervisedLearning } from './Supervised learning/supervised-learning-category';
import { unsupervisedLearning } from './Unsupervised learning/unsupervised-learning-category';
import { reinforcementLearning } from './Reinforcement learning/reinforcement-learning-category';
import { neuralNetworks } from './Neural networks/neural-networks-category';
import { ensembleLearning } from './Ensemble learning/ensemble-learning-category';

export const machineLearning: ConceptNode = {
  id: 'machine-learning',
  title: 'Machine Learning',
  description: 'Algorithms that improve automatically through experience',
  color: "bg-gradient-to-br from-blue-500 to-cyan-600",
  children: [
    supervisedLearning,
    unsupervisedLearning,
    reinforcementLearning,
    neuralNetworks,
    ensembleLearning
  ]
};
