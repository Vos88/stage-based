import { ConceptNode } from '../types';
import { supervisedLearning } from './supervised-learning';
import { unsupervisedLearning } from './unsupervised-learning-category';
import { reinforcementLearning } from './reinforcement-learning-category';
import { neuralNetworks } from './neural-networks-category';
import { ensembleLearning } from './ensemble-learning-category';

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

