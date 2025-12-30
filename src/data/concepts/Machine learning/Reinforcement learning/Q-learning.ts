import { ConceptNode } from '../../../types';

export const qLearning: ConceptNode = {
  id: 'q-learning',
  title: 'Q-Learning',
  description: 'Model-free reinforcement learning algorithm',
  color: "bg-gradient-to-br from-orange-400 to-yellow-500",
  overview: 'Q-learning learns optimal action-value function without requiring a model of the environment.',
  howItWorks: 'Updates Q-values using Bellman equation based on rewards received from actions in states.',
  applications: ['Game playing', 'Robot navigation', 'Trading strategies', 'Resource allocation'],
  advantages: ['Model-free', 'Guaranteed convergence', 'Off-policy learning'],
  limitations: ['Requires discrete state/action spaces', 'Slow convergence', 'Memory intensive for large state spaces'],
  children: []
};
