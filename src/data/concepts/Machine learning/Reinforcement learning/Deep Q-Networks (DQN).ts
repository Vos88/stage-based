import { ConceptNode } from '../../../types';

export const dqn: ConceptNode = {
  id: 'dqn',
  title: 'Deep Q-Networks (DQN)',
  description: 'Deep learning approach to Q-learning',
  color: "bg-gradient-to-br from-orange-500 to-red-500",
  overview: 'DQN uses deep neural networks to approximate Q-values for high-dimensional state spaces.',
  howItWorks: 'Combines Q-learning with deep networks, using experience replay and target networks for stability.',
  applications: ['Video game AI', 'Robotics', 'Autonomous vehicles', 'Strategic planning'],
  advantages: ['Handles high-dimensional states', 'End-to-end learning', 'Scales to complex problems'],
  limitations: ['Sample inefficient', 'Unstable training', 'Requires careful hyperparameter tuning'],
  children: []
};
