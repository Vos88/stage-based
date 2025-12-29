import { ConceptNode } from '../types';

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

export const policyGradient: ConceptNode = {
  id: 'policy-gradient',
  title: 'Policy Gradient Methods',
  description: 'Directly optimizing policy parameters',
  color: "bg-gradient-to-br from-yellow-500 to-orange-600",
  overview: 'Policy gradient methods directly optimize policy parameters using gradient ascent on expected returns.',
  howItWorks: 'Estimates gradient of expected return with respect to policy parameters and updates parameters accordingly.',
  applications: ['Continuous control', 'Natural language generation', 'Multi-agent systems', 'Portfolio optimization'],
  advantages: ['Handles continuous actions', 'Direct policy optimization', 'Stochastic policies'],
  limitations: ['High variance gradients', 'Sample inefficient', 'Local optima'],
  children: []
};

