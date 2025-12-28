import { ConceptNode } from '../types';
import { qLearning, dqn, policyGradient } from './reinforcement-learning';

export const reinforcementLearning: ConceptNode = {
  id: 'reinforcement-learning',
  title: 'Reinforcement Learning',
  description: 'Learning through interaction with environment via rewards and penalties',
  color: "bg-gradient-to-br from-orange-500 to-red-600",
  children: [
    qLearning,
    dqn,
    policyGradient
  ]
};

