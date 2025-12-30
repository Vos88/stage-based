import { ConceptNode } from '../../../types';
import { qLearning } from './Q-learning';
import { dqn } from './Deep Q-Networks (DQN)';
import { policyGradient } from './Policy Gradient Methods';

export const reinforcementLearning: ConceptNode = {
  id: 'reinforcement-learning',
  title: 'Reinforcement Learning',
  description: 'Learning through interaction with environment via rewards and penalties',
  color: "bg-gradient-to-br from-orange-500 to-amber-600",
  children: [
    qLearning,
    dqn,
    policyGradient
  ]
};
