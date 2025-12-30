import { ConceptNode } from '../../../types';

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
