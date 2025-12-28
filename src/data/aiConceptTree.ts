import { ConceptNode } from './types';
import { symbolicAi } from './concepts/symbolic-ai';
import { machineLearning } from './concepts/machine-learning';

export type { ConceptNode } from './types';

export const aiConceptTree: ConceptNode = {
  id: 'root',
  title: 'AI',
  description: 'Artificial Intelligence - The simulation of human intelligence in machines',
  color: "bg-gradient-to-br from-purple-600 to-blue-600",
  children: [
    symbolicAi,
    machineLearning
  ]
};
