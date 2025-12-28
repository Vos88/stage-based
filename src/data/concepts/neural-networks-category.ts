import { ConceptNode } from '../types';
import { mlp, cnn, transformers, rnn, gan, diffusion, autoencoders } from './neural-networks';

export const neuralNetworks: ConceptNode = {
  id: 'neural-networks',
  title: 'Neural Networks',
  description: 'Computing systems inspired by biological neural networks',
  color: "bg-gradient-to-br from-violet-500 to-indigo-600",
  children: [
    mlp,
    cnn,
    transformers,
    rnn,
    gan,
    diffusion,
    autoencoders
  ]
};

