import { ConceptNode } from '../../../types';

export const mlp: ConceptNode = {
  id: 'mlp',
  title: 'Multi-layer Perceptron',
  description: 'Feedforward artificial neural network',
  color: "bg-gradient-to-br from-indigo-400 to-blue-500",
  overview: 'MLPs consist of multiple layers of neurons with non-linear activation functions for learning complex patterns.',
  howItWorks: 'Forward propagates input through hidden layers and uses backpropagation to update weights.',
  applications: ['Pattern recognition', 'Function approximation', 'Classification', 'Regression'],
  advantages: ['Universal function approximator', 'Non-linear modeling', 'Flexible architecture'],
  limitations: ['Requires large amounts of data', 'Prone to overfitting', 'Black box nature'],
  children: []
};

export const cnn: ConceptNode = {
  id: 'cnn',
  title: 'Convolutional Neural Network',
  description: 'Deep learning architecture for processing grid-like data',
  color: "bg-gradient-to-br from-blue-400 to-indigo-500",
  overview: 'CNNs use convolutional layers to detect local features and pooling layers to reduce dimensionality.',
  howItWorks: 'Applies learnable filters across input to detect features, followed by pooling for translation invariance.',
  applications: ['Image classification', 'Object detection', 'Medical imaging', 'Computer vision'],
  advantages: ['Translation invariant', 'Parameter sharing', 'Hierarchical feature learning'],
  limitations: ['Requires large datasets', 'Computationally intensive', 'Not suitable for non-grid data'],
  children: []
};

export const nlp: ConceptNode = {
  id: 'nlp',
  title: 'Natural Language Processing (NLP)',
  description: 'AI for understanding and generating human language',
  color: "bg-gradient-to-br from-indigo-500 to-blue-600",
  overview: 'NLP combines computational linguistics with machine learning to process and understand human language.',
  howItWorks: 'Uses various techniques from tokenization to deep learning for language understanding and generation.',
  applications: ['Chatbots', 'Machine translation', 'Sentiment analysis', 'Document summarization'],
  advantages: ['Versatile applications', 'Improving rapidly', 'Transfer learning'],
  limitations: ['Context understanding', 'Ambiguity handling', 'Cultural and linguistic biases'],
  children: []
};

export const transformers: ConceptNode = {
  id: 'transformers',
  title: 'Transformers',
  description: 'Attention-based architecture for sequence modeling and generation',
  color: "bg-gradient-to-br from-blue-500 to-indigo-600",
  overview: 'Transformers use self-attention mechanisms to process sequences in parallel and capture long-range dependencies.',
  howItWorks: 'Attention mechanism computes weighted representations based on similarity between sequence elements.',
  applications: ['Machine translation', 'Text generation', 'Question answering', 'Code generation'],
  advantages: ['Parallel processing', 'Long-range dependencies', 'Transfer learning'],
  limitations: ['Memory intensive', 'Requires large datasets', 'Quadratic complexity with sequence length'],
  children: [nlp]
};

export const rnn: ConceptNode = {
  id: 'rnn',
  title: 'Recurrent Neural Network (RNN)',
  description: 'Neural network architecture for sequential data using temporal dependencies',
  color: "bg-gradient-to-br from-indigo-500 to-violet-600",
  overview: 'RNNs process sequential data by maintaining hidden states that capture information from previous time steps.',
  howItWorks: 'Uses recurrent connections to maintain memory of previous inputs while processing sequences.',
  applications: ['Time series prediction', 'Speech recognition', 'Language modeling', 'Sequence generation'],
  advantages: ['Handles variable-length sequences', 'Memory of past information', 'Parameter sharing across time'],
  limitations: ['Vanishing gradient problem', 'Sequential processing', 'Difficulty with long sequences'],
  children: []
};

export const gan: ConceptNode = {
  id: 'gan',
  title: 'Generative Adversarial Network (GAN)',
  description: 'Neural network architecture for generating realistic data through adversarial training',
  color: "bg-gradient-to-br from-blue-400 to-indigo-600",
  overview: 'GANs consist of generator and discriminator networks competing against each other to generate realistic data.',
  howItWorks: 'Generator creates fake data while discriminator learns to distinguish real from fake data.',
  applications: ['Image generation', 'Data augmentation', 'Style transfer', 'Super resolution'],
  advantages: ['High-quality generation', 'Unsupervised learning', 'Flexible data types'],
  limitations: ['Training instability', 'Mode collapse', 'Difficult to evaluate'],
  children: []
};

export const diffusion: ConceptNode = {
  id: 'diffusion',
  title: 'Diffusion Models',
  description: 'Generative models that learn to reverse a noise process to synthesize data',
  color: "bg-gradient-to-br from-violet-400 to-indigo-500",
  overview: 'Diffusion models learn to denoise data by reversing a gradual noise addition process.',
  howItWorks: 'Trains neural network to predict noise added at each step of forward diffusion process.',
  applications: ['Image generation', 'Audio synthesis', 'Video generation', 'Molecular design'],
  advantages: ['High-quality samples', 'Stable training', 'Controllable generation'],
  limitations: ['Slow sampling', 'Computationally expensive', 'Many sampling steps required'],
  children: []
};

export const autoencoders: ConceptNode = {
  id: 'autoencoders',
  title: 'Autoencoders',
  description: 'Neural networks for feature learning',
  color: "bg-gradient-to-br from-indigo-400 to-violet-500",
  overview: 'Autoencoders learn efficient data representations by encoding input to lower dimension and reconstructing it.',
  howItWorks: 'Encoder compresses input to latent representation, decoder reconstructs original input from latent code.',
  applications: ['Dimensionality reduction', 'Anomaly detection', 'Denoising', 'Data compression'],
  advantages: ['Unsupervised learning', 'Learns meaningful representations', 'Flexible architecture'],
  limitations: ['May lose important information', 'Requires careful architecture design', 'Prone to overfitting'],
  children: []
};

