import { ConceptNode } from '../types';
import { kMeans, gmm, dbscan, pca, tSne } from './unsupervised-learning';

export const clustering: ConceptNode = {
  id: 'clustering',
  title: 'Clustering',
  description: 'Grouping similar data points together',
  color: "bg-gradient-to-br from-cyan-500 to-sky-600",
  children: [
    kMeans,
    gmm,
    dbscan
  ]
};

export const featureExtraction: ConceptNode = {
  id: 'feature-extraction',
  title: 'Feature Extraction',
  description: 'Reducing dimensionality while preserving important information',
  color: "bg-gradient-to-br from-sky-500 to-blue-600",
  children: [
    pca,
    tSne
  ]
};

export const unsupervisedLearning: ConceptNode = {
  id: 'unsupervised-learning',
  title: 'Unsupervised Learning',
  description: 'Learning patterns from unlabeled data',
  color: "bg-gradient-to-br from-cyan-500 to-teal-600",
  children: [
    clustering,
    featureExtraction
  ]
};

