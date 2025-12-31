import { ConceptNode } from '../../../../types';

export const kMeans: ConceptNode = {
  id: 'k-means',
  title: 'K-Means',
  description: 'Partitioning data into k clusters',
  color: "bg-gradient-to-br from-cyan-400 to-sky-500",
  overview: `K-means clustering partitions data into k clusters by minimizing within-cluster sum of squares. The objective function is:

$$\\min_{\\{S_1, S_2, \\ldots, S_k\\}} \\sum_{i=1}^{k} \\sum_{x \\in S_i} ||x - \\mu_i||^2$$

where $S_i$ are the clusters, $\\mu_i$ is the centroid of cluster $i$, and $||x - \\mu_i||^2$ is the squared Euclidean distance.`,
  howItWorks: 'Iteratively assigns points to nearest centroid and updates centroids until convergence.',
  applications: ['Customer segmentation', 'Image segmentation', 'Market research', 'Data compression'],
  advantages: ['Simple and fast', 'Works well with globular clusters', 'Guaranteed convergence'],
  limitations: ['Requires specifying k', 'Sensitive to initialization', 'Assumes spherical clusters'],
  children: []
};
