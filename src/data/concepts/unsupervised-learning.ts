import { ConceptNode } from '../types';

export const kMeans: ConceptNode = {
  id: 'k-means',
  title: 'K-Means',
  description: 'Partitioning data into k clusters',
  color: "bg-gradient-to-br from-blue-500 to-indigo-600",
  overview: `K-means clustering partitions data into k clusters by minimizing within-cluster sum of squares. The objective function is:

$$\\min_{\\{S_1, S_2, \\ldots, S_k\\}} \\sum_{i=1}^{k} \\sum_{x \\in S_i} ||x - \\mu_i||^2$$

where $S_i$ are the clusters, $\\mu_i$ is the centroid of cluster $i$, and $||x - \\mu_i||^2$ is the squared Euclidean distance.`,
  howItWorks: 'Iteratively assigns points to nearest centroid and updates centroids until convergence.',
  applications: ['Customer segmentation', 'Image segmentation', 'Market research', 'Data compression'],
  advantages: ['Simple and fast', 'Works well with globular clusters', 'Guaranteed convergence'],
  limitations: ['Requires specifying k', 'Sensitive to initialization', 'Assumes spherical clusters'],
  children: []
};

export const gmm: ConceptNode = {
  id: 'gmm',
  title: 'Gaussian Mixture Models (GMM)',
  description: 'Probabilistic clustering with Gaussian distributions',
  color: "bg-gradient-to-br from-purple-500 to-violet-600",
  overview: 'GMM assumes data comes from a mixture of Gaussian distributions and estimates their parameters.',
  howItWorks: 'Uses Expectation-Maximization algorithm to estimate mixture components and assignment probabilities.',
  applications: ['Speech recognition', 'Computer vision', 'Density estimation', 'Anomaly detection'],
  advantages: ['Probabilistic output', 'Flexible cluster shapes', 'Handles overlapping clusters'],
  limitations: ['Computationally intensive', 'Sensitive to initialization', 'Requires choosing number of components'],
  children: []
};

export const dbscan: ConceptNode = {
  id: 'dbscan',
  title: 'DBSCAN',
  description: 'Density-based clustering algorithm',
  color: "bg-gradient-to-br from-teal-500 to-green-600",
  overview: 'DBSCAN groups together points in high-density areas and marks points in low-density areas as outliers.',
  howItWorks: 'Identifies core points with sufficient neighbors and expands clusters by connecting density-reachable points.',
  applications: ['Anomaly detection', 'Image processing', 'Social network analysis', 'Spatial data analysis'],
  advantages: ['Finds arbitrary shaped clusters', 'Automatically determines outliers', 'Robust to noise'],
  limitations: ['Sensitive to hyperparameters', 'Struggles with varying densities', 'Memory intensive'],
  children: []
};

export const pca: ConceptNode = {
  id: 'pca',
  title: 'Principal Component Analysis (PCA)',
  description: 'Linear dimensionality reduction technique',
  color: "bg-gradient-to-br from-indigo-500 to-purple-600",
  overview: `PCA reduces dimensionality by projecting data onto principal components that capture maximum variance. For a data matrix $X$ with $n$ samples and $p$ features, PCA finds the eigenvectors of the covariance matrix $\\Sigma = \\frac{1}{n-1}X^TX$.

The first principal component $w_1$ maximizes:

$$w_1 = \\arg\\max_{||w||=1} w^T\\Sigma w$$

Subsequent components are found by maximizing variance while being orthogonal to previous components.`,
  howItWorks: 'Computes eigenvectors of covariance matrix and projects data onto top eigenvectors.',
  applications: ['Data visualization', 'Feature reduction', 'Data compression', 'Noise reduction'],
  advantages: ['Reduces overfitting', 'Removes correlation', 'Computational efficiency'],
  limitations: ['Linear transformation only', 'Components may not be interpretable', 'Sensitive to scaling'],
  children: []
};

export const tSne: ConceptNode = {
  id: 't-sne',
  title: 't-SNE',
  description: 'Non-linear dimensionality reduction technique for visualization',
  color: "bg-gradient-to-br from-cyan-500 to-blue-600",
  overview: 't-SNE preserves local structure by modeling pairwise similarities in high and low dimensions.',
  howItWorks: 'Minimizes divergence between probability distributions of pairwise similarities in original and reduced space.',
  applications: ['Data visualization', 'Exploratory data analysis', 'Cluster visualization', 'Image analysis'],
  advantages: ['Preserves local structure', 'Reveals clusters', 'Non-linear mapping'],
  limitations: ['Computationally expensive', 'Non-deterministic', 'Hyperparameter sensitive'],
  children: []
};

