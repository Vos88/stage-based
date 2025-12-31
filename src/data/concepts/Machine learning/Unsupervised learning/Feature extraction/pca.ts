import { ConceptNode } from '../../../../types';

export const pca: ConceptNode = {
  id: 'pca',
  title: 'Principal Component Analysis (PCA)',
  description: 'Linear dimensionality reduction technique',
  color: "bg-gradient-to-br from-sky-400 to-blue-500",
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
