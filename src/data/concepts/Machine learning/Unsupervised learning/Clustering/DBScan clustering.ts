import { ConceptNode } from '../../../../types';

export const dbscan: ConceptNode = {
  id: 'dbscan',
  title: 'DBSCAN',
  description: 'Density-based clustering algorithm',
  color: "bg-gradient-to-br from-blue-400 to-cyan-500",
  overview: 'DBSCAN groups together points in high-density areas and marks points in low-density areas as outliers.',
  howItWorks: 'Identifies core points with sufficient neighbors and expands clusters by connecting density-reachable points.',
  applications: ['Anomaly detection', 'Image processing', 'Social network analysis', 'Spatial data analysis'],
  advantages: ['Finds arbitrary shaped clusters', 'Automatically determines outliers', 'Robust to noise'],
  limitations: ['Sensitive to hyperparameters', 'Struggles with varying densities', 'Memory intensive'],
  children: []
};
