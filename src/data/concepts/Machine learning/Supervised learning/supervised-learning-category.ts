import { ConceptNode } from '../../../types';
import { 
  polynomialRegression, 
  linearRegression, 
  ridgeLasso, 
  decisionTreeRegressor, 
  knnRegressor 
} from './Regression/regression-category';
import { 
  logisticRegression, 
  naiveBayes, 
  decisionTreeClassifier, 
  svm, 
  knn 
} from './Classification/classification-category';

export const regression: ConceptNode = {
  id: 'regression',
  title: 'Regression',
  description: 'Predicting continuous numerical values',
  color: "bg-gradient-to-br from-emerald-500 to-teal-600",
  children: [
    polynomialRegression,
    linearRegression,
    ridgeLasso,
    decisionTreeRegressor,
    knnRegressor
  ]
};

export const classification: ConceptNode = {
  id: 'classification',
  title: 'Classification',
  description: 'Predicting discrete class labels',
  color: "bg-gradient-to-br from-violet-500 to-purple-600",
  children: [
    logisticRegression,
    naiveBayes,
    decisionTreeClassifier,
    svm,
    knn
  ]
};

export const supervisedLearning: ConceptNode = {
  id: 'supervised-learning',
  title: 'Supervised Learning',
  description: 'Learning with labeled training data',
  color: "bg-gradient-to-br from-green-500 to-emerald-600",
  children: [
    regression,
    classification
  ]
};

