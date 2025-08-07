export interface ConceptNode {
  id: string;
  title: string;
  description: string;
  category: string;
  color: string;
  children?: ConceptNode[];
  content?: {
    codeExample?: string;
    codeLanguage?: string;
    links?: { title: string; url: string }[];
    image?: string;
    keyPoints?: string[];
  };
}

export const aiConceptTree: ConceptNode = {
  id: "artificial-intelligence",
  title: "Artificial Intelligence",
  description: "The simulation of human intelligence in machines that are programmed to think and learn like humans.",
  category: "ai-root",
  color: "258 96% 67%",
  content: {
    keyPoints: [
      "Intelligence demonstrated by machines",
      "Encompasses learning, reasoning, and perception",
      "Applications span from simple automation to complex decision-making",
      "Foundation for modern technological advancement"
    ],
    links: [
      { title: "MIT AI Course", url: "https://ocw.mit.edu/courses/6-034-artificial-intelligence-fall-2010/" },
      { title: "Stanford AI", url: "https://ai.stanford.edu/" }
    ]
  },
  children: [
    {
      id: "machine-learning",
      title: "Machine Learning",
      description: "A subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
      category: "ml-core",
      color: "215 100% 65%",
      content: {
        codeExample: `# Basic ML workflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)`,
        codeLanguage: "python",
        keyPoints: [
          "Learns patterns from data automatically",
          "Improves performance with more data",
          "Three main types: supervised, unsupervised, reinforcement"
        ]
      },
      children: [
        {
          id: "supervised-learning",
          title: "Supervised Learning",
          description: "Learning with labeled training data to predict outcomes for new data.",
          category: "supervised",
          color: "25 95% 60%",
          children: [
            {
              id: "regression",
              title: "Regression",
              description: "Predicts continuous numerical values.",
              category: "supervised",
              color: "25 95% 60%",
              content: {
                codeExample: `from sklearn.linear_model import LinearRegression
import numpy as np

# Create sample data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([2, 4, 6, 8, 10])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict
prediction = model.predict([[6]])
print(f"Prediction for x=6: {prediction[0]}")`,
                codeLanguage: "python",
                keyPoints: [
                  "Predicts continuous values",
                  "Examples: house prices, stock prices, temperature",
                  "Evaluates using metrics like MSE, MAE, R²"
                ]
              },
              children: [
                {
                  id: "linear-regression",
                  title: "Linear Regression",
                  description: "Models relationship between variables using a linear equation.",
                  category: "algorithms",
                  color: "195 85% 60%",
                  content: {
                    codeExample: `# Simple Linear Regression
y = mx + b

# Multiple Linear Regression  
y = b0 + b1*x1 + b2*x2 + ... + bn*xn`,
                    codeLanguage: "python"
                  }
                },
                {
                  id: "polynomial-regression",
                  title: "Polynomial Regression",
                  description: "Models non-linear relationships using polynomial equations.",
                  category: "algorithms",
                  color: "195 85% 60%"
                },
                {
                  id: "ridge-regression",
                  title: "Ridge Regression",
                  description: "Linear regression with L2 regularization to prevent overfitting.",
                  category: "algorithms",
                  color: "195 85% 60%"
                }
              ]
            },
            {
              id: "classification",
              title: "Classification",
              description: "Predicts discrete categories or classes.",
              category: "supervised",
              color: "25 95% 60%",
              content: {
                keyPoints: [
                  "Predicts discrete categories",
                  "Examples: spam detection, image recognition, medical diagnosis",
                  "Evaluates using accuracy, precision, recall, F1-score"
                ]
              },
              children: [
                {
                  id: "logistic-regression",
                  title: "Logistic Regression",
                  description: "Uses logistic function for binary classification.",
                  category: "algorithms",
                  color: "195 85% 60%"
                },
                {
                  id: "decision-trees",
                  title: "Decision Trees",
                  description: "Tree-like model for making decisions based on feature values.",
                  category: "algorithms",
                  color: "195 85% 60%"
                },
                {
                  id: "svm",
                  title: "Support Vector Machines",
                  description: "Finds optimal boundary to separate different classes.",
                  category: "algorithms",
                  color: "195 85% 60%"
                }
              ]
            }
          ]
        },
        {
          id: "unsupervised-learning",
          title: "Unsupervised Learning",
          description: "Finds hidden patterns in data without labeled examples.",
          category: "unsupervised",
          color: "315 85% 70%",
          children: [
            {
              id: "clustering",
              title: "Clustering",
              description: "Groups similar data points together.",
              category: "unsupervised",
              color: "315 85% 70%",
              children: [
                {
                  id: "k-means",
                  title: "K-Means",
                  description: "Partitions data into k clusters based on similarity.",
                  category: "algorithms",
                  color: "195 85% 60%"
                },
                {
                  id: "hierarchical",
                  title: "Hierarchical Clustering",
                  description: "Creates tree-like cluster structures.",
                  category: "algorithms",
                  color: "195 85% 60%"
                }
              ]
            },
            {
              id: "dimensionality-reduction",
              title: "Dimensionality Reduction",
              description: "Reduces the number of features while preserving important information.",
              category: "unsupervised",
              color: "315 85% 70%",
              children: [
                {
                  id: "pca",
                  title: "Principal Component Analysis",
                  description: "Finds principal components that explain variance in data.",
                  category: "algorithms",
                  color: "195 85% 60%"
                },
                {
                  id: "tsne",
                  title: "t-SNE",
                  description: "Non-linear dimensionality reduction for visualization.",
                  category: "algorithms",
                  color: "195 85% 60%"
                }
              ]
            }
          ]
        },
        {
          id: "reinforcement-learning",
          title: "Reinforcement Learning",
          description: "Learns optimal actions through trial and error interactions with an environment.",
          category: "reinforcement",
          color: "285 85% 65%",
          content: {
            codeExample: `# Q-Learning basics
import numpy as np

class QLearning:
    def __init__(self, states, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = np.zeros((states, actions))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate
    
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.q_table.shape[1])
        return np.argmax(self.q_table[state])`,
            codeLanguage: "python"
          },
          children: [
            {
              id: "q-learning",
              title: "Q-Learning",
              description: "Model-free reinforcement learning algorithm.",
              category: "algorithms",
              color: "195 85% 60%"
            },
            {
              id: "policy-gradient",
              title: "Policy Gradient",
              description: "Directly optimizes the policy function.",
              category: "algorithms",
              color: "195 85% 60%"
            }
          ]
        }
      ]
    },
    {
      id: "deep-learning",
      title: "Deep Learning",
      description: "ML using artificial neural networks with multiple layers to model complex patterns.",
      category: "deep-learning",
      color: "210 100% 55%",
      content: {
        codeExample: `import tensorflow as tf

# Simple neural network
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])`,
        codeLanguage: "python"
      },
      children: [
        {
          id: "neural-networks",
          title: "Neural Networks",
          description: "Networks of interconnected nodes inspired by biological neurons.",
          category: "deep-learning",
          color: "210 100% 55%",
          children: [
            {
              id: "cnn",
              title: "Convolutional Neural Networks",
              description: "Specialized for processing grid-like data such as images.",
              category: "computer-vision",
              color: "45 95% 65%"
            },
            {
              id: "rnn",
              title: "Recurrent Neural Networks",
              description: "Networks with memory for sequential data processing.",
              category: "nlp",
              color: "165 85% 55%"
            }
          ]
        }
      ]
    },
    {
      id: "natural-language-processing",
      title: "Natural Language Processing",
      description: "AI field focused on interaction between computers and human language.",
      category: "nlp",
      color: "165 85% 55%",
      children: [
        {
          id: "text-processing",
          title: "Text Processing",
          description: "Fundamental techniques for preparing and analyzing text data.",
          category: "nlp",
          color: "165 85% 55%"
        },
        {
          id: "language-models",
          title: "Language Models",
          description: "Models that understand and generate human language.",
          category: "nlp",
          color: "165 85% 55%"
        }
      ]
    },
    {
      id: "computer-vision",
      title: "Computer Vision",
      description: "AI field that trains computers to interpret and understand visual information.",
      category: "computer-vision",
      color: "45 95% 65%",
      children: [
        {
          id: "image-classification",
          title: "Image Classification",
          description: "Categorizing images into predefined classes.",
          category: "computer-vision",
          color: "45 95% 65%"
        },
        {
          id: "object-detection",
          title: "Object Detection",
          description: "Identifying and locating objects within images.",
          category: "computer-vision",
          color: "45 95% 65%"
        }
      ]
    }
  ]
};