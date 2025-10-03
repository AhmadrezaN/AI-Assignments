# AI Assignments

This repository contains my solutions related to the **Artificial Intelligence assignments**.
Each assignment covers a different AI concept, ranging from search algorithms to optimization, game-playing, and machine learning.

---

## 📂 Contents

* **CA1 – Search Algorithms (Sokoban-like Problem)**

  * Implemented uninformed search algorithms: **BFS, DFS, IDS**
  * Implemented informed search algorithms: **A*, Weighted A***
  * Applied heuristics (Manhattan distance, custom heuristic design)
  * Compared algorithms based on execution time, number of explored states, and solution quality

* **CA2 – Optimization & Game Playing**

  * **Part 1: Genetic Algorithm**

    * Applied GA to approximate Fourier series coefficients
    * Chromosome representation, population initialization, crossover, mutation, and selection strategies
    * Fitness evaluation using RMSE and comparison with the original function
  * **Part 2: Pentago Game (Adversarial Search)**

    * Implemented **Minimax algorithm** (with and without alpha-beta pruning)
    * Designed heuristic evaluation functions
    * Analyzed branching factor, node expansions, and win rates against random opponents

* **CA3 – Bsic Machine Learning Models for Grade Prediction**

  * Dataset: Student demographics, social, and academic data
  * Preprocessing: Missing values, categorical encoding, normalization
  * Models implemented:

    * **Naive Bayes**
    * **Decision Tree** (with pruning, feature importance)
    * **Random Forest**
    * **XGBoost**
    * **Decision Tree from scratch**
  * Evaluation metrics: **Accuracy, Precision, Recall, F1-score, Confusion Matrix**
  * Compared sklearn models with manual implementation

* **CA4 – Neural Networks for Image Classification**

  * **Part 1: Fully Connected Neural Network**
    * Implemented a multi-layer perceptron for CIFAR-10 image classification
    * Designed network architecture with approximately 500,000 ± 33,500 trainable parameters
    * Applied dropout regularization to prevent overfitting
    * Used CrossEntropy loss function and Adam optimizer
    * Trained for 60 epochs with performance monitoring
    * Analyzed training/validation loss and accuracy curves

  * **Part 2: Convolutional Neural Network (CNN)**
    * Designed and implemented CNN architecture for CIFAR-10 classification
    * Maintained comparable parameter count with FC network for fair comparison
    * Used convolutional layers, pooling layers, and fully connected layers
    * Implemented feature space analysis and visualization
    * Applied t-SNE for dimensionality reduction and cluster visualization
    * Analyzed feature maps and model interpretability
    * Compared 24 misclassified samples between predicted and actual labels

  * **Comparative Analysis**
    * Compared performance metrics between FC and CNN architectures
    * Analyzed learning curves, generalization capability, and feature learning
    * Evaluated on test set with accuracy measurements

* **CA5 – Text Clustering on English Song Lyrics**

  * **Text Preprocessing & Feature Extraction**
    * Implemented text cleaning: stop word removal, punctuation removal, lemmatization
    * Used SentenceTransformers with `all-MiniLM-L6-v2` model for feature vector extraction
    * Applied TF-IDF and embedding techniques for text representation

  * **Clustering Algorithms**
    * Implemented **K-Means** clustering with elbow method for optimal K selection
    * Applied **DBSCAN** for density-based clustering
    * Used **Hierarchical Clustering** for nested cluster structures
    * Compared clustering performance across different algorithms

  * **Dimensionality Reduction & Visualization**
    * Applied **PCA** for feature space reduction to 2D/3D
    * Visualized clusters using scatter plots and cluster analysis
    * Used t-SNE for non-linear dimensionality reduction

  * **Evaluation & Analysis**
    * Calculated clustering metrics: **Silhouette Score, Homogeneity**
    * Analyzed semantic similarity within clusters
    * Sampled and compared lyrics from each cluster for thematic analysis
    * Selected optimal clustering method based on performance metrics

---

* **HW1 – Genetic & Memetic Algorithms**

  * **Part 1: Polynomial Curve Fitting using Evolutionary Algorithms**
    * Implemented **Genetic Algorithm** for polynomial curve fitting to given data points
    * Implemented **Memetic Algorithm** combining GA with local search
    * Designed chromosome representation for polynomial coefficients
    * Applied selection, crossover, and mutation operators
    * Evaluated fitness using RMSE between fitted curve and data points
    * Tested on multiple datasets with varying polynomial degrees

  * **Part 2: Traveling Salesman Problem (TSP)**
    * Applied **Genetic Algorithm** to solve classic TSP
    * Implemented **Memetic Algorithm** with local optimization
    * Used permutation-based chromosome representation
    * Implemented crossover operators (PMX, OX) and mutation operators
    * Applied Euclidean distance calculation between city coordinates

* **HW2 – Set Covering Problem with Ant Colony Optimization**
  
  * **Max-Min Ant System (MMAS) Implementation**
    * Implemented MMAS algorithm for Set Covering Problem
    * Designed pheromone update strategies with maximum and minimum limits
    * Applied heuristic information based on set costs and coverage
    * Implemented solution construction using probabilistic selection

* **HW3 – Advanced Optimization Algorithms**

  * **Part 1: Quadratic Assignment Problem (QAP)**
    * Implemented **Simulated Annealing** for QAP optimization
    * Designed permutation-based solution representation
    * Applied cooling schedules and neighborhood search

  * **Part 2: Particle Swarm Optimization (PSO)**
    * Implemented **PSO algorithm** for function optimization
    * Designed particle position and velocity updates
    * Applied inertia weight and acceleration coefficients

  * **Part 3: QAP with PSO **
    * Applied **Particle Swarm Optimization** to Quadratic Assignment Problem
    * Designed hybrid approach combining PSO with local search
    * Compared performance with Simulated Annealing results RMSE and comparison with the original function

* **HW4 – Neural Networks & Finite State Machines**
  * **Perceptron Algorithm**: Implemented from scratch for 3D point classification with various learning rates
  * **Single Layer Neural Network**: Polynomial regression (degrees 1-4) with MSE analysis and overfitting study
  * **DFA with McCulloch-Pitts Neurons**: Designed neural networks to simulate deterministic finite automata

* **HW5 – Shallow Neural Networks**
  * **From-Scratch Implementation**: Built shallow neural network with 1000 hidden neurons (ReLU) and sigmoid output
  * **Gradient Descent**: Trained on synthetic Gaussian data with various learning rates and epochs
  * **Manual Implementation**: Used only NumPy without deep learning frameworks

* **HW6 – Image Captioning**
  * **CNN-RNN Architecture**: Used pre-trained ResNet18 for feature extraction and LSTM for caption generation
  * **Text Processing**: Implemented embedding, SOS/EOS tokens, and vocabulary indexing
  * **Training Strategies**: Compared frozen vs unfrozen ResNet18 layers using CrossEntropy loss and Adam optimizer
