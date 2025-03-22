# Credit-Card-Fraud-Detection
Enhanced Framework for Credit Card Fraud Detection Using Robust Feature Selection and a  Stacking Ensemble Model Approach


This project demonstrates how to detect fraudulent credit card transactions using advanced feature selection techniques and ensemble classification models. The workflow includes preprocessing, oversampling, dimensionality reduction, and hybrid model evaluation to ensure accurate and reliable fraud detection.

### Project Overview
The research work includes preprocessing of the highly imbalanced credit card fraud dataset, applying SMOTE and SMOTE-ENN for balancing, feature selection using Mutual Information and ANOVA, optimization via PSO (Particle Swarm Optimization), dimensionality reduction with Incremental PCA, and implementation of a stacked ensemble model combining Random Forest, Gradient Boosting, and MLP classifiers. The model is evaluated using a range of metrics including accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix.

### Dataset
The dataset consists of real-world anonymized credit card transactions.
It includes features obtained through PCA transformations and a binary class label indicating fraud (1) or legitimate (0).
Dataset used: Credit Card Fraud Detection Dataset - https://drive.google.com/file/d/1vDg0zzZNtiPd3x_5MGnJF82n36X0xTeU/view?usp=sharing
### Technologies Used
Python, TensorFlow / Keras, Scikit-learn, imbalanced-learn (SMOTE, SMOTEENN), pandas, NumPy, Matplotlib & Seaborn (for visualization), PSO (via PySwarm)

### Model Architecture
The final model is a stacked ensemble that integrates Random Forest Classifier, Gradient Boosting Classifier, Multi-Layer Perceptron (MLP) With feature inputs selected using statistical techniques and reduced using Incremental PCA. Oversampling is handled by SMOTE and SMOTEENN to address class imbalance.
![model_arch_2 drawio](https://github.com/user-attachments/assets/0628c444-2db6-404c-a0cf-9cb01babac03)


### Results and Evaluation Metrices
Our Proposed work Successfully handled data imbalance with oversampling techniques, Feature selection and PCA improved training time and model performance, The ensemble model achieved high accuracy and strong F1-score on fraudulent transaction detection, ROC-AUC and confusion matrix visuals show strong separation and reliability.

### Comparision Model
![model_comparision_based_on_metrices](https://github.com/user-attachments/assets/89358ba9-e0b9-4b1c-953c-765571697c01)

### ROC Curve
![roc_curve](https://github.com/user-attachments/assets/a6a24bde-e36f-4350-8591-f86d292753d6)

### Topsis Ranking
![topisis_rank](https://github.com/user-attachments/assets/e1622c5a-96dc-498f-95d6-40a59600445b)

### Loss Curve
![loss_credit](https://github.com/user-attachments/assets/ae3ee0ae-9513-4761-8874-1c359e5071a2)

### Accuracy Curve
![accuracy_credit](https://github.com/user-attachments/assets/14a548a2-9698-463c-975e-429fc23908e4)

### Feature Selection Score
![feature_selection_score](https://github.com/user-attachments/assets/92c0ca1a-b58a-42bf-b657-da2370db79b0)





