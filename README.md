# Activity-Recognition-using-HAR70-dataset

This project involves the development and evaluation of machine learning models to classify human activities using time-series sensor data. The models are tested across various class configurations (2-class, 4-class, 7-class), with and without regularization, and include grid search and visualization components.

---

## 📂 Model Training Notebooks

These notebooks represent the initial phase of the pipeline where various models are trained and evaluated to determine the best-performing one before proceeding to fine-tuning or deployment.

1. **`Project_research_2class.ipynb`**  
   - Trains a classifier to distinguish between two activity classes.  
   - Includes feature extraction using statistical and frequency-domain methods.  
   - Applies standard classification techniques (e.g., Random Forest, SVM, etc.).

2. **`Project_research_4class.ipynb`**  
   - Extended to four activity categories.  
   - Evaluates model performance on this more complex classification problem.

3. **`Project_research_7class.ipynb`**  
   - Full 7-class classification setup.  
   - Implements preprocessing, model training, and performance evaluation.

---

## 🚫 Models Without Regularization

4. **`RF2Class.ipynb`**  
   - Improves accuracy of binary classification using 5-fold cross-validation.  
   - Focused on overfitting detection in 2-class models.

5. **`RF4Class.ipynb`**  
   - Same approach as above, but for the 4-class model.

6. **`RF7Class.ipynb`**  
   - Applies similar techniques to the 7-class classification.

---

## 🔧 Optimization & Testing

7. **`GridSearchHAR.ipynb`**  
   - Performs hyperparameter tuning using `GridSearchCV` and `RandomSearchCV`.  
   - Helps identify optimal model parameters.

8. **`Test_7class.ipynb`**  
   - Final testing on the 7-class model using unseen test data (HARTH).  
   - Includes accuracy, confusion matrix, and other performance metrics.

---

## 📊 Visualization

9. **`Visualization.ipynb`**  
   - Visualizes raw sensor signals and class distribution.

---

## 🚀 Getting Started

### Prerequisites

Make sure the following libraries are installed:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
