# Ola Driver Attrition Prediction

This project builds a machine learning pipeline to predict driver attrition for Ola using historical driver and performance data. The notebook covers data loading, exploratory data analysis, preprocessing, class imbalance handling, model training, and model evaluation using standard classification metrics.

## Project Objective

The goal of this project is to identify drivers who are likely to leave the platform so that the business can take proactive retention actions. Driver attrition can directly affect service availability, customer experience, and operational costs, so predicting high-risk drivers can help improve workforce planning.

## Dataset

The project uses an `Ola.csv` dataset containing driver-level information. The dataset is loaded in Google Colab and analyzed using Python libraries such as Pandas, NumPy, Matplotlib, and Seaborn.

```python
df = pd.read_csv("Ola.csv")
```

If running in Google Colab, make sure the file is uploaded to the working directory or use the correct path:

```python
df = pd.read_csv("/content/Ola.csv")
```

## Key Steps

1. Import required libraries
2. Load and inspect the dataset
3. Perform exploratory data analysis
4. Check missing values and data types
5. Handle categorical and numerical features
6. Split data into training and testing sets
7. Apply SMOTE to handle class imbalance
8. Train machine learning classification models
9. Evaluate models using accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
10. Compare model performance and identify the best model

## Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Imbalanced-learn
- Google Colab

## Machine Learning Approach

The dataset is first cleaned and prepared for modeling. Since attrition datasets are often imbalanced, SMOTE is used to balance the target classes in the training data.

```python
from imblearn.over_sampling import SMOTE
```

Classification models are then trained and evaluated using multiple performance metrics. Special attention is given to recall and F1-score because correctly identifying drivers likely to leave is more important than only maximizing accuracy.

## Evaluation Metrics

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC score
- Confusion matrix
- ROC curve

These metrics help understand how well the model distinguishes between drivers who are likely to stay and drivers who are likely to leave.

## Business Impact

This project can help Ola:

- Identify drivers at high risk of attrition
- Improve driver retention strategies
- Reduce operational disruption
- Support data-driven decision making
- Prioritize engagement for drivers who need attention

## How to Run

1. Open the notebook in Google Colab.
2. Upload `Ola.csv` to the Colab working directory.
3. Install required dependencies if needed:

```python
!pip install imbalanced-learn
```

4. Run all cells from top to bottom.
5. Review the model evaluation results and visualizations.

## Expected Output

The notebook produces:

- Dataset overview
- EDA visualizations
- Processed training and testing data
- Balanced dataset using SMOTE
- Model performance comparison
- Confusion matrix
- ROC curve
- Final classification report

## Repository Structure

```text
.
├── Ola.csv
├── Ola_Driver_Attrition_Prediction.ipynb
└── README.md
```

## Conclusion

This project demonstrates an end-to-end machine learning workflow for driver attrition prediction. It combines data analysis, preprocessing, class imbalance handling, model training, and evaluation to build a practical business-focused classification solution.
