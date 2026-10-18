<div align="center">

# 💻 Laptop Price Prediction Using Machine Learning

### MINOR PROJECT — 05

**Developed by BHOLA SANKAR MAJHI**

Predicting laptop prices using machine learning, data analysis, and model evaluation.

<br>

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white)

</div>

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Objectives](#-project-objectives)
- [Dataset Information](#-dataset-information)
- [Technology Stack](#-technology-stack)
- [Project Workflow](#-project-workflow)
- [Data Preparation](#-data-preparation)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Machine Learning Models](#-machine-learning-models)
- [Regression Results](#-regression-results)
- [Price Classification](#-price-classification)
- [Key Findings](#-key-findings)
- [Limitations](#-limitations)
- [Future Improvements](#-future-improvements)
- [Project Structure](#-project-structure)
- [Installation and Usage](#-installation-and-usage)
- [Author](#-author)
- [References](#-references)

---

## 📖 Project Overview

Laptop prices vary significantly depending on hardware specifications, manufacturer, laptop category, and other technical characteristics.

This project applies Machine Learning techniques to analyze laptop specifications, identify patterns in pricing, and predict laptop prices.

The project follows an end-to-end workflow, including data inspection, cleaning, exploratory data analysis (EDA), feature preprocessing, model training, evaluation, and comparison.

Three regression algorithms are evaluated:

- Gradient Boosting Regressor
- Random Forest Regressor
- Linear Regression

A separate Logistic Regression experiment classifies laptops into Low, Medium, and High price categories.

**Primary focus:** Laptop price prediction using Gradient Boosting and comparative evaluation of regression models.

## 🎯 Problem Statement

Estimating a laptop's price from its specifications is a regression problem.

The objective is to develop and evaluate machine learning models that estimate laptop prices from available product features.

The project also investigates whether laptop specifications can help distinguish between different price categories.

## 🚀 Project Objectives

- Understand the dataset and its features.
- Inspect data quality and identify potential issues.
- Clean and prepare numerical and categorical data.
- Explore relationships between laptop specifications and prices.
- Create meaningful visualizations and interpret the findings.
- Train multiple regression models.
- Compare models using MAE, RMSE, and R².
- Perform a separate price-category classification experiment.
- Document limitations and opportunities for improvement.

## 📊 Dataset Information

| Property | Details |
|---|---|
| Dataset | Laptop Price Dataset |
| Source | Kaggle |
| Total Records | 1,303 |
| Total Columns | 13 |
| Prediction Target | `Price_euros` |
| Prediction Type | Regression |
| Price Categories | Low, Medium, High |
| Price Currency | Euro (€) |

**Dataset Source:** [Kaggle — Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)

### Feature Description

| Feature | Description |
|---|---|
| `laptop_ID` | Laptop identifier |
| `Company` | Manufacturer |
| `Product` | Product name |
| `TypeName` | Laptop category |
| `Inches` | Screen size |
| `ScreenResolution` | Display resolution |
| `Cpu` | Processor information |
| `Ram` | RAM capacity |
| `Memory` | Storage configuration |
| `Gpu` | Graphics processor |
| `OpSys` | Operating system |
| `Weight` | Laptop weight |
| `Price_euros` | Laptop price in euros |

The `Price_euros` column is the target variable for regression.

The `laptop_ID` column is excluded from model features because it serves as an identifier rather than a meaningful laptop specification.

> **Dataset note:** This dataset represents the prices recorded in the source dataset and should not be treated as a live or current laptop price database.

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical operations |
| Matplotlib | Data visualization |
| Seaborn | Statistical visualization |
| Scikit-learn | Preprocessing, model training, and evaluation |
| Google Colab | Notebook execution |
| Jupyter Notebook | Interactive development |
| GitHub | Version control and project presentation |

## 🔄 Project Workflow

The project follows these stages:

1. **Data Loading** — Load the Kaggle dataset into a Pandas DataFrame.
2. **Data Understanding** — Inspect dimensions, columns, data types, and descriptive statistics.
3. **Data Quality Checks** — Examine missing values, duplicates, identifiers, and invalid values.
4. **Data Cleaning** — Convert RAM and weight into numeric formats.
5. **Exploratory Data Analysis** — Study distributions, group averages, and feature relationships.
6. **Feature Preparation** — Separate the target, numerical features, and categorical features.
7. **Preprocessing** — Apply imputation and One-Hot Encoding through a preprocessing pipeline.
8. **Model Training** — Train three regression algorithms.
9. **Model Evaluation** — Compare predictions using MAE, RMSE, and R².
10. **Classification** — Predict Low, Medium, and High price categories using Logistic Regression.
11. **Interpretation** — Document findings, limitations, and future improvements.

## 🧹 Data Preparation

Data preparation was performed before model training.

### Data Quality

The original dataset was checked for:

- Missing values
- Duplicate rows
- Duplicate laptop identifiers
- Non-positive prices
- Invalid screen sizes

The initial checks found no missing values or duplicate rows.

### Feature Cleaning

The `Ram` and `Weight` columns contained numeric values combined with text units.

The units were removed and the columns were converted into numeric formats.

### Feature Preprocessing

The preprocessing pipeline includes:

- **Numerical features:** Median imputation
- **Categorical features:** Most-frequent imputation
- **Categorical encoding:** One-Hot Encoding
- **Unknown categories:** Handled using `handle_unknown="ignore"`

The regression dataset was divided into 80% training data and 20% testing data.

The preprocessing steps were fitted through the training pipeline to reduce data leakage risk.

## 📈 Exploratory Data Analysis

Exploratory Data Analysis was used to understand laptop prices and the relationships between prices and product specifications.

### Key Observations

**1. Price Distribution**

Laptop prices are concentrated mainly in the lower and middle price ranges, with a smaller number of expensive laptops.

**2. RAM and Price**

Average laptop prices generally increase with RAM capacity. However, RAM is not the only factor affecting price, and some RAM categories contain very few observations.

**3. Laptop Type and Price**

Workstations and gaming laptops have higher average prices than ordinary notebooks in this dataset.

**4. Screen Size and Price**

Screen size alone has a weak linear relationship with price. A larger screen does not automatically mean a more expensive laptop.

**5. Feature Relationships**

RAM has a stronger positive linear correlation with price than screen size or weight in this dataset.

These observations describe associations in the available data and do not establish causal relationships.

## 🤖 Machine Learning Models

### 1. Gradient Boosting Regressor

Gradient Boosting is the primary regression model in this project.

It builds an ensemble of decision trees sequentially, with later trees attempting to improve the predictions made by the existing ensemble.

**Configuration:**

- Estimators: 100
- Learning rate: 0.1
- Maximum tree depth: 3
- Random state: 42

### 2. Random Forest Regressor

Random Forest combines predictions from multiple decision trees.

It is included to evaluate whether an ensemble of trees can reduce prediction errors.

**Configuration:**

- Estimators: 100
- Random state: 42

### 3. Linear Regression

Linear Regression provides a baseline model for estimating the relationship between input features and laptop prices.

Comparing this model with tree-based models helps evaluate how well a simpler regression approach performs.

### Evaluation Metrics

| Metric | Interpretation | Preferred Direction |
|---|---|---|
| MAE | Average absolute prediction error | Lower |
| RMSE | Error metric that penalizes larger errors more strongly | Lower |
| R² | Proportion of target variance explained by the model | Generally higher |

## 📊 Regression Results

The following results were obtained from the model evaluation on the held-out test set.

| Model | MAE (€) | RMSE (€) | R² Score |
|---|---:|---:|---:|
| Gradient Boosting | 200.30 | 290.21 | 0.8342 |
| Random Forest | 184.01 | 309.82 | 0.8110 |
| Linear Regression | 190.01 | 282.91 | 0.8424 |

### Results Interpretation

**Random Forest**
- Achieved the lowest MAE of approximately €184.01.
- Produced the smallest average absolute prediction error among the evaluated models.

**Linear Regression**
- Achieved the lowest RMSE of approximately €282.91.
- Achieved the highest R² score of approximately 0.8424.

**Gradient Boosting**
- Achieved an MAE of approximately €200.30.
- Achieved an RMSE of approximately €290.21.
- Achieved an R² score of approximately 0.8342.

### Model Selection

There is no single model that performs best on every metric.

- Choose **Random Forest** when prioritizing MAE.
- Choose **Linear Regression** when prioritizing RMSE and R² in this experiment.
- Retain **Gradient Boosting** as the project's primary model for further evaluation.

A final production model should be selected based on the intended use case, validation results, and the relative importance of different prediction errors.

## 🏷️ Price Classification

In addition to regression, a separate Logistic Regression model was trained to classify laptops into three price categories.

### Category Definitions

| Category | Price Range |
|---|---|
| Low | Price ≤ €700 |
| Medium | €700 < Price ≤ €1,500 |
| High | Price > €1,500 |

### Classification Results

| Metric | Result |
|---|---:|
| Test Accuracy | 85.06% |
| Macro ROC-AUC | 0.9606 |
| Weighted ROC-AUC | 0.9581 |

### Class-wise ROC-AUC

| Category | ROC-AUC |
|---|---:|
| High | 0.96 |
| Low | 0.99 |
| Medium | 0.94 |

The classification results indicate that the model separated the three price categories reasonably well on the test set.

These results apply to the current dataset and evaluation setup. They do not guarantee the same performance on new or future laptop data.

## 💡 Key Findings

- Laptop specifications provide useful information for estimating prices.
- RAM capacity shows a positive association with price in the dataset.
- Laptop type and manufacturer are associated with differences in average prices.
- Screen size alone is not a strong linear indicator of price.
- Random Forest achieved the lowest MAE among the evaluated regression models.
- Linear Regression achieved the lowest RMSE and highest R².
- Logistic Regression achieved approximately 85.06% test accuracy for price-category classification.

## ⚠️ Limitations

- The dataset may not reflect current market prices.
- The prices are recorded in euros and may not represent other markets.
- Some categories contain very few examples.
- Product names and hardware configurations may be specific to the source dataset.
- Test-set performance may differ from performance on new data.
- Correlation and group averages do not prove causation.
- The reported results do not replace validation on a current, independent dataset.

## 🔮 Future Improvements

Potential improvements include:

- Apply cross-validation for more robust performance estimates.
- Perform systematic hyperparameter tuning.
- Analyze feature importance and prediction errors.
- Evaluate additional regression algorithms.
- Test the models on newer laptop price data.
- Develop a web application for interactive price estimation.
- Add model versioning, input validation, and monitoring before deployment.

## 📁 Project Structure

```text
BHOLA-SANKAR-MAJHI-Laptop-Price-Prediction-MINOR-PROJECT-5/
│
├── Laptop_Price_Prediction_Machine_Learning.ipynb
├── README.md
├── requirements.txt
├── .gitignore
└── reports/
    └── Laptop_Price_Prediction_Project_Report.pdf
```

The PDF report is optional until it has been created.

## ⚙️ Installation and Usage

### Prerequisites

- Python 3.10 or a compatible Python version
- Google Colab or Jupyter Notebook
- The Kaggle dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/BHOLA-SANKAR-MAJHI-Laptop-Price-Prediction-MINOR-PROJECT-5.git
```

Replace `YOUR-USERNAME` with your actual GitHub username.

### Step 2: Open the Project

```bash
cd BHOLA-SANKAR-MAJHI-Laptop-Price-Prediction-MINOR-PROJECT-5
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Download the Dataset

Download the CSV file from the Kaggle dataset page:

https://www.kaggle.com/datasets/muhammetvarl/laptop-price

### Step 5: Run the Notebook

Open `Laptop_Price_Prediction_Machine_Learning.ipynb` in Google Colab or Jupyter Notebook.

Upload `laptop_price.csv` to the expected location or update the notebook's dataset path.

Run the notebook cells in order.

## 👨‍💻 Author

**BHOLA SANKAR MAJHI**

**Project:** Laptop Price Prediction Using Machine Learning  
**Project Type:** Minor Project 5

This project was developed for academic learning and practical exploration of Machine Learning workflows.

## 📚 References

- [Kaggle — Laptop Price Dataset](https://www.kaggle.com/datasets/muhammetvarl/laptop-price)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [NumPy Documentation](https://numpy.org/doc/)

---

<div align="center">

**Laptop Price Prediction | Machine Learning Minor Project**

**Developed by BHOLA SANKAR MAJHI**

</div>
