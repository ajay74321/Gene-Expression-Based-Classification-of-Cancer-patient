# Gene-Expression-Based-Classification-of-Cancer-patient

## 🧬 Introduction
Cancer is marked by uncontrolled cell growth and progression through stages. Early detection and accurate stage classification are crucial for guiding treatment and improving outcomes.
This project develops machine learning models to classify cancer patients into early (+1) or late (–1) stage using ~60,000 gene expression features. Due to the high dimensionality, preprocessing and feature selection are essential to avoid overfitting.The workflow integrates statistical analysis, dimensionality reduction, and classifiers such as PCA, SVM, and AutoML to uncover patterns in gene expression data. This demonstrates how computational biology and machine learning can advance cancer diagnostics and support clinical decision-making.
## Cancer-Patient-Classification

## Project Description
In this competition, participants are required to **predict the stage of cancer patients** based on gene expression data.

## File Descriptions

- **train.csv**  
  Contains the training dataset with:  
  - `ID` – Patient/sample identifier  
  - `label` – Target variable (`+1` for early stage, `-1` for late stage)  
  - Gene expression values for ~60,000 genes across ~280 patients  

- **test.csv**  
  Contains the test dataset with:  
  - `ID` – Patient/sample identifier  
  - Gene expression values for ~60,000 genes across ~70 patients  
  - Participants need to **predict the label** for each patient  

- **sample.csv**  
  A sample submission file in the correct format containing:  
  - `ID`  
  - `label` (predicted)  

## Goal
Build a model to accurately predict the **cancer stage** of patients based on their gene expression profiles.

## Requirements 
- numpy==1.25.2
- pandas==2.1.1
- scikit-learn==1.3.2
- autogluon==0.8.2
- scipy==1.11.1

You can install all dependencies using:  

```bash
pip install -r requirements.txt
```

## Preprocessing
- Features and labels are separated.  
- **ANOVA F-test (SelectKBest)** is used to select the **top 10,000 most informative genes**.  
- Training data is split into **training and validation sets** for model evaluation.

## Data
The training and test data used in this project are provided in the dataset folder.

## Note
This project was part of a kaggle competition.


