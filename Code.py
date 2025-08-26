# Model for the CLassification of Cancer Patients
# Importing the necessary libraries 

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from autogluon.tabular import TabularPredictor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Loading the train and test datasets
train_data = pd.read_csv('train.csv')  # Training data
test_data = pd.read_csv('test.csv')    # Test data (no labels)

# Separating the features and the labels from training data
X = train_data.drop(columns=['id', 'Label'])  # Drop 'ID' and 'Label' columns
y = train_data['Label']  # Labels

# Applying ANOVA F-test to the select the top 10000 features
selector = SelectKBest(score_func=f_classif, k=10000)
X_selected = selector.fit_transform(X, y)

# selected features are indices and also the names
selected_indices = selector.get_support(indices=True)
selected_feature_names = X.columns[selected_indices]

# Creating DataFrames from the selected 10000 features
X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

# Combining both the selected features and labels into a single DataFrame for AutoGluon Model
train_selected_df = X_selected_df.copy()
train_selected_df['Label'] = y.values  # Add the labels back to the DataFrame

# Spliting the training data into training and validation sets
train_data, val_data = train_test_split(train_selected_df, test_size=0.2, random_state=42)

# Initializing the AutoGluon Tabular_Predictor with all the default presets
predictor = TabularPredictor(label='Label', eval_metric='roc_auc').fit(
    train_data=train_data,
    tuning_data=val_data,
    use_bag_holdout=True,  
    presets='best_quality'  
)

# Evaluating the model on the validation set
y_value_proba = predictor.predict_proba(val_data)  # Get predicted probabilities
# Assuming the binary classification of the data
auc_score = roc_auc_score(val_data['Label'], y_value_proba.iloc[:, 1])  
print("Validation AUC Score: ", auc_score)

# Making predictions on the test dataset
test_selected_df = test_data[selected_feature_names]  # Applying the same feature selection to test data
test_predictions = predictor.predict_proba(test_selected_df)  # Get predicted probabilities for the data

# Preparing the submission file along with 'id' and 'Label' columns
submission = pd.DataFrame({
    'id': test_data['id'],  # Including the ID column
    'Label': test_predictions.iloc[:, 1]  # Predicting probabilities
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to 'submission_21.csv'")