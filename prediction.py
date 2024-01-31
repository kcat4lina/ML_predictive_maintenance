import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Load historical PLC data
data = pd.read_csv('PLC_data.csv')

# Split data into features (x) and target variable (y)
x = data[['sensor1', 'sensor2', 'sensor3']]
y = data['equipment_failure']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Train a random forest classifier model
model = RandomForestClassifier()
model.fit(x_train, y_train)

# Make predictions on the testing data
predicted_failures = model.predict(x_test)

# Calculate ROC AUC score
ROC_AUC = roc_auc_score(y_test, predicted_failures)
print("ROC AUC:", ROC_AUC)
