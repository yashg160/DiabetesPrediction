import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# Read the data from the csv file
diabetesDF = pd.read_csv('diabetes.csv')


# Load the saved model from the pickle file
diabetesLoadedModel, means, stds = joblib.load('diabeteseModel.pkl')
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

# Prepare the testing data
testLabel = dfTest['Outcome']
testData = dfTest.drop('Outcome', 1)

# Normalize the data
testData = (testData - means) / stds

accuracyModel = diabetesLoadedModel.score(testData, testLabel)
print("accuracy = ", accuracyModel * 100, "%")

# Accuracy will be same if model was saved and loaded correctly. It is 78%

# Validation data that has not been tested before
sampleData = dfCheck[:1]

# prepare sample
sampleDataFeatures = sampleData.drop('Outcome', 1)
sampleDataFeatures = (sampleDataFeatures - means) / stds

predictionProbability = diabetesLoadedModel.predict_proba(sampleDataFeatures)
prediction = diabetesLoadedModel.predict(sampleDataFeatures)

print('Probability:', predictionProbability)
print('prediction:', prediction)
