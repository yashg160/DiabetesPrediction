import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.externals import joblib

# Read the data from the csv file
diabetesDF = pd.read_csv('diabetes.csv')

""" corr = diabetesDF.corr()
print(corr) """
""" sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns) """

""" ps = []
ns = []
for i in range(768):
    if (diabetesDF.iloc[i]['Outcome'] == 1):
        ps.append(diabetesDF.iloc[i]['Age'])
    else:
        ns.append(diabetesDF.iloc[i]['Age'])

print(np.mean(ps))
print(np.mean(ns))

sns.barplot(data=[ns, ps])
plt.show() """

# Split the data into training, testing and checking
dfTrain = diabetesDF[:650]
dfTest = diabetesDF[650:750]
dfCheck = diabetesDF[750:]

# Prepare the training data
trainLabel = dfTrain['Outcome']
trainData = dfTrain.drop('Outcome', 1)

# Prepare the testing data
testLabel = dfTest['Outcome']
testData = dfTest.drop('Outcome', 1)

# Normalize the data. Subtract the mean and divide by standard deviation
means = np.mean(trainData, axis=0)
stds = np.std(trainData, axis=0)

trainData = (trainData - means)/stds
testData = (testData - means)/stds

# np.mean(trainData, axis=0) => check that new means equal 0
# np.std(trainData, axis=0) => check that new stds equal 1

# print(means)
# print(stds)

# Create and train the Logistec Regression model
diabetesCheck = LogisticRegression()
diabetesCheck.fit(trainData, trainLabel)

# Check the accuracy
accuracy = diabetesCheck.score(testData, testLabel)
print("accuracy = ", accuracy * 100, "%")

coeff = list(diabetesCheck.coef_[0])
labels = list(trainData.columns)

features = pd.DataFrame()
features['Features'] = labels
features['importance'] = coeff
features.sort_values(by=['importance'], ascending=True, inplace=True)
features['positive'] = features['importance'] > 0
features.set_index('Features', inplace=True)
features.importance.plot(kind='barh', figsize=(
    11, 6), color=features.positive.map({True: 'blue', False: 'red'}))
plt.xlabel('Importance')
plt.show()

joblib.dump([diabetesCheck, means, stds], 'diabeteseModel.pkl')
