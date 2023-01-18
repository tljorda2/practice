#%%
from CBB_DataPrep import *
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#%%
# Splitting the data into X and y. Dropped any of the features we are not interested in.
X = Model_DataFrame.drop(columns=['Schl', 'Opp', 'School_x', 'Home_Win', 'School_y'],axis=1)
y = Model_DataFrame['Home_Win']

#%%
#Importing the metrics we are going to use
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
# Defining a dictionary of metrics
score_metrics = {'accuracy':make_scorer(accuracy_score),
                 'precision': make_scorer(precision_score),
                 'recall': make_scorer(recall_score),
                 'f1_score': make_scorer(f1_score)}

# Since I will be constantly repeating these lines of codes with few changes, I decided to make a function
# With the inputs being what is mostly going to be changed throughout the file.
def RunModel(model, folds=10):
    if type(folds) == int:
        pass
    else:
        raise TypeError('The number of folds has to be an integer')
    scores = cross_validate(model, X, y, cv=folds, scoring=score_metrics)
    print('Avg Accuracy: '+str(np.mean(scores['test_accuracy'])))
    print('Avg Precision: '+str(np.mean(scores['test_precision'])))
    print('Avg Recall: '+str(np.mean(scores['test_recall'])))
    print('Avg F1-Score: '+str(np.mean(scores['test_f1_score'])))
    return np.mean(scores['test_accuracy'])
#%%
'''K-Nearest Neighbors'''
from sklearn.model_selection import cross_validate
# Importing the KNN Module
from sklearn.neighbors import KNeighborsClassifier

#Create an instance of the model
knn = KNeighborsClassifier(n_neighbors=5)

#Evaluating the model via 10 fold cross validation
KNN_CF_Accuracy = RunModel(knn, 10)

# %%
'''Neural Network'''
# Importing the nn Module
from sklearn.neural_network import MLPClassifier

#Create an instance of the model
# Adjusted the max iterations since there was a convergence warning at 200 iterations
nn = MLPClassifier(hidden_layer_sizes=(5,3),random_state=1, max_iter=500)

#Evaluating the model via 10 fold cross validation
NeuralNet_CF_Accuracy = RunModel(nn, 10)
# %%
'''Decision Tree'''
# Importing the dt Module
from sklearn import tree

#Create an instance of the model
dt = tree.DecisionTreeClassifier()

#Evaluating the model via 10 fold cross validation
DecisionTree_CF_Accuracy = RunModel(dt, 10)
#%%
'''Logistic Regression'''
from sklearn.linear_model import LogisticRegression

log = LogisticRegression(max_iter=3000)
LogRegression_CF_Accuracy = RunModel(log,10)

#%%
'''Random Forest'''
from sklearn.ensemble import RandomForestClassifier
ran_forest = RandomForestClassifier(n_estimators=100, random_state=100)
RandomForest_CF_Accuracy = RunModel(ran_forest,10)

#%%
'''Naive Bayes'''
from sklearn.naive_bayes import GaussianNB
naive_bayes = GaussianNB()
NaiveBayes_CF_Accuracy = RunModel(naive_bayes, 10)
# %%
# Importing the voting classifier
from sklearn.ensemble import VotingClassifier

#Creating instances of different classifiers to be aggregated
clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()

#Applying the voting ensemble
eclf = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    voting='hard'
)

#Evaluate all Classifiers
for clf, label in zip([clf1,clf2,clf3,eclf], 
                      ['Logisitic Regression', 'Random Forest', 'Naive Bayes', 'Ensemble']
                      ):
  scores = cross_validate(clf, X, y, scoring=score_metrics, cv=10)
  print('Accuracy: %0.2f [%s]' % (np.mean(scores['test_accuracy']), label))

#%%
from statistics import fmean
Names = ['Classification Tree', 'Neural Network', 'K-Nearest Neighbors', 'Logistic Regression',
         'Random Forest', 'Naive Bayes']
Original_Accuracy = [DecisionTree_CF_Accuracy, NeuralNet_CF_Accuracy, KNN_CF_Accuracy,
                     LogRegression_CF_Accuracy, RandomForest_CF_Accuracy, NaiveBayes_CF_Accuracy]
# This compares the accuracy scores for all the models
ax = sns.barplot(x=Original_Accuracy, y=Names)
plt.xlabel(f'Accuracy Score')
plt.xlim(0.65,0.82)
plt.ylabel('Models')
plt.title(f'Original Dataset Accruacy Scores\nAverage: {round(fmean(Original_Accuracy), 4)}\nAxis Condensed')
plt.show()

# %%
