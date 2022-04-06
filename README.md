# Diabetes-ML-model
Supervised learning model to predict whether a patient has diabetes using scikit.learn

import numpy as np

import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import LabelEncoder,StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import accuracy_score,roc_auc_score,classification_report

from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,roc_curve

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

df=pd.read_csv('...diabetes.csv')
y = df['Outcome'].values
X = df.drop('Outcome', axis=1).values
df.head()
df.shape
df.info()

 #KNN classification Model
knn = KNeighborsClassifier(n_neighbors=27)#sqaure root of total observations knn.fit(X,y)

 ss=StandardScaler()#scaling required for KNN model so the algorithm is not dominated by higher magnitude variables 
 X=ss.fit_transform(X)
y_predict=knn.predict(X) #predict label for training data X_new_=np.array([1,85,66,29,0,26.6,.351,31]).reshape(1,-1)
new_point_prediction=knn.predict(X_new)#predict whether new data point individual is diabetic or not 
print(new_point_prediction)
 #Model Evaluation
#Accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15, random_state=42, stratify=y) knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
print("Accuracy Score:",knn.score(X_test, y_test))#since there is a class imbalance of 34% positives to 66% negatives, the accuracy score isn't very helpful

 #more nuanced model measures
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
precision = tp/(tp+fp),recall=tp/(tp+fn),f1=(2*precision*recall)/(precision+recall)

 #Logistic Regression
logreg=LogisticRegression() logreg.fit(X_train,y_train)
y_pred = logreg.predict(X_test) print(confusion_matrix(y_test, y_pred)) print(classification_report(y_test, y_pred))

#visually assess model using ROC curve
y_predicted_prob=logreg.predict_proba(X_test)[:,1] #predicted probabilities
fpr, tpr, thresholds = roc_curve(y_test, y_predicted_prob)#ROC values
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show() #best models have curve in top left and worst models have curve along diagonal #IOW, the larger the area under the ROC curve, the better the model

 #AUC score
roc_auc_score(y_test,y_predicted_prob)#want this close to 1

 #Gaussian Naive Bayes classification, recall features are assumed to be independent from each other #use this over Bernoulli and Multinomial naive Baye's since features are continuous and not discrete model = GaussianNB()
model.fit(X,y)
predicted= model.predict(X_new)
print ("Predicted Value:", predicted)

 #Gaussian Model Evaluation
model=GaussianNB()
model.fit(X_train,y_train) y_predicted_gaussian=model.predict(X_test)
print('Accuracy Score:',accuracy_score(y_test,y_predicted_gaussian)) print(confusion_matrix(y_test, y_predicted_gaussian)) print(classification_report(y_test, y_predicted_gaussian))
