import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import confusion_matrix as cm
from sklearn.linear_model import LogisticRegression as lr
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

getData = pd.read_csv("BankNote_Authentication.csv")

X = getData[["variance","skewness","curtosis","entropy"]]
Y = getData["class"]
xTrain, xTest, yTrain, yTest = tts(X, Y, test_size=0.25, random_state=0)

classifier = lr(random_state=0)
classifier.fit(xTrain,yTrain)
prediction = classifier.predict(xTest)
predictValue = []
actualValue = []
for p in prediction :
  if p == 0 :
    predictValue.append("Authorized")
  else :
    predictValue.append("Forged")
for a in yTest.ravel() :
  if a == 0 :
    actualValue.append("Authorized")
  else :
    actualValue.append("Forged")

labels = ["Authorized","Forged"]
matrix = cm(actualValue,predictValue,labels)
ax = plt.subplot()
sb.heatmap(matrix, annot=True, ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")
ax.xaxis.set_ticklabels(labels);ax.yaxis.set_ticklabels(labels)