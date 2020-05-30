from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
import pylab as plt 
import pickle
import numpy as np

data = pickle.loads(open("./files/confusion_matrix.pickle", "rb").read())
data = np.array(data)
actual = data[0]
predicted = data[1]
states=["sober","drunk"]
print("Accuracy Score :",accuracy_score(actual, predicted))
print("Report         :",classification_report(actual, predicted))

cm = confusion_matrix(actual, predicted, states)
print(cm)
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(cm)
plt.title("Confusion matrix of the classifier")
fig.colorbar(cax)
ax.set_xticklabels([""] + states)
ax.set_yticklabels([""] + states)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.text(10,10,"asdf", ha='center',va='center',size=10)
plt.show()