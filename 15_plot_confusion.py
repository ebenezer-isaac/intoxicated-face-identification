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
print("Report         :\n",classification_report(actual, predicted))

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
plt.savefig('./docs/confusion_matrix_1.png')
print("Confusion Matrix has been saved to ./docs/confusion_matrix_1.png and ./docs/confusion_matrix_2.png")
plt.show()

cm = confusion_matrix(actual, predicted, states)
fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cm)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Sober', 'Predicted Drunk'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Sober', 'Actual Drunk'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
plt.savefig('./docs/confusion_matrix_2.png')
plt.show()
