from sklearn.metrics import confusion_matrix, accuracy_score, classification_report 
import pylab as plt 
import pickle
import numpy as np

data = pickle.loads(open("./files/tensorflow_prediction.pickle", "rb").read())
data = np.array(data)
actual = data[0]
predicted = data[1]
states=["sober","drunk"]
print("Accuracy Score :",accuracy_score(actual, predicted))
print("Report         :\n",classification_report(actual, predicted))

cm = confusion_matrix(actual, predicted, states)
fig, ax = plt.subplots(figsize=(8, 8))
cax = ax.matshow(cm,cmap=plt.cm.RdYlGn)
plt.title("3D Convolution Neural Network Confusion Matrix")
fig.colorbar(cax)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted Sober', 'Predicted Drunk'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual Sober', 'Actual Drunk'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha='center', va='center', color='black')
plt.savefig('./docs/tensorflow_confusion.png')
print("Tensorflow Confusion Matrix has been saved to ./docs/tensorflow_confusion.png")
plt.show()
