from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
import pylab as plt 

logisticRegr = LogisticRegression(max_iter=1000,verbose=1)

print("Reading Files")
X_test = pickle.loads(open("./files/x_test.pickle", "rb").read())
print("X_test Size :",len(X_test))
Y_test = pickle.loads(open("./files/y_test.pickle", "rb").read())
print("Y_test Size :",len(Y_test))

X_test_shape, nx, ny, nz = X_test.shape
X_test_2d = X_test.reshape((X_test_shape,nx*ny*nz))
labels = []
for label in Y_test:
	labels.append(label[0])

logisticRegr = pickle.load(open("./files/loginstic_regression.pickle", 'rb'))
#predictions = logisticRegr.predict(x_test)
result = logisticRegr.score(X_test_2d, labels)
print(result)
