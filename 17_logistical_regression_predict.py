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
logisticRegr = pickle.load(open("./files/loginstic_regression.pickle", 'rb'))
print("Files Loaded")

#Data Preparation for 3D Image (remove nz for 2d)
X_test_shape, nx, ny, nz = X_test.shape
X_test_2d = X_test.reshape((X_test_shape,nx*ny*nz))
labels = []
for label in Y_test:
	labels.append(label[0])

states=["sober","drunk"]
actual = []
prediction = []
for (image,label) in zip(X_test_2d,labels):

	prediction.append(states[int(logisticRegr.predict(image.reshape(1,-1)))])
	actual.append(states[int(label)])
f = open("./files/logistical_regression_prediction.pickle", "wb")
f.write(pickle.dumps([actual,prediction]))
f.close()
print("Predictions saved to ./files/logistical_regression_prediction.pickle")
