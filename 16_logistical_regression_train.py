from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

logisticRegr = LogisticRegression(max_iter=1000,verbose=1)

print("Reading Files")
X_train = pickle.loads(open("./files/x_train.pickle", "rb").read())
print("X_train Size :",len(X_train))
Y_train = pickle.loads(open("./files/y_train.pickle", "rb").read())
print("Y_train Size :",len(Y_train))

X_train_shape, nx, ny, nz = X_train.shape
X_train_2d = X_train.reshape((X_train_shape,nx*ny*nz))
sober_count=0
drunk_count=0
labels = []
for label in Y_train:
	labels.append(label[0])

print("Training Logistic Regression Model\n\t This may take some time depending on your PC\n\nVerbose Logging Below:")
logisticRegr.fit(X_train_2d, labels)
print("Model has been trained")

pickle.dump(logisticRegr, open("./files/loginstic_regression.pickle", 'wb'))
print("Model has been saved to ./files/loginstic_regression.pickle")