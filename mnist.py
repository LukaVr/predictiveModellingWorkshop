import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.random.seed(42)
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

N = 6000 #60000

def sort_by_target(mnist):
    reorder_train = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[:N])]))[:, 1]
    reorder_test = np.array(sorted([(target, i) for i, target in enumerate(mnist.target[N:])]))[:, 1]
    mnist.data[:N] = mnist.data[reorder_train]
    mnist.target[:N] = mnist.target[reorder_train]
    mnist.data[N:] = mnist.data[reorder_test + N]
    mnist.target[N:] = mnist.target[reorder_test + N]

from sklearn.datasets import fetch_openml

mnist = fetch_openml('mnist_784', version=1, cache=True)
mnist.target = mnist.target.astype(np.int8) # fetch_openml() returns targets as strings
sort_by_target(mnist) # fetch_openml() returns an unsorted dataset
print(mnist.data.shape)
X, y = mnist["data"], mnist["target"]

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = mpl.cm.binary, interpolation="nearest")
plt.axis("off")
plt.savefig("5.png")
#plt.show()

X_train, X_test, y_train, y_test = X[:N], X[N:], y[:N], y[N:]
shuffle_index = np.random.permutation(N)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score

print("Classifying...")

from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)

never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=10, scoring="accuracy"))

clfs = [KNeighborsClassifier(n_neighbors=20),
        LogisticRegression(solver='lbfgs', max_iter=10000),
        Never5Classifier(),
        #SGDClassifier(max_iter=5, tol=-np.infty, random_state=42),
       ]

for clf in clfs:
	clf.fit(X_train, y_train_5)


print("Evaluating using cross-validation...")

for clf in clfs:
    print(clf.__class__.__name__)
	#print(cross_val_score(clf, X_train, y_train_5, cv=10, scoring="accuracy"))

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, accuracy_score

for clf in clfs:
    print("------------------------------")
    print(clf.__class__.__name__)
    y_train_pred = cross_val_predict(clf, X_train, y_train_5, cv=10)
    cm = confusion_matrix(y_train_5, y_train_pred)
    print("Confusion matrix:")
    print(cm)
    accuracy = accuracy_score(y_train_5, y_train_pred)
    precision = precision_score(y_train_5, y_train_pred)
    recall = recall_score(y_train_5, y_train_pred)
    print("Accuracy: {:.2f}\nPrecision: {:.2f}\nRecall: {:.2f}".format(accuracy, precision, recall))