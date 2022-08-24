
#MultiLabel MultiClass: Power Transformation Approach

from sklearn import datasets
X, y = datasets.load_iris(return_X_y=True)

# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset
from sklearn.naive_bayes import GaussianNB

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier = LabelPowerset(GaussianNB())

# train
classifier.fit(X_train, y_train)

# predict
predictions = classifier.predict(X_test)
#accuracy_score(y_test,predictions)

# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance
from sklearn.naive_bayes import GaussianNB

# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier = BinaryRelevance(GaussianNB())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=1)
# train
classifier.fit(X_train, y_train)
# predict
predictions = classifier.predict(X_test)

#from sklearn.metrics import accuracy_score
#accuracy_score(y_test,predictions)
