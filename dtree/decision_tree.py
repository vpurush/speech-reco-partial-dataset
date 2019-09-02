from sklearn.ensemble import RandomForestClassifier

classifiers = {}
def trainDTreeClassifier(twoCharSequence, x, y):
    clf = RandomForestClassifier()
    clf.fit(x, y)
    classifiers[twoCharSequence] = clf

def printDTreeClassifiers():
    print("classifiers", classifiers)

def predictDTree(key, x):
    # print("x.shape", x.shape, x)
    clf = classifiers[key]
    prediction = clf.predict(x)
    print("prediction", prediction)