X, y = make_classification(n_samples=1000, n_features=4,
                            n_informative=2, n_redundant=0,
                            random_state=0, shuffle=False)
clf = RandomForestClassifier(max_depth=2, random_state=0)
clf.fit(X, y)
RandomForestClassifier(max_depth=2, random_state=0)
prediction = (clf.predict([[0, 0, 0, 0]]))
print(prediction)
