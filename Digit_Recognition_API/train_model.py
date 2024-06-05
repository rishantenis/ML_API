# import pickle

# from sklearn.datasets import fetch_openml
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split

# X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)
# # X, y = fetch_openml(name: 'mnist_784', version=1, return_X_y=True)
# X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
# clf =RandomForestClassifier(n_jobs=-1)
# clf.fit(X_train, Y_test)

# print(clf.score(X_test, Y_test))

# with open('mnist_model.pkl', 'wb') as f:
#     pickle.dump(clf, f)


import pickle
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Fetch the MNIST data
X, y = fetch_openml(name='mnist_784', version=1, return_X_y=True)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
clf = RandomForestClassifier(n_jobs=-1)

# Train the classifier on the training data and labels
clf.fit(X_train, Y_train)

# Evaluate the classifier on the test data and labels
print(clf.score(X_test, Y_test))

# Save the trained model to a file
with open('mnist_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
