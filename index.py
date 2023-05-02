import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load your dataset
# X: patient symptoms (features)
# y: medical diseases/problems (labels)
X = np.array([
    [1, 0, 1],
    [0, 1, 0],
    [1, 1, 0],
    [1, 0, 0],
    [0, 1, 1],
    [0, 0, 1],
    [1, 1, 1],
    [1, 1, 0],
    [0, 1, 0],
    [1, 0, 1]
])
y = np.array([0, 1, 1, 0, 1, 0, 1, 1, 1, 0])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy of the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)