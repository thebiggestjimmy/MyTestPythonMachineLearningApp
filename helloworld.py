from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data from the open ML database
mnist = fetch_openml('mnist_784', version=1)

# Data preparation
# mnist.data contains the features (the pixel values for each image)
# mnist.target contains the labels (the digit each image represents)
X, y = mnist.data, mnist.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
# We use Logistic Regression here for simplicity
model = LogisticRegression(max_iter=1000, solver='lbfgs')
model.fit(X_train, y_train)

# Predictions and evaluation
y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred)}")
