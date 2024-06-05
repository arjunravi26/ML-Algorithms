import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from simple_linear_regression import SimpleLinearRegression

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 100)  # Independent variable
y = 2 * x + 1 + np.random.normal(0, 1, 100)  # Dependent variable (with noise)

# Create a DataFrame
df = pd.DataFrame({'x': x, 'y': y})
X = df['x']
y = df['y']
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=44,test_size=0.20)

linear_reg = SimpleLinearRegression()
train_score = linear_reg.fit(X_train,y_train)
print(train_score)
test_y = linear_reg.predict(X_test)
# print(X_test)
# print(test_y)
print(r2_score(X_test,test_y))


# Example usage:
# Generate some random data
np.random.seed(0)
# Generate synthetic data for the first feature
feature1 = 2 * np.random.rand(200, 1)

# Generate synthetic data for the second feature
feature2 = 2 * np.random.rand(200, 1)

# Concatenate the features vertically to create X with two features
X = np.concatenate((feature1, feature2), axis=1)  


y = 4 * np.random.rand(200, 1)

# Print the shape of y
print("Shape of feature1:", feature1.shape)
print("Shape of feature2:", feature2.shape)
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Instantiate and fit the model
model = LinearRegression()
score = model.fit(X[:150], y[:150])
print(score)

# Make predictions
y_pred = model.predict(X[150:])

print("Predictions:", y_pred)
print("real:", y[150:])
print(r2_score(y_pred,y[150:]))

