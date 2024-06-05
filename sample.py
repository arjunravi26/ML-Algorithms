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
