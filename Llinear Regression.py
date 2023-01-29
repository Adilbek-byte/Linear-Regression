import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the data into a pandas DataFrame
df = pd.read_csv('data.csv')

# Split the data into features and target
X = df[['feature1', 'feature2', 'feature3']]
y = df['target']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict the target values for the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
from sklearn.metrics import mean_squared_error
print('Mean Squared Error:', mean_squared_error(y_test, y_pred))
