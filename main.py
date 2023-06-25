import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Import the dataset
data = {
    'Age': [25, 35, 45, 55, 65, 75, 85],
    'Weight': [60, 70, 80, 90, 100, 110, 120],
    'BP': [120, 130, 140, 150, 160, 170, 180]
}

df = pd.DataFrame(data)

# Step 2: Split the data into training and testing sets
X = df[['Age', 'Weight']]
y = df['BP']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
