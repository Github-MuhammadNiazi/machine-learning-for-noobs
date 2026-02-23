#####################################################################################################################################
#                                                                                                                                   #
#                                          Task 001: Getting Started with Machine Learning                                          #
#    In this task, we will create a simple linear regression model to predict house prices based on their size (square footage).    #
#                                                                                                                                   #
#####################################################################################################################################

# Please research the following packages and explain their purpose in the context of machine learning as we will be using them alot, that's what I did when I am learning machine learning:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Creating custom tiny dataset
    # We are creating a simple dataset with two columns: 'sqft' (square footage of the house) and 'price' (the price of the house).
    # This dataset will be used to train our linear regression model to predict house prices based on their size.
data = pd.DataFrame({
    'sqft': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'price': [150000, 180000, 210000, 260000, 300000, 330000, 360000, 400000]
})

# Step 2: Visualize - SEE the relationship
    # Projecting data into 2D space
plt.scatter(data['sqft'], data['price'])
    # Adding labels to x-axis
plt.xlabel('Square Feet')
    # Adding labels to y-axis
plt.ylabel('Price')
    # Adding title to the plot
plt.title('House Price vs Size')
    # Show the plot
# plt.show()

# Step 3: Train a model
    # The reason we need to reshape the data is because sklearn expects a 2D array. Right now data['sqft'] is a 1D array => [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500]
    # By reshaping it to (-1, 1), we are converting it into a 2D array with one column and as many rows as needed.
    # The -1 means that the number of rows will be determined automatically based on the number of elements in the array.
    # The 1 means that we want one column. So after reshaping, x will look like this:
    # [[ 800],
    #  [1000],
    #  [1200],
    #  [1500],
    #  [1800],
    #  [2000],
    #  [2200],
    #  [2500]]
x = data['sqft'].values.reshape(-1, 1)  # Reshape for sklearn
y = data['price'].values

model = LinearRegression()
    # The fit method is used to train the model on the provided data.
    # It takes the input features (x) and the target variable (y) and finds the best-fitting line that minimizes the mean squared error between the predicted values and the actual values.
model.fit(x, y)

# Step 4: Make a prediction
    # Predicting the price of a house with 1600 sqft
    # We are using the value with double brackets [[1600]] because the predict method expects a 2D array as input, just like we reshaped our training data.
    # If we were to use a single bracket [1600], it would be treated as a 1D array, which would not be compatible with the model's expected input format.
predicted_price = model.predict([[1600]])
print(f"The predicted price of a house with 1600 sqft is: ${predicted_price[0]:.2f}")
