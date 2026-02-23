#####################################################################################################################################
#                                                                                                                                   #
#                                          Task 002: Adding Features to Our Model                                                   #
#    In this task, we will add a new feature to our linear regression model to predict house prices based on their size and age.    #
#                                                                                                                                   #
#####################################################################################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Add new features to the dataset making it more "rich" and "complex"
data = pd.DataFrame({
    'sqft': [800, 1000, 1200, 1500, 1800, 2000, 2200, 2500],
    'price': [150000, 180000, 210000, 260000, 300000, 330000, 360000, 400000],

    # *NEW* Adding new features to the dataset
    'age': [10, 5, 15, 20, 25, 30, 35, 40], # *NEW* Age of the house
    'location_score': [7, 8, 6, 9, 5, 8, 7, 9]# *NEW* A score representing the desirability of the location (1-10)
})

# Step 2: Visualize - SEE the relationship between multiple features
    # We will be using subplots to visualize the relationship between each feature and the price.
    # We did not use scatterplot because we have multiple features and we want to see the relationship between each feature and the price separately.
    # Here 1 is the number of rows, 3 is the number of columns, and figsize is the size of the figure.
    # We will be using a 1x3 grid of subplots to visualize the relationship between sqft, age, and location_score with price.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
features = ['sqft', 'age', 'location_score']

    # This is a loop that iterates over a list of features.
    # For each feature, it calculates the row and column indices for a subplot in a grid of subplots.
    # It then uses the indices to access a specific subplot and plots a scatter plot of the feature against the price using the data from the DataFrame.
    # It also sets the x-axis label, y-axis label, and title of the subplot.
for i, feature in enumerate(features):
    row = 0 # Since we know that we have only 3 features, we can use the modulus operator to determine the column index for each feature. The row index will always be 0 since we are using a single row of subplots.
    col = i % 3 # This will give us the column index for each feature (0, 1, 2) as we iterate through the features list.
    axes[col].scatter(data[feature], data['price']) # Plotting the feature against price in the corresponding subplot
    axes[col].set_xlabel(feature)
    axes[col].set_ylabel('Price')
    axes[col].set_title(f'Price vs {feature}')

plt.tight_layout() # Adjusts the spacing between subplots to prevent overlap
# plt.show()

# Step 3: Prepare the data for training
    # We need to select the features we want to use for training the model.
x = data[['sqft', 'age', 'location_score']] # In this case, we will use 'sqft', 'age', 'bedrooms' and 'location_score' as our features (X)
y = data['price']   # And 'price' as our target variable (y).

# Step 4: Train and test Split
    # We will be using the train_test_split function from sklearn to split our dataset into a training set and a testing set.
    # We do this to evaluate the performance of our model on unseen data.
from sklearn.model_selection import train_test_split

    # The test_size parameter specifies the proportion of the dataset to include in the test split (in this case, 20% for testing and 80% for training).
    # The random_state parameter is used to ensure that the split is reproducible. By setting it to 42 (or any decided value), we ensure that the same split will be used each time the code is run.
    # If we set random_state to a different value, we would get a different split of the data each time we run the code, which could lead to different results when evaluating the model's performance.
    # For example if we set random_state to 31
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Step 5: Train the model
model = LinearRegression()
model.fit(x_train, y_train)

# Step 6: See how feature affects the price
    # The coef_ attribute of the trained model contains the coefficients for each feature, which represent the change in the target variable (price) for a one-unit change in the feature, while keeping all other features constant.
    # I suggest looking up what coefficients mean in the context of linear regression to understand this better.
    # I understood it when I read that a coeffecient in current case would be something like:
    # Price = (coef1 * sqft) + (coef2 * age) + (coef3 * location_score) + intercept
    # This is how coefficient for each feature is obtained
    # Additionally intercept is just the value of price when all features are 0, which is not very meaningful in this context but it is still a part of the linear regression equation.
for feature, coef in zip(features, model.coef_):
    print(f"Each +1 in {feature} changes the price by ${coef:.0f}")

# Step 7: Test on unseen data
predicted_price = model.predict(x_test)
for actual, predicted in zip(y_test, predicted_price):
    print(f"Actual Price: ${actual:.0f}, Predicted Price: ${predicted:.0f}")