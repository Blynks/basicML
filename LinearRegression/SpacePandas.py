import pandas as pd
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
import seaborn as sns

# Load advertising data
data = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)

# Show first 5 tuples
print(data.head())
print()

# Show last 5 tuples
print(data.tail())
print()

# Use 3 features
X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

# Split data into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training set
regr.fit(X_train, y_train)

print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)

# Plot outputs
sns.pairplot(data, x_vars=['TV', 'Radio', 'Newspaper'], y_vars='Sales', size=7, aspect=0.7, kind='reg')
sns.plt.show()



