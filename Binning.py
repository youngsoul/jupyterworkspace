# Chapter 4, page 222 Binning


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from mglearn import datasets
from sklearn.preprocessing import OneHotEncoder


X, y = datasets.make_wave(n_samples=100)

line = np.linspace(-3,3, 1000, endpoint=False).reshape(-1,1)

reg_model = DecisionTreeRegressor(min_samples_split=3)
reg_model.fit(X,y)

plt.plot(line, reg_model.predict(line), label="decision tree regressor")

reg_model = LinearRegression()
reg_model.fit(X,y)

plt.plot(line, reg_model.predict(line), label="linear regression")

plt.plot(X[:,0], y, 'o', c='k')
plt.ylabel("Regression Output")
plt.xlabel("Input feature")
plt.legend(loc='best')


# create bins
bins = np.linspace(-3,3,11)

# digitize the values into bins
which_bin = np.digitize(X, bins=bins)

encoder = OneHotEncoder(sparse=False)

#encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)

X_binned = encoder.transform(which_bin)

# because we specified 10 bins to hold the X values, the transformed dataset X_binned now is made up of 10 features
# instead of numeric values for the continuous variable X, we bin'ed the values into 10 different possible bins
print(X_binned.shape)

line_binned = encoder.transform(np.digitize(line, bins=bins))

reg_model = LinearRegression()
reg_model.fit(X_binned, y)
plt.plot(line, reg_model.predict(line_binned), label='linear regression binned')

reg_model = DecisionTreeRegressor(min_samples_split=3)
reg_model.fit(X_binned, y)

plt.plot(line, reg_model.predict(line_binned), label="decision tree binned")


plt.show()
