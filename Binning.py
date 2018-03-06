# Chapter 4, page 222 Binning


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt
from mglearn import datasets
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures

X, y = datasets.make_wave(n_samples=100)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# create a 1000 input values (X predict) between -3 and 3
# and then predict the y value.
# plot the prediction for DecisionTreeRegressor and LinearRegressor
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
print(f"X_binned shape: {X_binned.shape}")

line_binned = encoder.transform(np.digitize(line, bins=bins))

reg_model = LinearRegression()
reg_model.fit(X_binned, y)
plt.plot(line, reg_model.predict(line_binned), label='linear regression binned', alpha=0.5)

reg_model = DecisionTreeRegressor(min_samples_split=3)
reg_model.fit(X_binned, y)

plt.plot(line, reg_model.predict(line_binned), label="decision tree binned", alpha=0.8)


# Interactions and Polynomials
# page 229

#include polynomials up to x** 10
# the default 'include_bias=True' adds a feature that is constantly 1

poly = PolynomialFeatures(degree=10, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

print(f"X_poly.shape: {X_poly.shape}")

poly_reg = LinearRegression().fit(X_poly, y)

# since line is our X input to predict, we have to transform this just like we did our training X
X_predict_line_poly = poly.transform(line)

plt.plot(line, poly_reg.predict(X_predict_line_poly), label='ploynomial linear regression')

plt.legend(loc='best')

plt.show()
