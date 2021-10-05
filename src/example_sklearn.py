from matplotlib import pyplot as plt
import numpy as np
import copy
from sklearn.linear_model import (
    LinearRegression, TheilSenRegressor, RANSACRegressor, HuberRegressor)
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
# Make sure that it X is 2D
X = X[:, np.newaxis]

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10

estimators = [('OLS', LinearRegression()),
              ('Theil-Sen', TheilSenRegressor(random_state=42)),
              ('RANSAC', RANSACRegressor(random_state=42)),
              ('HuberRegressor', HuberRegressor())]
colors = {'OLS': 'turquoise', 'Theil-Sen': 'gold', 'RANSAC': 'lightgreen', 'HuberRegressor': 'black'}
linestyle = {'OLS': '-', 'Theil-Sen': '-.', 'RANSAC': '--', 'HuberRegressor': '--'}
lw = 3

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
        ('Modeling Errors Only', X, y),
        ('Corrupt X, Small Deviants', X_errors, y),
        ('Corrupt y, Small Deviants', X, y_errors),
        ('Corrupt X, Large Deviants', X_errors_large, y),
        ('Corrupt y, Large Deviants', X, y_errors_large)]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, 'b+')

    for name, estimator in estimators:
        # print(this_X)
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)

        mse = mean_squared_error(model.predict(X_test), y_test)
        print(name, 'mse pipe:', mse)

        poly = PolynomialFeatures(3)
        X_poly = poly.fit_transform(this_X)
        X_poly = np.delete(X_poly, 2, axis=1)

        # reg = TheilSenRegressor(random_state=42).fit(X, y)
        estimator.fit(X_poly, this_y)


        X_poly_test = poly.fit_transform(X_test)
        X_poly_test = np.delete(X_poly_test, 2, axis=1)


        # print(estimator.coef_)
        mse = mean_squared_error(estimator.predict(X_poly_test), y_test)
        print(name, 'mse mine:', mse)

        print()
        # # y_plot = model.predict(x_plot[:, np.newaxis])
        # y_plot = estimator.predict(x_plot[:, np.newaxis])
        # plt.plot(x_plot, y_plot, color=colors[name], linestyle=linestyle[name],
        #          linewidth=lw, label='%s: error = %.3f' % (name, mse))

    legend_title = 'Error of Mean\nAbsolute Deviation\nto Non-corrupt Data'
    legend = plt.legend(loc='upper right', frameon=False, title=legend_title,
                        prop=dict(size='x-small'))
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()