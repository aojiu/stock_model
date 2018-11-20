import csv
import numpy as np
from sklearn.svm import SVR
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib import style

dates = [1,2,3,4]
real_dates = []
prices = [50,60,70,80]

def predict_prices(dates, prices):

    #fit the model  with data
    #data is a one dimension matrix
    dates = np.reshape(dates,(4,1))

    svr_lin = SVR(kernel = "linear", C = 1e3)
    # svr_poly = SVR(kernel = "poly", C = 1e3, degree = 2)
    svr_rbf = SVR(C = 1e3, gamma = 0.1)

    svr_lin.fit(dates, prices)
    # svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    #plot the fitted line of the model with the same training data
    plt.scatter(dates, prices, color = "black", label = "Data")
    #Plot the line of model
    plt.plot(dates, svr_rbf.predict(dates), color = "red", label = "RBF model")
    plt.plot(dates, svr_lin.predict(dates), color="green", label="Linear model")
    # plt.plot(dates, svr_poly.predict(dates), color="blue", label="Polynomial model")

    #Plot the axis
    plt.xlabel("Data")
    plt.ylabel("Prices")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()



predict_prices(dates, prices)
