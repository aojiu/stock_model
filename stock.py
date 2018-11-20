import csv
import numpy as np
from sklearn.svm import SVR
from sklearn import model_selection
import matplotlib.pyplot as plt
from matplotlib import style

dates = []
real_dates = []
prices = []
# date_train=[]
# prices_test = []

def read_data(file_name):#to reade the data in csv file
    with open(file_name,"r") as csvFile:
        csvReader = csv.reader(csvFile, delimiter = ",")
        next(csvReader)
        # next(csvReader)

        for row in csvReader:
            elem = []
            elem.append(int(row[0]))#store the date in the list
            elem.append(float(row[3]))#store the stock prices in the list
            dates.append(elem)
            real_dates.append(int(row[0]))
            prices.append(float(row[3]))
    print(dates)
# def split(date, prices, validation_size): #does not work because this is random selection
#     date_train, prices_test, date_test, prices_test = model_selection.train_test_split(date, prices,
#                                                                                             test_size=validation_size,
#                                                                                             random_state=42)

def predict_prices(dates, prices, x):

    #fit the model  with data
    print(prices)

    svr_lin = SVR(kernel = "linear", C = 1e3)
    # svr_poly = SVR(kernel = "poly", C = 1e3, degree = 2)
    svr_rbf = SVR(C = 1e3, gamma = 0.1)

    svr_lin.fit(dates, prices)
    # svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    #plot the fitted line of the model with the same training data
    plt.scatter(real_dates, prices, color = "black", label = "Data")
    #Plot the line of model
    plt.plot(dates, svr_rbf.predict(real_dates), color = "red", label = "RBF model")
    plt.plot(dates, svr_lin.predict(real_dates), color="green", label="Linear model")
    # plt.plot(dates, svr_poly.predict(dates), color="blue", label="Polynomial model")

    #Plot the axis
    plt.xlabel("Data")
    plt.ylabel("Prices")
    plt.title("Support Vector Regression")
    plt.legend()
    plt.show()
    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0]

read_data("HistoricalQuotes.csv")
predict_prices(dates, prices, 29)
