import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold

import statistics

import random
random.seed(1)

def f(x):
	return np.sin(np.pi*x)

def dataset_generator(sample_size,noise = "noiseless") :
    diff_base = 0.2
    x_initial = -1
    sample_size_base = 10

    ratio = sample_size/sample_size_base
    diff = diff_base/ratio

    x_array = np.array([])
    y_array = np.array([])
    for i in range(sample_size) :
        x = x_initial + (i*diff)

        x_array = np.append(x_array, np.array([x]))
        
        if noise == "noiseless" :
            y = f(x)
            y_array = np.append(y_array, np.array([y]))

        elif noise == "noisy" :
            y = f(x) + random.uniform(-1,1)
            y_array = np.append(y_array, np.array([y]))

    return x_array.reshape(-1, 1), y_array.reshape(-1, 1)


def dataset(file_name) :
    data = pd.read_csv("C:/Users/User/Desktop/tem/{a}.csv".format(a = file_name))

    x = data.iloc[:,0].to_numpy()
    y = data.iloc[:,8].to_numpy()

    return x.reshape(-1, 1), y.reshape(-1, 1)

def training_set(x,y,n_degree) :
    """if random :
        x_train, x, y_train, y = train_test_split(x, y, test_size = test_size, random_state = seed)"""

    poly = PolynomialFeatures(degree=n_degree, include_bias=False)
    poly_features = poly.fit_transform(x)
    reg = LinearRegression().fit(poly_features, y)

    w = reg.coef_
    b = reg.intercept_

    y_predict = reg.predict(poly_features)

    error = mean_squared_error(y, y_predict)
    rms_error = np.sqrt(error)

    return y_predict ,w ,b ,rms_error

def holdout(x,y,seed,train_size):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = (1 - train_size), random_state = seed)

    reg = LinearRegression().fit(x_train, y_train)

    w = reg.coef_
    b = reg.intercept_

    y_predict = reg.predict(np.array(x_test))

    error = mean_squared_error(y_test, y_predict)
    rms_error = np.sqrt(error)

    return y_predict ,w ,b ,rms_error

def cross_validation(x,y,seed,n_fold,n_degree) :
    poly = PolynomialFeatures(degree=n_degree, include_bias=False)
    poly_features = poly.fit_transform(x)

    kf = KFold(n_splits = n_fold, random_state = seed, shuffle = True)
    sum_rms_error = 0
    for train_index, test_index in kf.split(poly_features) :
        x_train, x_test = poly_features[train_index], poly_features[test_index]
        y_train, y_test = y[train_index], y[test_index]

        reg = LinearRegression().fit(x_train, y_train)

        y_predict = reg.predict(np.array(x_test))

        error = mean_squared_error(y_test, y_predict)
        rms_error = np.sqrt(error)
		
        sum_rms_error += rms_error
	
    return sum_rms_error/n_fold

#-----------------------------plot linear graph-----------------------------------

def plot_linear_graph(x,y,y_predict) :
    plt.plot(x,y,"o")
    plt.plot(x,y_predict,"-")
    plt.grid()
    plt.legend(["linear"], loc='best')
    plt.show()

#-----------------------------plot graph-----------------------------------

def plot_holdout_graph(x, y, mod) :
    size_list = []
    all_rms_error = []
    all_sd = []
    limit_size = 9
    limit_seed = 100
    for train_size in range(limit_size) :
        sum_rms_error = 0
        rms_all_seed = []
        for seed in range(limit_seed) :
            y_predict ,w ,b ,rms_error = holdout(x,y,seed + 1,0.1*(train_size+1))
            rms_all_seed.append(rms_error)
            sum_rms_error += rms_error

        size_list.append(0.1*(train_size+1))
        all_rms_error.append(sum_rms_error/limit_seed)
        all_sd.append(statistics.stdev(rms_all_seed))

    train_y_predict ,train_w ,train_b ,train_rms_error = training_set(x, y)

    if mod == "rms" :
        plt.plot(size_list,all_rms_error,"-")
        plt.plot(size_list,len(all_rms_error)*[train_rms_error],"--")
        plt.grid()
        plt.legend(["holdout rms average","training set"], loc='best')
        plt.xlabel('training set percentage split')
        plt.ylabel('root mean square error')
        plt.title('holdout')
        plt.show()
    elif mod == "sd" :
        plt.plot(size_list,all_sd,"-")
        plt.grid()
        plt.legend(["holdout rms S.D."], loc='best')
        plt.xlabel('training set percentage split')
        plt.ylabel('standard deviation')
        plt.title('holdout')
        plt.show()

def plot_cross_validation_graph(x, y, mod) :
    n_fold_list = []
    all_rms_error = []
    all_sd = []
    limit_n_fold = 50
    limit_seed = 100
    for n_fold in range(1,limit_n_fold) :
        sum_rms_error = 0
        rms_all_seed = []
        for seed in range(limit_seed) :
            rms_error = cross_validation(x,y,seed + 1,n_fold + 1)
            rms_all_seed.append(rms_error)
            sum_rms_error += rms_error

        n_fold_list.append(n_fold + 1)
        all_rms_error.append(sum_rms_error/limit_seed)
        all_sd.append(statistics.stdev(rms_all_seed))

    train_y_predict ,train_w ,train_b ,train_rms_error = training_set(x, y)

    if mod == "rms" :
        plt.plot(n_fold_list,all_rms_error,"-")
        plt.plot(n_fold_list,len(all_rms_error)*[train_rms_error],"--")
        plt.grid()
        plt.legend(["cross validation rms average","training set"], loc='best')
        plt.xlabel('k fold')
        plt.ylabel('root mean square error')
        plt.title('cross validation')
        plt.show()
    elif mod == "sd" :
        plt.plot(n_fold_list,all_sd,"-")
        plt.grid()
        plt.legend(["cross validation rms S.D."], loc='best')
        plt.xlabel('k fold')
        plt.ylabel('standard deviation')
        plt.title('cross validation')
        plt.show()

def accuracy_graph(x, y) :
    limit_seed = 100

    data_size_start = 100
    step = 100
    data_size_limit = 1000

    rms_train_array = []
    rms_holdout_array = []
    rms_cross_array = []

    sample_size_array = []

    for sample_size in range(data_size_start, data_size_limit+1, step) :
        train_data_retio = 1 - (sample_size / len(x))

        sum_rms_train = 0
        sum_rms_holdout = 0
        sum_rms_cross = 0

        sample_size_array.append(sample_size)

        for seed in range(limit_seed) :
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = train_data_retio, random_state = seed + 1)

            a_ ,a__ ,a___ ,train_rms_error = training_set(x_train, y_train)
            sum_rms_train += train_rms_error

            b_ ,b__ ,b___ ,holdout_rms_error = holdout(x_train,y_train,seed + 1,0.1)
            sum_rms_holdout += holdout_rms_error

            cross_rms_error = cross_validation(x_train,y_train,seed + 1,10)
            sum_rms_cross += cross_rms_error

        rms_train_array.append(sum_rms_train/limit_seed)
        rms_holdout_array.append(sum_rms_holdout/limit_seed)
        rms_cross_array.append(sum_rms_cross/limit_seed)

    plt.plot(sample_size_array,rms_train_array,"-")
    plt.plot(sample_size_array,rms_holdout_array,"-")
    plt.plot(sample_size_array,rms_cross_array,"-")
    plt.grid()
    plt.legend(["training set","holdout","cross validation"], loc='best')
    plt.xlabel('Sampling Size')
    plt.ylabel('root mean square error')
    plt.title('Accuracy Test')
    plt.show()

def precision_graph(x, y) :
    limit_seed = 100

    data_size_start = 100
    step = 100
    data_size_limit = 1000

    sd_train_array = []
    sd_holdout_array = []
    sd_cross_array = []

    sample_size_array = []

    for sample_size in range(data_size_start, data_size_limit+1, step) :
        train_data_retio = 1 - (sample_size / len(x))

        rms_all_seed_train = []
        rms_all_seed_holdout = []
        rms_all_seed_cross = []

        sample_size_array.append(sample_size)

        for seed in range(limit_seed) :
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = train_data_retio, random_state = seed + 1)

            a_ ,a__ ,a___ ,train_rms_error = training_set(x_train, y_train)
            rms_all_seed_train.append(train_rms_error)

            b_ ,b__ ,b___ ,holdout_rms_error = holdout(x_train,y_train,seed + 1,0.1)
            rms_all_seed_holdout.append(holdout_rms_error)

            cross_rms_error = cross_validation(x_train,y_train,seed + 1,10)
            rms_all_seed_cross.append(cross_rms_error)

        sd_train_array.append(statistics.stdev(rms_all_seed_train))
        sd_holdout_array.append(statistics.stdev(rms_all_seed_holdout))
        sd_cross_array.append(statistics.stdev(rms_all_seed_cross))

    plt.plot(sample_size_array,sd_train_array,"-")
    plt.plot(sample_size_array,sd_holdout_array,"-")
    plt.plot(sample_size_array,sd_cross_array,"-")
    plt.grid()
    plt.legend(["training set","holdout","cross validation"], loc='best')
    plt.xlabel('Sampling Size')
    plt.ylabel('Standard deviation')
    plt.title('Precision Test')
    plt.show()

#____________________________week 3_____________________________

def check_overfit(x, y, N,n_data) :
    train_rms_error_list = []
    cross_rms_error_list = []

    degree_list = []

    for n_degree in range(1,N+1) :
        y_predict ,w ,b ,train_rms_error = training_set(x, y, n_degree)
        train_rms_error_list.append(train_rms_error)

        cross_rms_error = cross_validation(x,y,1,10,n_degree)
        cross_rms_error_list.append(cross_rms_error)

        degree_list.append(n_degree)

    less_error = min(cross_rms_error_list)
    less_error_index = cross_rms_error_list.index(less_error)

    plt.plot(degree_list,train_rms_error_list,"-")
    plt.plot(degree_list,cross_rms_error_list,"-")
    plt.plot(degree_list[less_error_index],cross_rms_error_list[less_error_index],"o")
    plt.text(degree_list[less_error_index],cross_rms_error_list[less_error_index],"(" + str(degree_list[less_error_index]) + ")",
            color = 'green', fontweight = 'bold', size=20)
    plt.grid()
    plt.legend(["training set","cross validation 10 kfold","cross validation lowest error"], loc='best')
    plt.xlabel('Degree')
    plt.ylabel('root mean square error')
    plt.title('sin noiseless {n} sample'.format(n = n_data))
    plt.show()

def check_overfit_sample(n_degree,noise = "noiseless") :
    train_rms_error_list = []
    cross_rms_error_list = []

    sample_size_list = []

    for sample_size in range(10,100) :
        x, y = dataset_generator(sample_size,noise)

        y_predict ,w ,b ,train_rms_error = training_set(x, y, n_degree)
        train_rms_error_list.append(train_rms_error)

        cross_rms_error = cross_validation(x,y,1,10,n_degree)
        cross_rms_error_list.append(cross_rms_error)

        sample_size_list.append(sample_size)

    plt.plot(sample_size_list,train_rms_error_list,"-")
    plt.plot(sample_size_list,cross_rms_error_list,"-")
    plt.grid()
    plt.legend(["training set","cross validation 10 kfold"], loc='best')
    plt.xlabel('sample size')
    plt.ylabel('root mean square error')
    plt.title('sin {m} polynomial {n} degree'.format(n = n_degree, m = noise))
    plt.show()


if __name__ == '__main__':

    data_size = 10
    """file_name = "sin_noiseless_{n}sample".format(n = data_size)
    x_train, y_train = dataset(file_name)"""

    x_train, y_train = dataset_generator(data_size)
    #print(y_train)

    #y_predict ,w ,b ,rms_error = training_set(x_train, y_train,2)
    #y_predict ,w ,b ,rms_error = holdout(x_train,y_train,1,0.2)
    #rms_error = cross_validation(x_train,y_train,1,10,2)

    #plot_holdout_graph(x_train, y_train,"sd")
    #plot_cross_validation_graph(x_train, y_train,"rms")

    #accuracy_graph(x_train, y_train)
    #precision_graph(x_train, y_train)

    #check_overfit(x_train, y_train,20,data_size)
    check_overfit_sample(2,"noiseless")
