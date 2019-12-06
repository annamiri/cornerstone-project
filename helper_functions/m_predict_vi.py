import pandas as pd
import math

def avg_predict_vi(picture, prediction_function, number_of_classes, number_of_predictions):
    n = number_of_predictions
    m = number_of_classes

    # create an empty list to store the predictions in, the maximal probability is the predicted class
    l = []
    for i in range(n):
        predictions = prediction_function.predict(picture)[0]
        l.append(predictions)
    df = pd.DataFrame(l)
    #print(l)

    #create a list to store the mean and variance values in
    mean = df.mean()
    max_value = mean.argmax()
    var = df.var()
    class_predicted = max_value
    var_predicted = var[max_value]
    mean_predicted = mean[max_value]

    return class_predicted, mean_predicted, var_predicted

def average_predict_vi(dataset, prediction_function, number_of_classes, number_of_predictions):
    #create a list to store the results in
    r = []
    n = dataset.shape[0] #number of datapoints in the dataset

    for i in range(n):
        cl, p, var = avg_predict_vi(dataset[i:i+1], prediction_function, number_of_classes, number_of_predictions)
        std = math.sqrt(var)
        r.append([cl, p, var, std])

    return r
