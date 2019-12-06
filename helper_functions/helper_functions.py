import pandas as pd
import math

# define function to select only specific category
def select_max_category(data, label, category, max_nr_for_category):
    new_data = []
    new_label = []
    for i in range(len(label)):
        if label[i] == category:
            if len(new_label) <= max_nr_for_category-1:
                new_label.append(label[i])
                new_data.append(data[i])
    #new_data = np.array(new_data)
    #new_label = np.array(new_label)
    return new_data, new_label

def uncertainty_predict_vi(picture, prediction_function, number_of_classes, number_of_predictions):
    n = number_of_predictions
    m = number_of_classes

    # create an empty list to store the predictions in, the maximal probability is the predicted class
    l = []
    for i in range(n):
        predictions = prediction_function.predict(picture)
        l.append(predictions.argmax())

    #create a list to store the purity values in
    p = []
    for j in range(m):
        purity = l.count(j)/n
        p.append(purity)

    return l, p

def bern_predict_vi(dataset, prediction_function, number_of_classes, number_of_predictions, max_deviation=0.0, additional_class=None):
    #create a list to store the results in
    if additional_class == None:
        e = number_of_classes + 1 #to mark a datapoint which network is not certain about
    else:
        e = additional_class

    r = []
    cl = [] # create a  list just with the classes
    n = dataset.shape[0] #number of datapoints in the dataset

    for i in range(n):
        probabilities = uncertainty_predict_vi(dataset[i:i+1], prediction_function, number_of_classes, number_of_predictions)[1]
        p = max(probabilities)
        v = round(p*(1-p), 2)
        s = round(v**(1/2), 2)
        # get the class
        c = probabilities.index(max(probabilities))

        r.append([c, p, v, s])

    return r

def max_predict_vi(picture, prediction_function, number_of_classes, number_of_predictions):
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

def maximum_predict_vi(dataset, prediction_function, number_of_classes, number_of_predictions):
    #create a list to store the results in
    r = []
    n = dataset.shape[0] #number of datapoints in the dataset

    for i in range(n):
        cl, p, var = max_predict_vi(dataset[i:i+1], prediction_function, number_of_classes, number_of_predictions)
        std = math.sqrt(var)
        r.append([cl, p, var, std])

    return r
