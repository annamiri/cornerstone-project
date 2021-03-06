def uncertainty_predict(picture, prediction_function, number_of_classes, number_of_predictions):
    n = number_of_predictions
    m = number_of_classes

    # create an empty list to store the predictions in, the maximal probability is the predicted class
    l = []
    for i in range(n):
        predictions = prediction_function([picture, 1])[0]
        l.append(predictions.argmax())

    #create a list to store the purity values in
    p = []
    for j in range(m):
        purity = l.count(j)/n
        p.append(purity)

    return l, p

def b_predict_mc(dataset, prediction_function, number_of_classes, number_of_predictions, max_deviation=0.0, additional_class=None):
    #create a list to store the results in
    if additional_class == None:
        e = number_of_classes + 1 #to mark a datapoint which network is not certain about
    else:
        e = additional_class

    r = []
    cl = [] # create a  list just with the classes
    n = dataset.shape[0] #number of datapoints in the dataset

    for i in range(n):
        probabilities = uncertainty_predict(dataset[i:i+1], prediction_function, number_of_classes, number_of_predictions)[1]
        p = max(probabilities)
        v = round(p*(1-p), 2)
        s = round(v**(1/2), 2)
        # get the class
        c = probabilities.index(max(probabilities))
        # get the class if uncertainty is set

        #if max_deviation == 0:
        #    d = c
        #else:
        #    if s > max_deviation:
        #        d = e
        #    else:
        #        d = c
        #d = c

        #r.append([c, d, p, v, s])
        r.append([c, p, v, s])

        #cl.append(d)

    return r
