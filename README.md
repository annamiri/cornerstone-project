# What does this repository contain?
This repository contains the notebooks and scripts for my master thesis on Bayesian neural networks and uncertainty. The thesis is not publicly available.

Everything is written in Python 3 and kept in jupyter notebooks. The first letter of the name in the notebooks corresponds with the scenario I am looking at. In my thesis I check three scenarios as listed below.
•	A: Bayesian neural networks allow to conclude whether an input has been seen during training and can be classified or predicted
•	B: Bayesian neural networks perform better than standard neural networks on imbalanced datasets
•	C: Bayesian neural networks perform better than standard neural networks on small datasets 

The numbers following the letter in the name were used for versioning and do not contain any meaning. After the numbers, the letters MC or VI indicate whether the Monte Carlo Dropout or the Variational Inference method was used to approximate a Bayesian neural network. The then following letters do not contain information relevant to the reader. 

There are also three notebooks starting with the letter D, which does not correspondent to a scenario. In these notebooks the review of the results per scenario where done. The numbering and naming were done chronologically and the notebooks correspond as follows to the scenarios:

D_003 -> scenario A
D_002 -> scenario B
D_001 -> scenario C

The file helper_functions contains easy, self-written functions to use the prediction methods more conveniently.

# Remarks on methods
To approximate a Bayesian neural network I use two methods, the Monte Carlo Dropout and Variational Inference. For background on the Monte Carlo Dropout I recommend this blogpost by Yarin Gal http://mlg.eng.cam.ac.uk/yarin/blog_3d801aa532c1ce.html. Variationl Inference is described in many books, I particularly liked the definition and explanation in the book "Pattern Recognition and Machine Learning" by Christopher M. Bishop. 

To get the methods, function etc. right I relied on the book written by Oliver Dürr, Beate Sick and Elvis Morina which is called Probabilistic Deep Learning with Python.

# Remarks on code and packages
For the Monte Carlo Dropout method, TensorFlow 1 was used. To leave the learning phase on during prediction a special predict function was created in all of the scripts for the Monte Carlo dropout. 
For the Variational Inference TensorFlow 2 and TensorFlow Probability was used. To use TensorFlow 2 properly for my purposes, i,e. to simulate Bayesian neural networks and get varying results when predicting, the eager execution has to be disabled.
The mentioned helper functions were used in the notebooks starting with D to evaluate the different approximation and prediction methods in a more convenient way.
All the other user packages are standard packages used for data manipulation and plotting.
The notebooks can all be executed, the data is fetched from the internet via Keras. Since I stored and then reused all my models, the reader might want to be cautious of all the load and store statements in the script since this might not be wanted. However, you can easily ignore these steps.
When saving and loading the models it depends whether they use the Monte Carlo Dropout or the Variational Inference method. For the first one, the standard functions for saving and loading a model can be used. For the latter created with the TensorFlow Probability package, only the weights can be loaded again and therefore when reusing a model the user needs to define the model again and then load the weights. The reader can see these different steps in the respective notebooks.
