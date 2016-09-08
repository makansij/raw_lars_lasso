
import numpy as np 
import pandas as pd 
import scipy as sp
import operator

from sklearn import datasets, linear_model


# raw lasso:
# find predictor x_j with the highest correlation with the labels
# for b_j in range(start, end, stepsize):
#   - construct y=Xb
#   - get residuals r=y-y_hat
#   for x_j in (j's where j is not the index
#       of any of the predictors already in the model):
#       - regress the residuals on x_j using OLS
#       - compare:  1) correlation of current model with residuals AND
#           2) correlation of x_j with the residuals
#       - if 2) has higher correlation, then incorporate it into the model
#       (I think you actually use the feature/variable with the highest correlation)
# increase the joint least squares of all of the beta coefficients

# inputs: data matrix without NULLs
# outputs: LARS regression model which means:
#   all the beta coefficients

# helper functions that you'll need:
# 1) function to increase betas in direction of least squares
# 2) function to compute OLS and get residuals


class LARS_model():

    def __init__(self):
        self.coeff_dict={}

    def add_coeff_and_feature(self, feature, coeff):
        self.coeff_dict[feature]=coeff


def LARS(data, label_col):
    # input: "data" is the pandas dataframe, 
    #       "label_col" is the column in that dataframe that contains the labels.
    #   find predictor with the highest correlation with the label
    # output:
    #   LARS regression model
    current_lm_model=LARS_model()

    # Step 1:  find the predictor with the greatest correlation with the label
    feature_cols = filter(lambda aCol: aCol!=label_col, data.columns)
    data_matrix = data[feature_cols].as_matrix()

    for i in xrange(len(feature_cols)):
        if i==0:
            # find the feature with strongest correlation alone with the label
            y = data[label_col].values
            # y=y.values()
            max_corr=0.0
            max_corr_feature=''
            for col in feature_cols:
                x_j = data[col].values.reshape(len(data[col].values), 1)
                lm = linear_model.LinearRegression()
                print ' the fit is ', lm.fit(x_j, y)
                print ' the score is ', lm.score(x_j, y)
                print ' the coefficient is ', lm.coef_
                score = lm.score(x_j, y)
                coeff = lm.coef_
                if score > max_corr:
                    max_corr = score
                    max_corr_feature = col
            print ' feature with max corr is ', max_corr_feature
            # store the model with the new coefficient and feature
            current_lm_model.add_coeff_and_feature(max_corr_feature, coeff)
        else:
            i=0
            # increase betas in the joint least squares direction,
            #....while no model is found
            better_corr = False
            while (not better_corr):
                # get residuals
                # compute correlation of current model with the residuals (use OLS?)
                # compute correlation of each feature with the residuals (use OLS)
                # see if there exists a feature that is more correlated with the residuals than the current model
                # if so, then incorporate that feature into the model, and start increasing the betas in the
                # ....joint least squares direction

                # get residuals using OLS
                residuals = get_residuals(current_lm_model, data, label_col)
                # compute correlation of current model with the residuals
                curr_model_correlation = OLS(data[current_lm_model.coeff_dict.keys()].as_matrix(), residuals)
                # check if curr_model_correlation is zero???
                print ' curr_model_correlation is ', curr_model_correlation
                # compute correlation of each feature not yet in the model with the residuals 
                correlations = [(OLS(data[x_j].values.reshape(len(data[x_j].values), 1), residuals.reshape(len(residuals), 1)), x_j) \
                                for x_j in filter(lambda aCol: aCol not in current_lm_model.coeff_dict.keys(), feature_cols)]
                # get feature with max correlation and check that the correlation
                maximum_correlation = max(correlations, key=operator.itemgetter(0))
                if maximum_correlation[0] > curr_model_correlation:
                    # print '\n the feature ', maximum_correlation[1], ' has greater correlation with the residuals than the current model '
                    # print ' incorporating this new feature into the model \n'
                    current_lm_model.add_coeff_and_feature(maximum_correlation[1], 0)
                    better_corr=True
                else:
                    # print ' nothing greater found, increasing betas in joint least squares direction '
                    # increase the betas in the joint least squares direction
                    x = current_lm_model.coeff_dict.values()
                    keys = current_lm_model.coeff_dict.keys()
                    print ' before it is ', dict(zip(keys,x))
                    eps = np.zeros_like(x)                # create empty array [0, 0, 0]
                    index = np.argmin(gradient(x))        # get position of the lowest value in the array 
                    eps[index] = 0.1                      # go to that position in the empty array above, and replace it with epsilon   
                    x += eps                              # add the array with epsilon but otherwise empty, to the original array, x

                    print ' after it is ', dict(zip(keys,x))
                    current_lm_model.coeff_dict = dict(zip(keys,x))
                    
        # Step 2:  start increasing beta until there's another feature that
        #.... is more correlated with the residuals than the current model

    return current_lm_model


def get_residuals(current_lm_model, data, label_col):
    # data is a dataframe
    # y is 
    # current_lm_model is the current linear model as defined at beginning
    # y_true are the true labels.
    
    features = current_lm_model.coeff_dict.keys()
    coefficients = current_lm_model.coeff_dict.values()

    # grab only the columns that are currently in the model

    current_feature_cols = filter(lambda aCol: aCol in features, data.columns)
    data_matrix = data[current_feature_cols].as_matrix()
    betas_vector = np.matrix(coefficients)
    print ' data_matrix.shape[1] is ', data_matrix.shape[1]
    print ' betas_vector.shape[0] is ', betas_vector.shape[0]
    assert data_matrix.shape[1] == betas_vector.shape[0], ' dimensions don\'t match here '
    
    print ' we are using these features ', features

    y_hat = data_matrix*betas_vector
    assert data[label_col].values.shape[0] == data_matrix.shape[0], ' dimensions don\'t match ' 
    residuals = data[label_col].values.reshape(len(data[label_col].values), 1) - y_hat
    print ' the type of residuals is ', type(residuals)
    return residuals


def OLS(X, y):
    # inputs: X is a data matrix in the form of a numpy matrix
    #           y is vector of true y values
    # outputs: OLS model coefficients, and vector of residuals
    # fit, predict, and return the R^2 correlation 
    lm = linear_model.LinearRegression()
    lm.fit(X,y)
    correlation = lm.score(X,y)
    return correlation


def joint_least_squares(x):
    # "x" is an array of values
    # compute least squares
    pass


def gradient(x):
    # gradient/derivative of the squared function is 2 times each variable
    # it's like taking a partial derivative wrt each of the variables
    return np.array(2*x)

    
diabetes_data = datasets.load_diabetes()
diabetes_df=pd.DataFrame(diabetes_data.data, columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10'])
diabetes_target=diabetes_data.target.reshape(len(diabetes_data.target), 1)
target_df = pd.DataFrame(diabetes_target, columns=['target'])
diabetes_df=pd.concat([diabetes_df, target_df], axis=1)
current_lm_model=LARS(diabetes_df, 'target')

residuals = get_residuals(current_lm_model, diabetes_df, 'target')

