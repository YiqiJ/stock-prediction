from scipy.special import kv, gamma
from scipy.spatial.distance import pdist, cdist, squareform
import scipy
import matplotlib.pyplot as plt
import numpy as np

from validation import *

badex = np.load('badex.npy')
yahoops = np.load('yahoops.npy')

######################## Define Loss Function ############################


def loss(pred, true, cov):
    len_p = len(pred)
    len_t = len(true)
    assert(len_p == len_t)
    #std = np.sqrt(np.diag(cov))
    diff = np.abs(pred - true)
    #loss = np.sum((diff/true)*std)
    loss = np.sum((diff/true))

    trend_p = np.array([pred[i+1]-pred[i] for i in range(len_p-1)]) >= 0
    trend_t = np.array([true[i+1]-true[i] for i in range(len_t-1)]) >= 0
    trend_diff = np.abs(trend_p.astype(np.float32)-trend_t.astype(np.float32))

    weights = []
    for i in range(len_p-1):
        weights.append(3*(len_p-i))

    #weighted_trend_diff = np.array(weights)*trend_diff
    #final_weights = 1.5**np.sum(weighted_trend_diff)
    final_weights = 1.4**np.sum(trend_diff)
    return loss*final_weights

############################ Process Data #############################


def data(company, days=3):
    # days that we want to predict
    X = np.arange(372, 372+days).reshape(-1, 1)
    # training set
    X_train = np.arange(372).reshape(-1, 1)
    Y_train = company
    # here we assume the stock price is stationary
    mu_data_org = np.mean(Y_train)
    return X, X_train, Y_train, mu_data_org


def posterior_predictive(X, X_train, Y_train, kernel, mu_data, mu_a, sigma_ep=1e-8):
    K = kernel(X_train, X_train) + np.square(sigma_ep) * np.eye(len(X_train))
    K_s = kernel(X_train, X)
    K_ss = kernel(X, X) + np.square(sigma_ep) * np.eye(len(X))

    K_inv = np.linalg.inv(K)

    mu_s = mu_a + K_s.T @ K_inv @ (Y_train - mu_data)
    cov_s = K_ss - K_s.T @ K_inv @ K_s

    return mu_s, cov_s


####################### Construct Validatoin Set ######################
def train_valid_split(X_train, days=3):
    result = []
    for i in range(300, 372-days+1):
        result.append((np.arange(i-300, i), np.arange(i, i+days)))
    return result


X, X_train, Y_train, mu_data_org_Y = data(yahoops, days=9)
X, X_train, B_train, mu_data_org_B = data(badex, days=9)

####################### Cross Validation ########################
cases = []
cases_ind = []

for train_ind, valid_ind in train_valid_split(X_train, days=9):
    x_train, x_valid = X_train[train_ind], X_train[valid_ind]
    y_train, y_valid = Y_train[train_ind], Y_train[valid_ind]
    b_train, b_valid = B_train[train_ind], B_train[valid_ind]
    cases_ind.append(np.squeeze(np.concatenate([x_train, x_valid])))
    case_y = np.squeeze(np.concatenate([y_train, y_valid]))
    case_b = np.squeeze(np.concatenate([b_train, b_valid]))
    cases.append([case_y, case_b])
cases = np.array(cases)
cases_ind = np.array(cases_ind)

train_ind = cases_ind[:50]
test_ind = cases_ind[50:]

#Y_train_cases = cases[:50][:,0]
#Y_test_cases = cases[50:][:,0]
#B_train_cases = cases[:50][:,1]
#B_test_cases = cases[50:][:,1]
# print(Y_test_cases.shape)
# print(cases[:,0].shape)

Y_train_cases = cases[:, 0]
B_train_cases = cases[:, 1]

########################### Model Selection ##########################


def valid_prediction(test_cases, kernel, mu_data, mu_a, sigma_ep, days=3):
    prediction = []
    for c in range(len(test_cases)):
        x_train, x_valid = np.array(test_ind[c][:300]), np.array(
            test_ind[c][300:300+days])
        y_train, y_valid = np.array(test_cases[c][:300]), np.array(
            test_cases[c][300:300+days])
        # predict on validation set
        mu_pred, cov_pred = posterior_predictive(
            x_valid.reshape(-1, 1), x_train.reshape(-1, 1), y_train, kernel, mu_data, mu_a, sigma_ep)
        # prediction[i]的第一项是true stock price
        prediction.append((np.append(y_train[-1], mu_pred), cov_pred, np.append(
            x_train[-1], x_valid), np.append(y_train[-1], y_valid)))
    return prediction


########################### Yahoops ############################
# Yahoops
Y_rbf_kernel, Y_rbf_m_data, Y_rbf_m_a, Y_rbf_sigma_ep = hyperparameter_prediction2(
    Y_train_cases, kernel='rbf')
Y_matern_kernel, Y_matern_m_data, Y_matern_m_a, Y_matern_sigma_ep = hyperparameter_prediction2(
    Y_train_cases, kernel='matern')
Y_RQ_kernel, Y_RQ_m_data, Y_RQ_m_a, Y_RQ_sigma_ep = hyperparameter_prediction2(
    Y_train_cases, kernel='RQ')

print(Y_rbf_kernel, Y_rbf_m_data, Y_rbf_m_a, Y_rbf_sigma_ep)
print(Y_matern_kernel, Y_matern_m_data, Y_matern_m_a, Y_matern_sigma_ep)
print(Y_RQ_kernel, Y_RQ_m_data, Y_RQ_m_a, Y_RQ_sigma_ep)

############################# BadEx ##############################
B_rbf_kernel, B_rbf_m_data, B_rbf_m_a, B_rbf_sigma_ep = hyperparameter_prediction2(
    B_train_cases, kernel='rbf')
B_matern_kernel, B_matern_m_data, B_matern_m_a, B_matern_sigma_ep = hyperparameter_prediction2(
    B_train_cases, kernel='matern')
B_RQ_kernel, B_RQ_m_data, B_RQ_m_a, B_RQ_sigma_ep = hyperparameter_prediction2(
    B_train_cases, kernel='RQ')

print(B_rbf_kernel, B_rbf_m_data, B_rbf_m_a, B_rbf_sigma_ep)
print(B_matern_kernel, B_matern_m_data, B_matern_m_a, B_matern_sigma_ep)
print(B_RQ_kernel, B_RQ_m_data, B_RQ_m_a, B_RQ_sigma_ep)
