from kernel import *
from train import *
from graph import *

##################### Calculate Schedule Profit ######################
# Yahoops
Y_rbf_kernel = const(kernel_rbf(length_scale=2.8360292938593776,
                     sigma_f=2.7387552413737923), 0.3600983440680554)
Y_rbf_m_data = 263.4720132411851
Y_rbf_m_a = 263.4720132411851
Y_rbf_sigma_ep = 0.5364201458420647

Y_matern_kernel = const(kernel_matern(
    length_scale=2.8908740514242157, nu=1.5), 1.8529272697196413)
Y_matern_m_data = 264.0034790936386
Y_matern_m_a = 264.0034790936386
Y_matern_sigma_ep = 0.07548126917944822

Y_RQ_kernel = const(kernel_RQ(length_scale=2.566840462179793,
                    alpha=0.1084333054789699), 0.9874950708481125)
Y_RQ_m_data = 74.69269351401996
Y_RQ_m_a = 74.69269351401996
Y_RQ_sigma_ep = 0.04744000906119038

# BadEx
B_rbf_kernel = const(kernel_rbf(length_scale=2.986846505189094,
                     sigma_f=2.262087818715068), 0.5687633214902885)
B_rbf_m_data = 71.73933571135058
B_rbf_m_a = 71.73933571135058
B_rbf_sigma_ep = 0.33831835481169653

B_matern_kernel = const(kernel_matern(
    length_scale=2.9466499863544366, nu=2.5), 0.15884571359219246)
B_matern_m_data = 72.6956518895716
B_matern_m_a = 72.6956518895716
B_matern_sigma_ep = 0.030729084975174265

B_RQ_kernel = const(kernel_RQ(length_scale=2.294760286762172,
                    alpha=0.46925427454602964), 1.5463947613360227)
B_RQ_m_data = 74.28836883677167
B_RQ_m_a = 74.28836883677167
B_RQ_sigma_ep = 0.3066258858882307

####################### Predict Date ########################
X_out, X_train_out, Y_train_out, _ = data(yahoops, days=6)
X_out, X_train_out, B_train_out, _ = data(badex, days=6)

Y_rbf_m_s, Y_rbf_cov_s = posterior_predictive(
    X_out, X_train_out, Y_train_out, Y_rbf_kernel, Y_rbf_m_data, Y_rbf_m_a, Y_rbf_sigma_ep)
Y_matern_m_s, Y_matern_cov_s = posterior_predictive(
    X_out, X_train_out, Y_train_out, Y_matern_kernel, Y_matern_m_data, Y_matern_m_a, Y_matern_sigma_ep)
Y_RQ_m_s, Y_RQ_cov_s = posterior_predictive(
    X_out, X_train_out, Y_train_out, Y_RQ_kernel, Y_RQ_m_data, Y_RQ_m_a, Y_RQ_sigma_ep)


B_rbf_m_s, B_rbf_cov_s = posterior_predictive(
    X_out, X_train_out, B_train_out, B_rbf_kernel, B_rbf_m_data, B_rbf_m_a, B_rbf_sigma_ep)
B_matern_m_s, B_matern_cov_s = posterior_predictive(
    X_out, B_train_out, B_train_out, B_matern_kernel, B_matern_m_data, B_matern_m_a, B_matern_sigma_ep)
B_RQ_m_s, B_RQ_cov_s = posterior_predictive(
    X_out, X_train_out, B_train_out, B_RQ_kernel, B_RQ_m_data, B_RQ_m_a, B_RQ_sigma_ep)

print("Yahoops: ")
print("\n day 368 price: ", Y_train_out[-1], "\n this is the RBF prediction: \n",
      Y_rbf_m_s, "\n this is the covariance matrix for the RBF prediction: \n", Y_rbf_cov_s)
print("\n day 368 price: ", Y_train_out[-1], "\n this is the Matern prediction: \n",
      Y_matern_m_s, "\n this is the covariance matrix for the Matern prediction: \n", Y_matern_cov_s)
print("\n day 368 price: ", Y_train_out[-1], "\n this is the RQ prediction: \n",
      Y_RQ_m_s, "\n this is the covariance matrix for the RQ prediction: \n", Y_RQ_cov_s)
print("BadEx: ")
print("\n day 368 price: ", B_train_out[-1], "\n this is the RBF prediction: \n",
      B_rbf_m_s, "\n this is the covariance matrix for the RBF prediction: \n", B_rbf_cov_s)
print("\n day 368 price: ", B_train_out[-1], "\n this is the Matern prediction: \n",
      B_matern_m_s, "\n this is the covariance matrix for the Matern prediction: \n", B_matern_cov_s)
print("\n day 368 price: ", B_train_out[-1], "\n this is the RQ prediction: \n",
      B_RQ_m_s, "\n this is the covariance matrix for the RQ prediction: \n", B_RQ_cov_s)

################### Plot Prediction and Confidence Interval #################
plot_gp(Y_rbf_m_s, Y_rbf_cov_s, X_out)
plot_gp(Y_matern_m_s, Y_matern_cov_s, X_out)
plot_gp(Y_RQ_m_s, Y_RQ_cov_s, X_out)
plot_gp(B_rbf_m_s, B_rbf_cov_s, X_out)
plot_gp(B_matern_m_s, B_matern_cov_s, X_out)
plot_gp(B_RQ_m_s, B_RQ_cov_s, X_out)
