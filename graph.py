import matplotlib.pyplot as plt
import numpy as np
# Plot Gaussian Process


def plot_gp(mu, cov, X, samples=[]):
    X = X.reshape(-1)
    mu = mu.reshape(-1)

    # 95% confidence interval
    uncertainty = 1.96 * np.sqrt(np.diag(cov))

    plt.fill_between(X, mu + uncertainty, mu - uncertainty, alpha=0.1)
    for i in range(len(mu)):
        print("[", mu[i] - uncertainty[i], ", ", mu[i] + uncertainty[i], "] \n")
    plt.plot(X, mu, label='Mean')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='sample_{}'.format(i))

    plt.legend()


def plot_gp2(mu1, mu2, cov1, cov2, X, samples=[]):
    X = X.reshape(-1)
    mu1 = mu1.reshape(-1)
    mu2 = mu2.reshape(-1)

    # 95% confidence interval
    #uncertainty1 = 1.96 * np.sqrt(np.diag(cov1))
    #uncertainty2 = 1.96 * np.sqrt(np.diag(cov2))

    #plt.fill_between(X, mu1 + uncertainty1, mu1 - uncertainty1, alpha=0.1)
    plt.plot(X, mu1, label='Mean1')

    #plt.fill_between(X, mu2 + uncertainty2, mu2 - uncertainty2, alpha=0.1)
    plt.plot(X, mu2, label='Mean2')

    for i, sample in enumerate(samples):
        plt.plot(X, sample, lw=1, ls='--', label='sample_{}'.format(i))

    plt.legend()


def plot_pred(prediction, pred_x,  X_train, Y_train):
    """
    prediction: [(mu, cov, validx, validy), ...] prediction in validation set
    """
    pred_day = []
    x = []
    n = len(prediction)
    for i in range(0, n-1):
        mu_p, _, truex, truey = prediction[i]
        x.append(truex[1])
        pred_day.append(mu_p[1])
    lst_mu, _, lst_truex, lst_truey = prediction[n-1]
    pred_day = np.append(pred_day, list(lst_mu[1:]))

    plt.plot(X_train[300:], Y_train[300:], label='true')
    plt.plot(pred_x, prediction, label='pred')
