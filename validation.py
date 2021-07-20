from kernel import *
from mipego import mipego
from mipego.Surrogate import RandomForest
from mipego.SearchSpace import ContinuousSpace, NominalSpace, OrdinalSpace


def hyperparameter_prediction(cases_onecompany, kernel='rbf'):
    if kernel == 'rbf':
        K = kernel_rbf
    elif kernel == 'matern':
        K = kernel_matern
    elif kernel == 'RQ':
        K = kernel_RQ
    else:
        raise NotImplementedError

    def obj_func(x):
        '''
        Prediction loss.
        '''
        #scale, l, nu, ma, mb, sigma = x['scale'], x['l'], x['nu'], x['m_0'], x['m_1'], x['sigma']

        if kernel == 'rbf':
            scale, length_scale, sigma_f, m, sigma = x['scale'], x[
                'length_scale'], x['sigma_f'], x['m'], x['sigma']
            pred_kernel = const(K(length_scale, sigma_f), scale)
        elif kernel == 'matern':
            scale, length_scale, nu, m, sigma = x['scale'], x['length_scale'], x['nu'], x['m'], x['sigma']
            pred_kernel = const(K(length_scale, nu), scale)
        elif kernel == 'RQ':
            scale, length_scale, alpha, m, sigma = x['scale'], x['length_scale'], x['alpha'], x['m'], x['sigma']
            pred_kernel = const(K(length_scale, alpha), scale)
        else:
            raise NotImplementedError

        price = cases_onecompany
        x_valid = np.arange(300, 303).reshape(-1, 1)
        x_train = np.arange(300).reshape(-1, 1)
        loss_p = 0
        for c in range(len(cases_onecompany)):
            pred_p, cov = posterior_predictive(
                x_valid+c, x_train+c, price[c][:300], pred_kernel, m, m, sigma)
            loss_p += loss(pred_p, price[c][300:303], cov)
        #print('loss={}, options={}'.format(loss_p,x))
        return loss_p

    if kernel == 'rbf':
        scale = ContinuousSpace([0.1, 2], "scale")    # kernel scaling
        length_scale = ContinuousSpace(
            [0.1, 3], "length_scale")            # RBF length scale
        sigma_f = ContinuousSpace([0.1, 3], "sigma_f")          # RBF type
        m = ContinuousSpace([50, 350], "m")          # data mean
        sigma = ContinuousSpace([0, 1], "sigma")     # data noise
        dummy0 = OrdinalSpace([0, 1], "dummy0")         # to solve bug
        dummy1 = NominalSpace([0, 0.1], "dummy1")

        search_space = scale * length_scale * sigma_f * m * sigma * dummy0 * dummy1
    elif kernel == 'matern':
        scale = ContinuousSpace([0.1, 2], "scale")    # kernel scaling
        length_scale = ContinuousSpace(
            [0.1, 3], "length_scale")            # matern length scale
        nu = NominalSpace([1.5, 2.5], "nu")          # matern type
        m = ContinuousSpace([50, 350], "m")          # data mean
        sigma = ContinuousSpace([0, 1], "sigma")     # data noise
        dummy = OrdinalSpace([0, 1], "dummy")         # to solve bug

        search_space = scale * length_scale * nu * m * sigma * dummy
    elif kernel == 'RQ':
        scale = ContinuousSpace([0.1, 2], "scale")    # kernel scaling
        length_scale = ContinuousSpace(
            [0.1, 3], "length_scale")            # RQ length scale
        alpha = ContinuousSpace([0.1, 3], "alpha")          # RQ type
        m = ContinuousSpace([50, 350], "m")          # data mean
        sigma = ContinuousSpace([0, 1], "sigma")     # data noise
        dummy0 = OrdinalSpace([0, 1], "dummy0")         # to solve bug
        dummy1 = NominalSpace([0, 0.1], "dummy1")

        search_space = scale * length_scale * alpha * m * sigma * dummy0 * dummy1
    else:
        raise NotImplementedError

    model = RandomForest(levels=search_space.levels)
    opt = mipego(search_space, obj_func, model,
                 minimize=True,  # the problem is a minimization problem.
                 max_eval=100,  # we evaluate maximum 500 times
                 max_iter=100,  # we have max 500 iterations
                 infill='EI',  # Expected improvement as criteria
                 n_init_sample=10,  # We start with 10 initial samples
                 n_point=1,  # We evaluate every iteration 1 time
                 n_job=1,  # with 1 process (job).
                 optimizer='MIES',  # We use the MIES internal optimizer.
                 verbose=True, random_seed=None,
                 log_file='log.txt')
    print('run')
    incumbent, stop_dict = opt.run()
    return incumbent, stop_dict


def hyperparameter_prediction2(cases_onecompany, kernel='rbf'):
    incumbent, stop_dict = hyperparameter_prediction(cases_onecompany, kernel)
    if kernel == 'rbf':
        kernel = const(kernel_rbf(
            length_scale=incumbent[1], sigma_f=incumbent[2]), incumbent[0])
        mu_data = incumbent[3]
        mu_a = incumbent[3]
        sigma_ep = incumbent[4]
        print("scale: " + str(incumbent[0]) + " length_scale: " +
              str(incumbent[1]) + " sigma_f: " + str(incumbent[2]))
        return kernel, mu_data, mu_a, sigma_ep,
    elif kernel == 'matern':
        kernel = const(kernel_matern(
            length_scale=incumbent[1], nu=incumbent[2]), incumbent[0])
        mu_data = incumbent[3]
        mu_a = incumbent[3]
        sigma_ep = incumbent[4]
        print("scale: " + str(incumbent[0]) + " length_scale: " +
              str(incumbent[1]) + " nu: " + str(incumbent[2]))
        return kernel, mu_data, mu_a, sigma_ep
    elif kernel == "RQ":
        kernel = const(
            kernel_RQ(length_scale=incumbent[1], alpha=incumbent[2]), incumbent[0])
        mu_data = incumbent[3]
        mu_a = incumbent[3]
        sigma_ep = incumbent[4]
        print("scale: " + str(incumbent[0]) + " length_scale: " +
              str(incumbent[1]) + " alpha: " + str(incumbent[2]))
        return kernel, mu_data, mu_a, sigma_ep
    else:
        raise NotImplementedError
