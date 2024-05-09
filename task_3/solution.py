"""Solution."""
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
from scipy.stats import norm
# import additional ...
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
import matplotlib.pyplot as plt

# global variables
DOMAIN = np.array([[0, 10]])  # restrict \theta in [0, 10]
SAFETY_THRESHOLD = 4  # threshold, upper bound of SA


# TODO: implement a self-contained solution in the BO_algo class.
# NOTE: main() is not called by the checker.
class BO_algo():
    def __init__(self):
        """Initializes the algorithm with a parameter configuration."""
        # TODO: Define all relevant class members for your BO algorithm here.
        # Gaussian Processes
        kernel_f = ConstantKernel(constant_value=1) * RBF(length_scale=10, length_scale_bounds=(1, 1e3)) + WhiteKernel(noise_level=0.15)
        self.gp_f = GaussianProcessRegressor(kernel=kernel_f)

        kernel_v = ConstantKernel(constant_value=1) * RBF(length_scale=1e-3, length_scale_bounds=(1e-5, 1e-2)) + WhiteKernel(noise_level=1e-4)
        self.gp_v = GaussianProcessRegressor(kernel=kernel_v)

        self.prior_mean_v = 4
        self.xi = 0.87
        self.X = []
        self.f = []
        self.v = []


    def next_recommendation(self):
        """
        Recommend the next input to sample.

        Returns
        -------
        recommendation: float
            the next point to evaluate
        """
        # TODO: Implement the function which recommends the next point to query
        # using functions f and v.
        # In implementing this function, you may use
        # optimize_acquisition_function() defined below.
        recommendation = min(self.optimize_acquisition_function(), 10)
        recommendation = max(0, recommendation)

        self.xi = self.xi * 0.8
        return np.array([recommendation]).reshape(-1, 1)


    def optimize_acquisition_function(self):
        """Optimizes the acquisition function defined below (DO NOT MODIFY).

        Returns
        -------
        x_opt: float
            the point that maximizes the acquisition function, where
            x_opt in range of DOMAIN
        """

        def objective(x):
            return -self.acquisition_function(x)

        f_values = []
        x_values = []

        # Restarts the optimization 20 times and pick best solution
        for _ in range(20):
            x0 = DOMAIN[:, 0] + (DOMAIN[:, 1] - DOMAIN[:, 0]) * \
                 np.random.rand(DOMAIN.shape[0])
            result = fmin_l_bfgs_b(objective, x0=x0, bounds=DOMAIN,
                                   approx_grad=True)
            x_values.append(np.clip(result[0], *DOMAIN[0]))
            f_values.append(-result[1])

        ind = np.argmax(f_values)
        x_opt = x_values[ind].item()

        return x_opt

    def acquisition_function(self, x: np.ndarray):
        """Compute the acquisition function for x.

        Parameters
        ----------
        x: np.ndarray
            x in domain of f, has shape (N, 1)

        Returns
        ------
        af_value: np.ndarray
            shape (N, 1)
            Value of the acquisition function at x
        """
        x = np.atleast_2d(x)
        # TODO: Implement the acquisition function you want to optimize.
        mu_f, sigma_f = self.gp_f.predict(x, return_std=True)
        mu_v, sigma_v = self.gp_v.predict(x, return_std=True)
        mu_v += self.prior_mean_v

        best_f = max(self.f)

        improvement = mu_f - best_f - self.xi
        Z = improvement / sigma_f
        ei = improvement * norm.cdf(Z) + sigma_f * norm.pdf(Z)

        pf = norm.cdf((SAFETY_THRESHOLD - mu_v) / sigma_v)

        af_value = ei * pf
        return af_value

    def add_data_point(self, x: float, f: float, v: float):
        """
        Add data points to the model.

        Parameters
        ----------
        x: float
            structural features
        f: float
            logP obj func
        v: float
            SA constraint func
        """
        # TODO: Add the observed data {x, f, v} to your model.
        self.X.append(x)
        self.f.append(f)
        self.v.append(v)

        # Update surrogates
        self.gp_f.fit(np.array(self.X).reshape(-1, 1), np.array(self.f))
        self.gp_v.fit(np.array(self.X).reshape(-1, 1), np.array(self.v) - self.prior_mean_v)

    def get_solution(self):
        """
        Return x_opt that is believed to be the maximizer of f.

        Returns
        -------
        solution: float
            the optimal solution of the problem
        """
        # TODO: Return your predicted safe optimum of f.
        lambda_ = 55
        lagrangian = np.array(self.f).ravel() - lambda_ * np.array([np.max(v_, 0) for v_ in self.v]).ravel()

        id_max = np.argmax(lagrangian)
        solution = self.X[id_max]
        return solution
    
    def plot(self, f_fn, v_fn):
        """Plot objective and constraint posterior for debugging (OPTIONAL)."""
        # Ground Truth
        xs = np.linspace(DOMAIN[0,0], DOMAIN[0,1], num=1000)
        f_vals = np.array([f_fn(x) for x in xs])
        v_vals = np.array([v_fn(x) for x in xs])

        # Predictions
        f_pred, f_std = self.gp_f.predict(xs.reshape(-1, 1), return_std=True)
        v_pred, v_std = self.gp_v.predict(xs.reshape(-1, 1), return_std=True)
        v_pred += self.prior_mean_v


        fig, axes = plt.subplots(ncols=2, nrows=1)
        legend_fontsize = 8
        legend_alpha = 0.5

        axes[0].plot(xs, f_vals, 'k--', label="True Function")
        axes[0].plot(xs, f_pred, '-', color="tab:blue", lw=2, label="Objective Posterior")
        axes[0].fill_between(
            xs,
            (f_pred - f_std),
            (f_pred + f_std),
            alpha=0.2,
            label="Confidence Interval"
        )
        axes[0].plot(np.array(self.X).flatten(), self.f, '.', markersize=15, color="tab:orange", label="Data")
        axes[0].legend(loc="lower right", fontsize=legend_fontsize, framealpha=legend_alpha)
        axes[0].title.set_text("Objective Function")


        axes[1].plot(xs, v_vals, 'k--', label="True Function")
        axes[1].plot(xs, v_pred, '-', color="tab:blue", lw=2, label="Constraint Posterior")
        axes[1].fill_between(
            xs,
            (v_pred - v_std),
            (v_pred + v_std),
            alpha=0.2,
            label="Confidence Interval"
        )
        axes[1].plot(np.array(self.X).flatten(), self.v, '.', markersize=15, color="tab:orange", label="Data")
        axes[1].legend(loc="upper left", fontsize=legend_fontsize, framealpha=legend_alpha)
        axes[1].title.set_text("Constraint Function")

        plt.savefig('obj_and_const_functions.png')

# ---
# TOY PROBLEM. To check your code works as expected (ignored by checker).
# ---

def check_in_domain(x: float):
    """Validate input"""
    x = np.atleast_2d(x)
    return np.all(x >= DOMAIN[None, :, 0]) and np.all(x <= DOMAIN[None, :, 1])


def f(x: float):
    """Dummy logP objective"""
    mid_point = DOMAIN[:, 0] + 0.5 * (DOMAIN[:, 1] - DOMAIN[:, 0])
    return - np.linalg.norm(x - mid_point, 2)


def v(x: float):
    """Dummy SA"""
    return 2.0


def get_initial_safe_point():
    """Return initial safe point"""
    x_domain = np.linspace(*DOMAIN[0], 4000)[:, None]
    c_val = np.vectorize(v)(x_domain)
    x_valid = x_domain[c_val < SAFETY_THRESHOLD]
    np.random.seed(0)
    np.random.shuffle(x_valid)
    x_init = x_valid[0]

    # return x_init
    return np.array([x_init]).reshape(-1, 1)


def main():
    """FOR ILLUSTRATION / TESTING ONLY (NOT CALLED BY CHECKER)."""
    # Init problem
    agent = BO_algo()

    # Add initial safe point
    x_init = get_initial_safe_point()
    obj_val = f(x_init)
    cost_val = v(x_init)
    agent.add_data_point(x_init, obj_val, cost_val)

    # Loop until budget is exhausted
    for j in range(20):
        # Get next recommendation
        x = agent.next_recommendation()

        # Check for valid shape
        assert x.shape == (1, DOMAIN.shape[0]), \
            f"The function next recommendation must return a numpy array of " \
            f"shape (1, {DOMAIN.shape[0]})"

        # Obtain objective and constraint observation
        obj_val = f(x) + np.random.randn()
        cost_val = v(x) + np.random.randn()
        agent.add_data_point(x, obj_val, cost_val)

    # Validate solution
    solution = agent.get_solution()
    assert check_in_domain(solution), \
        f'The function get solution must return a point within the' \
        f'DOMAIN, {solution} returned instead'

    # Compute regret
    regret = (0 - f(solution))

    print(f'Optimal value: 0\nProposed solution {solution}\nSolution value '
          f'{f(solution)}\nRegret {regret}\nUnsafe-evals TODO\n')
    
    agent.plot(f, v)


if __name__ == "__main__":
    main()
