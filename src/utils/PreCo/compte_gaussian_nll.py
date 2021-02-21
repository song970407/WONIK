def compute_gaussian_nll(mu, log_var, target):
    var = log_var.exp()
    ll = -0.5 * log_var + (-0.5 * (mu - target).pow(2) / var)  # ignore constant term.
    nll = -1 * ll
    return nll
