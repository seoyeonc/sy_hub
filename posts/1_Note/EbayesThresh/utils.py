import numpy as np
from scipy.stats import norm, cauchy, laplace
from scipy.optimize import minimize


def beta_cauchy(x):
    """
    Find the function beta for the mixed normal prior with Cauchy
    tails.  It is assumed that the noise variance is equal to one.
    Calculate the function beta for the mixed normal prior with Cauchy tails.
    It is assumed that the noise variance is equal to one.
    
    Parameters:
    x : array-like
        Input values
        
    Returns:
    beta : array-like
        Calculated beta values
    """
    
    phix = norm.pdf(x)
    j = (x != 0)
    beta = x
    beta = np.where(j == False, -1/2, beta)
    beta[j] = (norm.pdf(0) / phix[j] - 1) / (x[j] ** 2) - 1
    
    return beta



def beta_laplace(x, s=1, a=0.5):
    """
    The function beta for the Laplace prior given parameter a and s (sd)
    """
    
    x = np.abs(x)
    
    xpa = x/s + s*a
    xma = x/s - s*a
    
    rat1 = 1/xpa
    
    if isinstance(xpa, np.ndarray):
        for i in range(len(xpa)):
            if xpa[i] < 35:
                rat1[i] = norm.cdf(-xpa[i]) / norm.pdf(xpa[i])
    else:
        if xpa < 35:
            rat1 = norm.cdf(-xpa) / norm.pdf(xpa)

    rat2 = 1/np.abs(xma)
    
    if isinstance(xma, np.ndarray):
        for i in range(len(xma)):
            if xma[i] > 35:
                xma[i] = 35
    else:
         if xma > 35:
                xma = 35
                
    if isinstance(xma, np.ndarray):
        for i in range(len(xma)):
            if xma[i] > -35:
                rat2[i] = norm.cdf(xma[i])/norm.pdf(xma[i])   
    else:
        if xma > -35:
            rat2 = norm.cdf(xma)/norm.pdf(xma)   

    beta = (a * s) / 2 * (rat1 + rat2) - 1
    
    return beta


def mad(x, center=None, constant=1.4826, na_rm=False, low=False, high=False):
    # NA 제거 옵션 처리
    if na_rm:
        x = x[~np.isnan(x)]

    # center 값 설정
    if center is None:
        center = np.median(x)

    # low 또는 high median 설정
    if low:
        center = np.percentile(x, 25)
    elif high:
        center = np.percentile(x, 75)

    # Median Absolute Deviation 계산
    mad_value = np.median(np.abs(x - center)) * constant
    return mad_value



def wfromt(tt, s=1, prior="laplace", a=0.5):
    """
    Find the weight that has posterior median threshold tt, 
    given s (sd) and a.
    
    tt - Threshold value or vector of values.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as tt. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    """
    pr = prior[0:1]
    if pr == "l":
        tma = tt / s - s * a
        wi = 1 / np.abs(tma)
        if isinstance(wi, (int, np.integer, np.ndarray)):
            wi[tma > -35] = norm.cdf(tma[tma > -35]) / norm.pdf(tma[tma > -35])
        wi = a * s * wi - beta_laplace(tt, s, a)
        
    if pr == "c":
        dnz = norm.pdf(tt)
        wi = 1 + (norm.cdf(tt) - tt * dnz - 1/2) / (np.sqrt(np.pi/2) * dnz * tt**2)
        if isinstance(wi, np.ndarray):
            for i in range(len(wi)):
                if not np.isfinite(wi[i]):
                    wi[i] = 1
        else:
            if not np.isfinite(wi):
                wi = 1
    return 1 / wi


def wfromx(x, s=1, prior="laplace", a=0.5, universalthresh=True):
    """
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    pr = prior[0:1]

    if pr == "c":
        s = 1

    if universalthresh:
        tuniv = np.sqrt(2 * np.log(len(x))) * s
        wlo = wfromt(tuniv, s, prior, a)
        wlo = np.max(wlo)
    else:
        wlo = 0

    if pr == "l":
        beta = beta_laplace(x, s, a)
    elif pr == "c":
        beta = beta_cauchy(x)

    whi = 1
    beta = np.minimum(beta, 1e20)

    shi = np.sum(beta / (1 + beta))
    if shi >= 0:
        shi =  1

    slo = np.sum(beta / (1 + wlo * beta))
    if slo <= 0:
        slo = wlo

    for _ in range(1,31):
        wmid = np.sqrt(wlo * whi)
        smid = np.sum(beta / (1 + wmid * beta))
        if smid == 0:
            smid = wmid
        if smid > 0:
            wlo = wmid
        else:
            whi = wmid
            
    return np.sqrt(wlo * whi)

def isotone(x, wt=None, increasing=False):
    """
    Find the weighted least squares isotone fit to the sequence x,
    the weights given by the sequence wt. If increasing == True,
    the curve is set to be increasing, otherwise to be decreasing.
    The vector ip contains the indices on the original scale of the
    breaks in the regression at each stage.

    Parameters:
        x (list or numpy.ndarray): Input sequence.
        wt (list or numpy.ndarray, optional): Weights for the sequence x. Defaults to None.
        increasing (bool, optional): If True, the curve is set to be increasing, otherwise decreasing.
            Defaults to False.

    Returns:
        list: Isotonic fit to the input sequence x.
    """
    nn = len(x)
    if nn == 1:
        x = x.copy()
    if not increasing:
        x = -(x.copy())

    ip = np.arange(1, nn+1)
    dx = np.diff(x)
    nx = len(x)

    while nx > 1 and np.min(dx) < 0:
        # Find all local minima and maxima
        jmax = np.where(np.concatenate((dx <= 0, [False])) & np.concatenate(([True], dx > 0)))[0] + 1
        jmin = np.where(np.concatenate((dx > 0, [True])) & np.concatenate(([False], dx <= 0)))[0] + 1

        for jb in range(len(jmax)):
            ind = np.arange(jmax[jb], jmin[jb])
            wtn = np.sum(wt[ind])
            x[jmax[jb]] = np.sum(wt[ind] * x[ind]) / wtn
            wt[jmax[jb]] = wtn
            x[jmax[jb]:jmin[jb]] = np.nan

        # Clean up within iteration, eliminating the parts of sequences that
        # were set to NA
        ind = ~np.isnan(x)
        x = x[ind]
        wt = wt[ind]
        ip = ip[ind]
        dx = np.diff(x)
        nx = len(x)

    # Final cleanup: reconstruct z at all points by repeating the pooled
    # values the appropriate number of times
    jj = np.zeros(nn, dtype=int)
    jj[ip - 1] = 1
    z = x[np.cumsum(jj) - 1]

    if not increasing:
        z = -z.copy()

    return z.tolist()

def wmonfromx(xd, prior="laplace", a=0.5, tol=1e-08, maxits=20):
    """
    Find the monotone marginal maximum likelihood estimate of the
    mixing weights for the Laplace prior with parameter a.  It is
    assumed that the noise variance is equal to one.
    Find the beta values and the minimum weight
    Current version allows for standard deviation of 1 only.
    """
    pr = prior[0:1]
    nx = len(xd)
    wmin = wfromt(np.sqrt(2 * np.log(len(xd))), prior=prior, a=a)
    winit = 1
    if pr == "l":
        beta = beta_laplace(xd, a=a)
    if pr == "c":
        beta = beta_cauchy(xd)
    """
    now conduct iterated weighted least squares isotone regression
    """
    w = np.repeat(winit, len(beta))
    for j in range(maxits):
        aa = w + 1 / beta
        ps = w + aa
        ww = 1 / aa ** 2
        wnew = isotone(ps, ww, increasing=False)
        wnew = np.maximum(wmin, wnew)
        wnew = np.minimum(1, wnew)
        zinc = np.max(np.abs(np.diff(wnew)))
        w = wnew
        if zinc < tol:
            return w

    warning("More iterations required to achieve convergence")
    return w


def threshold(x, t, hard=True):
    """
    Threshold the data x using threshold t.
    If hard=True, use hard thresholding.
    If hard=False, use soft thresholding.
    """
    if hard:
        z = x * (np.abs(x) >= t)
    else:
        z = np.sign(x) * np.maximum(0, np.abs(x) - t)
    return z


def negloglik_laplace(xpar, xx, ss, tlo, thi):
    """
    Marginal negative log likelihood function for Laplace prior. 
    Constraints for thresholds need to be passed externally.

    Parameters:
    xpar : array-like, shape (2,)
        Vector of two parameters:
        xpar[0]: a value between [0, 1] which will be adjusted to range of w 
        xpar[1]: inverse scale (rate) parameter ("a")
    xx : array-like
        Data
    ss : array-like
        Vector of standard deviations
    tlo : array-like
        Lower bound of thresholds
    thi : array-like
        Upper bound of thresholds

    Returns:
    neg_loglik : float
        Negative log likelihood
    """
    a = xpar[1]
    
    # Calculate the range of w given a, using negative monotonicity
    # between w and t
    wlo = wfromt(thi, ss, a=a)
    whi = wfromt(tlo, ss, a=a)
    wlo = np.max(wlo)
    whi = np.min(whi)
    
    loglik = np.sum(np.log(1 + (xpar[0] * (whi - wlo) + wlo) *
                           beta_laplace(xx, ss, a)))
    
    return -loglik


def postmean_cauchy(x, w):
    """
    Find the posterior mean for the quasi-Cauchy prior with mixing
    weight w given data x, which may be a scalar or a vector.
    """
    muhat = x.copy()  # Ensure x is a numpy array
    ind = (x == 0)
    x = x[~ind]  # Remove zeros from x
    ex = np.exp(-x**2/2)
    z = w * (x - (2 * (1 - ex))/x)
    z = z / (w * (1 - ex) + (1 - w) * ex * x**2)
    muhat[~ind] = z
    return muhat


def wpost_laplace(w, x, s=1, a=0.5):
    # Calculate the posterior weight for non-zero effect
    laplace_beta = beta_laplace(x, s, a)
    return 1 - (1 - w) / (1 + w * laplace_beta)


def postmean_laplace(x, s=1, w=0.5, a=0.5):
    """
    Find the posterior mean for the double exponential prior for
    given x, s (sd), w, and a.

    Args:
        x (float or numpy array): Input data.
        s (float): Standard deviation.
        w (float): Parameter.
        a (float): Parameter.

    Returns:
        float or numpy array: Posterior mean.
    """
    # Only allow a < 20 for input value.
    a = min(a, 20)
    
    # First find the probability of being non-zero
    w_post = wpost_laplace(w, x, s, a)
    
    # Now find the posterior mean conditional on being non-zero
    sx = np.sign(x)
    x = np.abs(x)
    xpa = x/s + s*a
    xma = x/s - s*a
    xpa[xpa > 35] = 35
    xma[xma < -35] = -35
    
    cp1 = norm.cdf(-xpa)
    cp2 = norm.cdf(xma)
    ef = np.exp(np.minimum(2 * a * x, 100))
    postmean_cond = x - a * s**2 * (2 * cp1 / (cp1 + ef * cp2) - 1)
    
    # Calculate posterior mean and return
    return sx * w_post * postmean_cond

def postmean(x, s=1, w=0.5, prior="laplace", a=0.5):
    """
    Find the posterior mean for the appropriate prior for given x, s (sd), w, and a.
    """
    pr = prior[0:1]
    if pr == "l":
        mutilde = postmean_laplace(x, s, w, a=a)
    elif pr == "c":
        if np.any(s != 1):
            raise ValueError("Only standard deviation of 1 is allowed for Cauchy prior.")
        mutilde = postmean_cauchy(x, w)
    else:
        raise ValueError("Unknown prior type.")
    return mutilde


def wandafromx(x, s=1, universalthresh=True):
    """
    Find the marginal max lik estimators of w and a given standard  deviation s,
    using a bivariate optimization; If universalthresh=TRUE, the thresholds will 
    be upper bounded by universal threshold adjusted by standard deviation. 
    The threshold is constrained to lie between 0 and sqrt ( 2 log (n)) *   s. 
    Otherwise, threshold can take any nonnegative value;  If running R, 
    the routine optim is used; in S-PLUS the routine is nlminb.
    """
    # Range for thresholds
    if universalthresh:
        thi = np.sqrt(2 * np.log(len(x))) * s
    else:
        thi = np.inf

    if isinstance(s, int):
        tlo = np.zeros(len(str(s)))
    else:
        tlo = np.zeros(len(s))
    lo = np.array([0, 0.04])
    hi = np.array([1, 3])
    startpar = np.array([0.5, 0.5])

    if 'optim' in globals():
        result = minimize(negloglik_laplace, startpar, method='L-BFGS-B', bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, thi, tlo))
        uu = result.x
    else:
        result = minimize(negloglik_laplace, startpar, bounds=[(lo[0], hi[0]), (lo[1], hi[1])], args=(x, s, thi, tlo))
        uu = result.x

    a = uu[1]
    wlo = wfromt(thi, s, a=a)
    whi = wfromt(tlo, s, a=a)
    wlo = np.max(wlo)
    whi = np.min(whi)
    w = uu[0] * (whi - wlo) + wlo
    return {'w': w, 'a': a}

def threshld(x, t, hard=True):
    """
    Threshold the data x using threshold t.
    If hard=True, use hard thresholding.
    If hard=False, use soft thresholding.
    x - a data value or a vector of data
    t - value of threshold to be used
    hard - specifies whether hard or soft thresholding is applied
    """
    if hard:
        z = x * (abs(x) >= t)
    else:
        z = np.sign(x) * np.maximum(0, abs(x) - t)
    return z



def tfromx(x, s=1, prior="laplace", bayesfac=False, a=0.5, universalthresh=True):
    """
    Given the data x, the prior, and any other parameters, find the
    threshold corresponding to the marginal maximum likelihood
    estimator of the mixing weight.
    
    x - Vector of data.
    s - A single value or a vector of standard deviations if the Laplace prior is used. If
    a vector, must have the same length as x. Ignored if Cauchy prior is used.
    prior - Specification of prior to be used; can be "cauchy" or "laplace".
    bayesfac - Specifies whether Bayes factor threshold should be used instead of posterior
    median threshold.
    a - Scale factor if Laplace prior is used. Ignored if Cauchy prior is used.
    universalthresh - If universalthresh = TRUE, the thresholds will be upper bounded by universal
    threshold; otherwise, the thresholds can take any non-negative values.
    """
    pr = prior[0:1]
    if pr == "c":
        s = 1
    if pr == "l" and np.isnan(a):
        wa = wandafromx(x, s, universalthresh)
        w = wa['w']
        a = wa['a']
    else:
        w = wfromx(x, s, prior=prior, a=a)
    return tfromw(w, s, prior=prior, bayesfac=bayesfac, a=a)
