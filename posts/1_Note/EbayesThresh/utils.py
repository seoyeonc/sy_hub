import numpy as np
from scipy.stats import norm, cauchy, laplace
from scipy.optimize import minimize_scalar


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
    for i in range(len(xpa)):
        if xpa[i] < 35:
            rat1[i] = norm.cdf(-xpa[i]) / norm.pdf(xpa[i])
    
    rat2 = 1/np.abs(xma)
    for i in range(len(xma)):
        if xma[i] > 35:
            xma[i] = 35
    for i in range(len(xma)):
        if xma[i] > -35:
            rat2[i] = norm.cdf(xma[i])/norm.pdf(xma[i])   
    
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
                           laplace.pdf(xx, scale=ss/a)))
    
    return -loglik