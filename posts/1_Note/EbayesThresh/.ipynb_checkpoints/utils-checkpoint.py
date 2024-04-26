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



# def threshold(x, t, hard=True):
#     """
#     Threshold the data x using threshold t.
    
#     Parameters:
#         x (array-like): Input data.
#         t (float): Threshold value.
#         hard (bool, optional): If True, use hard thresholding. If False, use soft thresholding.
    
#     Returns:
#         array-like: Thresholded data.
#     """
#     if hard:
#         z = x * (np.abs(x) >= t)
#     else:
#         z = np.sign(x) * np.maximum(0, np.abs(x) - t)
#     return z

# def isotone(x, wt=None, increasing=False):
#     if wt is None:
#         wt = [1] * len(x)
        
#     nn = len(x)
#     if nn == 1:
#         return x
    
#     if not increasing:
#         x = [-val for val in x]
    
#     ip = list(range(1, nn+1))
#     dx = [x[i] - x[i-1] for i in range(1, nn)]
#     nx = len(x)
    
#     while nx > 1 and min(dx) < 0:
#         jmax = [i+1 for i in range(nx-1) if dx[i] <= 0] + [nx]
#         jmin = [i+1 for i in range(nx-1) if dx[i] > 0] + [nx]
        
#         for jb in range(len(jmax)):
#             ind = list(range(jmax[jb]-1, jmin[jb]))
#             wtn = sum(wt[i] for i in ind)
#             x[jmax[jb]-1] = sum(wt[i] * x[i] for i in ind) / wtn
#             wt[jmax[jb]-1] = wtn
#             x[jmax[jb]:jmin[jb]] = [None] * (jmin[jb] - jmax[jb])
        
#         x = [val for val in x if val is not None]
#         wt = [val for val in wt if val is not None]
#         ip = [ip[i] for i in range(len(x))]
#         dx = [x[i] - x[i-1] for i in range(1, len(x))]
#         nx = len(x)
    
#     jj = [0] * nn
#     for i in range(nn):
#         jj[ip[i]-1] = 1
#     z = [x[sum(jj[:i])] for i in range(1, nn+1)]
    
#     if not increasing:
#         z = [-val for val in z]
    
#     return z
