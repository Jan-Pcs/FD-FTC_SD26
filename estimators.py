"""Post-processing estimators (single-fault hypotheses), all computable on line."""
import numpy as np
from boost_linear_smo_sim import P, derived, TAU_F
d=derived(); C=P['C']; R=P['R']; z=d['z']; I=d['I']; V=d['V']
E_SLIDE=1e-6     # sliding gate on |e_y| (V)
T_BUF=1e-3       # the integral of the switching term is referenced 1 ms before detection
def fa_hat(s):  return C*s['r']/(z*I)
def fs_hat(s):  return R*C*s['r']/z
def fR_hat(s):
    t=s['t']; td=s['det_t'][-1]; out=np.full_like(t,np.nan)
    if td<0: return out
    k0=np.searchsorted(t,td-T_BUF); m=t>=td
    out[m]=(R*C/V)*(s['eta'][m]-s['eta'][k0]); return out
def thetaC_hat(s):
    """LS of the integrated switching term on the observer's output derivative,
    eta = theta*yhat_dot + b, using only sliding samples after detection."""
    t=s['t']; td=s['det_t'][-1]; th=np.full_like(t,np.nan)
    if td<0: return th
    m=(t>=td)&(np.abs(s['e'])<E_SLIDE)
    x=s['yhd']; yv=s['eta']
    Sxx=np.cumsum(np.where(m,x*x,0)); Sx=np.cumsum(np.where(m,x,0)); S1=np.cumsum(m.astype(float))
    Sxy=np.cumsum(np.where(m,x*yv,0)); Sy=np.cumsum(np.where(m,yv,0))
    den=S1*Sxx-Sx**2; ok=(den>1e-9)&(t>=td)
    th[ok]=(S1[ok]*Sxy[ok]-Sx[ok]*Sy[ok])/den[ok]; return th
