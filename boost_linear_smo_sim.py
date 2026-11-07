"""Linear faulty boost model (Section I, eq. L_ss_faulty_model) observed with
(i) the first-order SMO of Edwards, Spurgeon & Patton (2000) in canonical form and
(ii) the proposed super-twisting SMO (STSMO). Deviation variables throughout."""
import numpy as np
from numba import njit

P = dict(Vin=12.0, L=2e-3, C=470e-6, R=10.0, U=0.5)
DU = 0.03          # dither amplitude (duty)
FD = 10.0          # dither frequency (Hz)
FPLUS = 3e6        # STA perturbation bound (V/s^2)
TAU_F = 0.5e-3     # residual low-pass filter (s)
THR = 2e4          # detection threshold on the residual (V/s^2)
T_ARM = 0.05       # detection armed after initial convergence (s)
RHO_ES = 0.05      # ES switching gain (duty units, multiplies |D2|)
A22S_ES = -2000.0  # ES stable design matrix A22^s (1/s)

def derived(p=P):
    Vin,L,C,R,U = p['Vin'],p['L'],p['C'],p['R'],p['U']
    V = Vin/(1-U); I = Vin/(R*(1-U)**2); z = R*(1-U)**2/L
    return dict(V=V, I=I, z=z, a12=(1-U)/L, a21=(1-U)/C, a22=1/(R*C), bL=V/L, bC=I/C,
                eR=V/(R*C), k0=V*C/(L*I), detA=(1-U)**2/(L*C))

@njit(cache=True)
def _fault(t, kind, t0, mag, mag2, tr0, tr1):
    if t < t0:
        return 0.0
    if kind == 1:          # step + linear drift from mag to mag2 between tr0 and tr1
        if t < tr0: return mag
        if t < tr1: return mag + (mag2-mag)*(t-tr0)/(tr1-tr0)
        return mag2
    return mag

@njit(cache=True)
def run(T, dt, decim, a12, a21, a22, bL, bC, eR, z, k0, D2, A12, A22,
        DU, FD, k1, k2, tauf, thr, tarm, rho_es, a22s, es_ei0,
        fa_kind, fa_t0, fa_m, fa_m2, fa_r0, fa_r1, fs_t0, fs_m, fC_t0, fC_m, fR_t0, fR_m,
        xh_i0, xh_v0):
    n = int(T/dt); m = n//decim
    out = np.zeros((m, 20))
    di = 0.0; dv = 0.0                      # plant deviations
    xi = xh_i0; xv = xh_v0                  # STSMO estimates (delta i_L, delta v_C)
    r = 0.0; ydf = 0.0; eta = 0.0           # residual LPF, LPF of yhat_dot, integral of switching term
    det_t = -1.0; eta_d = 0.0; Srp = 0.0; Spp = 0.0
    # ES observer in canonical coordinates x1 = di + k0*dv, x2 = dv
    x1h = (0.0 + es_ei0); x2h = 0.0          # es_ei0: initial hidden-state error (A) with opposite sign convention below
    k = 0
    for j in range(n):
        t = j*dt
        sq = 1.0 if (t*FD - np.floor(t*FD)) < 0.5 else -1.0
        du = DU*sq
        fa = _fault(t, fa_kind, fa_t0, fa_m, fa_m2, fa_r0, fa_r1)
        fs = _fault(t, 0, fs_t0, fs_m, 0.0, 0.0, 0.0)
        fC = _fault(t, 0, fC_t0, fC_m, 0.0, 0.0, 0.0)
        fR = _fault(t, 0, fR_t0, fR_m, 0.0, 0.0, 0.0)
        y = dv + fs
        # ---- STSMO
        e = y - xv
        sg = np.sign(e)
        nu = k2*sg
        yhd = a21*xi - a22*y - bC*du             # model output derivative (exact in sliding)
        dxv = yhd + k1*np.sqrt(abs(e))*sg
        dxi = -a12*y + bL*du + nu/a21
        # ---- ES first-order SMO (paper eqs. (8)-(11), output error ey = yhat - y)
        eyes = x2h - y
        nues = -rho_es*abs(D2)*np.sign(eyes)
        dx1h = z*x1h + A12*x2h - A12*eyes                       # B1bar = 0
        dx2h = a21*x1h + A22*x2h + D2*du - (A22 - a22s)*eyes + nues
        # ---- plant (linear faulty model)
        kap = a21*di - a22*dv - bC*du
        ddi = -a12*dv + bL*(du + fa)
        ddv = (1.0 + fC)*kap - bC*fa + eR*fR
        # ---- residual processing
        drr = (nu - r)/tauf
        dyd = (yhd - ydf)/tauf
        phi = (yhd - ydf)/tauf                   # = d/dt LPF(yhat_dot) = LPF(y_ddot)
        if det_t < 0.0 and t > tarm and abs(r) > thr:
            det_t = t; eta_d = eta
        if det_t >= 0.0:
            Srp += r*phi*dt; Spp += phi*phi*dt
        # ---- log
        if j % decim == 0 and k < m:
            x1 = di + k0*dv
            th = Srp/Spp if Spp > 1e-9 else 0.0
            out[k,0]=t; out[k,1]=du; out[k,2]=di; out[k,3]=dv; out[k,4]=y
            out[k,5]=xi; out[k,6]=xv; out[k,7]=e; out[k,8]=r; out[k,9]=eta
            out[k,10]=th; out[k,11]=fa; out[k,12]=fs; out[k,13]=fC; out[k,14]=fR
            out[k,15]=x1h - x1; out[k,16]=eyes; out[k,17]=yhd; out[k,18]=phi; out[k,19]=det_t
            k += 1
        # ---- Euler step
        di += dt*ddi; dv += dt*ddv
        xi += dt*dxi; xv += dt*dxv
        if abs(x1h) < 1e12:                      # freeze the (diverging) ES observer once it has blown up
            x1h += dt*dx1h; x2h += dt*dx2h
        r += dt*drr; ydf += dt*dyd; eta += dt*nu
    return out

COLS = ['t','du','di','dv','y','xi','xv','e','r','eta','theta','fa','fs','fC','fR','e1_es','ey_es','yhd','phi','det_t']

def simulate(T, fa=None, fs=None, fC=None, fR=None, dt=1e-7, decim=100,
             xh_i0=None, xh_v0=-1.0, es_e10=0.0, es_on=True, p=P):
    d = derived(p)
    k1 = 1.5*np.sqrt(FPLUS); k2 = 1.1*FPLUS
    if xh_i0 is None: xh_i0 = -0.5*d['I']
    fa = fa or dict(kind=0, t0=1e9, m=0.0, m2=0.0, r0=0.0, r1=1.0)
    fs = fs or dict(t0=1e9, m=0.0); fC = fC or dict(t0=1e9, m=0.0); fR = fR or dict(t0=1e9, m=0.0)
    A12 = -(d['k0']**2*d['a21'] + d['k0']*d['a22'] + d['a12'])
    A22 = -(d['k0']*d['a21'] + d['a22'])
    D2 = -d['bC']
    rho = RHO_ES if es_on else 0.0
    o = run(T, dt, decim, d['a12'], d['a21'], d['a22'], d['bL'], d['bC'], d['eR'], d['z'], d['k0'],
            D2, A12, A22, DU, FD, k1, k2, TAU_F, THR, T_ARM, rho, A22S_ES, es_e10,
            fa['kind'], fa['t0'], fa['m'], fa['m2'], fa['r0'], fa['r1'],
            fs['t0'], fs['m'], fC['t0'], fC['m'], fR['t0'], fR['m'], xh_i0, xh_v0)
    return {c: o[:, i] for i, c in enumerate(COLS)}
