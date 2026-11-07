import numpy as np, json
from boost_linear_smo_sim import *
from estimators import *
TF=0.175
sc={}
sc['nominal']=simulate(0.3)
sc['es']=simulate(0.012, es_e10=1e-3, decim=10)
sc['actuator']=simulate(0.6, fa=dict(kind=1,t0=TF,m=0.03,m2=0.06,r0=0.375,r1=0.525))
sc['sensor']=simulate(0.4, fs=dict(t0=TF,m=0.5))
sc['capacitor']=simulate(0.7, fC=dict(t0=TF,m=1/0.7-1))
sc['load']=simulate(0.7, fR=dict(t0=TF,m=0.5))
for k in ['actuator','sensor','capacitor','load','nominal']:
    s=sc[k]; s['fa_hat']=fa_hat(s); s['fs_hat']=fs_hat(s); s['fR_hat']=fR_hat(s); s['thC_hat']=thetaC_hat(s)
np.savez_compressed('results.npz', **{f'{k}__{c}':v for k,s in sc.items() for c,v in s.items()})
M={}
def settle(t,x,target,tol,tstart,tend):
    m=(t>=tstart)&(t<=tend); tt=t[m]; xx=x[m]
    bad=np.where(~(np.abs(xx-target)<=tol*abs(target)))[0]
    if not len(bad): return 0.0
    return (tt[bad[-1]+1]-tstart) if bad[-1]+1<len(tt) else np.nan
s=sc['nominal']; t=s['t']; e=s['e']; ei=s['di']-s['xi']
M['nom_ey_1mV_ms']=1e3*t[np.where(np.abs(e)>=1e-3)[0][-1]+1]
M['nom_ey_1uV_ms']=1e3*t[np.where(np.abs(e)>=1e-6)[0][-1]+1]
M['nom_ei_1mA_ms']=1e3*t[np.where(np.abs(ei)>=1e-3)[0][-1]+1]
m=t>0.05
M['nom_max_ey_after50']=float(np.max(np.abs(e[m]))); M['nom_max_ei_after50']=float(np.max(np.abs(ei[m]))); M['nom_max_r_after50']=float(np.max(np.abs(s['r'][m])))
M['nom_vC_range']=[float(d['V']+s['dv'].min()),float(d['V']+s['dv'].max())]; M['nom_iL_range']=[float(d['I']+s['di'].min()),float(d['I']+s['di'].max())]
M['nom_detected']=bool(s['det_t'][-1]>=0)
M['nom_vC_dip']=float(-np.min(s['dv'][(t>0.1)&(t<0.103)]-s['dv'][np.searchsorted(t,0.1)-1]))
s=sc['es']; t=s['t']; e1=s['e1_es']; ey=s['ey_es']
k=np.searchsorted(t,4e-3); M['es_ratio_4ms']=float(e1[k]/(e1[0]*np.exp(z*t[k])))
idx=np.where(np.abs(ey)>1e-3)[0]; idx=idx[t[idx]>1e-3]; M['es_slide_lost_ms']=float(1e3*t[idx[0]])
M['es_e1_at_loss']=float(e1[idx[0]]); M['es_e1_theory_lost_ms']=float(1e3*np.log(RHO_ES*abs(d['bC'])/d['a21']/1e-3)/z)
M['es_ey_10ms']=float(ey[np.searchsorted(t,10e-3)])
s=sc['actuator']; t=s['t']; fah=s['fa_hat']
M['act_det_us']=1e6*(s['det_t'][-1]-TF)
w=(t>TF)&(t<TF+0.01); M['act_min_spike']=float(fah[w].min()); M['act_spike_dur_ms']=float(1e3*(t[w][np.where(fah[w]<0)[0][-1]]-TF))
M['act_settle2pct_ms']=1e3*settle(t,fah,0.03,0.02,TF,0.37)
m=(t>0.25)&(t<0.375); M['act_mean_1']=float(fah[m].mean()); M['act_bias_iL']=float(np.mean(s['xi'][m]-s['di'][m])); M['act_bias_theory']=float(d['I']*0.03/(1-P['U']))
M['act_corr_err_mA']=float(1e3*np.mean(np.abs(s['xi'][m]+d['I']*fah[m]/(1-P['U'])-s['di'][m])))
m=(t>0.55); M['act_mean_2']=float(fah[m].mean())
m=(t>0.38)&(t<0.52); M['act_drift_maxerr']=float(np.max(np.abs(fah[m]-s['fa'][m]))); M['act_drift_theory']=float(0.2*(1/z+TAU_F))
s=sc['sensor']; t=s['t']; fsh=s['fs_hat']
M['sen_det_us']=1e6*(s['det_t'][-1]-TF); M['sen_ey_jump']=float(np.max(np.abs(s['e'][(t>TF)&(t<TF+1e-3)])))
idx=np.where((t>TF)&(np.abs(s['e'])>=1e-3))[0]; M['sen_reconv_ms']=float(1e3*(t[idx[-1]+1]-TF))
M['sen_settle2pct_ms']=1e3*settle(t,fsh,0.5,0.02,TF,0.4)
m=t>0.25; M['sen_mean']=float(fsh[m].mean()); M['sen_as_fa']=float(s['fa_hat'][m].mean()); M['sen_as_fa_theory']=0.5/(R*d['I'])
w=(t>TF)&(t<TF+0.01); M['sen_ext']=[float(fsh[w].min()),float(fsh[w].max())]
s=sc['capacitor']; t=s['t']; th=s['thC_hat']
M['cap_det_us']=1e6*(s['det_t'][-1]-TF); M['cap_theta_end']=float(th[-1]); M['cap_C_hat_uF']=float(1e6*(1-th[-1])*C)
M['cap_settle2pct_ms']=1e3*settle(t,th,0.3,0.02,s['det_t'][-1]+1e-4,0.7)
M['cap_theta_at_250ms']=float(th[np.searchsorted(t,0.25)])
M['cap_r_peak']=float(np.max(np.abs(s['r'][t>TF]))); M['cap_r_pre_end']=float(np.max(np.abs(s['r'][(t>0.24)&(t<0.25)])))
M['cap_fahat_peak']=float(np.max(np.abs(s['fa_hat'][t>TF])))
M['cap_r_at_onset']=float(np.max(np.abs(s['r'][(t>TF)&(t<0.19)])))
s=sc['load']; t=s['t']; fRh=s['fR_hat']
M['load_det_us']=1e6*(s['det_t'][-1]-TF); M['load_fR_end']=float(fRh[-1]); M['load_R_hat']=float(R/(1-fRh[-1]))
M['load_settle2pct_ms']=1e3*settle(t,fRh,0.5,0.02,TF,0.7)
M['load_r_peak']=float(np.max(np.abs(s['r'][t>TF]))); M['load_r_after20ms']=float(np.max(np.abs(s['r'][t>TF+0.02])))
M['load_eiss']=float(np.mean((s['xi']-s['di'])[t>0.6])); M['load_eiss_theory']=float(d['I']*0.5)
json.dump(M,open('metrics.json','w'),indent=1)
for k,v in M.items(): print(f'{k:24s} {v}')
