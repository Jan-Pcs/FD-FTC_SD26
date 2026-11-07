import numpy as np, json, os
os.makedirs('figures', exist_ok=True)
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from boost_linear_smo_sim import P, derived, THR, RHO_ES
d=derived(); z=d['z']; V=d['V']; I=d['I']; U=P['U']
BLUE='#2a78d6'; ORANGE='#eb6834'; INK='#0b0b0b'; MUTED='#8a8984'; GRID='#e4e3df'
plt.rcParams.update({'font.family':'serif','font.serif':['cmr10'],'mathtext.fontset':'cm',
    'axes.formatter.use_mathtext':True,'font.size':8.5,'axes.titlesize':8.5,'axes.labelsize':8.5,
    'legend.fontsize':7.5,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'axes.linewidth':0.6,
    'axes.edgecolor':'#52514e','xtick.color':'#52514e','ytick.color':'#52514e','axes.grid':True,
    'grid.color':GRID,'grid.linewidth':0.5,'grid.linestyle':'-','lines.linewidth':1.1,
    'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,'savefig.bbox':'tight','savefig.pad_inches':0.02})
R=np.load('results.npz'); S=lambda k,c: R[f'{k}__{c}']
def title(ax,s): ax.set_title(s,loc='left',pad=4)
TF=0.175
# ---------------- Fig: nominal run with dither
fig,ax=plt.subplots(1,2,figsize=(6.5,2.2))
t=S('nominal','t'); ax[0].plot(t,U+S('nominal','du'),color=BLUE); ax[0].set_ylim(0.45,0.55)
ax[0].set_xlabel('Time (s)'); ax[0].set_ylabel('Duty cycle $u$'); title(ax[0],'(a) Duty cycle with the test dither')
ax[1].plot(t,V+S('nominal','dv'),color=INK,lw=1.0,label='$v_C$ (plant)')
ax[1].plot(t,V+S('nominal','xv'),color=BLUE,lw=1.0,ls=(0,(4,2)),label=r'$\hat v_C$ (STSMO)')
ax[1].set_xlabel('Time (s)'); ax[1].set_ylabel('Voltage (V)'); title(ax[1],'(b) Output voltage and its estimate'); ax[1].legend(loc='lower center',ncol=2)
ax[1].set_ylim(19.0,28.5)
fig.tight_layout(w_pad=2.0); fig.savefig('figures/fig_nominal.pdf'); plt.close(fig)
# ---------------- Fig: convergence ES vs STSMO
fig,ax=plt.subplots(2,2,figsize=(6.5,4.3))
t=S('es','t'); e1=np.abs(S('es','e1_es')); ey=np.abs(S('es','ey_es')); tm=1e3*t
m=tm<=8
ax[0,0].semilogy(tm[m],e1[m],color=ORANGE,label='simulation')
ax[0,0].semilogy(tm[m],1e-3*np.exp(z*t[m]),color=INK,lw=0.9,ls=(0,(4,2)),label=r'$e_1(0)\,e^{zt}$')
tl=json.load(open('metrics.json'))['es_slide_lost_ms']
for a in (ax[0,0],ax[0,1]): a.axvline(tl,color=MUTED,lw=0.7)
ax[0,0].text(tl+0.1,2e-3,'sliding lost',color='#52514e',fontsize=7)
ax[0,0].set_xlabel('Time (ms)'); ax[0,0].set_ylabel(r'$|e_1|$ (A)'); title(ax[0,0],'(a) First-order SMO: hidden error'); ax[0,0].legend(loc='upper left')
ax[0,1].semilogy(tm[m],np.maximum(ey[m],1e-12),color=ORANGE)
ax[0,1].set_xlabel('Time (ms)'); ax[0,1].set_ylabel(r'$|y-\hat y|$ (V)'); title(ax[0,1],'(b) First-order SMO: output error'); ax[0,1].set_ylim(1e-7,1e2)
t=S('nominal','t'); tm=1e3*t; m=tm<=3
ax[1,0].semilogy(tm[m],np.maximum(np.abs(S('nominal','e')[m]),1e-12),color=BLUE)
ax[1,0].set_xlabel('Time (ms)'); ax[1,0].set_ylabel(r'$|y-\hat y|$ (V)'); title(ax[1,0],'(c) STSMO: output error'); ax[1,0].set_ylim(1e-10,3)
ax[1,1].semilogy(tm[m],np.maximum(np.abs(S('nominal','di')[m]-S('nominal','xi')[m]),1e-9),color=BLUE)
ax[1,1].set_xlabel('Time (ms)'); ax[1,1].set_ylabel(r'$|i_L-\hat i_L|$ (A)'); title(ax[1,1],'(d) STSMO: current estimation error'); ax[1,1].set_ylim(1e-5,10)
fig.tight_layout(h_pad=1.2,w_pad=2.0); fig.savefig('figures/fig_convergence.pdf'); plt.close(fig)
# ---------------- Fig: additive faults
def resid(ax,t,r,t0,t1,lbl):
    m=(t>=t0)&(t<=t1); ax.plot(t[m],r[m]/1e5,color=BLUE,lw=0.9)
    for s_ in (1,-1): ax.axhline(s_*THR/1e5,color=MUTED,lw=0.6)
    ax.set_xlabel('Time (s)'); ax.set_ylabel(r'$r$ ($10^5$ V/s$^2$)'); title(ax,lbl)
def est(ax,t,xh,x,t0,t1,lbl,ylab,zoom,ylim=None,zylim=None,bbox=(0.47,0.10,0.5,0.40)):
    m=(t>=t0)&(t<=t1)
    ax.plot(t[m],x[m],color=INK,lw=0.9,ls=(0,(4,2)),label='true'); ax.plot(t[m],xh[m],color=BLUE,lw=1.0,label='estimate')
    ax.set_xlabel('Time (s)'); ax.set_ylabel(ylab); title(ax,lbl); ax.legend(loc='upper left')
    if ylim: ax.set_ylim(*ylim)
    if zoom:
        ins=inset_axes(ax,width='100%',height='100%',bbox_to_anchor=bbox,bbox_transform=ax.transAxes,loc='lower left',borderpad=0)
        mz=(t>=zoom[0])&(t<=zoom[1]); ins.plot(1e3*(t[mz]-TF),x[mz],color=INK,lw=0.8,ls=(0,(4,2))); ins.plot(1e3*(t[mz]-TF),xh[mz],color=BLUE,lw=0.9)
        ins.set_title('first %d ms after onset'%round(1e3*(zoom[1]-TF)),fontsize=6.5,pad=2); ins.tick_params(labelsize=6,pad=1)
        if zylim: ins.set_ylim(*zylim)
fig,ax=plt.subplots(2,2,figsize=(6.5,4.5))
t=S('actuator','t')
resid(ax[0,0],t,S('actuator','r'),0.15,0.6,'(a) Actuator fault: residual')
ax[0,0].set_ylim(-3,9)
est(ax[0,1],t,S('actuator','fa_hat'),S('actuator','fa'),0.15,0.6,r'(b) Actuator fault: $\hat f_a$','Duty-cycle shift',(TF-0.0005,TF+0.004),ylim=(-0.06,0.075),zylim=(-0.05,0.04))
t=S('sensor','t')
resid(ax[1,0],t,S('sensor','r'),0.15,0.4,'(c) Sensor fault: residual'); ax[1,0].set_ylim(-3,5)
est(ax[1,1],t,S('sensor','fs_hat'),S('sensor','fs'),0.15,0.4,r'(d) Sensor fault: $\hat f_s$','Sensor offset (V)',(TF-0.0005,TF+0.005),ylim=(-1.0,1.2),zylim=(-3,6.5),bbox=(0.47,0.08,0.5,0.40))
fig.tight_layout(h_pad=1.2,w_pad=2.0); fig.savefig('figures/fig_additive.pdf'); plt.close(fig)
# ---------------- Fig: multiplicative faults
fig,ax=plt.subplots(2,2,figsize=(6.5,4.5))
t=S('capacitor','t')
resid(ax[0,0],t,S('capacitor','r'),0.15,0.7,'(a) Capacitor fault: residual')
th=S('capacitor','thC_hat'); m=(t>=0.15)&(t<=0.7)
ax[0,1].plot(t[m],np.where(t[m]>=TF,0.3,0.0),color=INK,lw=0.9,ls=(0,(4,2)),label='true')
ax[0,1].plot(t[m],th[m],color=BLUE,lw=1.0,label='estimate'); ax[0,1].set_ylim(-0.05,0.4)
ax[0,1].set_xlabel('Time (s)'); ax[0,1].set_ylabel(r'$\hat\theta_C=1-\hat C_f/C$'); title(ax[0,1],r'(b) Capacitor fault: $\hat\theta_C$'); ax[0,1].legend(loc='lower right')
t=S('load','t'); r=S('load','r'); m=(t>=TF-0.0003)&(t<=TF+0.004)
ax[1,0].plot(1e3*(t[m]-TF),r[m]/1e5,color=BLUE,lw=0.9)
for s_ in (1,-1): ax[1,0].axhline(s_*THR/1e5,color=MUTED,lw=0.6)
ax[1,0].set_xlabel('Time after onset (ms)'); ax[1,0].set_ylabel(r'$r$ ($10^5$ V/s$^2$)'); title(ax[1,0],'(c) Load fault: residual at the onset')
fR=S('load','fR_hat'); m=(t>=0.15)&(t<=0.7)
ax[1,1].plot(t[m],np.where(t[m]>=TF,0.5,0.0),color=INK,lw=0.9,ls=(0,(4,2)),label='true')
ax[1,1].plot(t[m],np.nan_to_num(fR[m],nan=0.0),color=BLUE,lw=1.0,label='estimate'); ax[1,1].set_ylim(-0.05,0.65)
ax[1,1].set_xlabel('Time (s)'); ax[1,1].set_ylabel(r'$\hat f_R$'); title(ax[1,1],r'(d) Load fault: $\hat f_R$'); ax[1,1].legend(loc='lower right')
fig.tight_layout(h_pad=1.2,w_pad=2.0); fig.savefig('figures/fig_multiplicative.pdf'); plt.close(fig)
print('ok')
