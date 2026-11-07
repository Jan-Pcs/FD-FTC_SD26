import numpy as np
import sympy as sp
s,t=sp.symbols('s t')
Vin,L,C,R,U=sp.symbols('V_in L C R U',positive=True)
V=Vin/(1-U); I=Vin/(R*(1-U)**2); z=R*(1-U)**2/L
A=sp.Matrix([[0,-(1-U)/L],[(1-U)/C,-1/(R*C)]]); B=sp.Matrix([V/L,-I/C]); Bf=B
ER=sp.Matrix([0,V/(R*C)]); Cm=sp.Matrix([[0,1]])
Delta=s**2+s/(R*C)+(1-U)**2/(L*C)
M=(s*sp.eye(2)-A).inv()
Gfa=sp.simplify((Cm*M*Bf)[0]); GfR=sp.simplify((Cm*M*ER)[0]); Gfs=1
dA=sp.Matrix([[0,0],[(1-U)/C,-1/(R*C)]]); dB=sp.Matrix([0,-I/C])
GmC=sp.simplify((Cm*M*(dA*M*B+dB))[0]); Gu=sp.simplify((Cm*M*B)[0])
ok=lambda e: sp.simplify(e)==0
# ---------- 1. ES canonical form
k0=V*C/(L*I)
T=sp.Matrix([[1,k0],[0,1]]); Ti=T.inv()
Ab=sp.simplify(T*A*Ti); Db=sp.simplify(T*Bf); Cb=sp.simplify(Cm*Ti)
print('k0 = VC/(LI) =',sp.simplify(k0),' ; L*I*k0 = C*V:',ok(L*I*k0-C*V))
print('Abar =',Ab.applyfunc(sp.factor)); print('Dbar =',Db.T,' Cbar =',Cb)
print('A11 == z:',ok(Ab[0,0]-z),'  A21 == (1-U)/C:',ok(Ab[1,0]-(1-U)/C))
print('A12 =',sp.factor(Ab[0,1]),'   A22 =',sp.factor(Ab[1,1]),'  A22 == -(z+1/(RC)):',ok(Ab[1,1]+z+1/(R*C)))
print('Bbar = T B =',sp.simplify(T*B).T)
# invariant zero via Rosenbrock
Ros=sp.Matrix([[s-A[0,0],-A[0,1],-Bf[0]],[-A[1,0],s-A[1,1],-Bf[1]],[0,1,0]])
print('Rosenbrock det zeros:',sp.solve(sp.simplify(Ros.det()),s))
# energy: dE/dt linearized with dv=0 -> rate
print('Vin/(L I) == z:',ok(Vin/(L*I)-z))
# ---------- 2. STSMO error dynamics (time domain, symbolic functions)
fa,fs,fR,fC,du=[sp.Function(n)(t) for n in ['f_a','f_s','f_R','f_C','du']]
di,dv,xi1,xv=[sp.Function(n)(t) for n in ['di','dv','xh_i','xh_v']]
k1,k2,sg,ab=sp.symbols('k1 k2 sg absr')   # sg = sign(e), ab = |e|^(1/2)
a12=(1-U)/L; a21=(1-U)/C; a22=1/(R*C); bL=V/L; bC=I/C; eR=V/(R*C)
# plant (linear faulty model, eq. L_faulty_model)
d_di=-a12*dv+bL*(du+fa)
kap=a21*di-a22*dv-bC*du
d_dv=(1+fC)*kap-bC*fa+eR*fR
y=dv+fs
# STSMO
d_xv=a21*xi1-a22*y-bC*du+k1*ab*sg
d_xi=-a12*y+bL*du+(1/a21)*k2*sg
ey=y-xv; ei=di-xi1
d_ey=sp.diff(fs,t)+d_dv-d_xv
d_ei=d_di-d_xi
phi_v=-bC*fa+eR*fR+fC*kap+a22*fs+sp.diff(fs,t)
phi_i=bL*fa+a12*fs
print('\n ey_dot == a21*ei + phi_v - k1|e|^1/2 sg :',ok(sp.expand(d_ey-(a21*ei+phi_v-k1*ab*sg))))
print(' ei_dot == phi_i - k2 sg / a21            :',ok(sp.expand(d_ei-(phi_i-k2*sg/a21))))
xi=a21*ei+phi_v
# xi_dot computed with ei_dot from the error equation
xi_dot=a21*(phi_i-k2*sg/a21)+sp.diff(phi_v,t)
Psi=sp.expand(xi_dot+k2*sg)
print(' Psi =',sp.simplify(Psi))
# identities
print(' a21*bL == z*I/C:',ok(a21*bL-z*I/C),'   a21*a12 == z/(RC) == det A:',ok(a21*a12-z/(R*C)),ok(A.det()-z/(R*C)))
# ---------- 3. Laplace: Psi(s) per additive fault vs Delta(s)*G_f(s)
F=sp.symbols('F')
Psi_fa=(I/C)*(z*F-s*F); Psi_fs=(s**2+s/(R*C)+z/(R*C))*F; Psi_fR=eR*s*F
print('\n Delta*G_fa == -(I/C)(s-z):',ok(Delta*Gfa+(I/C)*(s-z)),'   Psi_fa matches:',ok(Psi_fa-Delta*Gfa*F))
print(' Delta*G_fs == Delta       :',ok(Psi_fs-Delta*Gfs*F))
print(' Delta*G_fR == (V/(RC)) s  :',ok(Psi_fR-Delta*GfR*F))
print(' Delta*G_mC == s^2 G_u     :',ok(Delta*GmC-s**2*Gu))
# time-domain Psi terms check against operator forms
Psi_t=sp.simplify(Psi)
chk=Psi_t-( bC*(z*fa-sp.diff(fa,t)) + eR*sp.diff(fR,t) + (sp.diff(fs,t,2)+a22*sp.diff(fs,t)+(z/(R*C))*fs) + sp.diff(fC*kap,t) )
print(' Psi == (I/C)(z fa - fa\') + (V/RC) fR\' + (fs\'\'+fs\'/(RC)+z fs/(RC)) + d/dt(fC*kappa):',ok(sp.expand(chk)))
# ---------- 4. constant-fault static gains
print('\n Delta(0)G_fa(0) == z I/C:',ok((Delta*Gfa).subs(s,0)-z*I/C),'  Delta(0) == z/(RC):',ok(Delta.subs(s,0)-z/(R*C)))
print(' => fa_hat = C nu/(z I), fs_hat = R C nu / z ; fs read as fa ->',sp.simplify((z/(R*C))/(z*I/C)),' * fs  (= 1/(R I))')
# f_C: dv_dot = (1+fC) kappa (only fC)  -> Psi = fC kappa_dot = fC/(1+fC) * ydd
fCc=sp.symbols('f_Cc'); th=fCc/(1+fCc)
print(' theta_C = fC/(1+fC) == 1 - C_f/C with C_f = C/(1+fC):',ok(th-(1-1/(1+fCc))))
# current-estimate bias under fa: ei = -phi_v/a21
print(' ei (fa only) = -phi_v/a21 =',sp.simplify((bC*sp.Symbol('fa'))/a21),' ; == I fa/(1-U):',ok((bC/a21)-I/(1-U)))
num={Vin:12,L:2e-3,C:470e-6,R:10,U:0.5}
for nm,ex in [('Vbar',V),('Ibar',I),('z',z),('k0',k0),('A12',Ab[0,1]),('A21',Ab[1,0]),('A22',Ab[1,1]),('D2',Db[1]),('detA',A.det()),('zI/C',z*I/C),('V/(RC)',eR),('I/(1-U)',I/(1-U)),('2/RC',2/(R*C))]:
    print(f'   {nm:8s} = {float(ex.subs(num)):.6g}')
# ---------- 5. checks used in the report text (Sections III-IV)
print('\n--- report checks')
# energy: L*I*x1_dot = Vin*di - 2V/R*dv, independent of du (fault-free linear model)
dI,dV,dU=sp.symbols('dI dV dU')
x1dot=(-a12*dV+bL*dU)+k0*(a21*dI-a22*dV-bC*dU)
print(' L I x1dot == Vin di - 2V/R dv:',ok(sp.expand(L*I*x1dot-(Vin*dI-2*V/R*dV))))
print(' A12 == -(1-U)/L - k0 (z+1/(RC)):',ok(Ab[0,1]+(1-U)/L+k0*(z+1/(R*C))))
# nu_eq time-domain form: Delta(s) dY + (I/C)(s-z) dU == Delta * fault part  (since Delta*G_u = -(I/C)(s-z))
print(' Delta*G_u == -(I/C)(s-z):',ok(Delta*Gu+(I/C)*(s-z)))
nv={k:v for k,v in num.items()}
zz=float(z.subs(nv)); bc=float((I/C).subs(nv)); a21n=float(a21.subs(nv)); eRn=float(eR.subs(nv))
wn=float(sp.sqrt((1-U)**2/(L*C)).subs(nv)); zeta=float((1/(R*C)/(2*sp.sqrt((1-U)**2/(L*C)))).subs(nv))
print(' poles:',[complex(p) for p in sp.Poly(Delta.subs(nv),s).nroots()],' wn=%.1f zeta=%.3f  4/(zeta wn)=%.1f ms'%(wn,zeta,4e3/(zeta*wn)))
rho=0.05; print(' ES sliding loss: e1=%.3f A at t=%.3f ms'%(rho*bc/a21n, 1e3*np.log(rho*bc/a21n/1e-3)/zz) if False else '')
import math
print(' ES sliding loss: e1 = rho|D2|/A21 = %.3f A at t = ln(.../1mA)/z = %.3f ms'%(rho*bc/a21n, 1e3*math.log(rho*bc/a21n/1e-3)/zz))
tauf=0.5e-3; k2n=1.1*3e6; dt=1e-7
print(' wrong-way peak -fa/(z tau_f) = %.4f'%(-0.03/(zz*tauf)),'; drift lag fa_dot(1/z+tau_f) = %.2e'%(0.2*(1/zz+tauf)))
print(' 2%% band for a step (filtered): t = tau_f ln((1+1/(z tau_f))/0.02) = %.2f ms'%(1e3*tauf*math.log((1+1/(zz*tauf))/0.02)))
print(' floors: k2 dt^2 = %.2e V ; C k2 dt/(1-U) = %.2e A ; k2 dt/tau_f = %.0f V/s^2'%(k2n*dt**2, 470e-6*k2n*dt/0.5, k2n*dt/tauf))
print(' detection time ~ J_th tau_f / k2 = %.2f us'%(1e6*2e4*tauf/k2n))
print(' largest steady |Psi| (fa=0.06) = %.3g ; f+/that = %.2f'%(zz*bc*0.06, 3e6/(zz*bc*0.06)))
print(' load jump of xi = V fR/(RC) = %.0f V/s ; absorbed in ~ %.2f ms'%(eRn*0.5, 1e3*eRn*0.5/k2n))
print(' k1 = 1.5 sqrt(f+) = %.1f ; k2 = %.3g'%(1.5*math.sqrt(3e6), k2n))
print(' static gain G_u(0) = %.1f V per unit duty'%float((Gu.subs(s,0)).subs(nv)))
print(' fs read as fa: 0.5/(R I) = %.6f'%(0.5/(10*4.8)))
print(' C_f = 0.7*470 uF = %.1f uF ; f_C = %.4f'%(0.7*470, 1/0.7-1))
