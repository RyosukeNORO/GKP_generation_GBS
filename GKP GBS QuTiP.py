#!/usr/bin/env python
# coding: utf-8

# In[9]:


from qutip import destroy, ket2dm, wigner, squeeze, fock, fock_dm, tensor, qeye, isket
from qutip.measurement import measure, measurement_statistics
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors, colorbar, cm
import time


# In[10]:


def plot_wigner(rho, fig=None, ax=None):
    """
    Plot the Wigner function and the Fock state distribution given a density matrix for
    a harmonic oscillator mode.
    """
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,4))

    if isket(rho):  # ket状態を密度関数にする（必要かわからん）
        rho = ket2dm(rho)
    
    scale = np.sqrt(2)
    xvec = np.linspace(-5*scale,5*scale,100)

    W = wigner(rho, xvec, xvec)
    wlim = abs(W).max()

    ax.contourf(xvec/scale, xvec/scale, W, 60, norm=mpl.colors.Normalize(-wlim,wlim), cmap=mpl.cm.get_cmap('RdBu'))
    ax.set_xlabel('q', fontsize=16)
    ax.set_ylabel('p', fontsize=16)
    #ax.set_title()
    fig.tight_layout
    
    return fig, ax


# Target GKP state
# 
# $|GKP_{target} \rangle = S\hat(r)(c_0 |0\rangle + c_2 |2\rangle + c_4 |4\rangle), r = 0.294, c_0 = 0.669, c_2 = -0.216, c_4 = 0.711$

# In[34]:


dim = 30
gkp_target = (squeeze(dim,-0.294) * (0.669*fock(dim,0) - 0.216*fock(dim,2) + 0.711*fock(dim,4))).unit()
plot_wigner(gkp_target)


# Circuit returning the fidelity between target state and created state

# In[35]:


a = destroy(dim)
adag = a.dag()

def beam_splitter(theta):
    op = theta * (tensor(a,adag) - tensor(adag,a))
    return op.expm()

def rotation(phi):
    op = 1j * phi * adag * a
    return op.expm()


# In[36]:


def circuit(params):
    # 入力のスクイーズド状態の準備
    phi_input0 = squeeze(dim, params[0]) * fock(dim, 0)
    phi_input1 = squeeze(dim, params[1]) * fock(dim, 0)
    phi_input2 = squeeze(dim, params[2]) * fock(dim, 0)
    phi0 = tensor(phi_input0, phi_input1, phi_input2)

    # 干渉計
    R0 = tensor(rotation(params[3]), qeye(dim), qeye(dim))
    BS0 = tensor(beam_splitter(params[4]), qeye(dim))
    R1 = tensor(qeye(dim), rotation(params[5]), qeye(dim))
    BS1 = tensor(qeye(dim), beam_splitter(params[6]))
    R2 = tensor(rotation(params[7]), qeye(dim), qeye(dim))
    BS2 = tensor(beam_splitter(params[8]), qeye(dim))
    R3 = tensor(rotation(params[9]), qeye(dim), qeye(dim))
    #R3 = tensor(rotation(params[9]), rotation(params[10]), rotation(params[11]))
    phi1 = BS2 * R2 * BS1 * R1 * BS0 * R0 * phi0

    # 1,2番目のビットをFock基底で測定
    P = []
    for i in range(dim):
        for j in range(dim):
            P.append(tensor(qeye(dim), fock_dm(dim, i), fock_dm(dim, j)))

    collapsed_states, probs = measurement_statistics(phi1, P)
    # ポストセレクト
    n = np.array([2, 2]) # ポストセレクトの値
    index = n[0]*dim + n[1]

    if probs[index] == 0:
        print('There is no probability to measure', n)
    else:
        state = collapsed_states[index].ptrace(0) # 0番目の状態を残して他はトレースアウト
    return state


# cost function
# 
# $C(\theta) = 1 - |\langle \Psi_{target}|\Psi_{learnt}\rangle|^2$

# In[37]:


def cost(params):
    state = circuit(params)
    f = gkp_target.dag() * state * gkp_target
    fid = np.real(f.full()[0,0])
    return 1-fid


# In[38]:


def callback(params):
    params_history.append(params)
    co = cost(params)
    cost_history.append(co)
    print('step', len(cost_history)-1, 'cost', co)
    print(params)
    print(time.perf_counter())


# Learning

# In[39]:


params = np.random.normal(0, 1, [10]) # initial parameters
maxiter = 500

params_history = []
cost_history = []
params_history.append(params)
cost_history.append(cost(params))

method = "BFGS"
options = {"disp": True, "maxiter": maxiter}

start = time.perf_counter()
opt = scipy.optimize.minimize(cost, params, method='BFGS',
                              callback=lambda x: callback(x), options=options)
end = time.perf_counter()

print('time', end - start, 's')
nit = opt['nit']
plt.plot(np.linspace(0, nit, nit+1), cost_history)
plt.show()


# In[ ]:


learnt_state = circuit(params_history[-1])
plot_wigner(learnt_state)
plot_wigner(gkp_target)
print(params_history[-1])


# In[ ]:


cost_history_np = np.array(cost_history)
params_history_np = np.array(params_history)
library = 'QuTiP'
optname = '_scipyminimize'
statename = '_gkp'
cpuname = 'intel_core_i7_11gen'
fid = 1-cost_history_np.min()
amin = cost_history_np.argmin()
savename = library+optname+statename+'_'+str(fid)
np.save(savename+'_cost', cost_history_np)
np.save(savename+'_params', params_history_np)
f = open(savename+'.txt', 'a')
f.write('\n')
datalist = ['CPU: '+cpuname+'\n', savename+statename+'\n', 'Cutoff dimention: '+str(dim)+'\n',
            'Runtime: '+str(end-start)+'s\n', 'Fidelity: '+str(fid)+'\n',
            'argmin: '+str(amin)+'\n', 'Parameters: '+str(params_history[amin])]
f.writelines(datalist)
f.close()
print(amin, fid)
print(savename)


# In[ ]:




