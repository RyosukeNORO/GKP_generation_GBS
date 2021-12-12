#!/usr/bin/env python
# coding: utf-8
from qutip import destroy, ket2dm, wigner, squeeze, fock, fock_dm, tensor, qeye, isket
from qutip.measurement import measurement_statistics
import strawberryfields as sf
from strawberryfields.ops import Squeezed, Rgate, BSgate, MeasureFock
import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import time

#================================
# Functions
#================================

def plot_wigner(rho, fig=None, ax=None):
    """
    Plot the Wigner function and the Fock state distribution given a ket or density matrix for
    a harmonic oscillator mode.
    """
    
    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(4,4))

    if isket(rho):
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

def circuit(params):
    prog = sf.Program(3)
    with prog.context as q:
        Squeezed(params[0])            | q[0]
        Squeezed(params[1])            | q[1]
        Squeezed(params[2])            | q[2]
        
        Rgate(params[3])               | q[0]
        BSgate(theta=params[4], phi=0) | (q[0], q[1])
        Rgate(params[5])               | q[1]
        BSgate(theta=params[6], phi=0) | (q[1], q[2])
        Rgate(params[7])               | q[0]
        BSgate(theta=params[8], phi=0) | (q[0], q[1])
        Rgate(params[9])               | q[0]
        #Rgate(params[10])              | q[1]
        #Rgate(params[11])              | q[2]
        
        MeasureFock(select=[2,2])       | (q[1], q[2])
        
    eng = sf.Engine("fock", backend_options={"cutoff_dim": dim})
    state = eng.run(prog).state
    fid = state.fidelity(gkp_target, mode=0)
    return fid

def cost(params):
    fid = circuit(params)
    return 1-fid

def callback(params):
    params_history.append(params)
    co = cost(params)
    cost_history.append(co)
    print('step', len(cost_history)-1, 'cost', co)
    print(params)
    print(time.perf_counter())

#================================
# Generate the target state
#================================

dim = 15
gkp_target_qutip = (squeeze(dim,-0.294) * (0.669*fock(dim,0) - 0.216*fock(dim,2) + 0.711*fock(dim,4))).unit()
gkp_target = np.ravel(gkp_target_qutip.full())
plot_wigner(gkp_target_qutip)

#================================
# Learn
#================================

params = np.random.normal(0, 0.5, [10]) # initial parameters
maxiter = 100
print(params)

params_history = []
cost_history = []
params_history.append(params)
cost_history.append(cost(params))

method = "BFGS"
options = {"disp": True, "maxiter": maxiter}
count = 0

start = time.perf_counter()
opt = scipy.optimize.minimize(cost, params, method='BFGS',
                              callback=lambda x: callback(x), options=options)
end = time.perf_counter()

print('time', end - start, 's')
nit = opt['nit']
plt.plot(np.linspace(0, nit, nit+1), cost_history)
plt.show()

#================================
# Confirm the learnt state by strawberryfields
#================================

def circuit2(params):
    prog = sf.Program(3)
    with prog.context as q:
        Squeezed(params[0])            | q[0]
        Squeezed(params[1])            | q[1]
        Squeezed(params[2])            | q[2]
        
        Rgate(params[3])               | q[0]
        BSgate(theta=params[4], phi=0) | (q[0], q[1])
        Rgate(params[5])               | q[1]
        BSgate(theta=params[6], phi=0) | (q[1], q[2])
        Rgate(params[7])               | q[0]
        BSgate(theta=params[8], phi=0) | (q[0], q[1])
        Rgate(params[9])               | q[0]
        #Rgate(params[10])              | q[1]
        #Rgate(params[11])              | q[2]
        
        MeasureFock(select=[2,2])       | (q[1], q[2])
        
    eng = sf.Engine("fock", backend_options={"cutoff_dim": dim})
    state = eng.run(prog).state
    return state


quad_axis = np.linspace(-5, 5, 1000)
state = circuit2(params_history[-1])
W = state.wigner(mode=0, xvec=quad_axis * sf.hbar, pvec=quad_axis * sf.hbar)

color_range = np.max(W.real)
nrm = mpl.colors.Normalize(-color_range, color_range)
plt.axes().set_aspect("equal")
plt.contourf(quad_axis, quad_axis, W, 60, cmap=cm.RdBu, norm=nrm)
plt.xlabel("q", fontsize=15)
plt.ylabel("p", fontsize=15)
plt.title('wigner sf')
plt.tight_layout()
plt.show()

#================================
# Confirm the learnt state and calculate the success probability by QuTiP
#================================

def beam_splitter(theta):
    op = theta * (tensor(a,adag) - tensor(adag,a))
    return op.expm()

def rotation(phi):
    op = 1j * phi * adag * a
    return op.expm()

def circuit_qutip(params):
    # input squeezed state
    phi_input0 = squeeze(dim, params[0]) * fock(dim, 0)
    phi_input1 = squeeze(dim, params[1]) * fock(dim, 0)
    phi_input2 = squeeze(dim, params[2]) * fock(dim, 0)
    phi0 = tensor(phi_input0, phi_input1, phi_input2)

    # interferometer
    R0 = tensor(rotation(params[3]), qeye(dim), qeye(dim))
    BS0 = tensor(beam_splitter(params[4]), qeye(dim))
    R1 = tensor(qeye(dim), rotation(params[5]), qeye(dim))
    BS1 = tensor(qeye(dim), beam_splitter(params[6]))
    R2 = tensor(rotation(params[7]), qeye(dim), qeye(dim))
    BS2 = tensor(beam_splitter(params[8]), qeye(dim))
    R3 = tensor(rotation(params[9]), qeye(dim), qeye(dim))
    #R3 = tensor(rotation(params[9]), rotation(params[10]), rotation(params[11]))
    phi1 = R3 * BS2 * R2 * BS1 * R1 * BS0 * R0 * phi0

    # measure 2nd and 3rd modes by fock basis
    P = []
    for i in range(dim):
        for j in range(dim):
            P.append(tensor(qeye(dim), fock_dm(dim, i), fock_dm(dim, j)))

    collapsed_states, probs = measurement_statistics(phi1, P)
    # post-select
    n = np.array([2, 2]) # post-select value
    index = n[0]*dim + n[1]

    if probs[index] == 0:
        print('There is no probability to measure', n)
    else:
        state = collapsed_states[index].ptrace(0)
    return state, probs[index]

dim = 50
a = destroy(dim)
adag = a.dag()

learnt_state, prob = circuit_qutip(params_history[-1])

plot_wigner(learnt_state)
print('probability for measuring (2,2): ', prob)

#================================
# Save results
#================================

cost_history_np = np.array(cost_history)
params_history_np = np.array(params_history)
library = 'strawberryfields'
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
            'argmin: '+str(amin)+'\n', 'Parameters: '+str(params_history[amin])+'\n',
            'probability: '+str(prob)]
f.writelines(datalist)
f.close()
print(amin, fid)
print(savename)




