# -*- coding: utf-8 -*-
"""
Created on Sun Nov 14 22:27:15 2021

@author: Ryosuke
"""

from qutip import destroy, ket2dm, wigner, squeeze, fock, fock_dm, tensor, qeye, isket
from qutip.measurement import measurement_statistics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time

#================================
# 関数置き場
#================================

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

def beam_splitter(theta):
    op = theta * (tensor(a,adag) - tensor(adag,a))
    return op.expm()

def rotation(phi):
    op = 1j * phi * adag * a
    return op.expm()

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
    phi1 = R3 * BS2 * R2 * BS1 * R1 * BS0 * R0 * phi0

    # 1,2番目のビットをFock基底で測定
    P = []
    for i in range(dim):
        for j in range(dim):
            P.append(tensor(qeye(dim), fock_dm(dim, i), fock_dm(dim, j)))

    collapsed_states, probs = measurement_statistics(phi1, P)
    # ポストセレクト
    n = np.array([2, 2]) # ポストセレクトの値
    index = n[0]*dim + n[1]
    '''
    if probs[index] == 0:
        print('There is no probability to measure', n)
        state = None
    else:
        state = collapsed_states[index].ptrace(0) # 0番目の状態を残して他はトレースアウト
    '''
    return probs[index]

#================================
# main
#================================

dim = 50
a = destroy(dim)
adag = a.dag()

fid = '0.9996217593040437'
name = 'strawberryfields_scipyminimize_gkp_'
# load data
params_history = np.load(name+fid+'_params.npy')
cost_history = np.load(name+fid+'_cost.npy')

print(len(cost_history))
s = time.time()
prob_history = np.zeros_like(cost_history)
for i in range(len(cost_history)):
    prob_history[i] = circuit(params_history[i])
    print(i)
e = time.time()
print('time',e-s)
step = np.linspace(0, len(cost_history)-1, len(cost_history))

fig = plt.figure()
ax1 = fig.add_subplot(111)
ln1 = ax1.plot(step, cost_history, label='cost', color='blue')
ax2 = ax1.twinx()
ln2 = ax2.plot(step, prob_history, label='probapility', color='red')

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='upper right')

ax1.set_xlabel('step')
ax1.set_ylabel('cost')
ax2.set_ylabel('probability')
ax1.set_yscale('log')

fig.savefig(name+fid+'_cost_prob.png', dpi=200)