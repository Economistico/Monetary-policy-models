#!/usr/bin/env python
# coding: utf-8

# ### Real Business Cycle (RBC) Model Using the Sequence-Space Jacobian (SSJ) Library in Python
# 
# This notebook replicates the RBC model presented in Celso José Costa Junior's Understanding DSGE Models, using the equations in levels rather than their log-linearized form. The SSJ library works directly with equations in levels and automatically linearizes them around the steady state.
# 
# ### The model
# 
# #### Households:
# 
# Households maximize a standard CRRA utility function subject to a budget constraint:
# 
# \begin{equation}
#     \max_{C_t,L_t, K_{t+1}} \mathbf{E} \sum_{t=0}^{\infty} \beta^t \left( \frac{C_t^{1-\sigma}}{1-\sigma} - \frac{L_t^{1+\phi}}{1+\phi} \right)
# \end{equation}
# 
# subject to: $P_t(C_t+I_t) = W_tL_t + R_tK_t + \Pi_t$
# 
# Also, capital accumulates according to: $K_{t+1} = (1-\delta)K_t + I_t$
# 
# 
# 
# Where:
# - $C_t$: Household consumption at time t.
# - $L_t$: Labor supplied by households at time t.
# - $K_t$: Physical capital stock available at the beginning of period t.
# - $I_t$: Investment at time t.
# - $Y_t$: Aggregate output (GDP) at time t.
# - $W_t$: Nominal wage rate.
# - $R_t$: Nominal rental rate of capital.
# - $P_t$: Aggregate price level.
# - $\Pi_t$: Firms' profits, distributed to households.
# - $\beta$: Household discount factor, measuring the degree of patience.
# - $\sigma$: Coefficient of relative risk aversion (inverse of the intertemporal elasticity of substitution).
# - $\phi$: Inverse of the Frisch elasticity of labor supply.
# 
# 
# 
# 
# 
# #### Firms:
# \begin{equation}
#     \max_{L_t, K_t} \Pi_t = Y_tP_t - W_tL_t - R_tW_t 
# \end{equation}
# 
# 
# The production function is given by $Y_t = A_tK_t^{\alpha}L_t^{1-\alpha}$, where:
# 
# - $A_t$: Total factor productivity (TFP), representing the level of technology.
# 
# 
# #### Equilibrium:
# 
# \begin{equation}
#     Y_t=C_t+I_t
# \end{equation}
# 
# ### Equations derivated after dealing with first order conditions:
# 
# \begin{equation}
#     C_t^\sigma L_t^\phi = \frac{W_t}{P_t}
# \end{equation}
# 
# \begin{equation}
#     \left[\frac{\mathbb{E}(C_{t+1})}{C_t} \right]^\sigma = \beta \left[ (1-\delta) + \mathbb{E} \left(\frac{R_{t+1}}{P_{t+1}} \right)\right]
# \end{equation}
# 
# \begin{equation}
#     K_{t+1}=(1-\delta)K_t+I_t
# \end{equation}
# 
# \begin{equation}
#     Y_t = A_t K_t^{\alpha} L_t^{1-\alpha}
# \end{equation}
# 
# \begin{equation}
#     K_t = \alpha \frac{Y_t}{R_t/P_t}
# \end{equation}
# 
# \begin{equation}
#     L_t = (1-\alpha) \frac{Y_t}{W_t/P_t}
# \end{equation}
# 
# \begin{equation}
#     Y_t = C_t + I_t
# \end{equation}

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, combine, create_model


# In[ ]:


@simple
def firm(A, K, L, alpha):
    Y  = A * K(-1)**alpha * L**(1-alpha)  
    R = alpha    * Y / K(-1)              
    W  = (1-alpha)* Y / L                
    return Y, R, W

@simple
def household(W, L, sigma, phi):
    C = (W / L**phi)**(1/sigma)            
    return C

@simple
def capital(K, delta):
    I = K - (1-delta)*K(-1)               
    return I


@simple
def equilibrium(C, R, Y, I, beta, sigma, delta):
    euler = (C(+1)/C)**sigma - beta*((1-delta) + R(+1))   
    goods = Y - C - I                                      
    return euler, goods


# In[ ]:


rbc = create_model([firm, household, capital, equilibrium], name="RBC")

print(rbc)
print(f"Blocks: {rbc.blocks}")


# In[ ]:


unknowns = ['K', 'L']
targets = ['euler', 'goods']
inputs = ['A']


# In[ ]:


# It is used the same calibration that the book suggests as well as the steady state values
calibration = {"A": 1., "R": 0.04, "sigma": 2, "phi": 1.5, "delta": 0.025, "alpha": 0.35, "beta": 0.985}
unknowns_ss = {"K": 20, "L": 0.7}
targets_ss = {"goods": 0., "euler": 0.}


# In[ ]:


ss = rbc.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

print(ss)


# In[ ]:


print(f"Euler equation: {ss['euler']}")
print(f"Goods market clearing: {ss['goods']}")


# In[ ]:


G = rbc.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)


# In[ ]:


# Here it is imposed a productivity shock
T, impact, rho = 300, 0.01, 0.8
dZ = np.empty((T, 2))
dZ[:, 0] = impact * ss['A'] * rho**np.arange(T)

plt.plot(100*dZ[:50, 0]/ss['A'], label='regular shock', linewidth=2.5)
plt.title(r'Two TFP shocks')
plt.ylabel(r'% deviation from ss')
plt.xlabel(r'quarters')
plt.legend()
plt.show()


# In[ ]:


z_shock = 100 * dZ / ss['A']
dC = 100 * G['C']['A'] @ dZ / ss['C']
dY = 100 * G['Y']['A'] @ dZ / ss['Y']
dL = 100 * G['L']['A'] @ dZ / ss['L']
dK = 100 * G['K']['A'] @ dZ / ss['K']
dR = 100 * G['R']['A'] @ dZ / ss['R']
dW = 100 * G['W']['A'] @ dZ / ss['W']
dI = 100 * G['I']['A'] @ dZ / ss['I']


# In[ ]:


fig, axs = plt.subplots(2, 4, figsize=(12,6))
series = [(z_shock, 'Productivity shock', 'brown'), (dC, 'Consumption', 'blue'), (dY, 'Output', 'green'), (dL, 'Labor', 'red'), 
          (dK, 'Capital', 'black'), (dR, 'Real rental rate', 'orange'), (dW, 'Real wage', 'purple'), (dI, 'Investment', 'cyan')]

for ax, (data, title, color) in zip(axs.flatten(), series):
    ax.plot(data[:100,0], linewidth=2.5, color=color)
    ax.set_title(title)
    ax.set_xlabel('Quarters')
    ax.set_ylabel('% de. from ss')
    #ax.set_ylim(-1,6)
    ax.grid(True)

fig.suptitle('Impulse Response Functions (IRFs) after a positive productivity shock of 1%', fontsize=16)
plt.tight_layout()
plt.show()

