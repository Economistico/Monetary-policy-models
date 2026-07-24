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
# Where:
# - $C_t$: Consumption of goods at time t.
# - $L_t$: Number of hours worked at time t.
# - $K_t$: Physical capital stock available at the beginning of period t.
# - $I_t$: Level of investment at time t.
# - $W_t$: Level of wages at time t.
# - $R_t$: Return on capital at time t.
# - $P_t$: General price level at time t.
# - $\Pi_t$: Firms' profits, distributed to households at time t.
# - $\beta$: Intertemporal discount factor (measuring the degree of patience).
# - $\sigma$: Coefficient of relative risk aversion (inverse of the intertemporal elasticity of substitution).
# - $\phi$: Marginal disutility in respect of labor supply (inverse of the Frisch elasticity of labor supply).
# 
# Also, capital accumulates according to $K_{t+1} = (1-\delta)K_t + I_t$ , where:
# - $\delta$: Depreciation rate of physical capital.
# 
# 
# #### Firms:
# \begin{equation}
#     \max_{L_t, K_t} \Pi_t = Y_tP_t - W_tL_t - R_tW_t 
# \end{equation}
# 
# 
# The production function is given by $Y_t = A_tK_t^{\alpha}L_t^{1-\alpha}$ , where:
# 
# - $Y_t$: Aggregate output (GDP) at time t.
# - $A_t$: Total factor productivity (TFP), representing the level of technology at time t.
# - $\alpha$: The elasticityof the level of production with respect to capital (the level of participation of capital in the productive
# process).
#     - $(1-\alpha)$: the level of participation of labor.
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


# Use this package only if you want to visualize the DAG of he project 
get_ipython().system('pip install graphviz')


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from sequence_jacobian import simple, combine, create_model


# In[17]:


from sequence_jacobian import drawdag # Import only if you want to visualize the DAG of the project 


# In[18]:


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


# In[19]:


rbc = create_model([firm, household, capital, equilibrium], name="RBC")

print(rbc)
print(f"Blocks: {rbc.blocks}")


# In[20]:


unknowns = ['K', 'L']
targets = ['euler', 'goods']
inputs = ['A']


# In[21]:


# DAG: it shows the dynamic of the solution given to the RBC model.
drawdag(rbc, inputs, unknowns, targets)


# In[22]:


# It is used the same calibration that the book suggests as well as the steady state values
calibration = {"A": 1., "R": 0.04, "sigma": 2, "phi": 1.5, "delta": 0.025, "alpha": 0.35, "beta": 0.985}
unknowns_ss = {"K": 20, "L": 0.7}
targets_ss = {"goods": 0., "euler": 0.}


# In[23]:


ss = rbc.solve_steady_state(calibration, unknowns_ss, targets_ss, solver="hybr")

print(ss)


# In[24]:


print(f"Euler equation: {ss['euler']}")
print(f"Goods market clearing: {ss['goods']}")


# In[25]:


G = rbc.solve_jacobian(ss, unknowns, targets, inputs, T=300)

print(G)


# In[26]:


# Here it is imposed a productivity shock
T, impact, rho = 300, 0.01, 0.8
dZ = np.empty((T, 2))
dZ[:, 0] = impact * ss['A'] * rho**np.arange(T)

plt.plot(100*dZ[:50, 0]/ss['A'], label='regular shock', linewidth=2.5)
plt.title(r'TFP shock')
plt.ylabel(r'% deviation from ss')
plt.xlabel(r'quarters')
plt.legend()
plt.grid(True)
plt.show()


# In[27]:


z_shock = 100 * dZ / ss['A']
dC = 100 * G['C']['A'] @ dZ / ss['C']
dY = 100 * G['Y']['A'] @ dZ / ss['Y']
dL = 100 * G['L']['A'] @ dZ / ss['L']
dK = 100 * G['K']['A'] @ dZ / ss['K']
dR = 100 * G['R']['A'] @ dZ / ss['R']
dW = 100 * G['W']['A'] @ dZ / ss['W']
dI = 100 * G['I']['A'] @ dZ / ss['I']


# In[28]:


fig, axs = plt.subplots(2, 4, figsize=(12,6))
series = [(z_shock, 'Productivity shock', 'brown'), (dC, 'Consumption', 'blue'), (dY, 'Output', 'green'), (dL, 'Labor', 'red'), 
          (dK, 'Capital', 'black'), (dR, 'Return on capital', 'orange'), (dW, 'Level of wages', 'purple'), (dI, 'Investment', 'cyan')]

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

