#!/usr/bin/env python
# coding: utf-8

# # Coupled higgs equation 
# $$
# u_{tt} - u_{xx} + |u|^2 u - 2uv = 0
# $$
# $$
# v_{tt} + v_{xx} - (\left| u \right|^2)_{xx} = 0
# $$
# 
# where, $ u(x,t) $ represents a complex nucleon field and $ v(x,t) $ represents a real scalar meson field. The coupled Higgs field Equation describes a system of conserved scalar nucleon interaction with a neutral scalar meson.
# 
# solutions 
# 
# $$
# u_1(x, t) = ir e^{ir(\omega x + t)} \sqrt{1 + \omega^2} \tanh\left(\frac{r(k + x + \omega t)}{\sqrt{2}}\right)
# $$
# $$
# v_1(x, t) = r^2 \tanh^2\left(\frac{r(k + x + \omega t)}{\sqrt{2}}\right)
# $$
# 
# for $t = 0$
# 
# $$
# u_1(x, 0) = ir e^{ir \omega x} \sqrt{1 + \omega^2} \tanh\left(\frac{r(k + x)}{\sqrt{2}}\right)
# $$
# $$
# v_1(x, 0) = r^2 \tanh^2\left(\frac{r(k + x)}{\sqrt{2}}\right)
# $$
# 
# where 
# $k = 4, \omega = 5 , \alpha = 2, c = 2, r = 2$
# 

# In[1]:


import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

# Activation function class
class SinActv(nn.Module):
    def forward(self, input_):
        return torch.sin(input_)

# Base class for shared functionalities
class BaseNN(nn.Module):
    def __init__(self):
        super(BaseNN, self).__init__()

    def _make_layers(self, in_features, out_features, num_layers, activation):
        layers = [nn.Linear(in_features, out_features), activation()]
        for _ in range(1, num_layers):
            layers += [nn.Linear(out_features, out_features), activation()]
        return nn.Sequential(*layers)

# Model for U1
class ModelU1(BaseNN):
    def __init__(self, input_dim=2, shared_units=128, branch_units=32, output_dim=1, shared_layers=4, branch_layers=1, activation=SinActv):
        super(ModelU1, self).__init__()
        self.shared_layers = self._make_layers(input_dim, shared_units, shared_layers, activation)
        
        # Branch for real part
        self.branch_real = self._make_layers(shared_units, branch_units, branch_layers, activation)
        self.final_real = nn.Linear(branch_units, output_dim)
        
        # Branch for imaginary part
        self.branch_imag = self._make_layers(shared_units, branch_units, branch_layers, activation)
        self.final_imag = nn.Linear(branch_units, output_dim)

    def forward(self, x):
        shared_output = self.shared_layers(x)
        u_real = self.final_real(self.branch_real(shared_output))
        u_imag = self.final_imag(self.branch_imag(shared_output))
        return u_real, u_imag

# Model for V1
class ModelV1(BaseNN):
    def __init__(self, input_dim=2, units=128, output_dim=1, layers=5, activation=nn.Tanh):
        super(ModelV1, self).__init__()
        self.layers = self._make_layers(input_dim, units, layers, activation)
        self.final = nn.Linear(units, output_dim)

    def forward(self, x):
        return self.final(self.layers(x))

def grad(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), create_graph=True, retain_graph=True)[0]

def laplacian(field, x, t):
    field_x = grad(field, x)
    field_xx = grad(field_x, x)
    field_t = grad(field, t)
    field_tt = grad(field_t, t)
    return field_xx, field_tt

# Define the ODE system for the Coupled Higgs field equations
def coupled_higgs(u_real, u_imag, v, x, t):
    u_r_xx, u_r_tt = laplacian(u_real, x, t)
    u_i_xx, u_i_tt = laplacian(u_imag, x, t)
    v_xx, v_tt = laplacian(v, x, t)

    u_abs = u_real**2 + u_imag**2
    u_abs_xx, u_abs_tt = laplacian(u_abs, x, t)

    # Calculate the field equations
    du_eq_r = u_r_tt - u_r_xx + u_abs * u_real - 2 * u_real * v
    du_eq_i = u_i_tt - u_i_xx + u_abs * u_imag - 2 * u_imag * v
    dv_eq = v_tt + v_xx - u_abs_xx
    
    return du_eq_r, du_eq_i, dv_eq

# Function to calculate the real part of u1
def real_u1(x, t, k, omega, r):
    return np.real(1j * r * np.exp(1j * r * (omega * x + t)) * np.sqrt(1 + omega**2) *
                   np.tanh((r * (k + x + omega * t)) / np.sqrt(2)))

def imag_u1(x, t, k, omega, r):
    return np.imag(1j * r * np.exp(1j * r * (omega * x + t)) * np.sqrt(1 + omega**2) *
                   np.tanh((r * (k + x + omega * t)) / np.sqrt(2)))
    
def real_v1(x, t, k, omega, r):
    return (r * np.tanh((r * (k + x + omega * t)) / np.sqrt(2)) )**2

def compute_boundary_loss(model_u1, model_v1, x_boundary, t_boundary, mse_cost_function, k, omega, r):
    x_t_boundary = Variable(torch.from_numpy(np.hstack([x_boundary, t_boundary])).float(), requires_grad=False).to(device)
    
    pred_u_r, pred_u_i = model_u1(x_t_boundary)
    pred_v = model_v1(x_t_boundary)

    real_u1_val = torch.tensor(real_u1(x_boundary, t_boundary, k, omega, r), device=device).float().view(-1, 1)
    imag_u1_val = torch.tensor(imag_u1(x_boundary, t_boundary, k, omega, r), device=device).float().view(-1, 1)
    real_v1_val = torch.tensor(real_v1(x_boundary, t_boundary, k, omega, r), device=device).float().view(-1, 1)
 
    boundary_loss_ur = mse_cost_function(pred_u_r, real_u1_val)
    boundary_loss_ui = mse_cost_function(pred_u_i, imag_u1_val)
    boundary_loss_v = mse_cost_function(pred_v, real_v1_val)
    
    boundary_loss_u = torch.mean(boundary_loss_ur ** 2 + boundary_loss_ui ** 2)
    boundary_loss_v = boundary_loss_v ** 2
    
    return boundary_loss_u, boundary_loss_v

def compute_physics_loss(model_u1, model_v1, x, t, mse_cost_function):
    #x_t = Variable(torch.from_numpy(np.hstack([x, t])).float(), requires_grad=True).to(device)
    x_t = torch.cat([x, t], dim=1) 
    pred_u_r, pred_u_i = model_u1(x_t)
    pred_v = model_v1(x_t)
    
    du_eq_r, du_eq_i, dv_eq = coupled_higgs(pred_u_r, pred_u_i, pred_v, x, t)
    loss_pde_u = torch.mean(du_eq_r**2 + du_eq_i**2)  
    loss_pde_v = torch.mean(dv_eq**2)
    
    return loss_pde_u, loss_pde_v

# Check if CUDA is available and set the default device
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

model_u1 = ModelU1().to(device)
model_v1 = ModelV1().to(device)

num_epochs = 100000  # Number of training epochs
lr = 1e-3          # Learning rate
num_samples = 10000 # Number of samples for training
k = 0.5
omega = 5
r = 1.1 

optimizer_u1 = Adam(model_u1.parameters(), lr=lr)
optimizer_v1 = Adam(model_v1.parameters(), lr=lr)
mse_cost_function = torch.nn.MSELoss()
model_save_path = 'model_weights_testing_CHIGGS'
os.makedirs(model_save_path, exist_ok=True)
losses = []

# Training loop
for epoch in tqdm(range(num_epochs),
                  desc='Progress:',  # Empty description
                  leave=False,  # Do not leave the progress bar when done
                  ncols=75,  # Width of the progress bar
                  mininterval=0.1,
                  bar_format='{l_bar}{bar}|{remaining}',  # Only show the bar without any counters
                  colour='blue'):
    x_n = (torch.rand(num_samples, 1) * 1).to(device)  # x in range [-5, -3]
    t_n = (torch.rand(num_samples, 1) * 1).to(device)   
    x_n.requires_grad = True
    t_n.requires_grad = True
    x_bc_x0 = np.zeros((num_samples, 1))
    t_bc_x0 = np.random.uniform(0, 1, (num_samples, 1))
    x_bc_x1 = np.ones((num_samples, 1))
    t_bc_x1 = np.random.uniform(0, 1, (num_samples, 1))
    x_bc_t0 = np.random.uniform(0, 1, (num_samples, 1))
    t_bc_t0 = np.zeros((num_samples, 1))
    x_bc_t1 = np.random.uniform(0, 1, (num_samples, 1))
    t_bc_t1 = np.ones((num_samples, 1))
    
    optimizer_u1.zero_grad()
    optimizer_v1.zero_grad()

    physics_loss_u, physics_loss_v = compute_physics_loss(model_u1, model_v1, x_n, t_n, mse_cost_function)
    boundary_loss_u_x0, boundary_loss_v_x0 = compute_boundary_loss(model_u1, model_v1, x_bc_x0, t_bc_x0, mse_cost_function, k, omega, r)
    boundary_loss_u_x1, boundary_loss_v_x1 = compute_boundary_loss(model_u1, model_v1, x_bc_x1, t_bc_x1, mse_cost_function, k, omega, r)
    boundary_loss_u_t0, boundary_loss_v_t0 = compute_boundary_loss(model_u1, model_v1, x_bc_t0, t_bc_t0, mse_cost_function, k, omega, r)
    boundary_loss_u_t1, boundary_loss_v_t1 = compute_boundary_loss(model_u1, model_v1, x_bc_t1, t_bc_t1, mse_cost_function, k, omega, r)
    
    # Total loss 
    loss_u = physics_loss_u + boundary_loss_u_x0 + boundary_loss_u_x1 + boundary_loss_u_t0 + boundary_loss_u_t1 
    loss_v = physics_loss_v + boundary_loss_v_x0 + boundary_loss_v_x1 + boundary_loss_v_t0 + boundary_loss_u_t1 

    total_loss = loss_u + loss_v 
    total_loss.backward()
    optimizer_u1.step()
    optimizer_v1.step()
    
    # Print loss every few epochs
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss U: {loss_u.item()}, Loss V: {loss_v.item()}')
        model_u1_filename = os.path.join(model_save_path, f'C_HIGGS_U1_epoch_{epoch}.pth')
        torch.save(model_u1.state_dict(), model_u1_filename)
        
        model_v1_filename = os.path.join(model_save_path, f'C_HIGGS_V1_epoch_{epoch}.pth')
        torch.save(model_v1.state_dict(), model_v1_filename)
        
        df_losses = pd.DataFrame(losses)
        csv_file_path = 'loss_data/C_HIGGS_training_losses.csv'
        df_losses.to_csv(csv_file_path, index=False)
    
    if total_loss.item() < 0.01:
        print(f'Stopping early at epoch {epoch} due to reaching target loss.')
        break
    
    losses.append({
        'Epoch': epoch,
        'Loss U': loss_u.item(),
        'Loss V': loss_v.item(),
        'Total Loss': total_loss.item(),
        'Physics Loss': physics_loss_u.item() + physics_loss_v.item(),
        'Boundary Loss U': boundary_loss_u_x0.item() + boundary_loss_u_x1.item() + boundary_loss_u_t0.item() + boundary_loss_u_t1.item(),
        'Boundary Loss V': boundary_loss_v_x0.item() + boundary_loss_v_x1.item() + boundary_loss_v_t0.item() + boundary_loss_v_t1.item(),
    })

df_losses = pd.DataFrame(losses)
csv_file_path = 'loss_data/C_HIGGS_training_losses.csv'
df_losses.to_csv(csv_file_path, index=False)


