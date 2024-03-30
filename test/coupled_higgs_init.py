import torch
import torch.nn as nn
from torch.optim import AdamW, LBFGS
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter
import os
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)


def fourier_features(x, B):
    x_transformed = torch.matmul(x, B)
    return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed) ], dim=-1)

def init_fixed_frequency_matrix(size, scale=1.0):
    num_elements = size[0] * size[1]
    lin_space = torch.linspace(-scale, scale, steps=num_elements)
    B = lin_space.view(size).float()
    return B

def fourier_features(x, B):
    x_transformed = torch.matmul(x, B)
    return torch.cat([torch.sin(x_transformed), torch.cos(x_transformed)], dim=-1)

def init_fixed_frequency_matrix(size, scale=1.0):
    num_elements = size[0] * size[1]
    lin_space = torch.linspace(-scale, scale, steps=num_elements)
    B = lin_space.view(size).float()
    return B

class FourierFeatureNN(nn.Module):
    def __init__(self, input_dim=1, shared_units=128, output_dim=3, layers_per_path=2, scale=1.0, 
                 activation=nn.Tanh, path_neurons=None, ffn_neurons=None, device='cpu'):
        super(FourierFeatureNN, self).__init__()
        self.Bx = init_fixed_frequency_matrix((input_dim, shared_units // 2), scale=scale).to(device)
        self.Bt = init_fixed_frequency_matrix((input_dim, shared_units // 2), scale=scale).to(device)
        
        if path_neurons is None:
            path_neurons = [shared_units] * layers_per_path
        if ffn_neurons is None:
            ffn_neurons = [shared_units // 2]  # Example default

        # Define separate paths for x and t after Fourier transformation
        self.path_x = self._build_path(path_neurons, activation)
        self.path_t = self._build_path(path_neurons, activation)

        # Define the subsequent FFN after pointwise multiplication
        ffn_layers = [nn.Linear(path_neurons[-1], ffn_neurons[0])] + [
            layer for n in ffn_neurons[1:] for layer in (activation(), nn.Linear(ffn_neurons[ffn_neurons.index(n)-1], n))
        ]
        self.ffn = nn.Sequential(*ffn_layers)

        # Final layer to produce output
        self.final_layer = nn.Linear(ffn_neurons[-1], output_dim)

    def _build_path(self, neurons, activation):
        layers = []
        for i in range(len(neurons) - 1):
            layers.append(nn.Linear(neurons[i], neurons[i+1]))
            layers.append(activation())
        return nn.Sequential(*layers)

    def forward(self, x, t):
        # Apply Fourier feature transformations
        x_fourier = fourier_features(x, self.Bx)
        t_fourier = fourier_features(t, self.Bt)

        # Pass through separate paths
        x_path_output = self.path_x(x_fourier)
        t_path_output = self.path_t(t_fourier)

        # Pointwise multiplication of the separate path outputs
        combined_features = x_path_output * t_path_output

        # Pass through the fully connected network
        ffn_output = self.ffn(combined_features)

        # Final layer to produce output
        final_output = self.final_layer(ffn_output)
        output_1, output_2, output_3 = final_output.split(1, dim=-1)
        
        return output_1, output_2, output_3

def grad(x, t):
    return torch.autograd.grad(x, t, grad_outputs=torch.ones_like(x), create_graph=True)[0]

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

    u_abs = torch.square(u_real) + torch.square(u_imag)
    u_abs_xx, u_abs_tt = laplacian(u_abs, x, t)

    # Calculate the field equations
    du_r = u_r_tt - u_r_xx + u_abs * u_real - 2 * u_real * v
    du_i = u_i_tt - u_i_xx + u_abs * u_imag - 2 * u_imag * v
    dv = v_tt + v_xx - u_abs_xx
    
    return du_r, du_i, dv

def real_u1(x, t, k, omega, r):
    complex_exp = torch.exp(1j * r * (omega * x + t))
    tanh_val = torch.tanh((r * (k + x + omega * t)) / torch.sqrt(torch.tensor(2.0)))
    result = torch.real(1j * r * complex_exp * torch.sqrt(torch.tensor(1) + omega**2) * tanh_val)
    return result

def imag_u1(x, t, k, omega, r):
    complex_exp = torch.exp(1j * r * (omega * x + t))
    tanh_val = torch.tanh((r * (k + x + omega * t)) / torch.sqrt(torch.tensor(2.0)))
    result = torch.imag(1j * r * complex_exp * torch.sqrt(torch.tensor(1) + omega**2) * tanh_val)
    return result

def real_v1(x, t, k, omega, r):
    result = (r * torch.tanh((r * (k + x + omega * t)) / torch.sqrt(torch.tensor(2.0))))**2
    return result

def compute_physics_loss(model, x, t, u_r0, u_i0, v0, y1, device, mse_cost_function, k, omega, r):
    x.requires_grad = True
    t.requires_grad = True
    u_r0.requires_grad = True 
    u_i0.requires_grad = True
    v0.requires_grad = True 
    y1.requires_grad = True 

    pred_u_r, pred_u_i, pred_v = model(x, t)
    u_r = u_r0 + y1*pred_u_r 
    u_i = u_i0 + y1*pred_u_i
    v = v0 + y1*pred_v
    
    du_eq_r, du_eq_i, dv_eq = coupled_higgs(u_r, u_i, v, x, t)
    
    # Define target tensors of zeros with the same shape as the predictions
    zeros_r = torch.zeros_like(du_eq_r, device=device)
    zeros_i = torch.zeros_like(du_eq_i, device=device)
    zeros_v = torch.zeros_like(dv_eq, device=device)
    
    # Compute the MSE loss against zeros for each differential equation residual
    loss_r = mse_cost_function(du_eq_r, zeros_r)
    loss_i = mse_cost_function(du_eq_i, zeros_i)
    loss_v = mse_cost_function(dv_eq, zeros_v)
    
    # Return the scalar loss values for real part, imaginary part, and v
    return loss_r, loss_i, loss_v

def cyclic_iterator(items):
    return cycle(items)

def plot_predictions(epoch, model, device, k, omega, r, image_save_path):
    model.eval()  # Set the model to evaluation mode
    x = torch.linspace(0, 1, 300)
    t = torch.linspace(0, 1, 300)
    X, T = torch.meshgrid(x, t)  # Create a 2D grid of x and t
    X_flat = X.flatten().unsqueeze(-1).to(device)
    T_flat = T.flatten().unsqueeze(-1).to(device)
    x0 = torch.zeros_like(X_flat).to(device)
    x1 = torch.ones_like(X_flat).to(device)
    
    
    with torch.no_grad():
        pred_u_r, pred_u_i, pred_v = model(X_flat, T_flat) 
    
    pred_u_r = (1-X_flat)*real_u1(x0, T_flat, k, omega, r) + X_flat*real_u1(x1, T_flat, k, omega, r) + X_flat*(1-X_flat)*pred_u_r 
    pred_u_i = (1-X_flat)*imag_u1(x0, T_flat, k, omega, r) + (1-X_flat)*imag_u1(x1, T_flat, k, omega, r) + X_flat*(1-X_flat)*pred_u_i
    pred_v = (1-X_flat)*real_v1(x0, T_flat, k, omega, r) + (1-X_flat)*real_v1(x1, T_flat, k, omega, r) + X_flat*(1-X_flat)*pred_v
  
    pred_u_r = pred_u_r.cpu().reshape(X.shape).numpy()
    pred_u_i = pred_u_i.cpu().reshape(X.shape).numpy()
    pred_v = pred_v.cpu().reshape(X.shape).numpy()

    real_u1_analytical = real_u1(X_flat, T_flat, k, omega, r).cpu().reshape(X.shape).numpy()
    imag_u1_analytical = imag_u1(X_flat, T_flat, k, omega, r).cpu().reshape(X.shape).numpy()
    real_v1_analytical = real_v1(X_flat, T_flat, k, omega, r).cpu().reshape(X.shape).numpy()

    sigma = 10
    pred_v_smooth = gaussian_filter(pred_v, sigma=sigma)
    shrink = 0.3
    cmap = 'viridis'
    aspect = 50
    # Data for plotting
    data_to_plot = [
        (pred_u_r, 'Predicted Real Part of $u_1(x, t)$', 'Real part of $u_1$'),
        (pred_u_i, 'Predicted Imaginary Part of $u_1(x, t)$', 'Imag part of $u_1$'),
        (pred_v_smooth, 'Predicted Real Part of $v_1(x, t)$', 'Real part of $v_1$'),
        (real_u1_analytical, 'Analytical Real Part of $u_1(x, t)$', 'Real part of $u_1$'),
        (imag_u1_analytical, 'Analytical Imaginary Part of $u_1(x, t)$', 'Imag part of $u_1$'),
        (real_v1_analytical, 'Analytical Real Part of $v_1(x, t)$', 'Real part of $v_1$')
    ]
    fig = plt.figure(figsize=(24, 16))
    
    for idx, (data, title, zlabel) in enumerate(data_to_plot, start=1):
        ax = fig.add_subplot(2, 3, idx, projection='3d')
        surf = ax.plot_surface(X.numpy(), T.numpy(), data, cmap=cmap)
        fig.colorbar(surf, ax=ax, shrink=shrink, aspect=aspect)
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel(zlabel)

    plt.tight_layout()
    plt.savefig(f'{image_save_path}/chiggs_model_comparison_epoch_{epoch}.png',dpi = 100)
    plt.close(fig)
      # Close the figure to free memory

def seq2seq_training(model, model_save_path, image_save_path, mse_cost_function, device, num_epochs, lr, num_samples, r, k, omega, gamma):
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    items = [-2, -1, 0, 1]
    iterator = cyclic_iterator(items)
    print(' Starting Seq2Seq Training')
    factor = 0
    x = torch.linspace(0,1,num_samples).unsqueeze(-1).to(device)
    t = torch.linspace(0,1,num_samples).unsqueeze(-1).to(device)
    x0 = torch.zeros_like(x).to(device)
    x1 = torch.ones_like(x).to(device)
    y = 1-x 
    y1 = x*(1-x)
    u_r0 = y*real_u1(x0, t, k, omega, r) + x*real_u1(x1, t, k, omega, r)
    u_i0 = y*imag_u1(x0, t, k, omega, r) + x*imag_u1(x1, t, k, omega, r)
    v0 = y*real_v1(x0,t, k, omega, r) + x*real_v1(x1, t, k, omega, r)     
    
    for epoch in tqdm(range(num_epochs),
                  desc='Progress:',  
                  leave=False,  
                  ncols=75,
                  mininterval=0.1,
                  bar_format='{l_bar} {bar} | {remaining}',  # Only show the bar without any counters
                  colour='blue'): 
        
        model.train()
        optimizer.zero_grad()

        loss_ur, loss_ui, loss_v = compute_physics_loss(model, x, t, u_r0, u_i0, v0, y1, device, mse_cost_function, k, omega, r)
        total_loss = loss_ur + loss_ui + loss_v
        total_loss.backward()
        optimizer.step()

        if epoch % 1000 == 0:
            print(f' Epoch {epoch}, Factor {factor}, Loss U (real): {loss_ur.item()}, Loss U (imag): {loss_ui.item()}, Loss V: {loss_v.item()}')
            plot_predictions(epoch, model, device, k, omega, r, image_save_path)
            
    model_filename = os.path.join(model_save_path, f'C_HIGGS_first_training.pth')
    torch.save(model.state_dict(), model_filename)
    print('COMPLETED Seq2Seq Training')


def main():
    # Check if CUDA is available and set the default device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print("CUDA is available! Training on GPU.")
    else:
        print("CUDA is not available. Training on CPU.")

    model = FourierFeatureNN(input_dim=1, shared_units=128, output_dim=3, layers_per_path=2, scale=1.0, 
                         activation=nn.Tanh, path_neurons=[128, 128, 128, 128], ffn_neurons=[128, 128, 128], device=device).to(device)
    print(model)
    num_epochs_lbfgs = 50  # Number of training epochs
    num_samples_lbfgs = 1000*5 # Number of samples for training
    num_epochs_sq = 12000
    num_samples_sq = 3000
    lr = 1e-4
    r = 1.1
    omega = 3 
    k = 0.5
    gamma = 1e-3
    model_save_path = 'model_weights_test' 
    mse_cost_function = torch.nn.MSELoss()
    os.makedirs(model_save_path, exist_ok=True)
    image_save_path = 'results_test'
    os.makedirs(image_save_path, exist_ok=True)
    losses = []
    seq2seq_training(model, model_save_path, image_save_path, mse_cost_function, device, num_epochs_sq, lr, num_samples_sq, r, k, omega, gamma)
    #LBFGS_training(model, model_save_path, mse_cost_function, device, num_epochs_lbfgs, lr, num_samples_lbfgs, r, k, omega, gamma)
if __name__ == '__main__':
    main()

