import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.autograd import Variable
from tqdm import tqdm
import numpy as np
import os
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
                 activation=nn.Tanh, path_neurons=None, ffn_neurons=None, device=device):
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

def grad(y, x):
    return torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y), create_graph=True)[0]

def laplacian(field, x, t):
    field_x = grad(field, x)
    field_xx = grad(field_x, x)
    field_t = grad(field, t)
    field_tt = grad(field_t, t)
    return field_xx, field_tt

# Define the ODE system for the Coupled Higgs field equations
def coupled_higgs(u_real, u_imag, v, x, t, beta):
    u_r_xx, u_r_tt = laplacian(u_real, x, t)
    u_i_xx, u_i_tt = laplacian(u_imag, x, t)
    v_xx, v_tt = laplacian(v, x, t)

    u_abs = torch.square(u_real) + torch.square(u_imag)
    u_abs_xx, u_abs_tt = laplacian(u_abs, x, t)

    # Calculate the field equations
    du_eq_r = u_r_tt - u_r_xx + u_abs * u_real - 2 * u_real * v
    du_eq_i = u_i_tt - u_i_xx + u_abs * u_imag - 2 * u_imag * v
    dv_eq = v_tt + v_xx - beta*u_abs_xx
    
    return du_eq_r, du_eq_i, dv_eq

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

def compute_analytical_boundary_loss(model, x, t, mse_cost_function, k, omega, r):
    pred_u_r, pred_u_i, pred_v = model(x, t)

    real_u1_val = real_u1(x, t, k, omega, r)
    imag_u1_val = imag_u1(x, t, k, omega, r)
    real_v1_val = real_v1(x, t, k, omega, r)
 
    boundary_loss_ur = mse_cost_function(pred_u_r, real_u1_val)
    boundary_loss_ui = mse_cost_function(pred_u_i, imag_u1_val)
    boundary_loss_v = mse_cost_function(pred_v, real_v1_val)
    
    return boundary_loss_ur, boundary_loss_ui, boundary_loss_v

def compute_physics_loss(model, x, t, beta, mse_cost_function):
    x.requires_grad = True
    t.requires_grad = True
    pred_u_r, pred_u_i, pred_v = model(x, t)

    # Compute the differential equation residuals
    du_eq_r, du_eq_i, dv_eq = coupled_higgs(pred_u_r, pred_u_i, pred_v, x, t, beta)
    
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

# Check if CUDA is available and set the default device
if torch.cuda.is_available():
    print("CUDA is available! Training on GPU.")
else:
    print("CUDA is not available. Training on CPU.")

model = FourierFeatureNN(input_dim=1, shared_units=128, output_dim=3, layers_per_path=2, scale=1.0, 
                         activation=nn.Tanh, path_neurons=[128, 128, 64, 64], ffn_neurons=[64, 32, 16], device=device).to(device)

num_epochs = 20000  # Number of training epochs
lr = 1e-4          # Learning rate
num_samples = 1000 # Number of samples for training
r = 1.1
omega = 3
k = 0.5
lambda_ = 1e-3
beta = 1

#epoch = 5434
model_save_path = 'model_weights_fourier_single_improved'
#model_state = torch.load(os.path.join(model_save_path, f'C_HIGGS_epoch_{epoch}.pth'), map_location=device)
#model = FourierFeatureNN(input_dim=1, shared_units=128, output_dim=3, layers_per_path=2, scale=1.0, 
#                         activation=nn.Tanh, path_neurons=[128, 128, 64, 64], ffn_neurons=[64, 32, 16], device=device).to(device)
#model.load_state_dict(model_state)
#model.train()
optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
mse_cost_function = torch.nn.MSELoss()
os.makedirs(model_save_path, exist_ok=True)
losses = []

print('\n##################### model  #################\n')
print(model)

# Training loop
x_bc_x0 = torch.zeros((num_samples, 1)).to(device)
t_bc_x0 = torch.rand((num_samples, 1)).to(device)  # Uniformly distributed random values between 0 and 1
x_bc_x1 = torch.ones((num_samples, 1)).to(device)
t_bc_x1 = torch.rand((num_samples, 1)).to(device)  # Uniformly distributed random values between 0 and 1
x_bc_t0 = torch.rand((num_samples, 1)).to(device)  # Uniformly distributed random values between 0 and 1
t_bc_t0 = torch.zeros((num_samples, 1)).to(device)
x_bc_t1 = torch.rand((num_samples, 1)).to(device)  # Uniformly distributed random values between 0 and 1
t_bc_t1 = torch.ones((num_samples, 1)).to(device)


for epoch in tqdm(range(num_epochs),
                  desc='Progress:',  # Empty description
                  leave=False,  # Do not leave the progress bar when done
                  ncols=75,  # Width of the progress bar
                  mininterval=0.1,
                  bar_format='{l_bar}{bar}|{remaining}',  # Only show the bar without any counters
                  colour='blue'): 
    x_n = (torch.rand(num_samples, 1)).to(device)  # x in range [-5, -3]
    t_n = (torch.rand(num_samples, 1)).to(device)   
    x_dom = torch.rand((num_samples, 1)).to(device)
    t_dom = torch.rand((num_samples, 1)).to(device) 
   
    optimizer.zero_grad()

    physics_loss_ur, physics_loss_ui, physics_loss_v = compute_physics_loss(model, x_n, t_n, beta, mse_cost_function)
    boundary_loss_ur_x0, boundary_loss_ui_x0, boundary_loss_v_x0 = compute_analytical_boundary_loss(model, x_bc_x0, t_bc_x0, mse_cost_function, k, omega, r)
    boundary_loss_ur_x1, boundary_loss_ui_x1, boundary_loss_v_x1 = compute_analytical_boundary_loss(model, x_bc_x1, t_bc_x1, mse_cost_function, k, omega, r)
    boundary_loss_ur_t0, boundary_loss_ui_t0, boundary_loss_v_t0 = compute_analytical_boundary_loss(model, x_bc_t0, t_bc_t0, mse_cost_function, k, omega, r)
    boundary_loss_ur_t1, boundary_loss_ui_t1, boundary_loss_v_t1 = compute_analytical_boundary_loss(model, x_bc_t1, t_bc_t1, mse_cost_function, k, omega, r)
    domain_loss_ur_t, domain_loss_ui_t, domain_loss_v_t = compute_analytical_boundary_loss(model, x_dom, t_dom, mse_cost_function, k, omega, r)
   
    # Total loss 
    loss_ur = lambda_*(physics_loss_ur) + (1-lambda_)*(boundary_loss_ur_x0 + boundary_loss_ur_t0 + domain_loss_ur_t )
    loss_ui = lambda_*(physics_loss_ui) + (1-lambda_)*(boundary_loss_ui_x0 + boundary_loss_ui_t0 + domain_loss_ui_t ) 
    loss_v = lambda_*(physics_loss_v) + (1-lambda_)*(boundary_loss_v_x0 + boundary_loss_v_t0 + domain_loss_v_t )
    total_loss = loss_ur + loss_ui + loss_v
    
    total_loss.backward()
    optimizer.step()
    
    # Print loss every few epochs
    if epoch % 1000 == 0:
        beta += 0.05
        beta = np.minimum(1, beta)
        print(f'Epoch {epoch}, Beta {beta:.2f}, Loss U (real): {loss_ur.item()}, Loss U (imag): {loss_ui.item()}, Loss V: {loss_v.item()}')
        model_filename = os.path.join(model_save_path, f'C_HIGGS_epoch_{epoch}_omega_{omega}.pth')
        torch.save(model.state_dict(), model_filename)
        
        df_losses = pd.DataFrame(losses)
        csv_file_path = 'loss_data/C_HIGGS_fourier_training_losses.csv'
        df_losses.to_csv(csv_file_path, index=False)
    
    if total_loss.item() < 1e-3:
        print(f'Stopping early at epoch {epoch} due to reaching target loss.')
        model_filename = os.path.join(model_save_path, f'C_HIGGS_epoch_{epoch}.pth')
        torch.save(model.state_dict(), model_filename)
        break
    
model_filename = os.path.join(model_save_path, f'C_HIGGS_omega_{omega}.pth')
torch.save(model.state_dict(), model_filename)

