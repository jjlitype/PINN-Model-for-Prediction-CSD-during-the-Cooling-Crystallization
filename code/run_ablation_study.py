import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import argparse
import json
import random

def set_seed(seed):
    """Sets the seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===================================================================
# 1. 辅助函数 (Helper Functions) - [无变化]
# ===================================================================

def robust_string_to_numpy_array(array_string):
    """
    Converts a string representation of numbers into a 1D NumPy float array.
    """
    if not isinstance(array_string, str):
        try:
            return np.array(array_string, dtype=float)
        except Exception:
            return np.array([], dtype=float)
    cleaned_str = array_string.strip('[] ')
    cleaned_str_single_line = cleaned_str.replace('\n', ' ')
    try:
        numpy_numbers = np.fromstring(cleaned_str_single_line, sep=' ', dtype=float)
        return numpy_numbers
    except ValueError:
        number_strings = [val for val in cleaned_str_single_line.split(' ') if val]
        if not number_strings:
            return np.array([], dtype=float)
        try:
            return np.array(number_strings, dtype=float)
        except ValueError:
            return np.array([], dtype=float)

def approx_dirac_delta(L, L0_tensor, sigma_tensor):
    """ Gaussian approximation of Dirac delta: delta(L - L0) """
    return (1.0 / (sigma_tensor * torch.sqrt(torch.tensor(2.0 * np.pi, device=L.device)))) * \
           torch.exp(-0.5 * ((L - L0_tensor) / sigma_tensor)**2)

def bin_experiment_data(L_array, n_array, num_bins=60):
    if len(L_array) < 2:
        bins = np.linspace(0, 1, num_bins + 1)
        bin_centers = 0.5 * (bins[:-1] + bins[1:])
        return bin_centers, np.zeros(num_bins)

    dL_kde = np.mean(np.diff(L_array))
    L_min, L_max = L_array.min(), L_array.max()
    bins = np.linspace(L_min, L_max, num_bins + 1)
    bin_widths = np.diff(bins)
    digitized = np.digitize(L_array, bins) - 1
    
    binned_n_density = np.zeros(num_bins)
    for i in range(num_bins):
        mask = (digitized == i)
        if np.any(mask):
            particles_in_bin = np.sum(n_array[mask]) * dL_kde
            if bin_widths[i] > 0:
                binned_n_density[i] = particles_in_bin / bin_widths[i]
            
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    return bin_centers, binned_n_density

def process_csv(filepath, data_col_prefix, rho, D, u, num_bins=60, sample_size=None, seed=42):
    """Loads and processes CSV data, now configurable for data column and sample size."""
    df = pd.read_csv(filepath)
    
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=seed)

    df['t_s'] = df['elapsed_sec']
    df['Re'] = df['RPM'] * rho * D**2 / 60 / u
    X_list, n_list = [], []
    
    x_coords_col = f'{data_col_prefix}_x_coords'
    y_density_col = f'{data_col_prefix}_y_density'

    for _, row in df.iterrows():
        L_array = robust_string_to_numpy_array(row[x_coords_col])
        p_array = robust_string_to_numpy_array(row[y_density_col])
        if L_array.size == 0 or p_array.size == 0: continue

        N_abs_scalar = row['crystal_count']
        if pd.isna(N_abs_scalar) or N_abs_scalar <= 0: continue

        p_array = np.maximum(p_array, 0)
        n_array = N_abs_scalar * p_array
        bin_centers, binned_n = bin_experiment_data(L_array, n_array, num_bins=num_bins)

        X_current = np.stack([
            bin_centers,
            np.full(num_bins, row['t_s']),
            np.full(num_bins, row['supersaturation']),
            np.full(num_bins, row['Re']),
            np.full(num_bins, row['Temperature']),
            np.full(num_bins, row['seed'])
        ], axis=1)
        n_current = binned_n.reshape(-1, 1)

        X_list.append(X_current)
        n_list.append(n_current)

    if not X_list:
        return np.array([[]]).reshape(0, 6), np.array([[]]).reshape(0, 1)
    return np.vstack(X_list), np.vstack(n_list)

def build_sequence_dataset(X, n, seq_len):
    """Creates sliding-window sequences for recurrent models."""
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if X.shape[0] < seq_len:
        raise ValueError("Not enough samples to build sequences with the requested seq_len")

    sequences = []
    targets = []
    for start_idx in range(0, X.shape[0] - seq_len + 1):
        end_idx = start_idx + seq_len
        sequences.append(X[start_idx:end_idx])
        targets.append(n[end_idx - 1])

    return np.stack(sequences), np.stack(targets)

# ===================================================================
# 2. PINN 模型定义 (PINN Model Definition) - [无变化]
# ===================================================================

class PINN(nn.Module):
    def __init__(self, config, device):
        super(PINN, self).__init__()
        
        layers = config['layers']
        activation_map = {'Tanh': nn.Tanh(), 'ReLU': nn.ReLU(), 'SiLU': nn.SiLU()}
        activation_func = activation_map[config['activation']]

        net_layers = []
        for i in range(len(layers) - 1):
            net_layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers) - 2: # No activation or BN on the last layer
                net_layers.append(nn.BatchNorm1d(layers[i+1]))
                net_layers.append(activation_func)
        self.net = nn.Sequential(*net_layers)

        # Physical parameters
        self.log_kg = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.log_kb = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32), requires_grad=not config['freeze_b'])
        self.g0_unconstrained = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.b0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=not config['freeze_b'])
        self.Re_exp = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.log_lambda = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.device = device

    def forward(self, x):
        return self.net(x)

    def growth_rate(self, L, S, Re):
        kg = torch.exp(self.log_kg)
        g0 = 1 + 2 * torch.sigmoid(self.g0_unconstrained)
        lambda_ = torch.exp(self.log_lambda)
        S_term = torch.pow(torch.clamp(S, min=1e-9), g0)
        Re_term = torch.pow(torch.clamp(Re, min=1e-9), self.Re_exp)
        return kg * S_term * Re_term * (1 + lambda_ * L)

    def nucleation_rate(self, S):
        kb = torch.exp(self.log_kb)
        return kb * torch.pow(torch.clamp(S, min=1e-9), self.b0)

class PINN_RNN(nn.Module):
    def __init__(self, config, device):
        super(PINN_RNN, self).__init__()
        
        layers = config['layers']
        input_size = layers[0]
        hidden_size = layers[1]
        num_layers = len(layers) - 2
        
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.fc_out = nn.Linear(hidden_size, layers[-1])

        # Physical parameters
        self.log_kg = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.log_kb = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32), requires_grad=not config['freeze_b'])
        self.g0_unconstrained = nn.Parameter(torch.tensor(0.0, dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.b0 = nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=not config['freeze_b'])
        self.Re_exp = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.log_lambda = nn.Parameter(torch.tensor(np.log(0.01), dtype=torch.float32), requires_grad=not config['freeze_g'])
        self.device = device

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        rnn_out, _ = self.rnn(x)
        last_step_out = rnn_out[:, -1, :]
        output = self.fc_out(last_step_out)
        return output

    def growth_rate(self, L, S, Re):
        kg = torch.exp(self.log_kg)
        g0 = 1 + 2 * torch.sigmoid(self.g0_unconstrained)
        lambda_ = torch.exp(self.log_lambda)
        S_term = torch.pow(torch.clamp(S, min=1e-9), g0)
        Re_term = torch.pow(torch.clamp(Re, min=1e-9), self.Re_exp)
        return kg * S_term * Re_term * (1 + lambda_ * L)

    def nucleation_rate(self, S):
        kb = torch.exp(self.log_kb)
        return kb * torch.pow(torch.clamp(S, min=1e-9), self.b0)

def get_pinn_loss(model, x_batch, n_batch, L0_tensor, sigma_delta_tensor, norm_params, config):
    is_rnn = config['architecture'] == 'RNN'

    x_batch.requires_grad_(True)

    with torch.backends.cudnn.flags(enabled=False):
        pred_n_norm = model(x_batch)

    grads_X = torch.autograd.grad(
        pred_n_norm,
        x_batch,
        torch.ones_like(pred_n_norm),
        create_graph=True,
        retain_graph=True
    )[0]

    if is_rnn:
        # Use the last time step for physics-informed constraints.
        x_physics = x_batch[:, -1, :]
        grads_last = grads_X[:, -1, :]
    else:
        x_physics = x_batch
        grads_last = grads_X

    dndL_norm, dndt_norm = grads_last[:, 0:1], grads_last[:, 1:2]

    pred_n = pred_n_norm * norm_params['n_std'] + norm_params['n_mean']

    L_comp_norm = x_physics[:, 0:1]
    S_comp_norm = x_physics[:, 2:3]
    Re_comp_norm = x_physics[:, 3:4]
    L_comp = L_comp_norm * norm_params['L_std'] + norm_params['L_mean']
    S_comp = S_comp_norm * norm_params['S_std'] + norm_params['S_mean']
    Re_comp = Re_comp_norm * norm_params['Re_std'] + norm_params['Re_mean']

    dndt = dndt_norm * (norm_params['n_std'] / (norm_params['t_std'] + 1e-9))
    dndL = dndL_norm * (norm_params['n_std'] / (norm_params['L_std'] + 1e-9))

    G_val = model.growth_rate(L_comp, S_comp, Re_comp)
    B_val = model.nucleation_rate(S_comp)
    dGdL = torch.autograd.grad(G_val, L_comp, torch.ones_like(G_val), create_graph=True, retain_graph=True)[0]
    delta_L_L0 = approx_dirac_delta(L_comp, L0_tensor, sigma_delta_tensor)

    pbe_residual = dndt + G_val * dndL + pred_n * dGdL - B_val * delta_L_L0
    
    loss_fn = nn.MSELoss()
    loss_data = loss_fn(pred_n_norm, n_batch)
    loss_phys = loss_fn(pbe_residual, torch.zeros_like(pbe_residual))
    nonneg_penalty = torch.mean(torch.relu(-pred_n))
    
    total_loss = config['lambda_data'] * loss_data + config['lambda_phys'] * loss_phys + config['lambda_nonneg'] * nonneg_penalty

    return total_loss, loss_data, loss_phys, nonneg_penalty

# ===================================================================
# 3. 主训练流程 (Main Training Function) - [已修改]
# ===================================================================

def train_pinn(config):
    # --- Constants and Hyperparameters ---
    L0, SIGMA_DELTA = 50.0, 2.0
    D, rho, u = 0.06, 1080, 0.000771
    
    # --- Data Processing and Normalization ---
    X_train_data, n_train_data = process_csv(
        config['train_file'], config['data_col'], rho, D, u,
        num_bins=config['num_bins'], sample_size=config['sample_size'], seed=config['seed'])
    X_val_data, n_val_data = process_csv(
        config['val_file'], config['data_col'], rho, D, u,
        num_bins=config['num_bins'], seed=config['seed'])
    
    # <--- 新增/修改 Start
    # 如果是继续训练，我们需要加载旧的 norm_params，而不是重新计算
    if config['resume_from'] and os.path.exists(config['resume_from']):
        print(f"Loading normalization parameters from checkpoint: {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'])
        # 确保 checkpoint 包含 norm_params
        if 'norm_params' in checkpoint:
            norm_params = checkpoint['norm_params']
        else:
            raise KeyError("Checkpoint does not contain 'norm_params'. Cannot resume with consistent normalization.")
    else:
        print("Calculating new normalization parameters from training data.")
        norm_params = {f'{key}_mean': X_train_data[:, i].mean() for i, key in enumerate(['L','t','S','Re','T','seed'])}
        norm_params.update({f'{key}_std': X_train_data[:, i].std() for i, key in enumerate(['L','t','S','Re','T','seed'])})
        norm_params['n_mean'], norm_params['n_std'] = n_train_data.mean(), n_train_data.std()
    # <--- 新增/修改 End

    n_train_norm = (n_train_data - norm_params['n_mean']) / (norm_params['n_std'] + 1e-9)
    n_val_norm = (n_val_data - norm_params['n_mean']) / (norm_params['n_std'] + 1e-9)

    def norm_X(X, params):
        X_norm = X.copy()
        for i, key in enumerate(['L','t','S','Re','T','seed']):
            std = params[f'{key}_std']
            if std > 1e-9: X_norm[:, i] = (X[:, i] - params[f'{key}_mean']) / std
        return X_norm

    X_train_norm, X_val_norm = norm_X(X_train_data, norm_params), norm_X(X_val_data, norm_params)

    # --- Setup Logging and PyTorch Components ---
    run_name = f"exp_{config['exp_id']}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(f'runs/{run_name}')
    
    model_save_dir = f'./models/{run_name}'
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    best_model_path = os.path.join(model_save_dir, 'best.pth')
    last_model_path = os.path.join(model_save_dir, 'last.pth')
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    print(f"Initializing model with architecture: {config['architecture']}")
    if config['architecture'] == 'FCNN':
        model = PINN(config, device).to(device)
    elif config['architecture'] == 'RNN':
        model = PINN_RNN(config, device).to(device)
    else:
        raise ValueError(f"Unknown architecture specified: {config['architecture']}")

    # --- Setup Optimizers and Schedulers ---
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(adam_optimizer, mode='min', factor=0.5, patience=500, min_lr=1e-6)

    # <--- 新增/修改 Start
    # --- Checkpoint Loading for Resuming Training ---
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    if config['resume_from'] and os.path.exists(config['resume_from']):
        print(f"Resuming training from: {config['resume_from']}")
        checkpoint = torch.load(config['resume_from'])
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer and scheduler states if they exist in the checkpoint
        if 'optimizer_state_dict' in checkpoint:
            adam_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Loaded optimizer state.")
        else:
            print("Warning: Optimizer state not found in checkpoint. Initializing new optimizer.")
            
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print("Loaded scheduler state.")
        else:
            print("Warning: Scheduler state not found in checkpoint. Initializing new scheduler.")
        
        # Set the starting epoch and best loss to continue tracking
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('loss', float('inf'))
        
        print(f"Resuming from Epoch {start_epoch}, with best validation loss so far: {best_val_loss:.6f}")
    else:
        print("Starting a new training session.")
    # <--- 新增/修改 End
 
    early_stop_patience = config.get('early_stop_patience', 0)
    early_stop_min_delta = config.get('early_stop_min_delta', 0.0)

    if config['architecture'] == 'RNN':
        seq_len = config.get('seq_len', 5)
        X_train_seq, n_train_seq = build_sequence_dataset(X_train_norm, n_train_norm, seq_len)
        X_val_seq, n_val_seq = build_sequence_dataset(X_val_norm, n_val_norm, seq_len)

        X_train_tensor = torch.from_numpy(X_train_seq).float().to(device)
        n_train_tensor = torch.from_numpy(n_train_seq).float().to(device)
        train_dataset = TensorDataset(X_train_tensor, n_train_tensor)

        X_val_tensor = torch.from_numpy(X_val_seq).float().to(device)
        n_val_tensor = torch.from_numpy(n_val_seq).float().to(device)
    else:
        X_train_tensor = torch.from_numpy(X_train_norm).float().to(device)
        n_train_tensor = torch.from_numpy(n_train_norm).float().to(device)
        train_dataset = TensorDataset(X_train_tensor, n_train_tensor)

        X_val_tensor = torch.from_numpy(X_val_norm).float().to(device)
        n_val_tensor = torch.from_numpy(n_val_norm).float().to(device)

    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)
    
    L0_tensor = torch.tensor(L0, device=device, dtype=torch.float32)
    sigma_delta_tensor = torch.tensor(SIGMA_DELTA, device=device, dtype=torch.float32)

    # --- Training Phase 1: Adam ---
    print("\n--- Starting Adam Optimization Phase ---")
    kinetic_params_history = []

    # <--- 新增/修改 Start: 调整训练循环的起始点
    for epoch in range(start_epoch, config['adam_epochs']):
    # <--- 新增/修改 End
        model.train()
        for x_batch, n_batch in train_loader:
            adam_optimizer.zero_grad()
            loss, _, _, _ = get_pinn_loss(model, x_batch, n_batch, L0_tensor, sigma_delta_tensor, norm_params, config)
            loss.backward()
            adam_optimizer.step()
        
        model.eval()
        train_loss, train_data, train_phys, train_nonneg = get_pinn_loss(model, X_train_tensor, n_train_tensor, L0_tensor, sigma_delta_tensor, norm_params, config)
        val_loss, val_data, val_phys, val_nonneg = get_pinn_loss(model, X_val_tensor, n_val_tensor, L0_tensor, sigma_delta_tensor, norm_params, config)
        val_loss_item = val_loss.item()
        
        # <--- 新增/修改 Start: 在保存时加入优化器和调度器的状态
        save_dict = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': adam_optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'norm_params': norm_params,
            'config': config,
            'epoch': epoch,
            'loss': val_loss_item
        }
        torch.save(save_dict, last_model_path)
        # <--- 新增/修改 End

        improvement = best_val_loss - val_loss_item
        is_best = improvement > early_stop_min_delta
        if is_best:
            best_val_loss = val_loss_item
            epochs_without_improvement = 0
            torch.save(save_dict, best_model_path) # 保存包含所有信息的字典
        else:
            epochs_without_improvement += 1

        if epoch % 100 == 0 or epoch == config['adam_epochs'] - 1:
            # Extract and log kinetic parameters
            kg = torch.exp(model.log_kg).item()
            kb = torch.exp(model.log_kb).item()
            g0 = (1 + 2 * torch.sigmoid(model.g0_unconstrained)).item()
            b0 = model.b0.item()
            Re_exp = model.Re_exp.item()
            lambda_ = torch.exp(model.log_lambda).item()
            
            # Save to history
            kinetic_params_history.append({
                'epoch': epoch, 'kg': kg, 'kb': kb, 'g0': g0, 'b0': b0, 'Re_exp': Re_exp, 'lambda': lambda_,
                'train_loss_total': train_loss.item(), 'val_loss_total': val_loss.item(),
                'train_loss_data': train_data.item(), 'val_loss_data': val_data.item(),
                'train_loss_phys': train_phys.item(), 'val_loss_phys': val_phys.item(),
                'train_loss_nonneg': train_nonneg.item(), 'val_loss_nonneg': val_nonneg.item()
            })
            
            log_message = (f"Epoch {epoch}/{config['adam_epochs']} | Train Loss: {train_loss.item():.6f} | Val Loss: {val_loss_item:.6f}")
            if is_best:
                log_message += " | New best model saved!"
            print(log_message)
            
            # Log losses and metrics to TensorBoard
            writer.add_scalar('Loss/Total/Train', train_loss.item(), epoch)
            writer.add_scalar('Loss/Total/Validation', val_loss.item(), epoch)
            writer.add_scalar('Loss/Data/Train', train_data.item(), epoch)
            writer.add_scalar('Loss/Data/Validation', val_data.item(), epoch)
            writer.add_scalar('Loss/Physics/Train', train_phys.item(), epoch)
            writer.add_scalar('Loss/Physics/Validation', val_phys.item(), epoch)
            writer.add_scalar('Loss/NonNeg/Train', train_nonneg.item(), epoch)
            writer.add_scalar('Loss/NonNeg/Validation', val_nonneg.item(), epoch)
            
            # Log kinetic parameters
            writer.add_scalar('KineticParams/kg', kg, epoch)
            writer.add_scalar('KineticParams/kb', kb, epoch)
            writer.add_scalar('KineticParams/g0', g0, epoch)
            writer.add_scalar('KineticParams/b0', b0, epoch)
            writer.add_scalar('KineticParams/Re_exp', Re_exp, epoch)
            writer.add_scalar('KineticParams/lambda', lambda_, epoch)

            scheduler.step(val_loss)

        if early_stop_patience > 0 and epochs_without_improvement >= early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}. No validation loss improvement greater than {early_stop_min_delta} for {early_stop_patience} epochs.")
            break

    # --- Final Saving ---
    kinetic_params_df = pd.DataFrame(kinetic_params_history)
    kinetic_params_csv_path = os.path.join(model_save_dir, 'training_history.csv')
    kinetic_params_df.to_csv(kinetic_params_csv_path, index=False)
    print(f"\nTraining history (kinetic parameters and losses) saved to {kinetic_params_csv_path}")

    writer.close()
    print(f"\nTraining finished. Best model saved at {best_model_path}, last model at {last_model_path}")

# ===================================================================
# 4. 执行入口 (Execution Entry Point) - [已修改]
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Run PINN Ablation Study for Crystallization PBE.")
    
    parser.add_argument('--architecture', type=str, default='FCNN', choices=['FCNN', 'RNN'], help='Type of network architecture to use (Fully Connected or Recurrent).')
    
    # <--- 新增/修改 Start
    # 新增 --resume_from 参数，用于指定 checkpoint 文件路径
    parser.add_argument('--resume_from', type=str, default=None, help='Path to the checkpoint file to resume training from (e.g., ./models/run_name/last.pth).')
    # <--- 新增/修改 End

    parser.add_argument('--exp_id', type=str, default='A-0_baseline', help='Experiment ID from the ablation plan.')
    parser.add_argument('--layers', type=int, nargs='+', default=[6, 60, 60, 60, 1], help='Network layer structure.')
    parser.add_argument('--activation', type=str, default='Tanh', choices=['Tanh', 'ReLU', 'SiLU'], help='Activation function.')
    parser.add_argument('--lambda_data', type=float, default=1.0, help='Weight for the data loss term.')
    parser.add_argument('--lambda_phys', type=float, default=3e5, help='Weight for the physics loss term.')
    parser.add_argument('--lambda_nonneg', type=float, default=1e-2, help='Weight for the non-negativity penalty.')
    parser.add_argument('--freeze_g', action='store_true', help='Freeze growth (G) parameters.')
    parser.add_argument('--freeze_b', action='store_true', help='Freeze nucleation (B) parameters.')
    parser.add_argument('--train_file', type=str, default=r'./train.csv', help='Path to the training data CSV.')
    parser.add_argument('--val_file', type=str, default=r'/home/kemove/WorkpaceP2/junjie/PINN/test/250329_45-20_0.3R.csv', help='Path to the validation data CSV.')
    parser.add_argument('--data_col', type=str, default='kde_DiameterFmax', choices=['kde_DiameterFmax', 'kde_DiameterFmin'], help='Column prefix for CSD data (dmax or dmin).')
    parser.add_argument('--sample_size', type=int, default=None, help='Number of data points to sample. Default is all.')
    parser.add_argument('--optimizer', type=str, default='Adam', choices=['Adam', 'Adam_LBFGS'], help='Optimizer strategy.')
    parser.add_argument('--adam_epochs', type=int, default=20000, help='Number of epochs for Adam.')
    parser.add_argument('--lbfgs_epochs', type=int, default=5000, help='Number of epochs for L-BFGS (if used).')
    parser.add_argument('--lr', type=float, default=5e-3, help='Initial learning rate for Adam.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--num_bins', type=int, default=60, help='Number of bins for CSD discretization.')
    parser.add_argument('--seq_len', type=int, default=5, help='Sequence length for RNN architecture inputs.')
    parser.add_argument('--early_stop_patience', type=int, default=0, help='Epochs without significant validation loss improvement before stopping (0 disables early stopping).')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='Minimum improvement threshold for validation loss to reset early stopping patience.')

    args = parser.parse_args()
    config = vars(args)
    
    set_seed(config['seed'])

    print("\n--- Running Experiment with Configuration ---")
    print(json.dumps(config, indent=4))
    print("-------------------------------------------\n")

    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')

    train_pinn(config)

if __name__ == "__main__":
    main()

