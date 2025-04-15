import random
import copy
import logging
import torch
import os
import gc
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Configure logging to output informational messages with timestamps
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)

# Define constants used in the training and generation process
total_samples = 1024           # Total number of samples to generate
batch_size = 64                # Batch size for optimization
lr_g = 0.1                     # Learning rate for generator input (latent space)
lr_z = 0.01                    # Learning rate for generator weights
iters = 4000                   # Number of iterations for training
latent_dim = 256              # Dimensionality of the latent vector
eps = 1e-6                    # Small epsilon value for numerical stability

# Directory paths for saving checkpoints
GEN_CHECKPOINT_DIR = "generator_checkpoints"
DATASET_CHECKPOINT_DIR = "dataset_checkpoint"

# Create checkpoint directories if they do not already exist
os.makedirs(GEN_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(DATASET_CHECKPOINT_DIR, exist_ok=True)

# Define the Generator neural network architecture
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Linear transformation from latent space to high-dimensional feature map
        self.linear = nn.Linear(256, 401408)  # 128 * 56 * 56

        # Convolutional layers and upsampling
        self.conv2d_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)

        # Batch normalization layers
        self.batch_norm2d_1 = nn.BatchNorm2d(128)
        self.batch_norm2d_2 = nn.BatchNorm2d(64)
        self.batch_norm2d_3 = nn.BatchNorm2d(3, affine=False)

        # Activation functions
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.tahn = nn.Tanh()

    def forward(self, x):
        # Project latent vector to a feature map and reshape
        x = self.linear(x)
        x = x.view(x.shape[0], 128, 56, 56)

        # Apply batch normalization and upsample
        x = self.batch_norm2d_1(x)
        x = self.upsample(x)

        # Convolve, normalize, and activate
        x = self.conv2d_1(x)
        x = self.batch_norm2d_2(x)
        x = self.leaky_reLU(x)

        # Further upsample and convolve to 3-channel output
        x = self.upsample(x)
        x = self.conv2d_2(x)

        # Apply tanh and final normalization
        x = self.tahn(x)
        x = self.batch_norm2d_3(x)

        return x

# Wrapper class to apply a random spatial shift before convolution
class SwingConv2d(nn.Module):
    def __init__(self, original_module):
        super().__init__()
        self.original_module = original_module
        self.reflection_pad_2d = nn.ReflectionPad2d(1)

    def forward(self, x):
        # Apply random spatial shift by padding and cropping
        height_shift = random.randint(0, 2)
        width_shift = random.randint(0, 2)
        x_pad = self.reflection_pad_2d(x)
        x = x_pad[:, :, height_shift:height_shift+x.shape[2], width_shift:width_shift+x.shape[3]] 
        return self.original_module(x)

# Hook class to capture input/output stats from batch norm layers
class BatchNormActivationHook():
    def __init__(self, module):
        self.input = None
        self.output = None
        self.handle = module.register_forward_hook(self.hook)  # Register hook to layer

    def hook(self, module, input, output):
        # Save input and output for analysis
        self.input = input[0]
        self.output = output

    def remove(self):
        # Remove hook when done
        self.handle.remove()

# Function to safely save checkpoints using temporary file renaming
def safe_save(state, checkpoint_path):
    tmp_checkpoint_path = checkpoint_path + '.tmp'
    torch.save(state, tmp_checkpoint_path)
    os.replace(tmp_checkpoint_path, checkpoint_path)

# Initialize generator, latent vector, optimizers, and schedulers
def initialize_generator():
    generator = Generator().cuda()
    z = torch.randn(batch_size, latent_dim, device='cuda', requires_grad=True)  # Latent input
    opt_z = optim.Adam([z], lr=lr_g)  # Optimizer for latent vectors
    opt_g = optim.Adam(generator.parameters(), lr=lr_z)  # Optimizer for generator weights

    # Learning rate schedulers to adapt learning rates over time
    scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, patience=100)
    scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)

    return generator, z, opt_z, opt_g, scheduler_z, scheduler_g

# Basic loss function: mean squared error normalized by batch size
def loss_function(A, B):
    return F.mse_loss(A, B, reduction='sum') / B.size(0)

# Computes statistical loss between generated data and real stats (mean/std)
def compute_loss(x, bn_stats, hooks, input_mean, input_std):
    # Compute mean and std of generated images
    data_std, data_mean = torch.std_mean(x, dim=[2, 3])
    mean_loss = loss_function(input_mean, data_mean)
    std_loss = loss_function(input_std, data_std)

    # Compute loss for each batch norm layer in the model
    for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
        bn_input = hook.input
        bn_std_input, bn_mean_input = torch.std_mean(bn_input, dim=[0, 2, 3])
        mean_loss += loss_function(bn_mean, bn_mean_input)
        std_loss += loss_function(bn_std, bn_std_input)

    return mean_loss, std_loss

def train_generator(model, generator, z, opt_z, opt_g, scheduler_z, scheduler_g, hooks, bn_stats, checkpoint_path, load_checkpoint=False):
    start_iteration = 0

    # Optionally load generator training from a checkpoint
    if load_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        z = checkpoint['z'].to('cuda')
        opt_z.load_state_dict(checkpoint['opt_z_state_dict'])
        opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        scheduler_z.load_state_dict(checkpoint['scheduler_z_state_dict'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        log.info(f"Loaded generator checkpoint from {checkpoint_path}, resuming at iteration {start_iteration}")

    # Define target mean and std tensors for generator output
    input_mean = torch.zeros(batch_size, 3, device='cuda')
    input_std = torch.ones(batch_size, 3, device='cuda')

    for iteration in range(start_iteration, iters):
        # Zero gradients for model and optimizers
        model.zero_grad()
        opt_z.zero_grad()
        opt_g.zero_grad()

        # Generate images from latent vectors
        x = generator(z)

        # Run generated images through model to collect hook activations
        model(x)

        # Compute statistical loss between generator output and batch norm stats
        mean_loss, std_loss = compute_loss(x, bn_stats, hooks, input_mean, input_std)
        total_loss = mean_loss + std_loss

        # Backpropagation and optimizer steps
        total_loss.backward()
        opt_z.step()
        opt_g.step()
        scheduler_z.step(total_loss.item())  # Update scheduler based on loss

        # Periodically log progress and save generator checkpoint
        if (iteration + 1) % 100 == 0:
            log.info(f'{iteration + 1}/{iters}, Loss: {total_loss:.3f}, Mean: {mean_loss:.3f}, Std: {std_loss:.3f}')
            scheduler_g.step()  # Decay generator learning rate
            safe_save({
                'generator_state_dict': generator.state_dict(),
                'z': z.detach().cpu(),
                'opt_z_state_dict': opt_z.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'scheduler_z_state_dict': scheduler_z.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'iteration': iteration,
            }, checkpoint_path)
            log.info(f"Generator checkpoint saved at iteration {iteration + 1} to {checkpoint_path}")

        # Early stopping if loss reaches acceptable level
        if total_loss.item() <= 0.039:
            log.info(f'{iteration + 1}/{iters}, Loss: {total_loss:.3f}, Mean: {mean_loss:.3f}, Std: {std_loss:.3f}')
            safe_save({
                'generator_state_dict': generator.state_dict(),
                'z': z.detach().cpu(),
                'opt_z_state_dict': opt_z.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'scheduler_z_state_dict': scheduler_z.state_dict(),
                'scheduler_g_state_dict': scheduler_g.state_dict(),
                'iteration': iteration,
            }, checkpoint_path)
            log.info(f"Generator checkpoint saved at iteration {iteration + 1} to {checkpoint_path}")
            break

    return x  # Return final generated batch


def distill_data(model, load_dataset_checkpoint=True, dataset_checkpoint_filename='dataset_checkpoint.pt'):
    # Clone and prepare the model for evaluation
    model = copy.deepcopy(model).cuda().eval()
    dataset = []
    hooks = []
    bn_stats = []

    # Modify convolutional layers and attach hooks to batch norm layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.stride != (1, 1):
            # Replace downsampling conv layers with SwingConv2d to introduce jitter
            *parent_names, submodule_name = name.split('.')
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)
            setattr(parent, submodule_name, SwingConv2d(module))
        elif isinstance(module, nn.BatchNorm2d):
            # Register hook to capture input/output from batch norm layers
            hooks.append(BatchNormActivationHook(module))
            bn_stats.append((module.running_mean.detach().clone().cuda(),
                             torch.sqrt(module.running_var + eps).detach().clone().cuda()))

    # Load dataset checkpoint if it exists
    dataset_checkpoint_path = os.path.join(DATASET_CHECKPOINT_DIR, dataset_checkpoint_filename)
    current_index = 0
    if load_dataset_checkpoint and os.path.exists(dataset_checkpoint_path):
        checkpoint_data = torch.load(dataset_checkpoint_path, map_location='cpu')
        dataset = checkpoint_data['dataset']
        current_index = checkpoint_data['current_index']
        log.info(f"Loaded dataset checkpoint from {dataset_checkpoint_path} with current index {current_index}")

    total_batches = total_samples // batch_size

    # Loop through remaining batches to generate and store synthetic data
    for i in range(current_index, total_batches):
        log.info(f'Generate Image ({i * batch_size}/{total_samples})')
        
        # Define checkpoint path for current batch's generator
        checkpoint_filename = f"checkpoint_generator_i{i}.pt"
        generator_checkpoint_path = os.path.join(GEN_CHECKPOINT_DIR, checkpoint_filename)
        
        # Initialize generator and optimizers
        generator, z, opt_z, opt_g, scheduler_z, scheduler_g = initialize_generator()

        # Train generator and retrieve output
        x = train_generator(model, generator, z, opt_z, opt_g, scheduler_z, scheduler_g,
                            hooks, bn_stats, generator_checkpoint_path, load_checkpoint=True)
        
        dataset.append(x.detach().cpu().clone())

        # Save dataset progress to checkpoint
        safe_save({
            'dataset': dataset,
            'current_index': i + 1,
        }, dataset_checkpoint_path)
        log.info(f"Dataset checkpoint saved at index {i} to {dataset_checkpoint_path}")

        # Clean up to free memory
        gc.collect()
        torch.cuda.empty_cache()
        log.info("Cleared CPU, GPU, and Python caches.")

    # Clean up hooks after use
    for hook in hooks:
        hook.remove()

    # Concatenate all batches into one dataset tensor
    dataset = torch.cat(dataset)

    # Save final full dataset
    safe_save({
        'dataset': dataset,
    }, dataset_checkpoint_path.replace('.pt', '_final.pt'))
    log.info(f"Final full dataset checkpoint saved.")

    return dataset
