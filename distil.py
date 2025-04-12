import random
import copy
import logging
import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    log.addHandler(handler)


total_samples = 1024
batch_size = 64
lr_g=0.1
lr_z=0.01
iters=4000
latent_dim = 256
eps = 1e-6


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(256, 401408)
        self.conv2d_1 = nn.Conv2d(128, 64, 3, stride=1, padding=1)
        self.conv2d_2 = nn.Conv2d(64, 3, 3, stride=1, padding=1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.batch_norm2d_1 = nn.BatchNorm2d(128)
        self.batch_norm2d_2 = nn.BatchNorm2d(64)
        self.batch_norm2d_3 = nn.BatchNorm2d(3, affine=False)
        self.leaky_reLU = nn.LeakyReLU(0.2)
        self.tahn = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = x.view(x.shape[0], 128, 56, 56)
        x = self.batch_norm2d_1(x)
        x = self.upsample(x)
        x = self.conv2d_1(x)
        x = self.batch_norm2d_2(x)
        x = self.leaky_reLU(x)
        x = self.upsample(x)
        x = self.conv2d_2(x)
        x = self.tahn(x)
        x = self.batch_norm2d_3(x)
        return x
    
class SwingConv2d(nn.Module):
    def __init__(self, original_module):
        super().__init__()

        self.original_module = original_module
        self.reflection_pad_2d = nn.ReflectionPad2d(1)

    def forward(self, x):
        height_shift = random.randint(0, 2)
        width_shift = random.randint(0, 2)
        x_pad = self.reflection_pad_2d(x)
        x = x_pad[:, :, height_shift:height_shift+x.shape[2], width_shift:width_shift+x.shape[3]] 
        return self.original_module(x)
    
class BatchNormActivationHook():
    def __init__(self, module):
        self.input = None
        self.output = None
        self.handle = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.input = input[0]
        self.output = output

    def remove(self):
        self.handle.remove()

def initialize_generator():
    generator = Generator().cuda()
    z = torch.randn(batch_size, latent_dim, device='cuda', requires_grad=True)
    opt_z = optim.Adam([z], lr=lr_g)
    opt_g = optim.Adam(generator.parameters(), lr=lr_z)

    scheduler_z = optim.lr_scheduler.ReduceLROnPlateau(opt_z, min_lr=1e-4, patience=100)
    scheduler_g = optim.lr_scheduler.ExponentialLR(opt_g, gamma=0.95)

    return generator, z, opt_z, opt_g, scheduler_z, scheduler_g

def loss_function(A, B):
    return F.mse_loss(A, B, reduction='sum') / B.size(0)

def compute_loss(x, bn_stats, hooks):
    input_mean = torch.zeros(batch_size, 3, device='cuda')
    input_std = torch.ones(batch_size, 3, device='cuda')

    data_std, data_mean = torch.std_mean(x, dim=[2, 3])
    mean_loss = loss_function(input_mean, data_mean)
    std_loss = loss_function(input_std, data_std)

    for (bn_mean, bn_std), hook in zip(bn_stats, hooks):
        bn_input = hook.input
        bn_std_input, bn_mean_input = torch.std_mean(bn_input, dim=[0, 2, 3])
        mean_loss += loss_function(bn_mean, bn_mean_input)
        std_loss += loss_function(bn_std, bn_std_input)

    return mean_loss, std_loss

def train_generator(model, generator, z, opt_z, opt_g, scheduler_z, scheduler_g, hooks, bn_stats, checkpoint_path, load_checkpoint=False, iters=1000):
    start_iteration = 0
    if load_checkpoint and checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        opt_z.load_state_dict(checkpoint['opt_z_state_dict'])
        opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        start_iteration = checkpoint['iteration'] + 1
        log.info(f"Loaded checkpoint from {checkpoint_path}, resuming at iteration {start_iteration}")

    for iteration in range(start_iteration, iters):
        model.zero_grad()
        opt_z.zero_grad()
        opt_g.zero_grad()

        x = generator(z)
        model(x)

        mean_loss, std_loss = compute_loss(x, bn_stats, hooks)
        total_loss = mean_loss + std_loss

        total_loss.backward()
        opt_z.step()
        opt_g.step()
        scheduler_z.step(total_loss.item())

        if (iteration + 1) % 100 == 0:
            log.info(f'{iteration + 1}/{iters}, Loss: {total_loss:.3f}, Mean: {mean_loss:.3f}, Std: {std_loss:.3f}')
            scheduler_g.step()

        if (iteration + 1) % 100 == 0 and checkpoint_path:
            torch.save({
                'generator_state_dict': generator.state_dict(),
                'opt_z_state_dict': opt_z.state_dict(),
                'opt_g_state_dict': opt_g.state_dict(),
                'iteration': iteration,
            }, checkpoint_path)
            log.info(f"Checkpoint saved at iteration {iteration + 1}")

    return x

def distill_data(model, load_dataset_checkpoint=False, dataset_checkpoint_path='dataset_checkpoint.pt'):
    model = copy.deepcopy(model).cuda().eval()
    dataset = []
    hooks = []
    bn_stats = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and module.stride != (1, 1):
            *parent_names, submodule_name = name.split('.')
            parent = model
            for parent_name in parent_names:
                parent = getattr(parent, parent_name)
            setattr(parent, submodule_name, SwingConv2d(module))
        elif isinstance(module, nn.BatchNorm2d):
            hooks.append(BatchNormActivationHook(module))
            bn_stats.append((module.running_mean.detach().clone().cuda(),
                             torch.sqrt(module.running_var + eps).detach().clone().cuda()))

    # Check if we need to load an existing dataset checkpoint
    current_index = 0
    if load_dataset_checkpoint and os.path.exists(dataset_checkpoint_path):
        checkpoint_data = torch.load(dataset_checkpoint_path)
        dataset = checkpoint_data['dataset']
        current_index = checkpoint_data['current_index']
        log.info(f"Loaded dataset checkpoint from {dataset_checkpoint_path} with current index {current_index}")

    # Assume total_samples and batch_size are defined elsewhere.
    total_batches = total_samples // batch_size

    for i in range(current_index, total_batches):
        log.info(f'Generate Image ({i * batch_size}/{total_samples})')
        
        # Get new generator and its training components
        checkpoint_path = f'checkpoint_generator_i{i}.pt'
        generator, z, opt_z, opt_g, scheduler_z, scheduler_g = initialize_generator()

        # Load from generator checkpoint if available as needed:
        x = train_generator(model, generator, z, opt_z, opt_g, scheduler_z, scheduler_g,
                            hooks, bn_stats, checkpoint_path, load_checkpoint=True)
        
        dataset.append(x.detach().clone())

        # Save dataset checkpoint after each batch
        torch.save({
            'dataset': dataset,
            'current_index': i + 1,  # increment index for next batch
        }, dataset_checkpoint_path)
        log.info(f"Dataset checkpoint saved at index {i + 1}")

    # Remove hooks after processing
    for hook in hooks:
        hook.remove()

    dataset = torch.cat(dataset)
    return dataset