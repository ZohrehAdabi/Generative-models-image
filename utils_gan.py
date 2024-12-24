from pathlib import Path
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


def save_plot_fig_stats(stats, labels, colors, title, file_name, xmax=None):

    def plot_fig_loss(stats_hist, label, color):
        plt.plot(stats_hist, label=label, color=color) # marker='o', linestyle='dashed',linewidth=2, markersize=12)
    plt.figure()
    for stats, l, c in zip(stats, labels, colors):
        plot_fig_loss(stats, l, c)
    if xmax is not None: plt.hlines(y=.5, xmin=0, xmax=xmax, colors='gray', linestyles='--', lw=2, label=r'$D^*$')
    plt.legend()
    plt.title(title)
    plt.savefig(file_name)
    plt.close()

def plot_samples(samples, data, path):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    if data.shape[0] > 2000:
        data = data[np.linspace(0, data.shape[0]-1, 2000, dtype=int), :]
    plt.figure(dpi=150)
    # plt.xticks([])
    # plt.yticks([])
    plt.scatter(data[:, 0], data[:, 1], label='data', c='C9', marker='.', s=20)
    plt.scatter(samples[:, 0], samples[:, 1], label='sample', c='C4', marker='.', s=20)
     
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    # plt.xlim(-4, 4)
    # plt.ylim(-4, 4)
    plt.legend()
    plt.savefig(path)
    plt.close()

def save_animation(intermediate_smpl, path):
    def draw_frame(i):
        plt.clf()
        Xvis = intermediate_smpl[i]
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker=".", color='m', animated=True)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        return fig,

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=False)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=500)
    anim.save(path, writer=writer)
    # anim.save(path, fps=10)

def save_animation_quiver(intermediate_smpl, grad_g_z, dataset, grid_x, grid_y, epoch, path):

    def draw_frame(i):
        plt.clf()
        plt.scatter(dataset[:, 0], dataset[:, 1], label='data', c='C9', marker='.', s=20, animated=True)
        Xvis = intermediate_smpl[i]
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1],  label='sample', marker=".", color='C4', animated=True)
        # plt.xlim([-2.5, 2.5])
        # plt.ylim([-2.5, 2.5])
        # grad_g_z = intermediate_grad_g_z[i]
        u, v = grad_g_z[:, 0], grad_g_z[:, 1]
        magnitude = np.hypot(u, v)
        # plt.scatter(self.grid_x, self.grid_y, color='k', s=1)

        
        plt.quiver(grid_x, grid_y, u, v, magnitude, scale=None, cmap='plasma', pivot='tail', angles='xy', units='width', animated=True) 
        plt.title(f'epoch: {epoch+1:<6}, t: {i} | '+r'$\frac{\partial{D(G(z))}}{\partial{G(z)}}$')
        # plt.close()
        return fig,

    fig = plt.figure(dpi=150)
    # grad_g_z = intermediate_grad_g_z[0]
    # u, v = grad_g_z[:, 0], grad_g_z[:, 1]
    # magnitude = np.hypot(u, v)
    # plt.quiver(grid_x, grid_y, u, v, magnitude, scale=0.025, cmap='plasma', pivot='tail', angles='xy', units='width', animated=True) 
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=True)
    # Writer = animation.writers['ffmpeg']
    # writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    # anim.save(path, writer=writer)
    anim.save(path, fps=10)

def evaluate_scores():

    # inception_score
    # frechet_inception_distance
    # mode_score
    # rkl_divergence
    # missing modes
    
    pass

def save_config_json(save_dir, params, expr_id):

    save_config = f'{save_dir}/configs_train/'
    p = Path(save_config)
    p.mkdir(parents=True, exist_ok=True)
    save_config = str(p / f'{expr_id}.json')
    config_dict = {'save_dir': save_dir,
                   # 'method': params.method,
                    'model': params.model,
                    'dataset': params.dataset,
                    'z_dim': params.z_dim,
                    'data_dim': params.data_dim,
                    'loss_dsc': params.loss_dsc,
                    'loss_gen': params.loss_gen,
                    'batch_size': params.batch_size,
                    'normalize': params.normalize,
                    'n_epoch': params.n_epoch,
                    'lr_dsc': params.lr_dsc,
                    'lr_gen': params.lr_gen,
                    'seed': params.seed
                    }
    if params.method == 'GAN-wo-G':
        config_dict['lr_fk_dt'] =  params.lr_fk_dt
        
    with open(save_config, 'w') as outfile:
        json.dump(config_dict, outfile, indent=4)


def create_save_dir_training(save_dir, expr_id):

    p = Path(save_dir)
    
    path_save = p / 'saved_models'
    path_save.mkdir(parents=True, exist_ok=True)
    path_save = p / 'plot_samples_training' / expr_id
    path_save.mkdir(parents=True, exist_ok=True)

def create_save_dir(save_dir):

    p = Path(save_dir)

    path_save = p / 'plot_samples'
    path_save.mkdir(parents=True, exist_ok=True)

def construct_image_grid(n_timestep, images):

    k_row, k_col = 6, 6
    height, width, channels = images.shape[2:] # height == width
    p_row = 3
    p_col = 3
    new_img_data = np.zeros([n_timestep, height*k_row + p_row*(k_row-1), width*k_col + p_col*(k_col-1),
                            channels], dtype=np.uint8)
    pad = np.ones([1, 1, channels], dtype=np.uint8) * 200

    indices_col = [width* i for i in range(1, k_col)]
    indices_col = np.tile(indices_col, p_col)
    indices_row = [height* i for i in range(1, k_row)]
    indices_row = np.tile(indices_row, p_row)
    for i in range(n_timestep):
        img = images[i]
        img = img.reshape([k_row, -1, height, width, channels])
        img = np.concatenate(img, axis=1)
        img = np.concatenate(img, axis=1)
        img = np.insert(img, indices_col, pad, axis=1)
        img = np.insert(img, indices_row, pad, axis=0)
        new_img_data[i] = img

    return new_img_data

def create_hdf_file_training(save_dir, expr_id, dataset, n_samples, n_timestep_smpl=-1, n_sel_time=-1, resume_training=False):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs_training' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_loss = str(path_save) + f'/{expr_id}_df_loss_per_epoch.h5'
    file_sample = str(path_save) + f'/{expr_id}_df_sample_per_epoch.h5'
    
    if resume_training:
        return file_loss, file_sample
    
    
    if dataset.shape[0] > 36:
        dataset = dataset[np.linspace(0, dataset.shape[0]-1, 36, dtype=int), :, :, :]

    dataset = dataset.cpu().numpy()   # permute
    dataset = denormalize(dataset)
    dataset = dataset[None, :, :, :, :]

    new_data = construct_image_grid(1, dataset)    
    flat_img_len = np.prod(new_data.shape[1:])
    data = {'epoch': [0]  * flat_img_len}
    data[f'data_{0}'] = new_data[0].flatten()

    dfs = pd.DataFrame(data)#.astype(int)

    if n_sel_time!= -1: 
        sampling_timesteps = select_steps(n_timestep_smpl, n_sel_time)
    
    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples], 
                                                                     'n_sel_time':[n_sel_time], 'image_shape': [str(new_data.shape[1:])[1:-1].replace(' ', '')]}), format='t')
        if n_sel_time!= -1: 
            hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')

    f = pd.HDFStore(file_loss, 'w')
    f.close()
    
    return file_loss, file_sample

def create_hdf_file(save_dir, expr_id, dataset, n_samples, n_timestep_smpl=-1, n_sel_time=-1, train_name=None, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)

    if train_name is not None:
        expr_id += train_name 

    file_sample = str(path_save) +  (f'/{expr_id}_df_sample.h5' if test_name is None else f'/{expr_id}_df_sample_{test_name}.h5')
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    if dataset.shape[0] > 36:
        dataset = dataset[np.linspace(0, dataset.shape[0]-1, 36, dtype=int), :, :, :]

    dataset = dataset.cpu().numpy()   # permute
    dataset = denormalize(dataset)
    dataset = dataset[None, :, :, :, :]

    new_data = construct_image_grid(1, dataset)    
    # flat_img_len = np.prod(new_data.shape[1:])
    data = {}
    data[f'data_{0}'] = new_data[0].flatten()

    dfs = pd.DataFrame(data)#.astype(int)

    if n_sel_time!= -1: 
        sampling_timesteps = select_steps(n_timestep_smpl, n_sel_time)
    
    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples], 
                                                                     'n_sel_time':[n_sel_time], 'image_shape': [str(new_data.shape[1:])[1:-1].replace(' ', '')]}), format='t')
        if n_sel_time!= -1: 
            # hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': n_timestep_smpl}), format='t')
            hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')
    
    return file_sample

def get_fake_sample_grid(grid_size=10):
    
    xs = torch.linspace(-2.5, 2.5, steps=grid_size)
    ys = torch.linspace(-2.5, 2.5, steps=grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    fake_images = torch.tensor([])
    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        f_img = torch.hstack([x.view(-1, 1), y.view(-1, 1)])
        fake_images = torch.cat([fake_images, f_img])
    return fake_images, grid_x, grid_y

def select_steps(n_timestep_smpl, n_sel_time):
    
    steps = [0]
    num_tim_stp = n_timestep_smpl #intermediate_smpl.shape[0]
    prcntg_reducing = num_tim_stp // n_sel_time
    start = 1
    [steps.append(t) for t in np.linspace(start, num_tim_stp-2, n_sel_time-2, dtype=int)]
    steps.append(num_tim_stp-1)
    

    return steps

def select_samples_for_plot(intermediate_smpl, n_samples, n_timestep_smpl, n_sel_time, with_time=True):
    

    # if intermediate_smpl.shape[0] == n_timestep_smpl:
    #     n_timestep_smpl -= 1
    n_timestep_smpl = intermediate_smpl.shape[0]
    steps = select_steps(n_timestep_smpl, n_sel_time)
    selected = intermediate_smpl[steps, :, :].reshape(-1, intermediate_smpl.shape[-1])
    # attention to time reversibility in sampling time
    selected_with_time = np.hstack([selected, np.repeat(steps, n_samples).reshape(-1, 1)])
    
    if with_time:
        return selected_with_time
    else:
        return selected

def denormalize(img, dataset_name=None):

    if dataset_name == 'MNIST' or img.shape[1] == 1:

        mean = np.array([0.5])  
        std = np.array([0.5])
        out_img = img * std + mean
        out_img = np.clip(out_img, a_min=0, a_max=1)
        img = out_img.transpose(0, 2, 3, 1)
        gray_img = (img * 255).astype('uint8')
        return gray_img
    
    if dataset_name in ['CIFAR10', 'CelebA'] or img.shape[1] == 3:

        mean = np.array([0.5, 0.5, 0.5]).reshape([3, 1, 1])  
        std = np.array([0.5, 0.5, 0.5]).reshape([3, 1, 1])  
        out_img = img * std + mean
        denorm_image = np.clip(out_img, a_min=0, a_max=1) 
        img = denorm_image.transpose(0, 2, 3, 1)  # Convert to HWC format 
        color_img = (img* 255).astype('uint8')
        return color_img



def read_launch_for_terminal():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_terminal.txt', 'w') as f:
        for line in params_txt:
            if '//' in line and 'python' in line:
                line = line.replace('//', '')
                line = line.strip()
                f.write(f"{line}\n\n")

if __name__=='__main__':

    # from beta_scheduling import select_beta_schedule
    # n_timesteps = 10
    # beta = select_beta_schedule('linear', n_timesteps=n_timesteps).to('cuda')
    # x = torch.tensor(np.random.randn(20).reshape(-1, 2)).to('cuda')
    # t = torch.randint(low=0, high=n_timesteps, size=[x.shape[0]]).to('cuda')
    # bt = extract_at_time_t(beta, t)
    print()
    read_launch_for_terminal()