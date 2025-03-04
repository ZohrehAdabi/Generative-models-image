from pathlib import Path
import torch
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import json


def plot_samples(samples, data, path):
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()
    
    if data.shape[0] > 2000:
        data = data[np.linspace(0, data.shape[0]-1, 2000, dtype=int), :]
    plt.clf()
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
        Xvis = intermediate_smpl[i].cpu()
        fig = plt.scatter(Xvis[:, 0], Xvis[:, 1], marker=".", color='m', animated=True)
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
        return fig,

    fig = plt.figure()
    anim = animation.FuncAnimation(fig, draw_frame, frames=intermediate_smpl.shape[0], interval=20, blit=True)
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
                # 'beta_schedule': params.beta_schedule, 
                'n_timesteps': params.n_timesteps, 
                'model': params.model,
                'dataset': params.dataset,
                'time_dim': params.time_dim,
                'data_dim': params.data_dim,
                'batch_size': params.batch_size,
                # 'total_size': params.total_size,
                # 'normalize': params.normalize,
                'n_epoch': params.n_epoch,
                'lr': params.lr,
                'seed': params.seed
                }

    if params.method in ['DDPM']:
        config_dict['beta_schedule'] = params.beta_schedule
    if params.method in ['DDPM-Unlearning']:
        config_dict['beta_schedule'] = params.beta_schedule
        config_dict['unlrn_weight'] = params.unlrn_weight
        config_dict['norm_type'] = params.norm_type
        config_dict['std_loss'] = params.std_loss
        config_dict['std_weight'] = params.std_weight
    if params.method == 'DDPM-Hidden':
        config_dict['hid_inp_size'] = params.hid_inp_size
        config_dict['beta_schedule'] = params.beta_schedule
    if params.method == 'RBM':
        config_dict['kl_weight'] = params.kl_weight
        config_dict['allX'] = params.allX
    if params.method == 'RBM-Hidden':
        config_dict['burn_in'] = params.burn_in
        config_dict['lastX'] = params.lastX    
        config_dict['fixed_enc'] = params.fixed_enc    
        config_dict['noise_start'] = params.noise_start  
        
    if params.method == 'BoostingOne':
        config_dict['beta_schedule'] = params.beta_schedule
        config_dict['pred_goal'] =  params.pred_goal
        # config_dict['gamma'] =  params.gamma

    # if params.method in ['GSN']:
    #     config_dict[''] =  params.

    if params.method == 'Boosting':
        config_dict['beta_schedule'] = params.beta_schedule
        config_dict['learner_inp'] =  params.learner_inp
        config_dict['innr_epoch'] =  params.innr_epoch
        config_dict['grad_type'] =  params.grad_type
        config_dict['gamma'] =  params.gamma

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

def create_hdf_file_training(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples, resume_training=False):
            
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

    for i, img in enumerate(new_data):
        data[f'data_{i}'] = img.flatten()

    dfs = pd.DataFrame(data)#.astype(int)

    sampling_timesteps = select_steps(n_timestep_smpl, n_sel_time)
    
    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time],  
                                                                     'n_samples': [n_samples], 'image_shape': [str(new_data.shape[1:])[1:-1].replace(' ', '')]}), format='t')
        
        hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')

    f = pd.HDFStore(file_loss, 'w')
    f.close()
    
    return file_loss, file_sample

def create_hdf_file(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples, train_name=None, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)
    
    if train_name is not None:
        expr_id += train_name 

    file_sample = str(path_save) +  (f'/{expr_id}_df_sample.h5' if test_name is None else f'/{expr_id}_df_sample_{test_name}.h5')

    if dataset.shape[0] > 36:
        dataset = dataset[np.linspace(0, dataset.shape[0]-1, 36, dtype=int), :, :, :]
    dataset = dataset.cpu().numpy()   # permute
    dataset = denormalize(dataset)
    dataset = dataset[None, :, :, :, :]

    new_data = construct_image_grid(1, dataset)    
    # flat_img_len = np.prod(new_data.shape[1:])
    data = {}
    data[f'data_{0}'] = new_data[0].flatten()

    dfs = pd.DataFrame(data)
    
    sampling_timesteps = select_steps(n_timestep_smpl+1, n_sel_time)

    with pd.HDFStore(file_sample, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time],  
                                                                     'n_samples': [n_samples], 'image_shape': [str(new_data.shape[1:])[1:-1].replace(' ', '')]}), format='t')
        
        hdf_store_samples.append(key=f'df/sampling_timesteps', value=pd.DataFrame({'time': sampling_timesteps}), format='t')


    return file_sample

def select_steps(n_timestep_smpl, n_sel_time):
    
    steps = [0]
    num_tim_stp = n_timestep_smpl #intermediate_smpl.shape[0]
    prcntg_reducing = num_tim_stp // n_sel_time
    start = 1
    [steps.append(t) for t in np.linspace(start, num_tim_stp-2, n_sel_time-2, dtype=int)]
    steps.append(num_tim_stp-1)
    

    return steps[::-1]

def select_samples_for_plot(intermediate_smpl, n_samples, n_timestep_smpl, n_sel_time, with_time=False):
    

    # if intermediate_smpl.shape[0] == n_timestep_smpl:
    #     n_timestep_smpl -= 1
    n_timestep_smpl = intermediate_smpl.shape[0]
    steps = select_steps(n_timestep_smpl, n_sel_time)
    selected = intermediate_smpl[steps]#.reshape(-1, *intermediate_smpl.shape[2:])
    # attention to time reversibility in sampling time
    # selected_with_time = np.hstack([selected, np.repeat(steps, n_samples).reshape(-1, 1)])

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



def create_save_dir_regression_training(save_dir, expr_id):

    p = Path(save_dir)
    
    path_save = p / 'saved_models'
    path_save.mkdir(parents=True, exist_ok=True)
    path_save = p / 'plot_predictions_training' / expr_id
    path_save.mkdir(parents=True, exist_ok=True)

def create_save_dir_regression(save_dir):

    p = Path(save_dir)

    path_save = p / 'plot_predictions'
    path_save.mkdir(parents=True, exist_ok=True)

def create_hdf_file_regression_training(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs_training' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_loss = str(path_save) + f'/{expr_id}_df_loss_per_epoch.h5'
    file_prediction = str(path_save) + f'/{expr_id}_df_prediction_per_epoch.h5'
    
    df_loss_per_epoch = pd.DataFrame(columns=['epoch', 'itr', 'loss_itr', 'loss_epoch'])
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])

    
    
    with pd.HDFStore(file_prediction, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_samples': [n_samples]}), format='t')
        

    return file_loss, file_prediction, df_loss_per_epoch

def create_hdf_file_regression(save_dir, expr_id, dataset, n_timestep_smpl, n_sel_time, n_samples, test_name=None):
            
    p = Path(save_dir)
    path_save = p / 'saved_hdfs' 
    path_save.mkdir(parents=True, exist_ok=True)
    file_prediction = str(path_save) +  (f'/{expr_id}_df_prediction.h5' if test_name is None else f'/{expr_id}_df_prediction_{test_name}.h5')
    # df_samples_per_epoch = pd.DataFrame(columns=['epoch', 'data_x', 'data_y', 'intermediate_samples_x', 'intermediate_samples_y'])
    data = dataset
    if dataset.shape[0] > 2000:
        data = dataset[np.linspace(0, dataset.shape[0]-1, 2000, dtype=int), :]
    dfs = pd.DataFrame(data, columns=['data_x', 'data_y'])
    # if dfs.shape[0] > 2000:
    #     dfs = dfs.loc[np.linspace(0, dfs.shape[0]-1, 2000, dtype=int), :]
    
    
    with pd.HDFStore(file_prediction, 'w') as hdf_store_samples:
        hdf_store_samples.append(key=f'df/data', value=dfs, format='t', data_columns=True)
        hdf_store_samples.append(key=f'df/info', value=pd.DataFrame({'n_sel_time': [n_sel_time], 'n_samples': [n_samples]}), format='t')
        


    return file_prediction


def get_fake_sample_grid(grid_size=10):
    
    xs = torch.linspace(-2.5, 2.5, steps=grid_size)
    ys = torch.linspace(-2.5, 2.5, steps=grid_size)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='xy')
    fake_images = torch.tensor([])
    for i, (x, y) in enumerate(zip(grid_x, grid_y)):
        f_img = torch.hstack([x.view(-1, 1), y.view(-1, 1)])
        fake_images = torch.cat([fake_images, f_img])
    return fake_images, grid_x, grid_y


def extract_at_time_t(var, t):
    
    selected = torch.gather(var, 0, t)
    reshape = [t.shape[0], -1] 
    return selected.reshape(*reshape)#.to(x.device)

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