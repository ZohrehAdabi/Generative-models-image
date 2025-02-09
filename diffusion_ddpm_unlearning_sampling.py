
from common_imports_diffusion import *

from utils_diffusion import denormalize
from utils_diffusion import construct_image_grid

class DDPM_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None, 
            betas=None, 
            n_timestep_smpl=100, 
            expr_id=None, 
            training=False
        ):
        super(DDPM_Sampler, self).__init__()

        self.n_timestep_smpl= n_timestep_smpl
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
        self.evaluating = False
 
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 
    

    def sampling(self, n_samples=5, data_dim=2, params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()

        sample = torch.randn(n_samples, *data_dim).to(device)
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, *data_dim[::-1]])
        intermediate_smpl[-1] = denormalize(sample.cpu().numpy())
        
        noises = np.empty([self.n_timestep_smpl, n_samples, *data_dim[::-1]])
        nois_norm_list = []
        

        timestp = tqdm(timesteps, disable=self.training)
        for t in timestp: 
            times = torch.repeat_interleave(t, n_samples).reshape(-1, 1).long().to(device)
            with torch.no_grad():
                sample, predicted_noise = self.sampling_ddpm_step(sample, times)
                nois_norm_list.append([torch.linalg.norm(predicted_noise, axis=1).mean().cpu().numpy()])
                
                noises[t] = predicted_noise.permute(0, 2, 3, 1).cpu().numpy()         
                intermediate_smpl[t] = denormalize(sample.cpu().numpy())

        timestp.close() 
        sample_zero = intermediate_smpl[0]
        noise_norm = np.concatenate(nois_norm_list)  
        
        if not self.training and not self.evaluating: self.save_result(sample_zero, intermediate_smpl, noise_norm, noises, params)

        return sample_zero, intermediate_smpl, noises, noise_norm

    def evaluation(self, n_samples=5, real_dataloader=None, data_dim=2, params=None, device='cuda'):

        batch_size = params.batch_size
        n_batch = (n_samples//batch_size)+1
        n_smpl = (n_batch) * batch_size
        samples = np.empty([n_smpl, *data_dim[::-1]])
        self.evaluating = True
        for b in range(n_batch):
            sample_zero, _, _, _ = self.sampling(n_samples=batch_size, data_dim=data_dim, params=params, device=device)
            samples[b*batch_size: (b+1)*batch_size] = sample_zero
        
        import torchvision.models as models
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode
        inception_weights = models.Inception_V3_Weights.DEFAULT
        # inception_transforms = model_weights.transforms()
        inception_model = models.inception_v3(weights=inception_weights)
        inception_model.fc = nn.Identity()

        sample_loader = DataLoader(samples, batch_size=batch_size, shuffle=True, num_workers=1)
        
        if not params.use_torchmetric:
            transform_inception = transforms.Compose([
                transforms.ToPILImage(),  # Tensor to PIL Image
                transforms.Resize((299, 299), interpolation=InterpolationMode.BILINEAR),  # Resize to (299, 299)
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                transforms.ToTensor(),  # Convert back to tensor [0, 1]
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
            inception_model.to(device)
            inception_model.eval()
            features = np.zeros((len(dataloader.dataset), 2048))

            for i, x in enumerate(sample_loader):
                # Apply the transformation to all images in the batch
                transformed_x = torch.stack([transform_inception(img) for img in x.permute(0, 3, 1, 2)]).to(device)
                ftur = inception_model(transformed_x).cpu().numpy()
                features[i*batch_size : (i+1)*batch_size , :] = ftur

        else:
            from torchmetrics.image.fid import FrechetInceptionDistance
            fid = FrechetInceptionDistance(feature=64, reset_real_features=False, 
                                           input_img_size=(3, 299, 299), normalize=True).to(device)
            # generate two slightly overlapping image intensity distributions
            # imgs_dist1 = torch.randint(0, 200, (100, 3, 299, 299), dtype=torch.uint8)
            # imgs_dist2 = torch.randint(100, 255, (100, 3, 299, 299), dtype=torch.uint8)
            transform_inception = transforms.Compose([
                transforms.ToPILImage(),  # Tensor to PIL Image
                transforms.Resize((299, 299), interpolation=InterpolationMode.BILINEAR),  # Resize to (299, 299)
                transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
                transforms.ToTensor(),  # Convert back to tensor [0, 1]
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
            ])
            for b, ((x, _), smpl) in enumerate(zip(real_dataloader, sample_loader)):
                x = denormalize(x.cpu().numpy())
                transformed_x = torch.stack([transform_inception(img) for img in x]).to(device)
                transformed_smpl = torch.stack([transform_inception(img) for img in smpl.permute(0, 3, 1, 2)]).to(device)
                fid.update(transformed_x, real=True)
                fid.update(transformed_smpl, real=False)
            fid_value = fid.compute().cpu().numpy()

        return fid_value

    def samplinga_by_selected_model_t(self, sample_t=None, selected_t=0, n_timestep_smpl=40, 
                                      n_samples=5, data_dim=2, normalize=False, scaler=None, params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()

        if sample_t is None:
            sample = torch.randn(n_samples, data_dim).to(device)
        else:
            sample = torch.from_numpy(sample_t).to(torch.float).to(device)
            
        selected_t = torch.tensor([selected_t])
        
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, data_dim])
        intermediate_smpl[-1] = sample.cpu()
        noises = np.empty([self.n_timestep_smpl, n_samples, data_dim])
        noise_norm_list = []
        
        timestp = tqdm(timesteps, disable=self.training)
        for t in timestp: 
            times = torch.repeat_interleave(selected_t, n_samples).reshape(-1, 1).long().to(device)
            with torch.no_grad():
                sample, predicted_noise = self.sampling_ddpm_step_t(sample, times, t)
                noise_norm_list.append([torch.linalg.norm(predicted_noise, axis=1).mean().cpu().numpy()])
                noises[t, :, :] = predicted_noise.cpu().numpy()
            if normalize:
                 intermediate_smpl[t, :, :] = scaler.inverse_transform(sample.cpu())
            else:
                intermediate_smpl[t, :, :] = sample.cpu()
            # intermediate_smpl[t, :, :] = sample.cpu()
        timestp.close() 
        sample_zero = intermediate_smpl[0]
        noise_norm = np.concatenate(noise_norm_list)    
        self.save_result(sample_zero, intermediate_smpl, noise_norm, noises, params)

        return sample_zero, intermediate_smpl

    def sampling_ddpm_step(self, x, t):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        predicted_noise = self.model(x, t)

        mu = self.compute_mu_theta(x, t, predicted_noise)
        sigma2 = self.reverse_variance(t)
        x = mu + torch.sqrt(sigma2)[:, :, None, None] * torch.randn_like(x) * int((t>0).all())

        return x, predicted_noise
    
    def sampling_ddpm_step_t(self, x, time, t):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        predicted_noise = self.model(x, time)

        mu = self.compute_mu_theta(x, time, predicted_noise)
        sigma2 = self.reverse_variance(time)
        x = mu + torch.sqrt(sigma2) * torch.randn_like(x) * int(t>0)

        return x, predicted_noise
 
    def compute_mu_theta(self, x, t, predicted_noise):
        """
        DDPM  
          approximated posterior of forward process 
            mu = 1/(1-beta_t) (x_t - beta_t/sqrt(1 - alpha_t) * eps_theta)
        """
        b = self.betas[t]
        s_one_alpha = self.sqrt_one_minus_alphas[t]
        return (1/torch.sqrt(1-b))[:, :, None, None] * (x - (b/s_one_alpha)[:, :, None, None]  * predicted_noise)
    
    def reverse_variance(self, t):
        """
        DDPM
        No training for Sigma [matrix].
        sigma_t = beta_t [scalar]  or
        sigma_t = (1- alpha_{t-1})/(1 - alpha_t) * beta_t [approximated posterior of forward process variance]
        """ 
        rev_variance = self.betas[t]
        # rev_variance = rev_variance.clip(1e-20)
        return rev_variance 
    

    def save_result(self, sample, intermediate_smpl, noise_norm, noises, params):

        
        create_save_dir(params.save_dir)
        if params.save_fig:
            f_name = f'sample_{self.expr_id}' if params.test_name is None else f'sample_{self.expr_id}_{params.test_name}'
            plot_samples(sample, f'{params.save_dir}/plot_samples/{f_name}.png')
            save_animation(intermediate_smpl, f'{params.save_dir}/plot_samples/{f_name}.mp4')
            
        if params.save_hdf:

            n_sel_time = params.n_sel_time if params.n_sel_time <= self.n_timestep_smpl else self.n_timestep_smpl 
            file_sample = create_hdf_file(params.save_dir, self.expr_id, params.dataset_mini, 
                                            self.n_timestep_smpl, n_sel_time, params.n_samples, params.train_name, params.test_name)

            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            # step = self.n_timestep_smpl // params.n_sel_time

            new_data_zero = construct_image_grid(1, sample[None, :, :, :, :]) 
            data = {}   
            data[f'data_{0}'] = new_data_zero[0].flatten()
            dfs = pd.DataFrame(data)

            intermediate_smpl = select_samples_for_plot(intermediate_smpl, params.n_samples, self.n_timestep_smpl, params.n_sel_time)
            
            new_data = construct_image_grid(params.n_sel_time, intermediate_smpl)    
            flat_img_len = np.prod(new_data.shape[1:])
            data = {'sampling': [1]  * flat_img_len}

            for i, img in enumerate(new_data):
                data[f'data_{i}'] = img.flatten()
            dfims = pd.DataFrame(data) 

            noises = select_samples_for_plot(noises, params.n_samples, self.n_timestep_smpl, params.n_sel_time)
            new_noise = construct_image_grid(params.n_sel_time, noises) 
            flat_noise_len = np.prod(new_noise.shape[1:])
            noises = {'sampling': [1]  * flat_noise_len}

            for i, img in enumerate(new_noise):
                noises[f'data_{i}'] = img.flatten()
            dfng = pd.DataFrame(data) 

            dfni = pd.DataFrame({'sample_noise_norm':noise_norm})

            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/intermediate_smpl', value=dfims, format='t')
                hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 
                hdf_store_samples.append(key=f'df/noise_grid', value=dfng, format='t')
                hdf_store_samples.append(key=f'df/noise_info', value=dfni, format='t')

def ddpm_sampling(expr_id, n_timestep_smpl=-1, n_sel_time=10, n_samples=36, evaluate=False, train_name=None, test_name=None):

    from utils_plotly_diffusion import get_params

    method = expr_id.split('_')[0] 
    params = get_params(method, expr_id)
    params.save_fig, params.save_hdf = False, True
    
    params.n_timestep_smpl = n_timestep_smpl
    params.n_sel_time = n_sel_time
    params.n_samples = n_samples

    n_timesteps = params.n_timesteps
    dataset_name = params.dataset

    params.train_name = train_name
    params.test_name = test_name
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl
    params.use_torchmetric = True

    betas = select_beta_schedule(s=params.beta_schedule, n_timesteps=n_timestep_smpl).to(device)
    model = select_model_diffusion(model_info=params.model, data_dim=params.data_dim, time_dim=params.time_dim, n_timesteps=params.n_timestep_smpl, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    params.save_dir = f"{params.save_dir}/{dataset_name}"
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    x_batch = next(iter(dataloader))[0]
    params.dataset_mini = dataset_mini
    
    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    print(f"Last saved epoch => {expr_checkpoint['epoch']}")
    ddpm = DDPM_Sampler(model=model, dataloader=None, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    ddpm.sampling(n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)
    if evaluate:
        fid_value = ddpm.evaluation(n_samples=n_samples, real_dataloader=dataloader, data_dim=x_batch.shape[1:], 
                        params=params, device=device)
        if params.save_hdf:
            p = Path(save_dir)
            path_save = p / 'saved_hdfs' 
            path_save.mkdir(parents=True, exist_ok=True)
            
            if train_name is not None:
                expr_id += train_name 

            file_sample = str(path_save) +  (f'/{expr_id}_df_sample.h5' if test_name is None else f'/{expr_id}_df_sample_{test_name}.h5')
            
            dffid = pd.DataFrame({'fid': [fid_value]}).astype(int)
            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/fid', value=dffid, format='t')


if __name__=='__main__':

    method = 'DDPM-Unlearning'

    # expr_id = 'DDPM_beta_linear_T_100_UNetMNIST_3_32_GN_MNIST_t_dim_128'
    # ddpm_sampling(expr_id, n_timestep_smpl=100, evaluate=True)
    

    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)

    beta_schedule = params.beta_schedule
    n_timesteps = params.n_timesteps

    model_name = params.model
    data_dim= params.data_dim
    time_dim= params.time_dim

    unlrn_weight = params.unlrn_weight
    norm_type = params.norm_type

    dataset_name = params.dataset
    n_epochs = params.n_epoch
    n_samples = params.n_samples
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl
    params.test_name = None
    
    experiment_info = [
        ['method:', method],
        ['beta_schedule:', beta_schedule],
        ['n_timesteps:', n_timesteps],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['time_dim:', time_dim],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['n_timestep_smpl:', n_timestep_smpl],
        ['n_epochs:', n_epochs],
        ['lr:' , params.lr],
        ['unlrn_weight:' , unlrn_weight],
        ['norm_type:' , norm_type],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    if norm_type: 
        expr_id = f'DDPM-Unlearning_beta_{beta_schedule}_T_{n_timesteps}_{model_name}_{dataset_name}_t_dim_{time_dim}_norm_type_{norm_type}_unlrn_weight_{unlrn_weight}'
    else:
        expr_id = f'DDPM-Unlearning_beta_{beta_schedule}_T_{n_timesteps}_{model_name}_{dataset_name}_t_dim_{time_dim}_unlrn_weight_{unlrn_weight}'

        
    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    params.dataset_mini = dataset_mini
    x_batch = next(iter(dataloader))[0]

    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    ddpm = DDPM_Sampler(model=model, dataloader=None, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    ddpm.sampling(n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)






