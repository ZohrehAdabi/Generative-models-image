
from common_imports_diffusion import *

from utils_diffusion import denormalize
from utils_diffusion import construct_image_grid

class BoostingOne_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None, 
            betas=None, 
            n_timestep_smpl=100, 
            expr_id=None, 
            training=False
        ):
        super(BoostingOne_Sampler, self).__init__()

        self.n_timestep_smpl= n_timestep_smpl
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
        
 
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 
    

    def sampling(self, pred_goal, n_samples=5, data_dim=2, solver='euler',  params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()
        self.data_dim = data_dim
        gamma = 0.001

        sample = torch.randn(n_samples, *data_dim).to(device)
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, *data_dim[::-1]])
        intermediate_smpl[-1] = denormalize(sample.cpu().numpy())
        
        grad_or_noises = np.empty([self.n_timestep_smpl, n_samples, *data_dim[::-1]])
        grad_or_noise_norm_list = []
        
        if pred_goal=='noise':
            
            selected_t = torch.tensor([1])
            timestp = tqdm(timesteps, disable=self.training)
            
            for t in timestp: 
                
                times = torch.repeat_interleave(selected_t, n_samples).reshape(-1, 1).long().to(device)
                with torch.no_grad():
                    sample, predicted_noise = self.sampling_ddpm_step(sample, times)
                    grad_or_noise_norm_list.append([torch.linalg.norm(predicted_noise, axis=1).mean().cpu().numpy()])
                    grad_or_noises[t] = predicted_noise.permute(0, 2, 3, 1).cpu().numpy()

                    intermediate_smpl[t] = denormalize(sample.cpu().numpy())
                
            timestp.close()  
            sample_zero = intermediate_smpl[0]

        else: #pred_goal=='grad'

            if solver=='euler':
                timestp = tqdm(timesteps, disable=self.training)
                for t in timestp: 
                    # times = torch.repeat_interleave(t, n_samples).reshape(-1, 1).long().to(device)
                    with torch.no_grad():

                        predicted_grad = self.sampling_grad_step_euler(sample)
                        sample = sample + 0.025 * predicted_grad

                        grad_or_noise_norm_list.append([torch.linalg.norm(predicted_grad, axis=1).mean().cpu().numpy()])
                        grad_or_noises[t] = predicted_noise.permute(0, 2, 3, 1).cpu().numpy()
                    
                    intermediate_smpl[t] = denormalize(sample.cpu().numpy())

                timestp.close()  
                sample_zero = intermediate_smpl[0]

            elif solver=='ode':
                
                with torch.no_grad():
                    sample = self.sampling_grad_step_ode(sample.cpu()).reshape([-1, n_samples, *data_dim])

                intermediate_smpl[t] = denormalize(sample.cpu().numpy())
                    
                sample_zero = intermediate_smpl[0]


        timestp.close() 
        sample_zero = intermediate_smpl[0]
        grad_or_noise_norm = np.concatenate(grad_or_noise_norm_list)  
        
        if not self.training: self.save_result(sample_zero, intermediate_smpl, grad_or_noise_norm, grad_or_noises, params)

        return sample_zero, intermediate_smpl, grad_or_noises, grad_or_noise_norm

    def sampling_ddpm_step(self, x, t):

        """
        DDPM
        x_{t} = mu_theta  +  sigma * z
        z ~ N(0, I)
        """
        
        predicted_noise = self.model(x)

        mu = self.compute_mu_theta(x, t, predicted_noise)
        sigma2 = torch.ones(x.shape[0], 1).to(x.device) * 0.0006
        #sigma2 = self.reverse_variance(t)
        x = mu + torch.sqrt(sigma2)[:, :, None, None] * torch.randn_like(x) * int((t>0).all())

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
    

    def sampling_grad_step_euler(self, x):

        pred_grad = self.model(x)
        return pred_grad

    def sampling_grad_step_ode(self, x):

        from scipy.integrate import solve_ivp

        solution = solve_ivp(self.ode_func, (1, 0.001), x.flatten(), args=(self.n_timestep_smpl,),
                                                rtol=1e-5, atol=1e-5, method='RK45', dense_output=True)
        # nfe = solution.nfev
        t = np.linspace(0, 1, self.n_timestep_smpl)
        bacward_ode = solution.sol(t).T
        
        return bacward_ode
    
    def ode_func(self, t, x, N):
            
        # time = int(t * (N - 1) / 1)
        inp = torch.tensor(x.reshape([-1, *self.data_dim])).to(torch.float).to(self.model.device)
        predicted_grad = - self.model(inp).cpu().numpy().flatten()
        
        return predicted_grad
    

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

def boosting_one_sampling(expr_id, n_timestep_smpl=-1, n_sel_time=10, n_samples=36, train_name=None, test_name=None):

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

    betas = select_beta_schedule(s=params.beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=params.model, data_dim=params.data_dim, time_dim=params.time_dim, n_timesteps=params.n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    params.save_dir = f"{params.save_dir}/{dataset_name}"
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    x_batch = next(iter(dataloader))[0]
    params.dataset_mini = dataset_mini
    
    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    print(f"Last saved epoch => {expr_checkpoint['epoch']}")
    boosting_one = BoostingOne_Sampler(model=model, dataloader=None, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    boosting_one.sampling(pred_goal=params.pred_goal, n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)


if __name__=='__main__':

    method = 'BoostingOne'

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
    pred_goal = params.pred_goal

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
        ['pred_goal:', pred_goal],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['n_timestep_smpl:', n_timestep_smpl],
        ['n_epochs:', n_epochs],
        ['lr:' , params.lr],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'BoostingOne_T_{n_timesteps}_{model_name}_{dataset_name}_pred_goal_{pred_goal}' 

        
    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    params.dataset_mini = dataset_mini
    x_batch = next(iter(dataloader))[0]

    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    boosting_one = BoostingOne_Sampler(model=model, dataloader=None, betas=betas, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    boosting_one.sampling(pred_goal=pred_goal, n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)






