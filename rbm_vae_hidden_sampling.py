
from common_imports_diffusion import *

from utils_diffusion import denormalize
from utils_diffusion import construct_image_grid

class RBM_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None, 
            n_timestep_smpl=100, 
            expr_id=None, 
            training=False
        ):
        super(RBM_Sampler, self).__init__()

        self.n_timestep_smpl= n_timestep_smpl
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
    
    def add_noise(self, mu, logvar):
        
        # eps = torch.randn_like(mu).to(mu.device)
        # return mu + eps * std.to(mu.device)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def sampling(self, n_samples=5, data_dim=2, solver='euler',  params=None, device='cuda'):

        self.model.to(device)
        self.model.eval()
        self.data_dim = data_dim

        sample = torch.randn(n_samples, *data_dim).to(device)
        timesteps = torch.arange(self.n_timestep_smpl-1, -1, -1)
        intermediate_smpl = np.empty([self.n_timestep_smpl+1, n_samples, *data_dim[::-1]])
        intermediate_smpl[-1] = denormalize(sample.cpu().numpy())
        
        # grad_or_noises = np.empty([self.n_timestep_smpl, n_samples, *data_dim[::-1]])
        # grad_or_noise_norm_list = []
        

        timestp = tqdm(timesteps, disable=self.training)
        # mu, logvar = torch.randn_like(sample), torch.randn_like(sample)
        # hidden = torch.zeros_like(sample)
        hidden = torch.randn_like(sample)
        
        with torch.no_grad():

            for t in timestp: 

                # sample, logvar, hidden = self.model(sample, hidden, burn_in=(t==self.n_timestep_smpl-1)) 
                sample, logvar, hidden = self.model(sample, hidden, burn_in=False) 
                intermediate_smpl[t] = denormalize(sample.cpu().numpy())
                sample = self.add_noise(sample, logvar)

            sample, logvar, hidden = self.model(sample, hidden, burn_in=False) 
            intermediate_smpl[t] = denormalize(sample.cpu().numpy())

        timestp.close() 
        sample_zero = intermediate_smpl[0]
        # grad_or_noise_norm = np.concatenate(grad_or_noise_norm_list)  
        
        if not self.training: self.save_result(sample_zero, intermediate_smpl, params)

        return sample_zero, intermediate_smpl


    

    def save_result(self, sample, intermediate_smpl, params):

        
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





            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/intermediate_smpl', value=dfims, format='t')
                hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 



def rbm_sampling(expr_id, n_timestep_smpl=-1, n_sel_time=10, n_samples=36, train_name=None, test_name=None):

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

    # betas = select_beta_schedule(s=params.beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=params.model, data_dim=params.data_dim, time_dim=params.time_dim, n_timesteps=params.n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    params.save_dir = f"{params.save_dir}/{dataset_name}"
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    x_batch = next(iter(dataloader))[0]
    params.dataset_mini = dataset_mini
    
    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    print(f"Last saved epoch => {expr_checkpoint['epoch']}")
    rbm = RBM_Sampler(model=model, dataloader=None, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    rbm.sampling(n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)


if __name__=='__main__':

    method = 'RBM-Hidden'

    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)

    # beta_schedule = params.beta_schedule
    n_timesteps = params.n_timesteps

    model_name = params.model
    data_dim= params.data_dim
    time_dim= params.time_dim

    dataset_name = params.dataset
    n_epochs = params.n_epoch
    n_samples = params.n_samples

    burn_in = params.burn_in
    allX = params.allX
    fixed_enc = params.fixed_enc

    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl
    params.test_name = None
    
    experiment_info = [
        ['method:', method],
        # ['beta_schedule:', beta_schedule],
        ['n_timesteps:', n_timesteps],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['time_dim:', time_dim],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['n_timestep_smpl:', n_timestep_smpl],
        ['n_epochs:', n_epochs],
        ['lr:' , params.lr],
        ['burn_in:', burn_in],
        ['allX:', allX],
        ['fixed_enc:', fixed_enc],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'RBM-Hidden_T_{n_timesteps}_{model_name}_{dataset_name}' 
    if burn_in > 0:
        expr_id += f'_burn_in_{burn_in}'
    if allX:
        expr_id += '_allX'
    if fixed_enc:
        expr_id += '_fixed_enc'
        
    # betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, data_dim=data_dim, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    params.dataset_mini = dataset_mini
    x_batch = next(iter(dataloader))[0]

    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    rbm = RBM_Sampler(model=model, dataloader=None, n_timestep_smpl=n_timestep_smpl, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    rbm.sampling(n_samples=n_samples, data_dim=x_batch.shape[1:], params=params, device=device)






