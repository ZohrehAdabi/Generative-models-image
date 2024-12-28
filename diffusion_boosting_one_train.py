from common_imports_diffusion import *

from diffusion_boosting_one_sampling import BoostingOne_Sampler
from utils_diffusion import construct_image_grid

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class BoostingOne_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            betas=None, 
            n_timesteps=100, 
            expr_id=None
        ):
        super(BoostingOne_Model, self).__init__()

        self.n_timesteps= n_timesteps
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        
        
 
        self.betas = betas
        # Make alphas 
        self.alphas = torch.cumprod(1 - self.betas, axis=0)
        self.alphas = torch.clip(self.alphas, 0.0001, 0.9999)

        # required for self.add_noise
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_one_minus_alphas = torch.sqrt(1 - self.alphas) 
    

    def add_noise_flow_grad(self, x, t, sigma=0.0001):
        
        
        eps_noise = torch.randn_like(x) 
        x_noisy = (1-(1-sigma) * t) * x + eps_noise * t
        # x_noisy = x + eps_noise * t
        grad = -(-(1-sigma) * x + eps_noise)
        return x_noisy, grad 
    
    def add_noise_flow_noise(self, x, t, alpha=0.9996):
        
        eps_noise = torch.randn_like(x) 
        x_noisy = alpha * x + eps_noise * t
    
        return x_noisy, eps_noise
    

    def train(self, n_epochs=5, lr=0.001, device='cuda'):

        self.model.to(device)
        self.model.train()

        loss_fn = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr)# betas=(0.9, 0.999), weight_decay=1e-4)

        len_dataloader = len(self.dataloader)
        start_epoch = 0
        loss = 0
        if params.resume and resume_file_exists:
            optimizer.load_state_dict(expr_checkpoint['optimizer_state_dict'])
            start_epoch = expr_checkpoint['epoch'] + 1
            if start_epoch==n_epochs or start_epoch==(n_epochs-1) or start_epoch > n_epochs :
                if params.stop_epoch==-1:
                    print(f'start_epoch: {start_epoch} is equal (or close)(or greater than) to n_epochs: {n_epochs}')
                    n_epochs = int(input('Enter a valid stop_epoch:'))
                else:
                    n_epochs = params.stop_epoch
            loss = expr_checkpoint['loss']

        loss_hist = []
        
        epochs = tqdm(range(start_epoch, n_epochs), unit="epoch", mininterval=0, disable=False)
        if params.resume: epochs.set_postfix_str(f"| epoch: [{start_epoch}/{n_epochs}]| Last Loss {loss:<6.3f}")

        norm_t_noise = 0.025 # beta1
        norm_t_grad = 0.05 # 2/T
        loss_epoch = 'None'

        for epoch in epochs:
            loss_hist_epoch = []
            # epochs.set_description(f"Epoch {epoch}")
            for itr, (x_batch, _) in enumerate(self.dataloader):
                
                x_batch = x_batch.to(device)
                if pred_goal=='grad':
                    with torch.no_grad():
                        x_noisy, grad = self.add_noise_flow_grad(x_batch, norm_t_grad)
                    predicted_grad = self.model(x_noisy)
                    loss = loss_fn(grad, predicted_grad)
                else: #pred_goal=='noise'
                    with torch.no_grad():
                        x_noisy, eps_noise = self.add_noise_flow_noise(x_batch, norm_t_noise)
                    predicted_noise = self.model(x_noisy)
                    loss = loss_fn(eps_noise, predicted_noise)

                optimizer.zero_grad()
                loss.backward()
                # nn.utils.clip_grad_norm_(self.parameters(), 1)
                optimizer.step()
                loss_hist_epoch.append(loss.item())
                epochs.set_postfix_str(f"| epoch [{epoch+1}/{n_epochs}]| itr: [{itr + 1:<3}/{len_dataloader}]| loss {loss_hist_epoch[-1]:<5.3f} | Loss {loss_epoch:<5.3}")
            # save progress
            if True: 
                
                loss_hist.append(np.mean(loss_hist_epoch))
                epochs.set_postfix_str(f"| epoch [{epoch+1}/{n_epochs}]| itr: {epoch*len_dataloader + 1:<6}| Loss {loss_hist[-1]:<6.3f}")

                
                if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                    # print(f'epoch {epoch+1}, loss: {loss.item()}')
                    if params.save_model:
                        
                        checkpoint = {'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss_hist[-1]}
                        torch.save(checkpoint, f'{params.save_dir}/saved_models/{self.expr_id}.pt')

                    if params.save_hdf:

                        df_loss_per_itr = pd.DataFrame(columns=['itr', 'loss_itr'])
                        df_loss_per_epoch = pd.DataFrame(columns=['epoch', 'loss_epoch'])
                        df_loss_per_epoch['epoch'] = [epoch + 1]
                        df_loss_per_epoch['loss_epoch'] = [np.mean(loss_hist_epoch)]
                        df_loss_per_itr['itr'] = np.arange(epoch*len_dataloader+1, (epoch+1)*len_dataloader+1)
                        df_loss_per_itr['loss_itr'] = loss_hist_epoch

                        with pd.HDFStore(file_loss, 'a') as hdf_store_loss:
                            hdf_store_loss.append(key=f'df/loss_epoch', value=df_loss_per_epoch, format='t')
                            hdf_store_loss.append(key=f'df/loss_itr', value=df_loss_per_itr, format='t') 
                       

                if (epoch + 1)% params.save_freq_img == 0 and params.validation:
                    epochs.set_postfix_str(f"| Validation ...| [{epoch+1}/{n_epochs}]| Loss train {loss_hist[-1]:<6.3f}")
                    self.save_result(epoch, loss_hist, x_batch.shape[1:])
                        
                    self.model.train()
                
        epochs.close()  
        print(f'\n{Fore.YELLOW}{np.mean(loss_hist):<6.3f}{Fore.RESET}\n')

    def train_ddpm_step(self, x):

        with torch.no_grad():
            t = torch.randint(0, self.n_timesteps, size=[len(x), 1]).to(x.device) # [0, T-1]  beta start from t=1 to T
            # t = torch.randint(0, self.n_timesteps, size=[1]).reshape(x.shape[0], 1).to(x.device)
            x_noisy, eps_noise = self.forward_noising_process(x, t)

        predicted_noise = self.model(x_noisy, t)

        return eps_noise, predicted_noise

    def save_result(self, epoch, loss_hist, data_dim):
        
        sampler = BoostingOne_Sampler(model=self.model, betas=self.betas, n_timestep_smpl=n_timestep_smpl, training=True)
        samples_zero, intermediate_smpl, noises, noise_norm = sampler.sampling(pred_goal=params.pred_goal, n_samples=params.n_samples, 
                                                                                    data_dim=data_dim, device=device)
        if params.save_fig:
            plot_samples(samples_zero, dataset_mini, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}.png')
            plt.figure()
            plt.plot(loss_hist)
            plt.title(f'epoch: {epoch+1}')
            plt.savefig( f'{params.save_dir}/plot_samples_training/{self.expr_id}/loss.png')
            plt.close()
        
        if params.save_hdf:
            
            
            # step = self.n_timesteps // params.n_sel_time
            # dfs = pd.DataFrame(intermediate_smpl[::step, :, :].reshape(-1, data_dim), columns=['x', 'y'])
            intermediate_smpl = select_samples_for_plot(intermediate_smpl, params.n_samples, n_timestep_smpl, params.n_sel_time)
            
            new_data = construct_image_grid(params.n_sel_time, intermediate_smpl)    
            flat_img_len = np.prod(new_data.shape[1:])
            data = {'epoch': [epoch + 1]  * flat_img_len}

            for i, img in enumerate(new_data):
                data[f'data_{i}'] = img.flatten()

            dfims = pd.DataFrame(data) 

            # predicted noises          
            # noises = select_samples_for_plot(noises, params.n_samples, n_timestep_smpl, params.n_sel_time)
            # new_noise = construct_image_grid(params.n_sel_time, noises)    
            # flat_noise_len = np.prod(new_noise.shape[1:])
            # noises = {'epoch': [epoch + 1]  * flat_noise_len}

            # for i, img in enumerate(new_noise):
            #     noises[f'data_{i}'] = img.flatten()
            # dfng = pd.DataFrame(data) 
            

            new_data_zero = construct_image_grid(1, samples_zero[None, :, :, :, :]) 
            flat_img_len = np.prod(new_data_zero.shape[1:])
            data_zero = {'epoch': [epoch + 1]  * flat_img_len}
            data_zero[f'data_{0}'] = new_data_zero[0].flatten()
            dfs = pd.DataFrame(data_zero) 
            
            dfni = pd.DataFrame({'sample_noise_norm':noise_norm, 'epoch': np.repeat(epoch + 1, noise_norm.shape[0])})

            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/intermediate_smpl_epoch_{epoch + 1:06}', value=dfims, format='t')
                hdf_store_samples.append(key=f'df/samples_epoch_{epoch + 1:06}', value=dfs, format='t')
                # hdf_store_samples.append(key=f'df/noise_grid_epoch_{epoch + 1:06}', value=dfng, format='t')
                hdf_store_samples.append(key=f'df/noise_info_epoch_{epoch + 1:06}', value=dfni, format='t')
            
                
            


if __name__=='__main__':

    method = 'BoostingOne'
    
    params = parse_args(method, 'train')
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
    params.normalize = True
    batch_size = params.batch_size
    n_timestep_smpl =  n_timesteps if params.n_timestep_smpl==-1 else params.n_timestep_smpl

    n_epochs = params.n_epoch
    lr = params.lr

    experiment_info = [
        ['method:', method],
        ['beta_schedule:', beta_schedule],
        ['n_timesteps:', n_timesteps],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['time_dim:', time_dim],
        ['pred_goal:', pred_goal],
        ['dataset_name:', dataset_name],
        ['batch_size:', batch_size],
        ['n_timestep_smpl:', n_timestep_smpl],
        ['n_epochs:', n_epochs],
        ['lr:' , lr],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    

    expr_id = f'BoostingOne_T_{n_timesteps}_{model_name}_{dataset_name}_pred_goal_{pred_goal}' 
    
    betas = select_beta_schedule(s=beta_schedule, n_timesteps=n_timesteps).to(device)
    model = select_model_diffusion(model_info=model_name, time_dim=time_dim, n_timesteps=n_timesteps, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])

    resume_file_exists = os.path.exists(f'{params.save_dir}/saved_models/{expr_id}.pt')
    if params.resume and resume_file_exists:
        expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
        model.load_state_dict(expr_checkpoint['model_state_dict'])
    else:
        save_config_json(save_dir, params, expr_id)
        create_save_dir_training(params.save_dir, expr_id)
        
    
    if params.save_hdf:
        file_loss, file_sample = \
                create_hdf_file_training(params.save_dir, expr_id, dataset_mini, n_timestep_smpl,  params.n_sel_time, params.n_samples, params.resume)

    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')

    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    
    print(f'\n {expr_id}\n')


    # torch.set_default_device(device)
    boosting_one = BoostingOne_Model(model=model, dataloader=dataloader, betas=betas, n_timesteps=n_timesteps, expr_id=expr_id)
    boosting_one.train(n_epochs=n_epochs, lr=lr, device=device)

    save_config_json(save_dir, params, expr_id)

   




