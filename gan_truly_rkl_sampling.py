
from common_imports_gan import *

from utils_gan import denormalize
from utils_gan import construct_image_grid

class GAN_Sampler(nn.Module):

    def __init__(
            self, 
            model,
            dataloader=None,  
            expr_id=None, 
            training=False
        ):
        super(GAN_Sampler, self).__init__()

   
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        self.training = training
        
    

    def sampling(self, n_samples=5, z_dim=2, params=None, device='cuda'):

        D = self.model.discriminator.to(device)
        G = self.model.generator.to(device)
        D.eval()
        G.eval()
        with torch.no_grad():  
            # sample = torch.randn(n_samples, data_dim).to(device)

            z = torch.randn(n_samples, z_dim, device=device)
            sample = G(z).detach()   

            sample_score = D(sample).cpu().numpy()
            # sample_score = D(sample).mean().numpy()

            sample = denormalize(sample.cpu().numpy())
            data_score = []
            for itr, (x_batch, _) in enumerate(self.dataloader):
                x_batch = x_batch.to(device)
                d_score = D(x_batch).cpu().numpy()
                data_score.append(d_score)
            data_score = np.concatenate(data_score)
            # data_score = np.mean(data_score)

        if not self.training: self.save_result(sample, sample_score, data_score, params)

        return sample, sample_score, data_score



    def save_result(self, sample, sample_score, data_score, params):


            
        create_save_dir(params.save_dir)
        if params.save_fig:
            f_name = f'sample_{self.expr_id}' if params.test_name is None else f'sample_{self.expr_id}_{params.test_name}'
            plot_samples(sample, f'{params.save_dir}/plot_samples/{f_name}.png')
            
            
        if params.save_hdf:
            
            file_sample = create_hdf_file(params.save_dir, self.expr_id, params.dataset_mini, 
                                            params.n_samples, -1, -1, params.train_name, params.test_name)

            import warnings
            warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
            
            sample_score_mean = round(np.mean(sample_score).item(), 4)
            sample_score_std = round(np.std(sample_score).item(), 4)
            data_score_mean = round(np.mean(data_score).item(), 4)
            data_score_std = round(np.std(data_score).item(), 4)

            sample = sample[None, :, :, :, :]

            new_data = construct_image_grid(1, sample)    
            data = {}
            data[f'data_{0}'] = new_data[0].flatten()
            dfs = pd.DataFrame(data)

            # dfssc = pd.DataFrame(sample_score, columns=['score']) # scores of all samples generated at this epoch
            # dfdsc = pd.DataFrame(data_score, columns=['score']) # scores of all data in at this epoch
            dfsci = pd.DataFrame({'data_score_mean': [data_score_mean], 'data_score_std': [data_score_std], 
                                  'sample_score_mean': [sample_score_mean], 'sample_score_std': [sample_score_std]})

            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/samples', value=dfs, format='t') 
                # hdf_store_samples.append(key=f'df/scores_sample', value=dfssc, format='t')
                # hdf_store_samples.append(key=f'df/scores_data', value=dfdsc, format='t')
                hdf_store_samples.append(key=f'df/score_data_sample_info', value=dfsci, format='t')



def gan_sampling(expr_id, n_samples=36, train_name=None, test_name=None):

    from utils_plotly_gan import get_params

    method = expr_id.split('_')[0] 
    params = get_params(method, expr_id)
    params.save_fig, params.save_hdf = False, True
    
    params.n_samples = n_samples
    dataset_name = params.dataset

    params.train_name = train_name
    params.test_name = test_name

    
    model = select_model_gan(model_info=params.model, data_dim=params.data_dim, z_dim=params.z_dim, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    params.save_dir = f"{params.save_dir}/{dataset_name}"
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])
    x_batch = next(iter(dataloader))[0]
    params.dataset_mini = dataset_mini
    
    expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_checkpoint['model_state_dict'])

    print(f"Last saved epoch => {expr_checkpoint['epoch']}")
    gan = GAN_Sampler(model=model, dataloader=dataloader, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    gan.sampling(n_samples=n_samples, z_dim=params.z_dim, params=params, device=device)



if __name__=='__main__':

    method = 'GAN-RKL'
    expr_id = 'GAN-RKL_GANMNIST_2_16_MNIST_z_dim_8_lr_dsc_1e-4_lr_gen_1e-4_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    gan_sampling(expr_id)
    params = parse_args(method, 'sampling')
    params.method = method
    set_seed(params.seed)

    
    save_dir = f'{save_dir}/{method}'
    params.save_dir = f'{save_dir}/{params.dataset}/'
    p = Path(params.save_dir)
    p.mkdir(parents=True, exist_ok=True)

    model_name = params.model
    data_dim= params.data_dim
    z_dim= params.z_dim

    loss_dsc = params.loss_dsc
    loss_gen = params.loss_gen

    dataset_name = params.dataset
    n_samples = params.n_samples

    n_epochs = params.n_epoch
    lr_gen = params.lr_gen
    lr_dsc = params.lr_dsc
    params.test_name = None

    lc_dsc = [] if not params.lc_dsc else [float(item) if item.strip() not in ['1', '1.0'] else int(item) for item in params.lc_dsc.replace('  ', ' ').strip().split(' ')] # if type of input arg was string
    lc_gen = [] if not params.lc_dsc else [float(item) if item.strip() not in ['1', '1.0'] else int(item)  for item in params.lc_gen.replace('  ', ' ').strip().split(' ')] 

    lc_dsc_id = '_'.join([str(i) for i in lc_dsc]) 
    lc_gen_id = '_'.join([str(i) for i in lc_gen])

    experiment_info = [
        ['method:', method],
        ['model:', model_name],
        ['data_dim:', data_dim],
        ['z_dim:', z_dim],
        ['loss_dsc:', loss_dsc],
        ['loss_gen:', loss_gen],
        ['dataset_name:', dataset_name],
        ['batch_size:', params.batch_size],
        ['n_epochs:', n_epochs],
        ['lr_gen:' , lr_gen],
        ['lr_dsc:' , lr_dsc],
        ['n_samples:' , n_samples],
        ['seed:', params.seed]
    ]
    experiment_info = tabulate(experiment_info, tablefmt='plain')
    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    

    expr_id = f'GAN-RKL_{model_name}_{dataset_name}_z_dim_{z_dim}_lr_dsc_{lr_dsc:.0e}_lr_gen_{lr_gen:.0e}_loss_dsc_{loss_dsc}'.replace("e-0", "e-")
   
    if lc_dsc:
        expr_id += f'_lc_{lc_dsc_id}'

    expr_id += f'_loss_gen_{loss_gen}'
    if lc_dsc:
        expr_id += f'_lc_{lc_gen_id}'

        
    
    model = select_model_gan(model_info=model_name, data_dim=data_dim, z_dim=z_dim, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=params.batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])

    expr_info = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
    model.load_state_dict(expr_info['model_state_dict'])

    gan = GAN_Sampler(model=model, dataloader=dataloader, expr_id=expr_id)
    print(f'\n {expr_id}\n')
    gan.sampling(n_samples=n_samples, z_dim=z_dim, params=params, device=device)






