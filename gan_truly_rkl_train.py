

from common_imports_gan import *

from gan_truly_rkl_sampling import GAN_Sampler
from utils_gan import construct_image_grid

import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)


class GAN_Model(nn.Module):

    def __init__(
            self, 
            model,
            dataloader, 
            loss_dsc=None, 
            loss_gen=None, 
            expr_id=None
        ):
        super(GAN_Model, self).__init__()

        self.loss_dsc = self.get_loss_discriminator(loss_dsc)
        self.loss_gen = self.get_loss_generator(loss_gen)
        self.model = model
        self.dataloader = dataloader
        self.expr_id = expr_id
        
    def log(self, x):
        
        """custom log function to prevent log of zero(infinity/NaN) problem."""
        return torch.log(torch.max(x, self.eps))
           
    def get_loss_discriminator(self, loss_name):

        log = self.log


        if loss_name in ['stan', 'heur']:

            def loss(real_score, fake_score, rc=0.5, fc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))]
                fake_part = -log(1.0 - fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part
            
        elif loss_name == 'comb':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] 
                fake_part = -log(1.0 - fake_score) 

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part
        elif loss_name == 'comb_mins_fs':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] + log[D(G(z))]
                fake_part = -log(1.0 - fake_score) - ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        elif loss_name == 'comb_plus_fs': # correct

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[D(x)]
                real_part = -log(real_score)

                # max log[1 - D(G(z))] -log[D(G(z))]
                fake_part = -log(1.0 - fake_score) + ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        elif loss_name == 'rvrs':

            def loss(real_score, fake_score, rc=0.5, fc=0.5, ffc=0.5, lc=1.0):
                # max log[1/D(x)]
                real_part = 1/real_score

                # max log[1 - D(G(z))] - log[D(G(z))]
                fake_part = -log(1 - fake_score) + ffc * log(fake_score)

                loss = rc * torch.mean(real_part) + fc * torch.mean(fake_part)
                return lc * loss
                # return loss, real_part, fake_part

        return loss
        
    def get_loss_generator(self, loss_name):
        
        log = self.log

        if loss_name == 'heur':

            def loss(fake_score, lc=0.5):
                # min -log[D(G(z))]
                loss = -log(fake_score)
                return  lc * torch.mean(loss)
            
        elif loss_name == 'stan':

            def loss(fake_score, lc=0.5):
                # min log[1 - D(G(z))]
                loss = log(1 - fake_score)
                return lc * torch.mean(loss)
            
        elif loss_name in ['comb', 'rvrs']:
            def loss(fake_score, ffc=1., lc=0.5):
                # min log[1 - D(G(z))] - log[D(G(z))]
                loss = log(1 - fake_score) - ffc * log(fake_score)
                return  lc * torch.mean(loss)
            
            
        return loss
    
    def train(self, n_epochs=5, lr_dsc=0.001, lr_gen=0.001, lc_dsc=[0.5, 0.5, 1., 1.], lc_gen=[1., 1.], device='cuda'):

        self.D = self.model.discriminator.to(device)
        self.G = self.model.generator.to(device)
        self.D.train()
        self.G.train()

        self.eps = eps.to(device)   

        optimizer_dsc = torch.optim.Adam(self.D.parameters(), lr=lr_dsc) # betas=(0.5, 0.999)
        optimizer_gen = torch.optim.Adam(self.G.parameters(), lr=lr_gen) # betas=(0.9, 0.999), weight_decay=1e-4)

        
        self.loss_coef_dsc = lc_dsc
        self.loss_coef_gen = lc_gen
        
        len_dataloader = len(self.dataloader)
        start_epoch = 0
        stats = {}
        if params.resume and resume_file_exists:
            optimizer_dsc.load_state_dict(expr_checkpoint['optimizer_dsc_state_dict'])
            optimizer_gen.load_state_dict(expr_checkpoint['optimizer_gen_state_dict'])
            start_epoch = expr_checkpoint['epoch'] + 1
            if start_epoch==n_epochs or start_epoch==(n_epochs-1) or start_epoch > n_epochs :
                if params.stop_epoch==-1:
                    print(f'start_epoch: {start_epoch} is equal (or close)(or greater than) to n_epochs: {n_epochs}')
                    n_epochs = int(input('Enter a valid stop_epoch:'))
                else:
                    n_epochs = params.stop_epoch

            stats['l_dsc'] = expr_checkpoint['loss_dsc']
            stats['l_gen'] = expr_checkpoint['loss_gen']
            stats['s_real'] = expr_checkpoint['real_score']
            stats['s_fake'] = expr_checkpoint['fake_score']

        loss_dsc_hist = []
        loss_gen_hist = []
        real_score_hist = []
        fake_score_hist = []

        row_df_loss = 0
        
        epochs = tqdm(range(start_epoch, n_epochs), unit="epoch", mininterval=0, disable=False)
        if params.resume: 
            epochs.set_postfix_str(f"| epoch: [{start_epoch}/{n_epochs}] Last: "+
                                       f"|Loss D {stats['l_dsc'] :<6.3f}|Loss G {stats['l_gen']:.3f}"+
                                       f"|D(x) {stats['s_real']:<6.3f}|D(G(z)) {stats['s_fake']:<6.3f}")
        for epoch in epochs:
            loss_dsc_hist_epoch, loss_gen_hist_epoch = [], []
            # epochs.set_description(f"Epoch {epoch}")
            for itr, (x_batch, _) in enumerate(self.dataloader):

                x_batch = x_batch.to(device)
                z = torch.randn(x_batch.shape[0], z_dim, device=device)
                # Training discriminator
                for k in range(1):
                    
                    loss_dsc = self.train_discriminator(x_batch, z)
                
                    optimizer_dsc.zero_grad()
                    loss_dsc.backward()
                    optimizer_dsc.step()

                loss_dsc_hist_epoch.append(loss_dsc.item())
                # Training generator
                for k in range(1):
                                      
                    loss_gen = self.train_generator(z)

                    optimizer_gen.zero_grad()
                    loss_gen.backward()
                    optimizer_gen.step()

                loss_gen_hist_epoch.append(loss_gen.item())
            # save progress
            if True: 
                with torch.no_grad():  
                    z = torch.randn(x_batch.shape[0], z_dim, device=device)
                    real_score = self.D(x_batch).mean().item()
                    fake_score = self.D(self.G(z)).mean().item()

                real_score_hist.append(real_score)
                fake_score_hist.append(fake_score)
                loss_dsc_hist.append(np.mean(loss_dsc_hist_epoch))
                loss_gen_hist.append(np.mean(loss_gen_hist_epoch))
                
                epochs.set_postfix_str(f"| epoch: [{epoch+1}/{n_epochs}]| itr: {epoch*len(self.dataloader) + 1:<6}"+
                                       f"|Loss D {loss_dsc_hist[-1]:<6.3f}|Loss G {loss_gen_hist[-1]:.3f}"+
                                       f"|D(x) {real_score:<6.3f}|D(G(z)) {fake_score:<6.3f}")
            
                

                if (epoch + 1)% params.save_freq == 0 or epoch == n_epochs-1:
                    # print(f'epoch {epoch+1}, loss: {loss.item()}')
                    if params.save_model:
                        checkpoint = {'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_dsc_state_dict': optimizer_dsc.state_dict(),
                                    'optimizer_gen_state_dict': optimizer_gen.state_dict(),
                                    'loss_dsc': loss_dsc_hist[-1], 
                                    'loss_gen': loss_gen_hist[-1],
                                    'real_score': real_score_hist[-1], 
                                    'fake_score': fake_score_hist[-1]}
                        torch.save(checkpoint, f'{params.save_dir}/saved_models/{self.expr_id}.pt')
                        
                    if params.save_hdf:

                        df_loss_score_per_itr = pd.DataFrame(columns=['itr', 'loss_dsc_itr', 'loss_gen_itr'])
                        df_loss_score_per_epoch = pd.DataFrame(columns=['epoch', 'loss_dsc_epoch', 'loss_gen_epoch', 'real_score_epoch', 'fake_score_epoch'])
                        df_loss_score_per_epoch['epoch'] = [epoch + 1]
                        df_loss_score_per_epoch['loss_dsc_epoch'] = [np.mean(loss_dsc_hist_epoch)]
                        df_loss_score_per_epoch['loss_gen_epoch'] = [np.mean(loss_gen_hist_epoch)]
                        df_loss_score_per_epoch['real_score_epoch'] = [real_score]
                        df_loss_score_per_epoch['fake_score_epoch'] = [fake_score]
                        df_loss_score_per_itr['itr'] = np.arange(epoch*len_dataloader+1, (epoch+1)*len_dataloader+1)
                        df_loss_score_per_itr['loss_dsc_itr'] = loss_dsc_hist_epoch
                        df_loss_score_per_itr['loss_gen_itr'] = loss_gen_hist_epoch

                        with pd.HDFStore(file_loss, 'a') as hdf_store_loss:
                                hdf_store_loss.append(key=f'df/loss_epoch', value=df_loss_score_per_epoch, format='t')
                                hdf_store_loss.append(key=f'df/loss_itr', value=df_loss_score_per_itr, format='t')     
                
                if (epoch + 1)% params.save_freq_img == 0 and params.validation:
                    epochs.set_postfix_str(f"| Validation ... | [{start_epoch+1}/{n_epochs}]| train stats:"+
                                       f"|Loss D {loss_dsc_hist[-1]:<6.3f}|Loss G {loss_gen_hist[-1]:.3f}"+
                                       f"|D(x) {real_score:<6.3f}|D(G(z)) {fake_score:<6.3f}")
                    self.save_result(epoch, loss_dsc_hist, loss_gen_hist, real_score_hist, fake_score_hist, x_batch.shape[1:], z_dim)
                        
                    self.model.train()
                
        epochs.close()  
        print(f'\n{Fore.YELLOW}Loss D {np.mean(loss_dsc_hist):<6.3f}|Loss G {np.mean(loss_gen_hist):<6.3f}'+
              f'|D(x) {real_score:<6.3f}|D(G(z)) {fake_score:<6.3f}{Fore.RESET}\n')

    def train_discriminator(self, x, z):

        real_data = x
        real_score = self.D(real_data) # D(x)

        # z = torch.randn(x.shape[0], z_dim, device=device)
        fake_data = self.G(z).detach()
        fake_score = self.D(fake_data)  # D(G(z))
    
        loss = self.loss_dsc(real_score, fake_score, *self.loss_coef_dsc)
        return  loss
   
    def train_generator(self, z):

        # z = torch.randn(batch_size, z_dim, device=device)
        fake_data = self.G(z)
        fake_score = self.D(fake_data)  # D(G(z))

        loss = self.loss_gen(fake_score, *self.loss_coef_gen)
        return loss


    def save_result(self, epoch, loss_dsc_hist, loss_gen_hist, real_score_hist, fake_score_hist, data_dim, z_dim):
        
        sampler = GAN_Sampler(model=self.model, dataloader=self.dataloader, training=True)
        samples, sample_score, data_score = sampler.sampling(n_samples=params.n_samples, z_dim=z_dim, device=device)
        sample_score_mean = round(np.mean(sample_score).item(), 4)
        sample_score_std = round(np.std(sample_score).item(), 4)
        data_score_mean = round(np.mean(data_score).item(), 4)
        data_score_std = round(np.std(data_score).item(), 4)
              

        if params.save_fig:
            plot_samples(samples, dataset_mini, f'{params.save_dir}/plot_samples_training/{self.expr_id}/{epoch+1}.png')
            plt.figure()
            plt.plot(loss_dsc_hist, label='D', color='C2') # marker='o', linestyle='dashed',linewidth=2, markersize=12)
            plt.plot(loss_gen_hist, label='G', color='C3')
            plt.legend()
            plt.title(f'epoch: {epoch+1:<6}| D(x): {data_score_mean:<6}| D(G(z)): {sample_score_mean:<6}')
            plt.savefig( f'{params.save_dir}/plot_samples_training/{self.expr_id}/loss.png')
            plt.close()

            plt.figure()
            plt.plot(real_score_hist, label='D(x)', color='C9') # marker='o', linestyle='dashed',linewidth=2, markersize=12)
            plt.plot(fake_score_hist, label='D(G(z))', color='C1')
            plt.hlines(y=.5, xmin=0, xmax=len(fake_score_hist), colors='gray', linestyles='--', lw=2, label=r'$D^*$')
            plt.legend()
            plt.title(f'epoch: {epoch+1:<6}|samples=> D(x): {data_score_mean:<6}| D(G(z)): {sample_score_mean:<6}')
            plt.savefig( f'{params.save_dir}/plot_samples_training/{self.expr_id}/score.png')
            plt.close()


        
        if params.save_hdf:
            
            
            samples = samples[None, :, :, :, :]

            new_data = construct_image_grid(1, samples)    
            flat_img_len = np.prod(new_data.shape[1:])
            data = {'epoch': [epoch + 1]  * flat_img_len}
            data[f'data_{0}'] = new_data[0].flatten()
            dfs = pd.DataFrame(data)

            # dfssc = pd.DataFrame(sample_score, columns=['score']) # scores of all samples generated at this epoch
            # dfdsc = pd.DataFrame(data_score, columns=['score']) # scores of all data in at this epoch
            dfsci = pd.DataFrame({'data_score_mean': [data_score_mean], 'data_score_std': [data_score_std], 
                                  'sample_score_mean': [sample_score_mean], 'sample_score_std': [sample_score_std], 'epoch': [epoch + 1]})
            
            with pd.HDFStore(file_sample, 'a') as hdf_store_samples:
                hdf_store_samples.append(key=f'df/samples_epoch_{epoch + 1:06}', value=dfs, format='t')
                # hdf_store_samples.append(key=f'df/scores_sample_epoch_{epoch + 1:06}', value=dfssc, format='t') # all samples
                # hdf_store_samples.append(key=f'df/scores_data_epoch_{epoch + 1:06}', value=dfdsc, format='t') # all data
                hdf_store_samples.append(key=f'df/score_data_sample_info_{epoch + 1:06}', value=dfsci, format='t')
                
                



if __name__=='__main__':


    
    method = 'GAN-RKL'
    
    params = parse_args(method, 'train')
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
    batch_size = params.batch_size
    params.normalize = True
    
    n_epochs = params.n_epoch
    lr_gen = params.lr_gen
    lr_dsc = params.lr_dsc

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
        ['batch_size:', batch_size],
        ['n_epochs:', n_epochs],
        ['lr_dsc:' , lr_dsc],
        ['lr_gen:' , lr_gen],
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

    
    save_config_json(save_dir, params, expr_id)


    model = select_model_gan(model_info=model_name, data_dim=data_dim, z_dim=z_dim, device=device)
    dataloader = select_dataset(dataset_name=dataset_name, batch_size=batch_size)
    dataset_mini = torch.cat([next(iter(dataloader))[0], next(iter(dataloader))[0]])

    resume_file_exists = os.path.exists(f'{params.save_dir}/saved_models/{expr_id}.pt')
    if params.resume and resume_file_exists:
        expr_checkpoint = torch.load(f'{params.save_dir}/saved_models/{expr_id}.pt', weights_only=False)
        model.load_state_dict(expr_checkpoint['model_state_dict'])
    else:
        save_config_json(save_dir, params, expr_id)
        create_save_dir_training(params.save_dir, expr_id)
    
    if params.save_hdf:          # params.n_timestep_smpl is not used for this GAN, is just for avoiding error   
        file_loss, file_sample = \
                create_hdf_file_training(params.save_dir, expr_id, dataset_mini, params.n_samples, params.n_timestep_smpl, -1, params.resume)

    print(f'\n{Fore.LIGHTBLACK_EX}{model}{Fore.RESET}\n')

    print(f'\n{Fore.MAGENTA}{experiment_info}{Fore.RESET}\n')
    
    print(f'\n {expr_id}\n')
    
    gan = GAN_Model(model=model, dataloader=dataloader, loss_dsc=loss_dsc, loss_gen=loss_gen, expr_id=expr_id)
    gan.train(n_epochs=n_epochs, lr_dsc=lr_dsc, lr_gen=lr_gen, lc_dsc=lc_dsc, lc_gen=lc_gen, device=device)

    save_config_json(save_dir, params, expr_id)

   




