

import matplotlib.pyplot as plt
# import torch
import numpy as np
from types import SimpleNamespace
import plotly.express as px 
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.figure_factory as ff
import scipy.stats as ss
import pandas as pd
import json

from utils_gan import get_fake_sample_grid

def show_latex_in_plotly():
    from plotly import offline
    from IPython.display import display, HTML
    ## Tomas Mazak's workaround
    offline.init_notebook_mode()
    display(HTML(
        '<script type="text/javascript" async src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG"></script>'
    ))
def read_launch_for_terminal():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_terminal.txt', 'w') as f:
        for line in params_txt:
            if '//' in line and 'python' in line:
                line = line.replace('//', '')
                line = line.strip()
                f.write(f"{line}\n\n")

def read_launch():
    with open('./.vscode/launch.json') as f:
        params_txt = f.readlines()
    with open('./.vscode/launch_plotly.json', 'w') as f:
        for line in params_txt:
            if '//' not in line:
                f.write(line)
    with open('./.vscode/launch_plotly.json', 'r') as f:
        launch = json.load(f)
    # print(lunch.keys())
    config = launch['configurations'][2:]
    config_name_dict = {}
    for i in range(len(config)):
        config_name_dict[config[i]['name'] ] = i
    # print(config_name)
    return config, config_name_dict

def configuration_from_lunch(config, config_name_dict, config_name):
    config_params = config[config_name_dict[config_name]]['args']
    params_dict = {}
    cnfg_len = len(config_params)
    for i, p in enumerate(config_params):
        if '--' in p:
            if i < cnfg_len-1:
                if '--' not in config_params[i+1]:
                    params_dict[p[2:]] = config_params[i+1]
                else:
                    params_dict[p[2:]] = True
            else: # last params
                params_dict[p[2:]] = True
    return params_dict

def get_config(method, config_name):

    config, config_name_dict = read_launch()
    params_dict = configuration_from_lunch(config, config_name_dict, config_name)
    expr_id = f"{method}_beta_{params_dict['beta_schedule']}_T_{params_dict['n_timesteps']}_{params_dict['model']}" + \
            f"_{params_dict['dataset']}_t_dim_{params_dict['time_dim']}"
    return params_dict, expr_id



def get_params(method, expr_id):

    dataset_name, config = get_dataset_name(method, expr_id)
    params = SimpleNamespace(**config)
    return params

def get_dataset_name(method, expr_id):
    with open(f'./saved_result/{method}/configs_train/{expr_id}.json', 'r') as f:
        config = json.load(f)
    dataset_name = config['dataset']
    # num_class = 1
    # if dataset_name[0].isdigit():
    #     num_class = int(dataset_name[0])
    #     if dataset_name[1].isdigit():
    #         num_class = int(dataset_name[0:2])
     
    return dataset_name, config
    




def read_hdf_loss(file):

    # df_loss = pd.read_hdf(file) 
    with pd.HDFStore(file) as df:
        df_loss_epoch = df['df/loss_epoch']
        df_loss_itr = df['df/loss_itr']  
    
    # df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    df_loss_epoch = df_loss_epoch.reset_index(drop=True)
    df_score = df_loss_epoch.loc[:, ['epoch', 'real_score_epoch', 'fake_score_epoch']]

    return df_loss_epoch.loc[:, ['epoch', 'loss_dsc_epoch', 'loss_gen_epoch']], df_loss_itr.reset_index(drop=True), df_score

def read_hdf_all_epochs_train(file):

    try:
        hdf_file = pd.HDFStore(file)   
        # print(smple.keys())
        keys = hdf_file.keys()
        
        image_shape = hdf_file['/df/info']['image_shape'].values[0]
        image_shape = list(map(int, image_shape.split(',')))
        n_sel_time = hdf_file['/df/info']['n_sel_time'].values[0]

        df_sample =[]
        df_score_info = pd.DataFrame()
        df_score_info_hist = pd.DataFrame()
        
        df_fake_data = pd.DataFrame()
        # df_intermediate_smpl = pd.DataFrame()
        df_intermediate_smpl_list = []
        
        epoch_list = []
        for k in keys:
            if 'samples' in k: 
                smpl = hdf_file[k]['data_0'].to_numpy().reshape(image_shape)
                df_sample.append(smpl)
                if 'epoch_' in k:
                    epoch_list.append(int(k.split('/')[2].split('_')[2]))
            elif 'intermediate_smpl'in k:
                # df_intermediate_smpl = pd.concat([df_intermediate_smpl if not df_intermediate_smpl.empty else None, hdf_file[k]])
                df_intermediate_smpl_list.append(hdf_file[k])
            elif 'score_data_sample_info' in k:
                df_score_info = pd.concat([df_score_info if not df_score_info.empty else None, hdf_file[k]])
            elif 'score_data_sample_hist' in k: # score of samples generated iteratively
                df_score_info_hist = pd.concat([df_score_info_hist if not df_score_info_hist.empty else None, hdf_file[k]])
            # elif 'grad_g_z_at_sampling' in k:
            #     # df_grad_g_z_at_sampling = pd.concat([df_grad_g_z_at_sampling if not df_grad_g_z_at_sampling.empty else None, hdf_file[k]]) 
          
            elif 'fake_data' in k:
                df_fake_data = pd.concat([df_fake_data if not df_fake_data.empty else None, hdf_file[k]])

        df_data = hdf_file['/df/data']
        n_samples = hdf_file['/df/info']['n_samples'].values[0]
        # grid_size = hdf_file['/df/info']['grid_size'].values[0]
        # timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
        hdf_file.close()
    except:
        print(f'Error in reading hdf file {file}')
        hdf_file.close()

    df_data = df_data['data_0'].to_numpy().reshape(image_shape)
    if df_sample:
        df_sample = np.stack(df_sample) 
    if not epoch_list:
        epoch_list = [0]

    return df_sample, df_data, df_intermediate_smpl_list, df_score_info.reset_index(drop=True), \
            df_score_info_hist, epoch_list, n_samples


def add_all_zeros_animation(df_sample, epoch_list):
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    
    if df_sample.ndim == 3:
        df_sample = df_sample[None, :, :, :]
    px_anim_sample = px.imshow(df_sample.squeeze(-1), animation_frame=0, binary_string=True, binary_compression_level=4)

    all_zero_sample_frames = [go.Frame(data=[smpl_frame.data[0].update(name='sample')]
                                , traces=[0], name=f'zero_epoch_{e}', group=f'all_zero')
                                            for e, smpl_frame in zip(epoch_list, px_anim_sample.frames)]
    return all_zero_sample_frames 

def add_data_trace(fig, df_data):

    # data
    fig.add_trace(px.imshow(df_data.squeeze(), binary_string=True).data[0].update(name='data', visible=False))
    
    gray = df_data.squeeze()
    rgb = np.stack([gray, gray, gray], axis=-1)
    alpha = np.full(gray.shape, 25, dtype=np.uint8)  # 255 Fully opaque
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    fig.add_trace(px.imshow(rgba, binary_string=True).data[0].update(name='data_transparnce', visible=False))
    
    return fig

def add_sammple_trace(fig, all_zero_sample_frames):

    # data, sample_zero at first epoch , grad at first epoch
    fig.add_trace(all_zero_sample_frames[0].data[0].update(name=f'sample_{0}', visible=True))
       
    return fig

def add_statistics_to_fig(fig, df_loss_itr, df_loss, df_score):

    # add traces of statistics
  
    x = df_loss_itr['itr']
    y_dsc = (np.cumsum(df_loss_itr['loss_dsc_itr']) / x)
    y_dsc[0] = df_loss_itr['loss_dsc_itr'][0]
    y_gen = (np.cumsum(df_loss_itr['loss_gen_itr']) / x)
    y_gen[0] = df_loss_itr['loss_gen_itr'][0]
    fig.add_trace(go.Scatter(x=x, y=df_loss_itr['loss_dsc_itr'], name="Loss D", mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen')))
    fig.add_trace(go.Scatter(x=x, y=df_loss_itr['loss_gen_itr'], name="Loss G", mode='lines', visible=False, showlegend=True, line=dict(width=1, color='crimson'))) 
    fig.add_trace(go.Scatter(x=x, y=y_dsc, name="Loss D mean", mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.add_trace(go.Scatter(x=x, y=y_gen, name="Loss G mean", mode='lines', visible=False, showlegend=True, line=dict(width=2, color='deeppink')))
    x = df_loss['epoch']
    y_dsc = (np.cumsum(df_loss['loss_dsc_epoch']) / x)
    y_dsc[0] = df_loss['loss_dsc_epoch'][0]
    y_gen = (np.cumsum(df_loss['loss_gen_epoch']) / x)
    y_gen[0] = df_loss['loss_gen_epoch'][0]
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_dsc_epoch'], name='Loss D', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen'))) # lightsalmon
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_gen_epoch'], name='Loss G', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='crimson'))) # indianred
    fig.add_trace(go.Scatter(x=x, y=y_dsc, name='Loss D mean' , mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.add_trace(go.Scatter(x=x, y=y_gen, name='Loss G mean' ,mode='lines', visible=False, showlegend=True, line=dict(width=2, color='deeppink')))
    fig.update_layout(title=dict(text='Loss'))

    
    fig.add_trace(go.Scatter(x=x, y=df_score['real_score_epoch'], name='D(x)', mode='lines',visible=False, showlegend=True, line=dict(width=1, color='darkturquoise')))
    fig.add_trace(go.Scatter(x=x, y=df_score['fake_score_epoch'], name='D(G(z))', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='darkorange')))
    y_real_score_mean = np.cumsum(df_score['real_score_epoch'])/x
    y_fake_score_mean = np.cumsum(df_score['fake_score_epoch'])/x

    # plot std of scores in epochs, not std of mean of scores
    num_epoch = x.shape[0]
    def std_mean_for_score(df, y_score_mean):
        y = df
        std_mean = np.empty([num_epoch])
        for i in range(num_epoch):
            std_mean[i] = np.std(y.iloc[:i+1])
        y_upper, y_lower = y_score_mean + std_mean, y_score_mean - std_mean
        y_with_std = pd.concat([y_upper, y_lower[::-1].reset_index(drop=True)]).reset_index(drop=True)
        return y_with_std
    x = df_score['epoch']
    y_real_with_std = std_mean_for_score(df_score['real_score_epoch'], y_real_score_mean)    
    y_fake_with_std = std_mean_for_score(df_score['fake_score_epoch'], y_fake_score_mean)
    x_x_revr = pd.concat([x, x[::-1].reset_index(drop=True)]).reset_index(drop=True)
    fig.add_trace(go.Scatter(x=x_x_revr, y=y_real_with_std, fill='toself', name='D(x) std',  visible=False, showlegend=True, fillcolor='rgba(0,176,246,0.2)', line_color='rgba(255,255,255,0)'))
    fig.add_trace(go.Scatter(x=x_x_revr, y=y_fake_with_std, fill='toself', name='D(G(z)) std',  visible=False, showlegend=True, fillcolor='rgba(231,107,243,0.2)', line_color='rgba(255,255,255,0)'))

    fig.add_trace(go.Scatter(x=x, y=y_real_score_mean, name='D(x) mean', mode='lines',visible=False, showlegend=True, line=dict(width=2, color='deepskyblue'))) #dodgerblue darkturquoise deepskyblue
    fig.add_trace(go.Scatter(x=x, y=y_fake_score_mean, name='D(G(z)) mean', mode='lines', visible=False, showlegend=True, line=dict(width=2, color='orangered'))) # indianred

    fig.add_trace(go.Scatter(x=x, y=np.ones(num_epoch) * 0.5, name='D*', mode='lines', visible=False, showlegend=True, line=dict(width=2, color='slategray', dash='dash')))
    
    
    return fig

def visibility_traces(fig):
    # visibility of traces when coresponding button is clicked
    trace_visibility = dict()
    for trace in fig.data:
        if trace.name not in  trace_visibility.keys():
            trace_visibility[trace.name] = False
        elif 'Loss' in trace.name:
            trace_visibility[f'{trace.name} epoch'] = False

    # trace_names_example = [['data', 'sample', 'grad', 
    # 'Loss D', 'Loss G', 'Loss D mean', 'Loss G mean', 
    # 'Loss D', 'Loss G', 'Loss D mean', 'Loss G mean', 
    # 'D(x)', 'D(G(z))', 'D(x) std', 'D(G(z)) std', 'D(x) mean', 'D(G(z)) mean', 'D*']
    
    trace_visibility['data'] = False
    

    trace_vis_loss_itr = trace_visibility.copy()
    trace_vis_loss_itr.update({f'Loss D': True, f'Loss G': True, f'Loss D mean': True, f'Loss G mean': True, 'data': False})
    trace_vis_loss_epoch = trace_visibility.copy()
    trace_vis_loss_epoch.update({f'Loss D epoch': True, f'Loss G epoch': True, f'Loss D mean epoch': True, f'Loss G mean epoch': True, 'data': False})
    trace_vis_score_epoch = trace_visibility.copy()
    trace_vis_score_epoch.update({f'D(x)': True, f'D(G(z))': True, f'D*': True, 'data': False})
    trace_vis_score_mean = trace_visibility.copy()
    trace_vis_score_mean.update({f'D(x) std': True, f'D(G(z)) std': True, f'D(x) mean': True, f'D(G(z)) mean': True, f'D*': True, 'data': False})
    
    stats_trace_vis = [trace_vis_loss_itr, trace_vis_loss_epoch, trace_vis_score_epoch, trace_vis_score_mean]

    return trace_visibility, stats_trace_vis

def visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics, just_loss=False, training=False):

    # store idx of other buttons that adding to the layout for making visible True or False
    stats_idx = n_button_animations
    pause_idx = n_button_animations + n_button_statistics
    if just_loss:
        back_stats_idx = pause_idx 
    else:
        back_stats_idx = pause_idx + 1 if training else pause_idx
    back_detail_idx = back_stats_idx + 1
    select_dropdown_idx = back_detail_idx + 1 
     
    # buttons visibility
    button_visibility = {}
    for i in range(n_all_buttons):
        button_visibility[f'updatemenus[{i}].visible'] = False 
    button_visibility[f'updatemenus[{pause_idx}].visible'] = True if training else False
    button_visibility[f'updatemenus[{select_dropdown_idx}].visible'] = True

    return button_visibility, stats_idx, pause_idx, back_stats_idx, back_detail_idx, select_dropdown_idx

def add_play_show_buttons_slider(trace_visibility, button_visibility, button_info, slider_info, epoch_list, training=False):
    
    play_button_list = []
    select_button_list = [] #drop down button

    # Play buttons for sampling
    frame_spec = {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 
              'transition': {'duration': 20, 'easing': 'cubic-in-out'}}
    
    if training:
        zero_play_button = dict(buttons=[dict(label=f"{f'&#9654; zeros':^16}", method='animate', args=[f'all_zero' , frame_spec])])
        zero_play_button.update(button_info)
        zero_play_button.update(dict(x=-0.22, y=0.001))

        play_button_list.append(zero_play_button)
    # slider for zero play button
    
    all_zero_slider = dict(steps=[dict(args=[[f'zero_epoch_{e}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate' }], 
                                method='animate', label=f'{e}' ) for e in epoch_list] )
    all_zero_slider.update(slider_info)

    visibility_layout = {'title': f'Sampling'}
    visibility_layout.update(button_visibility)
    visibility_layout[f'updatemenus[0].visible'] = True if training else False# show play zeros buttons
    visibility_layout['sliders'] = [all_zero_slider] 

    # add fist button to dropdwon 
    trace_vis_init = trace_visibility.copy()
    trace_vis_init.update({f'sample_0': True})
    select_button_list.append(dict(label=f'Show all samples' if training else 'Show sample ', method='update', 
                                args=[{'visible': list(trace_vis_init.values())}, visibility_layout]))

    vis_layout = visibility_layout.copy()
    vis_layout[f'updatemenus[0].visible'] = False if training else True


    # select_button_list.append(dict(label=f'', method='update',  args=[None]))

    return play_button_list, select_button_list, all_zero_slider

def add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, n_button_animations, button_info, method,
                      back_stats_idx, pause_idx, select_dropdown_idx, just_loss=False):

    # Show all stats buttons (loss, score)
    stats_button_list = []
    show_stats = {'title': method, 'width':1000, 'height': 450, 'plot_bgcolor': None, 
                    'xaxis': {'range': None, 'autorange': True, 'title': 'x', 'showgrid': True,'showticklabels': True},
                    'yaxis': {'range': None, 'autorange': True, 'title': 'y', 'showgrid': True,'showticklabels': True, 
                               'type': 'log'} }

    show_stats['sliders'] = []
    show_stats.update(button_visibility) 
    show_stats[f'updatemenus[{back_stats_idx}].visible'] = True
    show_stats[f'updatemenus[{pause_idx}].visible'] = False if not just_loss else True
    show_stats[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    
    stats_lables = ['Loss itr', 'Loss epoch', 'Score', 'Score mean']
    for i in range(n_button_statistics):  
        j = i + n_button_animations
        show_stats[f'updatemenus[{j}].visible'] = True 

        stats_vis = stats_trace_vis[i]
        stats_button = dict(buttons=[dict(label=f'{stats_lables[i]}', method='update', args=[{'visible': list(stats_vis.values())}, 
                                                                                            {'title': f'{stats_lables[i]}'}])])
        stats_button.update(button_info)
        stats_button.update(dict(visible=False, x=0.15+((i+1))/8, y=1.02, pad={"r": 10, "t": 25, "l": 10}))
        stats_button_list.append(stats_button)
    # Show loss in select button [dropdown]
    stats_vis = stats_trace_vis[0]
    show_stats_button = dict(label=f'Show Loss-Score', method='update', args=[{'visible': list(stats_vis.values())}, show_stats])

    return stats_button_list, show_stats_button

def add_data_button(button_visibility, trace_visibility, pause_idx):

    trace_vis = trace_visibility.copy()
    trace_vis.update({f'data': True})
    
    # Show data
    show_data = {'title': f'Data', 'height': 500} 
    show_data.update({  'xaxis': {'title': '', 'showticklabels': False, 'autorange':True, 'range': None}, 
                        'yaxis': {'title': '', 'showticklabels': False, 'autorange':'reversed', 'range': None}
                        })
            
    show_data['sliders'] = []
    show_data.update(button_visibility) 
    
    # show_smpl_grd_ns[f'updatemenus[{back_smpl_grad_noise_norm_idx}].visible'] = True
    show_data[f'updatemenus[{pause_idx}].visible'] = False
    # show_smpl_grd_ns[f'updatemenus[{select_dropdown_idx}].visible'] = False 

    # Show smpl score in select button [dropdown]

    show_data_button = dict(label=f'data', method='update', args=[{'visible': list(trace_vis.values())}, 
                                                                  show_data])

    return show_data_button
     
def add_detail_button(trace_visibility, button_visibility, config, back_detail_idx, pause_idx, select_dropdown_idx):

    # show details  in select button [dropdown ]
    show_detail = {}
    show_detail.update(button_visibility)
    show_detail[f'updatemenus[{back_detail_idx}].visible'] = True
    show_detail[f'updatemenus[{pause_idx}].visible'] = False
    show_detail[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    # Formatted string with LaTeX

    title_text = r"$\begin{aligned}"
    for k, v in config.items():
        if 'save_dir' in k:
            continue
        key = k.replace("_", "-")
        value = str(v).replace("_", "-")
        if  value.isdigit():
            title_text += rf" & \text{{{key}}}  \quad & = \quad & {value}  \\ "
        else:
            title_text += rf" & \text{{{key}}}  \quad & = \quad & \text{{{value}}}  \\ "
    title_text += r"\end{aligned}$"  # End LaTeX block
    show_detail['title.text'] = title_text
    show_detail['sliders'] = []
    
    show_detail.update({'margin.t':100, 'title.y': 0.6, 'title.x':0.2,
                        'xaxis': {'title': '', 'showticklabels': False, 'range': [-2.5, 2.5]}, 
                        'yaxis': {'title': '', 'showticklabels': False, 'range': [-2.5, 2.5]}})
    show_detail.update(dict(plot_bgcolor='white', height=450))
    hide_traces = trace_visibility.copy()
    hide_traces['data'] = False
    show_detail_button = dict(label=f'Show details', method='update', args=[{'visible': list(hide_traces.values())}, show_detail])
            #    [dict(label=f'Hide Loss-Score', method='update', args=[{'y': [], 'x': []}, hide_loss]),
            #    dict(label=f'1 details', method='relayout', args=[{'title': f'GAN 1'}])]
    return show_detail_button

def add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, method,
                     back_detail_idx, stats_idx, select_dropdown_idx, back_stats_idx):

    hide_info = {'margin.t': 60, 'title.y': 0.98, 'title.x':0.001, 'plot_bgcolor': 'white', 'width':750, 'height': 500,
                'xaxis': {'range': None, 'autorange': True, 'title': '', 'showgrid': False, 'showticklabels': False},
                'yaxis': {'range': None, 'autorange': 'reversed', 'title': '', 'showgrid': False, 'showticklabels': False, 'type': 'linear'},
                }

    slider_steps = dict(steps=[dict(args=[[None]], method='animate', label=f'{j}' ) for j in range(2)])
    slider_steps.update(slider_info)
    slider_steps.update({'currentvalue': dict(font_size=12, prefix= f'', xanchor='left')})
    # hide_info['sliders'] = [slider_steps]
    hide_info['sliders'] = []
    trace_vis = trace_visibility.copy()
    trace_vis['data_transparnce'] = True
    # add back button [stats]
    hide_stats = {'title': method}
    hide_stats.update(hide_info)
    # changing vsibility to False must be after changing title and slider to slider work properly
    for i in range(n_button_statistics):
        hide_stats[f'updatemenus[{stats_idx+i}].visible'] = False

    
    hide_stats[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_stats[f'updatemenus[{back_stats_idx}].visible'] = False
    back_stats_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_vis.values())}, hide_stats])]) 
    back_stats_button.update(button_info)
    back_stats_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))

    # add back button  [details]
    hide_detail = {'title.text': method}
    hide_detail.update(hide_info) 
    hide_detail[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_detail[f'updatemenus[{back_detail_idx}].visible'] = False
    back_detail_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_vis.values())}, hide_detail])]) 
    back_detail_button.update(button_info)
    back_detail_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))
    
    return back_stats_button, back_detail_button
    


def prepare_plotly_fig_quiver_grad(df_sample, df_data, df_score_info, df_score_info_hist,
                                df_loss_itr, df_loss, df_score,
                                epoch_list, quiver_name, method, config,  just_loss=False, training=True,
                                width=750, height=570):

    # The traces must be added in the correct order for the button's visibility to function properly
   
    num_epochs = len(epoch_list)
    colors = dict(sample='#AB63FA', grad='DarkSlateGrey', data='#19D3F3') # samples-grads-data
    if not just_loss:
        # create frames for each animation [epochs], all zeros samples are in one df: df_sample
        all_zero_sample_frames = add_all_zeros_animation(df_sample, epoch_list)

    fig = go.Figure() 

    if not just_loss:
        # data, sample_zero at first epoch , grad at first epoch
        fig = add_sammple_trace(fig, all_zero_sample_frames)

    # add traces of statistics
    if training:
        fig = add_statistics_to_fig(fig, df_loss_itr, df_loss, df_score)
    # add data
    fig = add_data_trace(fig, df_data)
    # visibility of traces when coresponding button is clicked
    trace_visibility, stats_trace_vis = visibility_traces(fig)
    if just_loss:
        n_button_animations = 0
    else:
        n_button_animations = (1 if training else 0)  # 1: for all_zeros button # equals to plays buttons
    
    n_button_statistics = 4 if training else 0# loss itr, loss epoch, score epoch, score mean & std
    
    if training:
        n_all_buttons = n_button_animations + n_button_statistics + 2 + 2  # show stats, show detail, show eval, 
    else:
        n_all_buttons = n_button_animations + 2                                                                       # back_stats, back_detail, back_eval
    # n_traces_animation = 1 + 1 +  num_epochs # 1:data, 1: grad first epoch, len(epoch_list): samples of each epoch
    # n_traces_statistics = 4 + 4 + 2 + 4 + 1 # 4: loss itr 4: loss epoch, 2: score epoch, 4: score mean & std, 1 is for D*

    # store idx of other buttons that adding to the layout for making visible True or False
    # buttons visibility
    button_visibility, stats_idx, pause_idx, \
    back_stats_idx, back_detail_idx, select_dropdown_idx = visibility_buttons_and_idx(n_all_buttons, n_button_animations, 
                                                                                      n_button_statistics, just_loss, training)

    button_info = dict(visible=True, type='buttons', direction='down', showactive=False, pad={"r": 10, "t": 10, "l": 10}, 
                         xanchor="right", yanchor="bottom", bgcolor='seashell')
    slider_info = dict(active=0, name=f'all_epoch', visible=True, currentvalue=dict(font_size=12, prefix=f'zero of epoch: ', xanchor='left'),
                 transition=dict(duration=300, easing='cubic-in-out'), pad=dict(b=10, t=60), len=1, x=0.0, y=0.08, xanchor='left', yanchor='top')
    
    all_zero_slider = dict()
    if not just_loss:
        # add play buttons, show buttons
        play_button_list, select_button_list, all_zero_slider = add_play_show_buttons_slider(trace_visibility, button_visibility, 
                                                                                      button_info, slider_info, epoch_list, training)
    if training:          
        # show all stats buttons (loss, score)
        stats_button_list, show_stats_button = add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, 
                                                                n_button_animations, button_info, method,
                                                                    back_stats_idx, pause_idx, select_dropdown_idx, just_loss)

    
    # show details  in select button [dropdown ]
    show_detail_button = add_detail_button(trace_visibility, button_visibility, config, back_detail_idx, pause_idx, select_dropdown_idx)
    
    show_data_button = add_data_button(button_visibility, trace_visibility, pause_idx)
    
    if not just_loss:
        # add pause button
        pause_button = dict(buttons=[dict(label=f"{f'&#9724;':^5}", method='animate',  args=[[None],  {'frame':{'duration': 0, 'redraw': False}, 
                                                'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'linear'}}])])
        pause_button.update(button_info)
        if not training:
            pause_button.update(dict(visible=False))
        pause_button.update(dict(x=-0.12, y=0.001))

    # [ back buttons ]
    back_stats_button, back_detail_button = add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, 
                                                        method, back_detail_idx, stats_idx, select_dropdown_idx, back_stats_idx)
    

    # all buttons  
    if training:
        if just_loss:
            button_list = stats_button_list + [back_stats_button] + [back_detail_button]     
            updatemenues = button_list + [dict(buttons=[show_detail_button]+[show_stats_button]+[show_data_button],
                                            visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
                                            x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')]
        else:   
            button_list = play_button_list + stats_button_list + [pause_button] + [back_stats_button] + [back_detail_button]     
            updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_detail_button]+[show_stats_button]+select_button_list[1:] + [show_data_button],
                                            visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
                                            x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')]
    else:
        button_list = play_button_list  + [pause_button]  + [back_detail_button]     
        updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_detail_button]+select_button_list[1:] + [show_data_button],
                                        visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
                                        x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')]
    # print(len(updatemenues))  
    fig.layout=go.Layout(
            updatemenus=updatemenues,
            sliders=[all_zero_slider])
    if not just_loss:
        frames = all_zero_sample_frames
        fig.frames = frames
    # Update layout for better aesthetics
    # scaleanchor='y', scaleratio=1,
    # scaleanchor='x', scaleratio=1,
    fig.update_layout(
        title=dict(text=method, pad={"r": 10, "t": 10, "l": 20, "b":10}, x=0.001, xanchor="left", y=0.98, yanchor="top"),
        xaxis=dict(title=dict(text="x", font_size=12), range=None,  showgrid=True, autorange=True),
        yaxis=dict(title=dict(text="y", font_size=12), range=None,  showgrid=True, autorange=True),
        # plot_bgcolor='rgba(100, 100, 100, 0)',
        margin=dict(l=80, t=60, b=20, r=150),
        showlegend=True,
        width=width,
        height=height
    )
    return fig

 

def plot_animation(expr_id, training=True, quiver_name=None, train_name=None, test_name=None):

    method = expr_id.split('_')[0] 
    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_sample_per_epoch.h5' if training else f'{path}/{expr_id}_df_sample.h5'  
    method_type = f"{method} [{config['loss_dsc']} {config['loss_gen']}]" + ("[training]"  if training else " [sampling]" )
    
    if train_name is not None:
        expr_id += train_name 
    if test_name is not None:
       file = f'{path}/{expr_id}_df_sample_{test_name}.h5' 


    
    df_sample, df_data, df_intermediate_smpl_list, df_score_info, df_score_info_hist,\
                                                    epoch_list, n_samples = read_hdf_all_epochs_train(file)

    df_loss, df_loss_itr, df_score = None, None, None 
    if training:
        # get statistics (loss, score) df
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_itr, df_score = read_hdf_loss(file)

    just_loss = True if len(df_sample)==0 else False
    fig_all = prepare_plotly_fig_quiver_grad(df_sample, df_data,  df_score_info, df_score_info_hist,
                                            df_loss_itr, df_loss, df_score, epoch_list, quiver_name, 
                                            method_type, config, just_loss, training)
          
    show_latex_in_plotly()
    return fig_all


 


if __name__=='__main__':
    # file = r'saved_result\\2spirals\\saved_hdfs_training\\DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64_df_sample_per_epoch.h5'
    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    expr_id = 'DDPM_beta_linear_T_40_ToyDDPM_4_64_swissroll_t_dim_1'
    expr_id = 'FlowMatching_T_40_ToyFlowMatching_4_64_swissroll_t_dim_1_gamma_0.025'
    expr_id = 'Regression_ToyRegressionNet_4_64_swissroll'
    expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_500_gamma_0.025'
    expr_id = 'GAN-RKL_ToyGAN_4_64_swissroll_z_dim_1_lr_dsc_1e-4_lr_gen_1e-4_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    expr_id = 'GAN-RKL_GANMNIST_2_16_MNIST_z_dim_8_lr_dsc_1e-4_lr_gen_1e-4_loss_dsc_stan_lc_0.5_0.5_1_loss_gen_heur_lc_1'
    method = expr_id.split('_')[0]
    # fig_sample, fig_loss = plot_animation(method, expr_id, training=True, all_epochs=True, quiver_name='grad')
    fig_sample = plot_animation(method, expr_id, training=True, quiver_name='grad')
    # fig_sample, fig_loss = plot_animation_regression(method, expr_id, training=False, all_zeros=True)