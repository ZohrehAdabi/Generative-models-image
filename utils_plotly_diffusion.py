

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

from utils_diffusion import get_fake_sample_grid

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




def prepare_df_zeros_for_plotly(df_sample, df_data, epoch_list, n_samples):

    df_sample['epoch'] = np.repeat(epoch_list, n_samples)
    df_sample['type'] = [f'sample_{i}' for i in range(1, n_samples+1)] * len(epoch_list)
    if df_data.shape[0] > 2000:
        df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    n_data = df_data.shape[0]
    df_data = pd.concat([df_data] * len(epoch_list)).reset_index(drop=True)
    df_data['epoch'] = np.repeat(epoch_list, n_data)
    df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * len(epoch_list)

    animation_frame = 'epoch'
    return df_sample, df_data, animation_frame

def prepare_df_for_plotly(df_sample, df_data, timesteps_list, n_sel_time):

    n_smpl = df_sample.shape[0] // n_sel_time
    
    df_sample['time'] = np.repeat(timesteps_list, n_smpl)
    df_sample['type'] = [f'sample_{i}' for i in range(1, n_smpl+1)] * n_sel_time
    if df_data.shape[0] > 2000:
        df_data = df_data.loc[np.linspace(0, df_data.shape[0]-1, 2000, dtype=int), :]
    n_data = df_data.shape[0]
    df_data = pd.concat([df_data] * n_sel_time).reset_index(drop=True)
    df_data['time'] = np.repeat(timesteps_list, n_data)
    df_data['type'] = [f'data_{i}' for i in range(1, n_data+1)] * n_sel_time

    animation_frame = 'time'
    return df_sample, df_data, animation_frame

def prepare_plotly_fig(df_sample, df_data, animation_frame='time'):

    if animation_frame=='time':
        df_sample = df_sample[::-1].reset_index(drop=True)
    fig_smpl = px.scatter(df_sample, x='x', y='y', animation_frame=animation_frame, width=600, height=600, 
                  animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]
                  )  
    fig_data = px.scatter(df_data, x='data_x', y='data_y', animation_frame=animation_frame, width=600, height=600, 
                    animation_group='type', range_x=[-2.5, 2.5], range_y=[-2.5, 2.5])
    fig_smpl['data'][0]['showlegend'] = True
    fig_data['data'][0]['showlegend'] = True

    for trace in fig_smpl.data:   # first display
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')  # ''

    for fr in fig_smpl.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#AB63FA'), name='sample')

    for trace in fig_data.data:   # first display
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')  # ''

    for fr in fig_data.frames:# on animation frames
        fr['data'][0]['showlegend'] = True    
        for trace in fr.data:    
            trace.update(marker=dict(size=5, color='#19D3F3'), name='data')

    return fig_smpl, fig_data


def add_all_zeros_animation_old(df_sample, epoch_list, colors, quiver_name='grad'):
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    
    
    px_anim_sample = px.scatter(df_sample, x='x', y='y', animation_frame='epoch')
    xy = df_grid_samples
    x, y = xy['x'].to_numpy(),  xy['y'].to_numpy() # inputs of ff.create_quiver must be numpy or pandas not tensor to function faster
    quiver_data = []
    for dfg in df_grad_g_z_list:
        u, v = dfg['u'],  dfg['v']
        # u, v = u.reshape(-1, grid_size), v.reshape(-1, grid_size)
        scale = np.sqrt(np.linalg.norm(x) / np.linalg.norm(u)) + np.sqrt(np.linalg.norm(y) / np.linalg.norm(v))
                    # u is added to x and v is added to y.
        fig_quiver = ff.create_quiver(x, y, u, v, scale=scale, name=f'{quiver_name}', arrow_scale=.2,
                                                angle=40*np.pi/180, line=dict(width=1, color=colors['grad']))
        quiver_data.append([fig_quiver.data[0]])

    all_zero_sample_frames = [go.Frame(data=[go.Scatter(x=smpl_frame.data[0].x, y=smpl_frame.data[0].y,  mode='markers', showlegend=True, 
                                             line=dict(color=colors['sample']), name=f'sample', hovertemplate='x=%{x}<br>y=%{y}'),
                                             ] + grad_frame,  
                            traces=[1, 2], name=f'zero_epoch_{e}', group=f'all_zero')  
                            for e, smpl_frame, grad_frame in zip(epoch_list, px_anim_sample.frames, quiver_data)]

    return all_zero_sample_frames 

def add_all_zeros_animation(df_sample, epoch_list, colors):
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    
    
    px_anim_sample = px.scatter(df_sample, x='x', y='y', animation_frame='epoch')

    all_zero_sample_frames = [go.Frame(data=[go.Scatter(x=smpl_frame.data[0].x, y=smpl_frame.data[0].y,  mode='markers', showlegend=True, 
                                             line=dict(color=colors['sample']), name=f'sample', hovertemplate='x=%{x}<br>y=%{y}'),
                                             ], traces=[1], name=f'zero_epoch_{e}', group=f'all_zero')  
                                            for e, smpl_frame in zip(epoch_list, px_anim_sample.frames)]

    return all_zero_sample_frames 

def add_epochs_animation_slider(df_intermediate_smpl_list, df_grad_or_noise_grid_list, df_grid_samples, 
                                epoch_list, timesteps_list, colors, quiver_name='grad'):
    # add animations of all epochs
    animation_list = []
    xy = df_grid_samples
    x, y = xy['x'].to_numpy(),  xy['y'].to_numpy() # inputs of ff.create_quiver must be numpy or pandas not tensor to function faster
    scale = 0.1
    for i, (e, dfs, dfg) in enumerate(zip(epoch_list, df_intermediate_smpl_list, df_grad_or_noise_grid_list)): #traces: data samples grads

        df_grad_frames_grous = dfg.groupby(dfg.time)
        quiver_data = []
        for t in timesteps_list:
            df_grad_frame = df_grad_frames_grous.get_group(t)
            u, v = df_grad_frame['u'],  df_grad_frame['v']
            # scale = np.sqrt(np.linalg.norm(x) / np.linalg.norm(u)) + np.sqrt(np.linalg.norm(y) / np.linalg.norm(v))
            fig_quiver = ff.create_quiver(x, y, -u, -v, scale=scale, name=f'{quiver_name}', arrow_scale=.2,
                                                angle=40*np.pi/180, line=dict(width=1, color=colors['grad']))
            quiver_data.append([fig_quiver.data[0]])

        # last sample will be at the end of animation
        px_anime_sample = px.scatter(dfs, x='x', y='y', animation_frame='time')
        sample_frames = [go.Frame(data=[go.Scatter(x=frame['data'][0].x, y=frame['data'][0].y,  mode='markers', showlegend=True,
                                                line=dict(color=colors['sample']), name=f'sample_{e}', hovertemplate='x=%{x}<br>y=%{y}')
                                     ]+ grad_frame, 
                                     traces=[2*(i+1), 2*(i+1)+1], name=f'frame_{e}_{t}', group=f'epoch_{e}')  
                                     for t, frame, grad_frame in zip(timesteps_list, px_anime_sample.frames, quiver_data)]

        animation_list.append(sample_frames)

    return animation_list

def add_data_sample_trace(fig, df_data, all_zero_sample_frames, animation_list, epoch_list, colors, quiver_name='grad'):

    # data, sample_zero at first epoch , grad at first epoch
    fig.add_trace(go.Scatter(x=df_data['data_x'], y=df_data['data_y'], mode='markers', line=dict(color=colors['data']), name=f'data',
                            visible=True, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    fig.add_trace(go.Scatter(x=all_zero_sample_frames[0].data[0].x, y=all_zero_sample_frames[0].data[0].y, mode='markers', 
                             line=dict(color=colors['sample']), name=f'sample', visible=True, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    
    
    # sample trace of all epoch in epoch_list
    for e, anim in zip(epoch_list, animation_list):
        fig.add_trace(go.Scatter(x=anim[0].data[0].x, y=anim[0].data[0].y, mode='markers', 
                                line=dict(color=colors['sample']), name=f'sample_{e}', visible=False, showlegend=True))
        fig.add_trace(go.Scatter(x=anim[0].data[1].x, y=anim[0].data[1].y, mode='lines', 
                             line=dict(color=colors['grad']), name=f'{quiver_name}_{e}', visible=False, showlegend=True, hovertemplate='x=%{x}<br>y=%{y}'))
    
    return fig

def add_statistics_to_fig(fig, df_loss_itr, df_loss):

    # add traces of statistics
  
    x = df_loss_itr['itr']
    fig.add_trace(go.Scatter(x=x, y=df_loss_itr['loss_itr'], name="Loss", mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen')))
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss_itr['loss_itr']) / x), name="Loss mean", mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    
    x = df_loss['epoch']
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_epoch'], name='Loss', mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen'))) # lightsalmon
    fig.add_trace(go.Scatter(x=x, y=(np.cumsum(df_loss['loss_epoch']) / x), name='Loss mean' , mode='lines', visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    fig.update_layout(title=dict(text='Loss'))

    return fig

def add_smpl_grads_or_noises_to_fig(fig, df_grads_or_noises_info, timesteps_list, epoch_list):
    x = timesteps_list
    for e, df in zip(epoch_list, df_grads_or_noises_info):
        fig.add_trace(go.Scatter(x=x, y=df['sample_noise_norm'], name=f"norm_{e}", mode='lines+markers', visible=False, showlegend=True, line=dict(width=1.5)))   
    return fig

def visibility_traces(fig, epoch_list):
    # visibility of traces when coresponding button is clicked
    trace_visibility = dict()
    for trace in fig.data:
        if trace.name not in  trace_visibility.keys():
            trace_visibility[trace.name] = False
        elif 'Loss' in trace.name:
            trace_visibility[f'{trace.name} epoch'] = False

    # ['data', 'sample', 'sample_50', 'grad_50', 'sample_100', 'grad_100', 'sample_150', 'grad_150', 
    #  'sample_200', 'grad_200', 'sample_250', 'grad_250', 'sample_300', 'grad_300', 'sample_350', 'grad_350', 
    #  'sample_400', 'grad_400', 'sample_450', 'grad_450', 'sample_500', 'grad_500', 'sample_550', 'grad_550', 
    #  'sample_600', 'grad_600', 'sample_650', 'grad_650', 'sample_700', 'grad_700', 'sample_750', 'grad_750', 
    #  'sample_800', 'grad_800', 'sample_850', 'grad_850', 'sample_900', 'grad_900', 'sample_950', 'grad_950', 
    #  'sample_1000', 'grad_1000', 'Loss', 'Loss mean', 'Loss', 'Loss mean', 
    #  'norm_50', 'norm_100', 'norm_150', 'norm_200', 'norm_250', 'norm_300', 'norm_350', 'norm_400', 'norm_450', 
    #  'norm_500', 'norm_550', 'norm_600', 'norm_650', ...]
    
    trace_visibility['data'] = True

    trace_vis_loss_itr = trace_visibility.copy()
    trace_vis_loss_itr.update({f'Loss': True, f'Loss mean': True, 'data': False})
    trace_vis_loss_epoch = trace_visibility.copy()
    trace_vis_loss_epoch.update({f'Loss epoch': True, f'Loss mean epoch': True, 'data': False})

    stats_trace_vis = [trace_vis_loss_itr, trace_vis_loss_epoch]
    smpl_grad_or_noise_norm_trace_vis = trace_visibility.copy()
    smpl_grad_or_noise_norm_trace_vis['data'] = False
    # smpl_score_trace_vis['D*'] = True
    for e in epoch_list:
        smpl_grad_or_noise_norm_trace_vis[f'norm_{e}'] = True

    return trace_visibility, stats_trace_vis, smpl_grad_or_noise_norm_trace_vis

def visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics):

    # store idx of other buttons that adding to the layout for making visible True or False
    
    stats_idx = n_button_animations
    pause_idx = n_button_animations + n_button_statistics
    back_stats_idx = pause_idx + 1
    back_detail_idx = back_stats_idx + 1
    back_smpl_grad_noise_norm_idx = back_detail_idx + 1
    select_dropdown_idx = back_smpl_grad_noise_norm_idx + 1 
     
    # buttons visibility
    button_visibility = {}
    for i in range(n_all_buttons):
        button_visibility[f'updatemenus[{i}].visible'] = False 
    button_visibility[f'updatemenus[{pause_idx}].visible'] = True
    button_visibility[f'updatemenus[{select_dropdown_idx}].visible'] = True

    return button_visibility, stats_idx, back_smpl_grad_noise_norm_idx, pause_idx, \
                back_stats_idx, back_detail_idx, select_dropdown_idx

def add_play_show_buttons_slider(trace_visibility, button_visibility, timesteps_list, button_info, 
                                 slider_info, frame_spec, epoch_list, quiver_name='grad'):
    
    play_button_list = []
    select_button_list = [] #drop down button

    # Play buttons for sampling
       

    zero_play_button = dict(buttons=[dict(label=f"{f'&#9654; zeros':^16}", method='animate', args=[f'all_zero' , frame_spec])])
    zero_play_button.update(button_info)
    zero_play_button.update(dict(x=-0.22, y=0.001))

    play_button_list.append(zero_play_button)
    # slider for zero play button
    
    all_zero_slider = dict(steps=[dict(args=[[f'zero_epoch_{e}'], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate' }], 
                                method='animate', label=f'{e}' ) for e in epoch_list] )
    all_zero_slider.update(slider_info)

    visibility_layout = {'title': f'All zeros'}
    visibility_layout.update(button_visibility)
    visibility_layout[f'updatemenus[0].visible'] = True # show play zeros buttons
    visibility_layout['sliders'] = [all_zero_slider] 

    # add fist button to dropdwon 
    trace_vis_init = trace_visibility.copy()
    trace_vis_init.update({f'sample': True, f'grad': True})
    select_button_list.append(dict(label=f'Show all zeros', method='update', 
                                args=[{'visible': list(trace_vis_init.values())}, visibility_layout]))

    vis_layout = visibility_layout.copy()
    vis_layout[f'updatemenus[0].visible'] = False
   # add play buttons for all epochs
    for i, e in enumerate(epoch_list):

        slider_steps = dict(steps=[dict(args=[[f'frame_{e}_{t}'], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate' }], 
                                    method='animate', label=f'{t}' ) for t in timesteps_list])
        slider_steps.update(slider_info)
        slider_steps.update({'currentvalue': dict(font_size=12, prefix= f'Epoch{e}: ', xanchor='left') , 'name': f'e_{e}'})
        
        play_button = dict(buttons=[dict(label=f"{f'&#9654; {e}':^20}", method='animate', args=[f'epoch_{e}', frame_spec])])
        play_button.update(button_info)
        play_button.update(dict(visible=False, x=-0.22, y=0.001))
        play_button_list.append(play_button)
        vis = {}
        vis.update(vis_layout)
        vis['title'] =  f'Epoch {e}'
        vis['sliders'] = [slider_steps]
        vis[f'updatemenus[{i+1}].visible'] = True # first play button is for all zeros
        
        trace_vis_epoch = trace_visibility.copy()
        trace_vis_epoch.update({f'sample_{e}': True})
        trace_vis_epoch.update({f'{quiver_name}_{e}': True})

        select_button_list.append(dict(label=f'Show epoch {e}', method='update', 
                                       args=[{'visible': list(trace_vis_epoch.values())}, vis]))  
        

    # select_button_list.append(dict(label=f'', method='update',  args=[None]))

    return play_button_list, select_button_list, all_zero_slider

def add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, n_button_animations, button_info, method,
                      back_stats_idx, pause_idx, select_dropdown_idx):

    # Show all stats buttons (loss, score)
    stats_button_list = []
    show_stats = {'title': method, 'width':1000, 'height': 450, 'xaxis.autorange': True, 'yaxis.autorange': True} 
                  #'xaxis.range': [-2, 9], 'yaxis.range': [-3, 7]
    show_stats['sliders'] = []
    show_stats.update(button_visibility) 
    show_stats[f'updatemenus[{back_stats_idx}].visible'] = True
    show_stats[f'updatemenus[{pause_idx}].visible'] = False
    show_stats[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    
    stats_lables = ['Loss itr', 'Loss epoch']
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
    show_stats_button = dict(label=f'Show Loss', method='update', args=[{'visible': list(stats_vis.values())}, show_stats])

    return stats_button_list, show_stats_button

def add_sampling_grad_noise_norm_button(button_visibility, smpl_grad_noise_trace_vis, back_smpl_grad_noise_norm_idx, 
                                        pause_idx, select_dropdown_idx, quiver_name='grad'):

    # Show sample score (validation) 
    show_smpl_grd_ns = {'title': f'Sample {quiver_name} norm', 'width':1000, 'height': 450, 'xaxis.autorange': True, 'yaxis.autorange': True} 
              #'xaxis.range': [-2, 9], 'yaxis.range': [-3, 7]
    show_smpl_grd_ns['sliders'] = []
    show_smpl_grd_ns.update(button_visibility) 
    show_smpl_grd_ns[f'updatemenus[{back_smpl_grad_noise_norm_idx}].visible'] = True
    show_smpl_grd_ns[f'updatemenus[{pause_idx}].visible'] = False
    show_smpl_grd_ns[f'updatemenus[{select_dropdown_idx}].visible'] = False 

    # Show smpl score in select button [dropdown]

    show_smpl_grd_ns_button = dict(label=f'Show {quiver_name} norm', method='update', args=[{'visible': list(smpl_grad_noise_trace_vis.values())}, show_smpl_grd_ns])

    return show_smpl_grd_ns_button
    
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
    
    show_detail.update({'margin.t':100, 'title.y': 0.6, 'title.x':0.2, 'xaxis.range': [-2.5, 2.5], 'yaxis.range': [-2.5, 2.5], 
                            'xaxis.tickvals': [], 'yaxis.tickvals': [], 'xaxis.title': '', 'yaxis.title': ''
                            }) 
    show_detail.update(dict(plot_bgcolor='white', height=450))
    hide_traces = trace_visibility.copy()
    hide_traces['data'] = False
    show_detail_button = dict(label=f'Show details', method='update', args=[{'visible': list(hide_traces.values())}, show_detail])
            #    [dict(label=f'Hide Loss-Score', method='update', args=[{'y': [], 'x': []}, hide_loss]),
            #    dict(label=f'1 details', method='relayout', args=[{'title': f'GAN 1'}])]
    return show_detail_button

def add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, method,
                     back_detail_idx, stats_idx, back_smpl_score_idx, select_dropdown_idx, back_stats_idx):

    hide_info = {'margin.t': 60, 'title.y': 0.98, 'title.x':0.001, 'showlegend': True, 'plot_bgcolor': None, 'width':750, 'height': 570,
                'xaxis.tickvals': None, 'yaxis.tickvals': None, 'xaxis.range': [-2.5, 2.5], 'yaxis.range': [-2.5, 2.5], 
                'xaxis.autorange': False, 'yaxis.autorange': False, 'xaxis.title': 'x', 'yaxis.title': 'y'}

    slider_steps = dict(steps=[dict(args=[[None]], method='animate', label=f'{j}' ) for j in range(2)])
    slider_steps.update(slider_info)
    slider_steps.update({'currentvalue': dict(font_size=12, prefix= f'', xanchor='left')})
    hide_info['sliders'] = [slider_steps]

    # add back button [stats]
    hide_stats = {'title': method}
    hide_stats.update(hide_info)
    # changing vsibility to False must be after changing title and slider to slider work properly
    for i in range(n_button_statistics):
        hide_stats[f'updatemenus[{stats_idx+i}].visible'] = False

    
    hide_stats[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_stats[f'updatemenus[{back_stats_idx}].visible'] = False
    back_stats_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_visibility.values())}, hide_stats])]) 
    back_stats_button.update(button_info)
    back_stats_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))

    # add back button  [details]
    hide_detail = {'title.text': method}
    hide_detail.update(hide_info) 
    hide_detail[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_detail[f'updatemenus[{back_detail_idx}].visible'] = False
    back_detail_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_visibility.values())}, hide_detail])]) 
    back_detail_button.update(button_info)
    back_detail_button.update(dict(visible=False, x=0.1, y=1.02, font=dict(size=15)))

    # add back button  [sample score]
    hide_smpl_grad_noise = {'title.text': method}
    hide_smpl_grad_noise.update(hide_info) 
    hide_smpl_grad_noise[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_smpl_grad_noise[f'updatemenus[{back_smpl_score_idx}].visible'] = False
    back_smpl_grad_noise_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_visibility.values())}, hide_smpl_grad_noise])]) 
    back_smpl_grad_noise_button.update(button_info)
    back_smpl_grad_noise_button.update(dict(visible=False, x=0.20, y=1.02, font=dict(size=15)))
    
    return back_stats_button, back_detail_button, back_smpl_grad_noise_button
    

def read_hdf_all_epochs_train(file):

    # try:
    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()
    # for k in keys:
    #     print(k)
    df_sample = pd.DataFrame()
    df_grads_or_noises_info = pd.DataFrame()
    # df_grad_or_noise_hist = []
    # df_grad_g_z = pd.DataFrame()
    # df_grad_g_z_at_sampling = pd.DataFrame()
    df_fake_data = pd.DataFrame()
    # df_intermediate_smpl = pd.DataFrame()
    df_intermediate_smpl_list = []
    df_grad_or_noise_grid_list = []
    df_grad_or_noise_info_hist = []
    epoch_list = []
    for k in keys:
        if 'samples' in k: 
            df_sample = pd.concat([df_sample if not df_sample.empty else None, hdf_file[k]])
            epoch_list.append(int(k.split('/')[2].split('_')[2]))
        elif 'intermediate_smpl'in k:
            # df_intermediate_smpl = pd.concat([df_intermediate_smpl if not df_intermediate_smpl.empty else None, hdf_file[k]])
            df_intermediate_smpl_list.append(hdf_file[k])
        
        # elif 'grad_info' in k or 'noise_info' in k:
        #     df_grads_or_noises_info = pd.concat([df_grads_or_noises_info if not df_grads_or_noises_info.empty else None, hdf_file[k]])
        elif 'grad_info' in k or 'noise_info' in k: # noise or grad predicted for of samples generated iteratively
            df_grad_or_noise_info_hist.append(hdf_file[k])
    
        elif 'grad_grid' in k or 'noise_grid' in k:
            # df_grad_g_z = pd.concat([df_grad_g_z if not df_grad_g_z.empty else None, hdf_file[k]])
            df_grad_or_noise_grid_list.append(hdf_file[k])
        

    df_data = hdf_file['/df/data']
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    grid_size = hdf_file['/df/info']['grid_size'].values[0]
    timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
    hdf_file.close()
    # except:
    #     print(f'Error in reading hdf file {file}')
    #     hdf_file.close()

    return df_sample, df_data, df_intermediate_smpl_list, df_grad_or_noise_info_hist, \
            df_grad_or_noise_grid_list, epoch_list, timesteps_list, n_samples, grid_size

def read_hdf_loss(file):

    df_loss = pd.read_hdf(file)   
    df_loss_itr = pd.DataFrame(df_loss.loc[0, ['loss_itr']].values[0], columns=['loss_itr'])
    for i in range(1, df_loss.shape[0]):
        df_loss_itr = pd.concat([df_loss_itr, pd.DataFrame(df_loss.loc[i, ['loss_itr']].values[0], columns=['loss_itr'])])
    df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    return df_loss.loc[:, ['epoch', 'loss_epoch']], df_loss_itr

def prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, df_grid_samples, 
                                    df_grad_or_noise_grid_list, df_grads_or_noises_info, 
                                    df_loss_itr, df_loss, epoch_list, timesteps_list, 
                                    quiver_name, method, config,
                                   width=750, height=570, range_x=[-2.5, 2.5], range_y=[-2.5, 2.5]):

    # The traces must be added in the correct order for the button's visibility to function properly
   
    num_epochs = len(epoch_list)
    colors = dict(sample='#AB63FA', grad='DarkSlateGrey', data='#19D3F3', fake='mediumorchid') # samples-grads-data
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    all_zero_sample_frames = add_all_zeros_animation(df_sample, epoch_list, colors)

    # add animations of all epochs
    animation_list = add_epochs_animation_slider(df_intermediate_smpl_list, df_grad_or_noise_grid_list, df_grid_samples, 
                                                                        epoch_list, timesteps_list, colors, quiver_name)

    fig = go.Figure() 
    # data, sample_zero at first epoch , grad at first epoch
    fig = add_data_sample_trace(fig, df_data, all_zero_sample_frames, animation_list, epoch_list, colors, quiver_name)

    # add traces of statistics
    fig = add_statistics_to_fig(fig, df_loss_itr, df_loss)
    # add smpl grad_or_noise norm trace
    fig = add_smpl_grads_or_noises_to_fig(fig, df_grads_or_noises_info, timesteps_list, epoch_list)

    # visibility of traces when coresponding button is clicked
    trace_visibility, stats_trace_vis, smpl_grad_noise_trace_vis = visibility_traces(fig, epoch_list)
    # n_button_animations equals to plays buttons
    n_button_animations = len(animation_list) + 1  # 1: for all_zeros button 
    n_button_statistics = 2 # loss itr, loss epoch
    n_all_buttons = n_button_animations + n_button_statistics + 1 + 2 + 2  # 1: show sample score val# show stats, show detail, show eval, 
                                                                            # back_stats, back_detail, back_eval
    # n_traces_animation = 1 + 1 +  2*num_epochs # 1:data, 1: grad first epoch, len(epoch_list): samples of each epoch
    # n_traces_statistics = 2 + 2 +   # 2: loss itr 2: loss epoch

    # store idx of other buttons that adding to the layout for making visible True or False
    # buttons visibility
    button_visibility, stats_idx, back_smpl_grad_noise_norm_idx, pause_idx, \
    back_stats_idx, back_detail_idx, select_dropdown_idx = visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics)

    frame_spec = {'frame': {'duration': 300, 'redraw': False}, 'fromcurrent': True, 'mode': 'immediate', 
              'transition': {'duration': 20, 'easing': 'cubic-in-out'}}
    
    button_info = dict(visible=True, type='buttons', direction='down', showactive=False, pad={"r": 10, "t": 10, "l": 10}, 
                         xanchor="right", yanchor="bottom", bgcolor='seashell')
    slider_info = dict(active=0, name=f'all_epoch', visible=True, currentvalue=dict(font_size=12, prefix=f'zero of epoch: ', xanchor='left'),
                 transition=dict(duration=300, easing='cubic-in-out'), pad=dict(b=10, t=60), len=1, x=0.0, y=0.08, xanchor='left', yanchor='top')
    # add play buttons, show buttons
    play_button_list, select_button_list, all_zero_slider = add_play_show_buttons_slider(trace_visibility, button_visibility, timesteps_list, 
                                                                                         button_info, slider_info, frame_spec, epoch_list, quiver_name)
    

    # show all stats buttons (loss, score)
    stats_button_list, show_stats_button = add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, 
                                                             n_button_animations, button_info, method,
                                                                back_stats_idx, pause_idx, select_dropdown_idx)

    # add smple grad_noise_norm [validation]
    show_smpl_grad_noise_norm_button = add_sampling_grad_noise_norm_button(button_visibility, smpl_grad_noise_trace_vis, back_smpl_grad_noise_norm_idx, pause_idx, select_dropdown_idx, quiver_name)
    # show details  in select button [dropdown ]
    show_detail_button = add_detail_button(trace_visibility, button_visibility, config, back_detail_idx, pause_idx, select_dropdown_idx)
    
    # add pause button
    pause_button = dict(buttons=[dict(label=f"{f'&#9724;':^5}", method='animate',  args=[[None],  {'frame':{'duration': 0, 'redraw': False}, 
                                            'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'linear'}}])])
    pause_button.update(button_info)
    pause_button.update(dict(x=-0.12, y=0.001))

    # [ back buttons ] 
    back_stats_button, back_detail_button, back_smpl_score_button = add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, 
                                                                    method, back_detail_idx, stats_idx, back_smpl_grad_noise_norm_idx, select_dropdown_idx, back_stats_idx)
    
    button_list = play_button_list + stats_button_list + [pause_button] + [back_stats_button] + [back_detail_button] + [back_smpl_score_button]

    # all buttons          
    updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_detail_button]+[show_stats_button]+select_button_list[1:]+[show_smpl_grad_noise_norm_button],
                                        visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
                                        x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')]
    # print(len(updatemenues))  
    fig.layout=go.Layout(
            updatemenus=updatemenues,
            sliders=[all_zero_slider])
    frames = all_zero_sample_frames 
    for i in range(len(animation_list)):
        frames += animation_list[i]
    fig.frames = frames
    # Update layout for better aesthetics
    # scaleanchor='y', scaleratio=1,
    # scaleanchor='x', scaleratio=1,
    fig.update_layout(
        title=dict(text=method, pad={"r": 10, "t": 10, "l": 20, "b":10}, x=0.001, xanchor="left", y=0.98, yanchor="top"),
        xaxis=dict(title=dict(text="x", font_size=12), range=range_x,  showgrid=True, autorange=False),
        yaxis=dict(title=dict(text="y", font_size=12), range=range_y,  showgrid=True, autorange=False),
        # plot_bgcolor='rgba(100, 100, 100, 0)',
        margin=dict(l=80, t=60, b=20, r=150),
        showlegend=True,
        width=width,
        height=height
    )
    return fig
 

def plot_animation(method, expr_id, training=True, epoch_key=-1, all_epochs=False, quiver_name=None, test_name=None):

    dataset_name, config = get_dataset_name(method, expr_id)
    path = f'saved_result/{method}/{dataset_name}/saved_hdfs_training' if training else f'saved_result/{method}/{dataset_name}/saved_hdfs'
    file = f'{path}/{expr_id}_df_sample_per_epoch.h5' if training else f'{path}/{expr_id}_df_sample.h5'  
    method_type = f"{method}"
    
    if test_name is not None:
       file = f'{path}/{expr_id}_df_sample_{test_name}.h5' 


    if training and all_epochs:
        df_sample, df_data, df_intermediate_smpl_list, df_grads_or_noises_info, \
            df_grad_or_noise_grid_list, epoch_list, timesteps_list, n_samples, grid_size = read_hdf_all_epochs_train(file)

        grid_samples, grid_x, grid_y = get_fake_sample_grid(grid_size) # for ploting quiver
        df_grid_samples = pd.DataFrame(grid_samples.numpy(), columns=['x', 'y'])
        # get statistics (loss, score) df
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_itr = read_hdf_loss(file)

        fig_all = prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, df_grid_samples, 
                                                            df_grad_or_noise_grid_list, df_grads_or_noises_info, 
                                                            df_loss_itr, df_loss, epoch_list, timesteps_list, 
                                                            quiver_name, method_type, config)
          
    show_latex_in_plotly()
    return fig_all





def read_hdf_regression(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()[3:-1]
    # for k in keys:
    #     print(k)
    df_sample = hdf_file[keys[-1]]
    df_data = hdf_file['/df/data']
   
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    error = hdf_file['/df/error']['error'].values[0]

    hdf_file.close()

    return df_sample, df_data, n_samples, error

def read_hdf_prediction_regression_train(file):

    hdf_file = pd.HDFStore(file)   
    # print(smple.keys())
    keys = hdf_file.keys()
    # for k in keys:
    #     print(k)
    df_prediction = pd.DataFrame()
    epoch_list = []
    error_list = []
    for k in keys:
        if 'prediction' in k:
            df_prediction = pd.concat([df_prediction if not df_prediction.empty else None, hdf_file[k]])
            epoch_list.append(int(k.split('/')[2].split('_')[2]))
        elif 'error' in k:
            error_list.append(hdf_file[k]['error'].values[0])

    df_data = hdf_file['/df/data']
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    hdf_file.close()
    return df_prediction, df_data, epoch_list, n_samples, error_list

def read_hdf_loss_regression(file):

    df_loss = pd.read_hdf(file)   
    df_loss_itr = pd.DataFrame(df_loss.loc[0, ['loss_itr']].values[0], columns=['loss_itr'])
    for i in range(1, df_loss.shape[0]):
        df_loss_itr = pd.concat([df_loss_itr, pd.DataFrame(df_loss.loc[i, ['loss_itr']].values[0], columns=['loss_itr'])])
    df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    return df_loss.loc[:, ['epoch', 'loss_epoch']], df_loss_itr

def create_fig_loss_regression(df_loss, df_loss_itr):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=df_loss_itr['loss_itr'], name="Loss itr", mode='lines', showlegend=True, line=dict(width=1, color='#AB63FA')))
    fig.add_trace(go.Scatter(x=df_loss_itr['itr'], y=(np.cumsum(df_loss_itr['loss_itr']) / df_loss_itr['itr']), name="Loss mean itr", mode='lines', showlegend=True, line=dict(width=1, color='darkviolet')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], name='Loss epoch', mode='lines', showlegend=True, line=dict(width=2, color='lightsalmon')))
    fig.add_trace(go.Scatter(x=df_loss['epoch'], y=(np.cumsum(df_loss['loss_epoch']) / df_loss['epoch']), name='Loss mean' ,mode='lines', line=dict(width=2, color='fuchsia')))
    fig.update_layout(title=dict(text='Loss'))
    fig.update_layout(width=800, height=320, margin=dict(l=20, r=20, b=5, t=30, pad=5))
    return fig


def create_fig_prediction_regression(df_prediction, df_data, error):

    fig = go.Figure()
    # fig.add_trace(go.Scatter(x=df_loss['epoch'], y=df_loss['loss_epoch'], mode='lines', name='Loss', line=dict(width=1, color='DarkSlateGrey')))
    # fig.update_layout(title=dict(text='Loss per epoch'))
    fig.add_trace(go.Scatter(x=df_data['data_x'], y=df_data['data_y'], name="data", mode='markers', showlegend=True, line=dict(width=1, color='#19D3F3')))
    fig.add_trace(go.Scatter(x=df_prediction['x'], y=df_prediction['y'], name="pred", mode='markers', showlegend=True, line=dict(width=1, color='AB63FA')))
    

    fig.update_layout(title=dict(text=f'MSE: {error}'), xaxis_range=[-2.5, 2.5], yaxis_range=[-2.5, 2.5])
    fig.update_layout(width=500, height=500, margin=dict(l=20, r=20, b=20, t=30, pad=5))
    return fig
  
def final_plot(fig_sample, fig_data):

    fig_all = go.Figure(
        data=fig_data.data + fig_sample.data,
        frames=[
                go.Frame(data=fr_data.data + fr_smpl.data, name=fr_smpl.name)
                for fr_data, fr_smpl in zip(fig_data.frames, fig_sample.frames)
        ],
        layout=fig_sample.layout
        )
    fig_all.update_layout(title=dict(text='Sampling'), width=500, height=500, margin=dict(l=20, r=20, b=5, t=30, pad=5)) 
    fig_all['layout'].updatemenus[0]['pad'] = {'r': 10, 't': 40}
    fig_all['layout'].sliders[0]['pad'] = {'r': 10, 't': 25}
    return fig_all



 


if __name__=='__main__':
    # file = r'saved_result\\2spirals\\saved_hdfs_training\\DDPM_beta_linear_T_300_ToyNetwork2_2spirals_t_dim_64_df_sample_per_epoch.h5'
    expr_id = 'DDPM_beta_linear_T_300_ToyNetwork2_25gaussians_t_dim_64'
    expr_id = 'DDPM_beta_linear_T_40_ToyDDPM_4_64_swissroll_t_dim_1'
    # expr_id = 'FlowMatching_T_40_ToyFlowMatching_4_64_swissroll_t_dim_1_gamma_0.025'
    # expr_id = 'Regression_ToyRegressionNet_4_64_swissroll'
    # expr_id = 'Boosting_T_40_ToyBoosting_4_64_swissroll_t_dim_1_innr_ep_500_gamma_0.025'
    expr_id = 'DDPM-Hidden_beta_linear_T_40_ToyDDPMHidden_4_64_swissroll_t_dim_1_h_size_2'
    expr_id = 'DDPM_beta_linear_T_40_UNetMNIST_2_16_MNIST_t_dim_16'
    method = expr_id.split('_')[0]
    fig = plot_animation(method, expr_id, training=True, all_epochs=True, quiver_name='noise')
    # fig_sample, fig_loss = plot_animation_regression(method, expr_id, training=False, all_zeros=True)