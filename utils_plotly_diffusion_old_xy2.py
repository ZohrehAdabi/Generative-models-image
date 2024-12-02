

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





def add_all_zeros_animation(df_sample, epoch_list, colors):
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    
    
    px_anim_sample = px.imshow(df_sample.squeeze(), animation_frame=0, binary_string=True, binary_compression_level=4)

    all_zero_sample_frames = [go.Frame(data=[smpl_frame.data[0].update(name='sample', xaxis="x", yaxis="y")]
                                , traces=[0], name=f'zero_epoch_{e}', group=f'all_zero')
                                            for e, smpl_frame in zip(epoch_list, px_anim_sample.frames)]

    return all_zero_sample_frames 

def add_epochs_animation_slider(df_intermediate_smpl_list, epoch_list, timesteps_list):
    # add animations of all epochs
    animation_list = []

    for i, (e, dfs) in enumerate(zip(epoch_list, df_intermediate_smpl_list)): #traces: data samples

        # last sample will be at the end of animation
        px_anime_sample = px.imshow(dfs.squeeze(), animation_frame=0, binary_string=True, binary_compression_level=4)
        sample_frames = [go.Frame(data=[frame.data[0].update(name=f'sample_{e}', visible=True, xaxis="x", yaxis="y") ], 
                                     traces=[1+i], name=f'frame_{e}_{t}', group=f'epoch_{e}')  
                                     for t, frame in zip(timesteps_list, px_anime_sample.frames)]

        animation_list.append(sample_frames)

    return animation_list

def add_data_trace(fig, df_data):

    # data
    fig.add_trace(px.imshow(df_data.squeeze(), binary_string=True).data[0].update(name='data', xaxis="x", yaxis="y", visible=False))
    
    gray = df_data.squeeze()
    rgb = np.stack([gray, gray, gray], axis=-1)
    alpha = np.full(gray.shape, 25, dtype=np.uint8)  # 255 Fully opaque
    rgba = np.concatenate([rgb, alpha[..., None]], axis=-1)
    fig.add_trace(px.imshow(rgba, binary_string=True).data[0].update(name='data_transparnce', xaxis="x", yaxis="y", visible=False))
    
    return fig

def add_sample_trace(fig, all_zero_sample_frames, animation_list, epoch_list):

    # sample_zero at first epoch , grad at first epoch
    fig.add_trace(all_zero_sample_frames[0].data[0].update(name=f'sample', visible=True))
    # sample trace of all epoch in epoch_list
    for e, anim in zip(epoch_list, animation_list):
        fig.add_trace(anim[0].data[0].update(name=f'sample_{e}')).update_traces(visible=False, xaxis="x", yaxis="y", selector=dict(name=f'sample_{e}'))

    return fig

def add_statistics_to_fig(fig, df_loss_itr, df_loss, xaxis='x2', yaxis='y2'):

    # add traces of statistics
  
    x = df_loss_itr['itr']
    y = (np.cumsum(df_loss_itr['loss_itr']) / x)
    y[0] = df_loss_itr['loss_itr'][0]
    fig.add_trace(go.Scatter(x=x, y=df_loss_itr['loss_itr'], name="Loss", xaxis=xaxis, yaxis=yaxis, mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen')))
    fig.add_trace(go.Scatter(x=x, y=y, name="Loss mean", mode='lines', xaxis=xaxis, yaxis=yaxis, visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    
    x = df_loss['epoch']
    y = (np.cumsum(df_loss['loss_epoch']) / x)
    y[0] = df_loss['loss_epoch'][0]
    fig.add_trace(go.Scatter(x=x, y=df_loss['loss_epoch'], name='Loss', xaxis=xaxis, yaxis=yaxis, mode='lines', visible=False, showlegend=True, line=dict(width=1, color='limegreen'))) # lightsalmon
    fig.add_trace(go.Scatter(x=x, y=y, name='Loss mean' , mode='lines', xaxis=xaxis, yaxis=yaxis, visible=False, showlegend=True, line=dict(width=2, color='darkgreen')))
    
    # fig.add_trace(go.Scatter(x=np.arange(100), y=np.arange(100), name='Line' , mode='lines', xaxis=xaxis, yaxis=yaxis, visible=False, showlegend=False, line=dict(width=2, color='red')))
  

    return fig

def add_smpl_grads_or_noises_info_to_fig(fig, df_grads_or_noises_info, timesteps_list, epoch_list, xaxis='x2', yaxis='y2'):
    x = timesteps_list
    for e, df in zip(epoch_list, df_grads_or_noises_info):
        fig.add_trace(go.Scatter(x=x, y=df['sample_noise_norm'], name=f"norm_{e}", xaxis=xaxis, yaxis=yaxis, mode='lines+markers', visible=False, showlegend=True, line=dict(width=1.5)))   
    return fig

def visibility_traces(fig, epoch_list):
    # visibility of traces when coresponding button is clicked
    trace_visibility = dict()
    for trace in fig.data:
        if trace.name not in  trace_visibility.keys():
            trace_visibility[trace.name] = False
        elif 'Loss' in trace.name:
            trace_visibility[f'{trace.name} epoch'] = False

    # ['data', 'sample', 'sample_50','sample_100',   'sample_150', 
    #  'sample_200', , 'sample_250',  'sample_300',  'sample_350', 
    #  'sample_400', , 'sample_450',  'sample_500',  'sample_550', 
    #  'sample_600', , 'sample_650',  'sample_700',  'sample_750', 
    #  'sample_800', , 'sample_850',  'sample_900',  'sample_950', 
    #  'sample_1000',, 'Loss', 'Loss mean', 'Loss', 'Loss mean', 
    #  'norm_50', 'norm_100', 'norm_150', 'norm_200', 'norm_250', 'norm_300', 'norm_350', 'norm_400', 'norm_450', 
    #  'norm_500', 'norm_550', 'norm_600', 'norm_650', ...]

    
    # trace_visibility['Line'] = True
    trace_vis_loss_itr = trace_visibility.copy()
    trace_vis_loss_itr.update({f'Loss': True, f'Loss mean': True, 'data': False})
    trace_vis_loss_epoch = trace_visibility.copy()
    trace_vis_loss_epoch.update({f'Loss epoch': True, f'Loss mean epoch': True, 'data': False})

    stats_trace_vis = [trace_vis_loss_itr, trace_vis_loss_epoch]
    smpl_grad_or_noise_norm_trace_vis = trace_visibility.copy()
    smpl_grad_or_noise_norm_trace_vis['data'] = False
  
    for e in epoch_list:
        smpl_grad_or_noise_norm_trace_vis[f'norm_{e}'] = True

    return trace_visibility, stats_trace_vis, smpl_grad_or_noise_norm_trace_vis

def visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics):

    # store idx of other buttons that adding to the layout for making visible True or False
    
    stats_idx = n_button_animations
    pause_idx = stats_idx + n_button_statistics
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
    
    all_zero_slider = dict(steps=[dict(args=[[f'zero_epoch_{e}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate' }], 
                                method='animate', label=f'{e}' ) for e in epoch_list] )
    all_zero_slider.update(slider_info)

    visibility_layout = {'title': f'All zeros', 'height': 570}
    # visibility_layout.update({
    #                     'xaxis':  {'range': None, 'autorange': True, 'title': 'x1', 'showgrid': False,'showticklabels': False},
    #                     'yaxis':  {  'range': None, 'autorange': 'reversed', 'title': 'y1', 'showgrid': False,'showticklabels': False},
    #                     'xaxis2': {'range': None, 'autorange': True, 'title': 'x2', 'showgrid': False,'showticklabels': False},
    #                     'yaxis2': {  'range': None, 'autorange': True, 'title': ' y2', 'showgrid': False,'showticklabels': False},
    #                     })
    visibility_layout.update({
                                'xaxis':  {'visible': True, }, #'visible': True,  'overlaying':'x' 'overlaying':None 'domain': [0, 1], 
                                'yaxis':  {'visible': True, }, #'visible': True,  'overlaying':'y' 'overlaying':None 'domain': [0, 1], 
                                'xaxis2': {'visible': False,}, #'visible': False, 'overlaying':'x' 'overlaying':None 'domain': [0, 0], 
                                'yaxis2': {'visible': False,}  #'visible': False, 'overlaying':'y' 'overlaying':None 'domain': [0, 0], 
                                })
    visibility_layout.update(button_visibility)
    
    visibility_layout[f'updatemenus[0].visible'] = True # show play zeros buttons
    
    visibility_layout['sliders'] = [all_zero_slider] 
    

    # add fist button to dropdwon 
    trace_vis_init = trace_visibility.copy()
    trace_vis_init.update({f'sample': True})
    select_button_list.append(dict(label=f'Show all zeros', method='update', 
                                args=[{'visible': list(trace_vis_init.values())}, visibility_layout]))

    vis_layout = visibility_layout.copy()
    vis_layout[f'updatemenus[0].visible'] = False
   # add play buttons for all epochs
    for i, e in enumerate(epoch_list):

        slider_steps = dict(steps=[dict(args=[[f'frame_{e}_{t}'], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate' }], 
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
        vis[f'updatemenus[{i+1}].visible'] = True # first play button is for all zeros
        vis['sliders'] = [slider_steps]

        trace_vis_epoch = trace_visibility.copy()
        trace_vis_epoch.update({f'sample_{e}': True})
        # trace_vis_epoch.update({f'{quiver_name}_{e}': True})
        # print(i)
        select_button_list.append(dict(label=f'Show epoch {e}', method='update', 
                                       args=[{'visible': list(trace_vis_epoch.values())}, vis]))  
        

    # select_button_list.append(dict(label=f'', method='update',  args=[None]))

    return play_button_list, select_button_list, all_zero_slider

def add_stats_buttons(button_visibility, stats_trace_vis, n_button_statistics, n_button_animations, button_info, method, 
                      back_stats_idx, pause_idx, select_dropdown_idx):

    # Show all stats buttons (loss, score)
    stats_button_list = []
    show_stats = {'title': method, 'width':1000, 'height': 450, 'plot_bgcolor': None, 
                        # 'xaxis2': {'range': None, 'autorange': True, 'title': 'x', 'showgrid': True,'showticklabels': True},
                        # 'yaxis2': {'range': None, 'autorange': True, 'title': 'y', 'showgrid': True,'showticklabels': True},
                        # 'xaxis':  {'range': None, 'autorange': True, 'title': ' ', 'showgrid': False,'showticklabels': False},
                        # 'yaxis':  {'range': None, 'autorange': True, 'title': ' ', 'showgrid': False,'showticklabels': False},
                        }
    show_stats.update({
                                'xaxis':  {'visible': False,  }, #'overlaying':'x', 'domain': [0, 0.5], 
                                'yaxis':  {'visible': False,  }, #'overlaying':'y', 'domain': [0, 1]  , 
                                'xaxis2': {'visible': True, }, #'overlaying':'x', 'domain': [0.5, 1], 
                                'yaxis2': {'visible': True, }  #'overlaying':'y', 'domain': [0, 1]  , 
                                })
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
                                                                                            {'title': f'{stats_lables[i]}',
                                                                                             }])])
        stats_button.update(button_info)
        stats_button.update(dict(visible=False, x=0.15+((i+1))/8, y=1.02, pad={"r": 10, "t": 25, "l": 10}))
        stats_button_list.append(stats_button)
    # Show loss in select button [dropdown]
    show_stats['sliders'] = []
    stats_vis = stats_trace_vis[0]
    show_stats_button = dict(label=f'Show Loss', method='update', args=[{'visible': list(stats_vis.values())}, show_stats])

    return stats_button_list, show_stats_button

def add_sampling_grad_noise_norm_button(button_visibility, smpl_grad_noise_trace_vis, back_smpl_grad_noise_norm_idx, 
                                        pause_idx, select_dropdown_idx, quiver_name='grad'):

    # Show sample score (validation) 
    show_smpl_grd_ns = {'title': f'Sample {quiver_name} norm', 'width':1000, 'height': 450, 'plot_bgcolor': None, 
                        # 'xaxis2': {'visible': True, 'range': None, 'autorange': True, 'title': 'x', 'showgrid': True,'showticklabels': True},
                        # 'yaxis2': {'visible': True, 'range': None, 'autorange': True, 'title': 'y', 'showgrid': True,'showticklabels': True},
                        # 'xaxis': {'visible': False, 'range': None, 'autorange': True, 'title': ' ', 'showgrid': False,'showticklabels': False},
                        # 'yaxis': {'visible': False, 'range': None, 'autorange': True, 'title': ' ', 'showgrid': False,'showticklabels': False},
                            }
    
    show_smpl_grd_ns.update(button_visibility) 
    show_smpl_grd_ns[f'updatemenus[{back_smpl_grad_noise_norm_idx}].visible'] = True
    show_smpl_grd_ns[f'updatemenus[{pause_idx}].visible'] = False
    show_smpl_grd_ns[f'updatemenus[{select_dropdown_idx}].visible'] = False 
    show_smpl_grd_ns['sliders'] = []
    # Show smpl score in select button [dropdown]

    show_smpl_grd_ns_button = dict(label=f'Show {quiver_name} norm', method='update', 
                                   args=[{'visible': list(smpl_grad_noise_trace_vis.values())}, show_smpl_grd_ns])

    return show_smpl_grd_ns_button

def add_data_button(button_visibility, trace_visibility, pause_idx):

    trace_vis = trace_visibility.copy()
    trace_vis.update({f'data': True})
    
    # Show data
    show_data = {'title': f'Data', 'height': 500} 
    show_data.update({'xaxis.showticklabels': False, 'yaxis.showticklabels': False, 'xaxis.title': '', 'yaxis.title': '',
                        'xaxis.autorange':True, 'yaxis.autorange':'reversed'})
            
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
    show_detail.update({'margin.t':100, 'title.y': 0.6, 'title.x':0.2, 'xaxis.range': [-2.5, 2.5], 'yaxis.range': [-2.5, 2.5], 
                            'xaxis.title': '', 'yaxis.title': '',# 'xaxis.tickvals': [], 'yaxis.tickvals': [], 
                            'xaxis.showticklabels': False, 'yaxis.showticklabels': False
                            }) 
    show_detail.update(dict(plot_bgcolor='white', height=450))
    hide_traces = trace_visibility.copy()
    hide_traces['data'] = False
    show_detail['sliders'] = []
    show_detail_button = dict(label=f'Show details', method='update', args=[{'visible': list(hide_traces.values())}, show_detail])
            #    [dict(label=f'Hide Loss-Score', method='update', args=[{'y': [], 'x': []}, hide_loss]),
            #    dict(label=f'1 details', method='relayout', args=[{'title': f'GAN 1'}])]
    return show_detail_button

def add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, method,
                     back_detail_idx, stats_idx, back_smpl_score_idx, select_dropdown_idx, back_stats_idx):

    hide_info = {'margin.t': 60, 'title.y': 0.98, 'title.x':0.001, 'plot_bgcolor': None, 'width':750, 'height': 500,
                # 'xaxis': { 'range': None, 'autorange': True, 'title': ' ', 'showgrid': True, 'showticklabels': False},
                # 'yaxis': { 'range': None, 'autorange': 'reversed', 'title': ' ', 'showgrid': True, 'showticklabels': False},
                # 'xaxis2': {'range': None, 'autorange': True, 'title': ' ', 'showgrid': False, 'showticklabels': False},
                # 'yaxis2': {'range': None, 'autorange': True, 'title': ' ', 'showgrid': False, 'showticklabels': False},
                }
    hide_info.update({
                    'xaxis':  {'visible': True, }, #'domain': [0, 0.5]  'overlaying':'x', 
                    'yaxis':  {'visible': True, }, #'domain': [0, 1]    'overlaying':'y', 
                    'xaxis2': {'visible': False,}, #'domain': [0.5, 1]  'overlaying':'x', 
                    'yaxis2': {'visible': False,}  #'domain': [0, 1]    'overlaying':'y', 
                                })
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

    # add back button  [sample score]
    hide_smpl_grad_noise = {'title.text': method}
    hide_smpl_grad_noise.update(hide_info) 
    hide_smpl_grad_noise[f'updatemenus[{select_dropdown_idx}].visible'] = True
    hide_smpl_grad_noise[f'updatemenus[{back_smpl_score_idx}].visible'] = False
    back_smpl_grad_noise_button = dict(buttons=[dict(label=f"{f'&#129092;':^5}", method='update', args=[{'visible': list(trace_vis.values())}, hide_smpl_grad_noise])]) 
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
    image_shape = hdf_file['/df/info']['image_shape'].values[0]
    image_shape = list(map(int, image_shape.split(',')))
    n_sel_time = hdf_file['/df/info']['n_sel_time'].values[0]

    df_sample = []
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
            smpl = hdf_file[k]['data_0'].to_numpy().reshape(image_shape)
            df_sample.append(smpl)
            epoch_list.append(int(k.split('/')[2].split('_')[2]))
        elif 'intermediate_smpl'in k:
            # df_intermediate_smpl = pd.concat([df_intermediate_smpl if not df_intermediate_smpl.empty else None, hdf_file[k]])
            
            intmd_smple = hdf_file[k].iloc[:, 1:].to_numpy().T.reshape(n_sel_time, *image_shape)
            df_intermediate_smpl_list.append(intmd_smple)
        
        # elif 'grad_info' in k or 'noise_info' in k:
        #     df_grads_or_noises_info = pd.concat([df_grads_or_noises_info if not df_grads_or_noises_info.empty else None, hdf_file[k]])
        elif 'grad_info' in k or 'noise_info' in k: # noise or grad predicted for of samples generated iteratively
            df_grad_or_noise_info_hist.append(hdf_file[k])
    
        elif 'grad_grid' in k or 'noise_grid' in k:
            # df_grad_g_z = pd.concat([df_grad_g_z if not df_grad_g_z.empty else None, hdf_file[k]])
            intmd_gr_ns = hdf_file[k].iloc[:, 1:].to_numpy().T.reshape(n_sel_time, *image_shape)
            df_grad_or_noise_grid_list.append(intmd_gr_ns)
        

    df_data = hdf_file['/df/data']
    n_samples = hdf_file['/df/info']['n_samples'].values[0]
    
    timesteps_list = hdf_file['/df/sampling_timesteps']['time'].to_numpy()
    hdf_file.close()
    # except:
    #     print(f'Error in reading hdf file {file}')
    #     hdf_file.close()
    df_data = df_data['data_0'].to_numpy().reshape(image_shape)
    df_sample = np.stack(df_sample)

    return df_sample, df_data, df_intermediate_smpl_list, df_grad_or_noise_info_hist, \
            df_grad_or_noise_grid_list, epoch_list, timesteps_list, n_samples

def read_hdf_loss(file):

    df_loss = pd.read_hdf(file)   
    df_loss_itr = pd.DataFrame(df_loss.loc[0, ['loss_itr']].values[0], columns=['loss_itr'])
    for i in range(1, df_loss.shape[0]):
        df_loss_itr = pd.concat([df_loss_itr, pd.DataFrame(df_loss.loc[i, ['loss_itr']].values[0], columns=['loss_itr'])])
    df_loss_itr['itr'] = np.arange(df_loss_itr.shape[0])

    return df_loss.loc[:, ['epoch', 'loss_epoch']], df_loss_itr.reset_index(drop=True)

def prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, 
                                    df_grad_or_noise_grid_list, df_grads_or_noises_info, 
                                    df_loss_itr, df_loss, epoch_list, timesteps_list, 
                                    quiver_name, method, config,
                                   width=750, height=570, range_x=[0, 207], range_y=[0, 207]):

    # The traces must be added in the correct order for the button's visibility to function properly
   
    num_epochs = len(epoch_list)
    colors = dict(sample='#AB63FA', grad='DarkSlateGrey', data='#19D3F3', fake='mediumorchid') # samples-grads-data
    # create frames for each animation [epochs], all zeros samples are in one df: df_sample
    all_zero_sample_frames = add_all_zeros_animation(df_sample, epoch_list, colors)

    # add animations of all epochs
    animation_list = add_epochs_animation_slider(df_intermediate_smpl_list, epoch_list, timesteps_list)

    fig = go.Figure() 
    # data, sample_zero at first epoch , grad at first epoch

    fig = add_sample_trace(fig, all_zero_sample_frames, animation_list, epoch_list)

    # add traces of statistics
    fig = add_statistics_to_fig(fig, df_loss_itr, df_loss)
    # add smpl grad_or_noise norm trace
    fig = add_smpl_grads_or_noises_info_to_fig(fig, df_grads_or_noises_info, timesteps_list, epoch_list)
    # add data
    fig = add_data_trace(fig, df_data)
    # visibility of traces when coresponding button is clicked
    trace_visibility, stats_trace_vis, smpl_grad_noise_trace_vis = visibility_traces(fig, epoch_list)
    # n_button_animations equals to plays buttons
    n_button_animations = len(animation_list) + 1  # 1: for all_zeros button 
    n_button_statistics = 2 # loss itr, loss epoch
    n_all_buttons = n_button_animations + n_button_statistics + 2  + 2 + 1 # 1: show sample noise norm  # show stats, show detail, show eval, 
                                                                                                        # back_stats, back_detail, back_eval
    # n_traces_animation = 1 + 1 +  2*num_epochs # 1:data, 1: grad first epoch, len(epoch_list): samples of each epoch
    # n_traces_statistics = 2 + 2 +   # 2: loss itr 2: loss epoch

    # store idx of other buttons that adding to the layout for making visible True or False
    # buttons visibility
    button_visibility, stats_idx, back_smpl_grad_noise_norm_idx, pause_idx, \
    back_stats_idx, back_detail_idx, select_dropdown_idx = visibility_buttons_and_idx(n_all_buttons, n_button_animations, n_button_statistics)

    frame_spec = {'frame': {'duration': 300, 'redraw': True}, 'fromcurrent': True, 'mode': 'immediate', 
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
    # add data button
    show_data_button = add_data_button(button_visibility, trace_visibility, pause_idx)
    # add pause button
    pause_button = dict(buttons=[dict(label=f"{f'&#9724;':^5}", method='animate',  args=[[None],  {'frame':{'duration': 0, 'redraw': False}, 
                                            'mode': 'immediate', 'fromcurrent': True, 'transition': {'duration': 0, 'easing': 'linear'}}])])
    pause_button.update(button_info)
    pause_button.update(dict(x=-0.12, y=0.001))

    # [ back buttons ] 
    back_stats_button, back_detail_button, back_smpl_score_button = add_back_buttons(trace_visibility, n_button_statistics, button_info, slider_info, 
                                                                    method, back_detail_idx, stats_idx, back_smpl_grad_noise_norm_idx, select_dropdown_idx, back_stats_idx)
    


    # button_list =  play_button_list + stats_button_list + [pause_button] + [back_stats_button]
    button_list = play_button_list + stats_button_list + [pause_button] + [back_stats_button] + [back_detail_button] + [back_smpl_score_button]
    # all buttons          
    # updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_stats_button]+select_button_list[1:],
    #                                     visible=True, showactive=False, direction='down', pad={"r": 10, "t": 20, "l": 10},
    #                                     x=-0.1, xanchor="right", y=1.05, yanchor="top", bgcolor='seashell')] 
                    
    updatemenues = button_list + [dict(buttons=select_button_list[0:1]+[show_detail_button]+[show_stats_button]+select_button_list[1:]+[show_smpl_grad_noise_norm_button] + [show_data_button],
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
    # 'xaxis.domain'=[0.0, 1.0] , 'xaxis.anchor'='y2'    
    fig.update_layout(
        title=dict(text=method, pad={"r": 10, "t": 10, "l": 20, "b":10}, x=0.001, xanchor="left", y=0.98, yanchor="top"),
        xaxis=dict(title=dict(text="x1", font_size=12), showgrid=True, autorange=True, anchor='y',
                   domain=[0, 1], 
                   ),  #  range=range_x, 
        xaxis2=dict(title=dict(text="x2", font_size=12), showgrid=True, autorange=True, anchor='y2',  
                    showticklabels=True, visible=True,
                    domain=[0, 1], 
                    # overlaying='x',
                    overlaying=None,
                    ),
        yaxis=dict(title=dict(text="y1", font_size=12), showgrid=True, autorange='reversed', anchor='x',
                   domain=[0, 1],
                   ),  #  range=range_y, 
        yaxis2=dict(title=dict(text="y2", font_size=12), showgrid=True, autorange=True, anchor='x2',  
                    showticklabels=True, visible=True,
                    domain=[0, 1], 
                    # overlaying='y',  
                    overlaying=None, 
                    ),
        # plot_bgcolor='rgba(100, 100, 100, 0)',
        margin=dict(l=80, t=60, b=20, r=150),
        showlegend=True,
        width=width,
        height=height
    )
    fig.update_layout(coloraxis_showscale=False)  # Optional: Hide color scale for cleaner view
    fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)

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
            df_grad_or_noise_grid_list, epoch_list, timesteps_list, n_samples  = read_hdf_all_epochs_train(file)

       
        file = f'{path}/{expr_id}_df_loss_per_epoch.h5'
        df_loss, df_loss_itr = read_hdf_loss(file)

        fig_all = prepare_plotly_fig_quiver_grad(df_sample, df_data, df_intermediate_smpl_list, 
                                                            df_grad_or_noise_grid_list, df_grads_or_noises_info, 
                                                            df_loss_itr, df_loss, epoch_list, timesteps_list, 
                                                            quiver_name, method_type, config)
          
    show_latex_in_plotly()
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