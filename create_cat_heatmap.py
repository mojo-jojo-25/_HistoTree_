from __future__ import print_function
import argparse
import os
import pdb
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import matplotlib.colors as mcolors
from vis_utils.heatmap_utils import initialize_wsi, drawCatHeatmap, drawHeatmap
from wsi_core.batch_process_utils import initialize_df
import torch
import torch.nn as nn
import umap
import matplotlib.pyplot as plt
import openslide
import numpy as np
from scipy.special import softmax
from PIL import Image
# from mil_models import create_model
from PIL import Image, ImageOps
import cv2
import scienceplots
import json

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict

def load_params(df_entry, params):
    for key in params.keys():
        if key in df_entry.index:
            dtype = type(params[key])
            val = df_entry[key]
            val = dtype(val)
            if isinstance(val, str):
                if len(val) > 0:
                    params[key] = val
            elif not np.isnan(val):
                params[key] = val
            else:
                pdb.set_trace()

    return params
def parse_config_dict(args, config_dict):
    if args.save_exp_code is not None:
        config_dict['exp_arguments']['save_exp_code'] = args.save_exp_code
    if args.overlap is not None:
        config_dict['patching_arguments']['overlap'] = args.overlap
    return config_dict
def hex_to_rgb_mpl_255(hex_color):
    rgb = mcolors.to_rgb(hex_color)
    return tuple([int(x*255) for x in rgb])

def get_default_cmap(n=32):
    colors = [
        '#696969', '#556b2f', '#a0522d', '#483d8b',
        '#008000', '#008b8b', '#000080', '#7f007f',
        '#8fbc8f', '#b03060', '#ff0000', '#ffa500',
        '#00ff00', '#8a2be2', '#00ff7f', '#FFFF54',
        '#00ffff', '#00bfff', '#f4a460', '#adff2f',
        '#da70d6', '#b0c4de', '#ff00ff', '#1e90ff',
        '#f0e68c', '#0000ff', '#dc143c', '#90ee90',
        '#ff1493', '#7b68ee', '#ffefd5', '#ffb6c1'
    ]

    colors = colors[:n]
    label2color_dict = dict(zip(range(n), [hex_to_rgb_mpl_255(x) for x in colors]))
    return label2color_dict


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='HistoTree')
    parser.add_argument('--save_exp_code', type=str, default='test')
    parser.add_argument('--overlap', type=float, default=None)
    parser.add_argument('--path_file', type=str, default='lung_files/test_0.txt', help='')
    parser.add_argument('--path_WSI', type=str, default='', help='')
    parser.add_argument('--path_graph', type=str,
                        default= '', help='')
    parser.add_argument('--vis_folder', type=str,
                        default= '/', help='')
    parser.add_argument('--config_file', type=str,
                        default='configs/heatmap_config_camelyon.yaml', help='')

    args = parser.parse_args()

    path_graph = args.path_graph
    vis_folder = args.vis_folder

    filenames = pd.read_csv(args.path_file, sep='\t', names=['filename', 'label'])


    config_path = os.path.join('heatmaps/configs', args.config_file)
    config_dict = yaml.safe_load(open(config_path, 'r'))
    config_dict = parse_config_dict(args, config_dict)

    args = config_dict
    patch_args = argparse.Namespace(**args['patching_arguments'])
    data_args = argparse.Namespace(**args['data_arguments'])
    exp_args = argparse.Namespace(**args['exp_arguments'])
    heatmap_args = argparse.Namespace(**args['heatmap_arguments'])
    sample_args = argparse.Namespace(**args['sample_arguments'])

    patch_size = tuple([patch_args.patch_size for i in range(2)])
    step_size = tuple((np.array(patch_size) * (1 - patch_args.overlap)).astype(int))
    print('patch_size: {} x {}, with {:.2f} overlap, step size is {} x {}'.format(patch_size[0], patch_size[1],
                                                                                  patch_args.overlap,
                                                                                  step_size[0], step_size[1]))
    preset = data_args.preset
    def_seg_params = {'seg_level': -1, 'sthresh': 15, 'mthresh': 11, 'close': 2, 'use_otsu': False,
                      'keep_ids': 'none', 'exclude_ids': 'none'}
    def_filter_params = {'a_t': 50.0, 'a_h': 8.0, 'max_n_holes': 10}
    def_vis_params = {'vis_level': -1, 'line_thickness': 250}
    def_patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}


    if preset is not None:
        preset_df = pd.read_csv(preset)
        for key in def_seg_params.keys():
            def_seg_params[key] = preset_df.loc[0, key]

        for key in def_filter_params.keys():
            def_filter_params[key] = preset_df.loc[0, key]

        for key in def_vis_params.keys():
            def_vis_params[key] = preset_df.loc[0, key]

        for key in def_patch_params.keys():
            def_patch_params[key] = preset_df.loc[0, key]

    slides = sorted(os.listdir(data_args.data_dir))
    slides = [slide for slide in slides if data_args.slide_ext in slide]
    df = initialize_df(slides, def_seg_params, def_filter_params, def_vis_params, def_patch_params,
                       use_heatmap_args=True)

    mask = df['process'] == 1
    process_stack = df[mask].reset_index(drop=True)
    total = len(process_stack)
    print('\nlist of slides to process: ')
    print(process_stack.head(len(process_stack)))

    os.makedirs(exp_args.raw_save_dir, exist_ok=True)

    blocky_wsi_kwargs = {'top_left': None, 'bot_right': None, 'patch_size': patch_size, 'step_size': patch_size,
                         'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                         'use_center_shift': heatmap_args.use_center_shift}
    attentions_scores = []
    for ind in filenames.index:
        if ind == 1:
            break
        slide_name = filenames['filename'][ind]
        slide_label = filenames['label'][ind]
        if data_args.slide_ext not in slide_name:
            slide_name += data_args.slide_ext
        print('\nprocessing: ', slide_name)

        slide_id = slide_name.replace(data_args.slide_ext, '')

        if isinstance(data_args.data_dir, str):
            slide_path = os.path.join(data_args.data_dir, slide_name)


        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)


        if os.path.exists(os.path.join(r_slide_save_dir, '{}_scores.png'.format(slide_id))):
                        continue



        mask_file = os.path.join(r_slide_save_dir, slide_id + '_mask.pkl')

        seg_params = def_seg_params.copy()

        filter_params = def_filter_params.copy()
        vis_params = def_vis_params.copy()

        keep_ids = str(seg_params['keep_ids'])
        if len(keep_ids) > 0 and keep_ids != 'none':
            seg_params['keep_ids'] = np.array(keep_ids.split(',')).astype(int)
        else:
            seg_params['keep_ids'] = []

        exclude_ids = str(seg_params['exclude_ids'])
        if len(exclude_ids) > 0 and exclude_ids != 'none':
            seg_params['exclude_ids'] = np.array(exclude_ids.split(',')).astype(int)
        else:
            seg_params['exclude_ids'] = []

        for key, val in seg_params.items():
            print('{}: {}'.format(key, val))

        for key, val in filter_params.items():
            print('{}: {}'.format(key, val))

        for key, val in vis_params.items():
            print('{}: {}'.format(key, val))

        print('Initializing WSI object')

        r_slide_save_dir = os.path.join(exp_args.raw_save_dir, exp_args.save_exp_code, slide_id)
        os.makedirs(r_slide_save_dir, exist_ok=True)

        wsi_object = initialize_wsi(slide_path, seg_mask_path=mask_file, seg_params=seg_params,
                                    filter_params=filter_params)

        wsi_ref_downsample = wsi_object.level_downsamples[patch_args.patch_level]

        vis_patch_size = tuple(
            (np.array(patch_size) * np.array(wsi_ref_downsample) * patch_args.custom_downsample).astype(int))

        patch_info = open(os.path.join(path_graph, slide_id, 'c_idx.txt'), 'r')
        patch_info = patch_info.readlines()
        coords = []
        xmax, ymax = 0, 0
        for patch in patch_info:
            x, y = patch.strip('\n').split('\t')
            if xmax < int(x): xmax = int(x)
            if ymax < int(y): ymax = int(y)
            coords.append((int(x), int(y)))
        coords = np.array(coords)

        assign_matrix = torch.load(os.path.join(vis_folder, '{}_s_matrix_ori.pt'.format(slide_id)))
        m = nn.Softmax(dim=1)
        assign_matrix = m(assign_matrix)

        att_matrix = torch.load(os.path.join(vis_folder, '{}_attn.pt'.format(slide_id)))
        att_matrix = torch.mm(assign_matrix, att_matrix.transpose(1, 0))
        att_matrix = att_matrix.cpu()

        inst_logits = torch.load(os.path.join(vis_folder, '{}_inst_logits.pt'.format(slide_id)))
        inst_logits = torch.mm(assign_matrix, inst_logits)
        inst_logits = inst_logits.cpu()

        weighted_logits = att_matrix * inst_logits
        bag_logits = torch.mean(weighted_logits, dim=0)
        pred = torch.argmax(bag_logits)

        m = nn.Softmax(dim=1)
        assign_matrix = m(weighted_logits)
        scores = assign_matrix[:, pred]
        scores = scores.detach().numpy()

        wsi_kwargs = {'patch_size': patch_size, 'step_size': step_size,
                      'custom_downsample': patch_args.custom_downsample, 'level': patch_args.patch_level,
                      'use_center_shift': heatmap_args.use_center_shift}


        probs = np.load(os.path.join(vis_folder, '{}_distances.npy'.format(slide_id)))
        labels = np.argmax(probs, axis=1)

        mask_path = os.path.join(r_slide_save_dir, '{}_mask.jpg'.format(slide_id))
        if vis_params['vis_level'] < 0:
            best_level = wsi_object.wsi.get_best_level_for_downsample(128)
            vis_params['vis_level'] = best_level
        vis_params['line_thickness'] = 250
        mask = wsi_object.visWSI(**vis_params, number_contours=False, annot_display=False)
        mask.save(mask_path)

        heatmap_1 = drawHeatmap(scores, coords, slide_path, wsi_object=wsi_object, cmap=heatmap_args.cmap,
                                alpha=heatmap_args.alpha,
                                use_holes=True, binarize=False, vis_level=-1, blank_canvas=False,
                                thresh=-1, patch_size=vis_patch_size, convert_to_percentiles=True)


        heatmap_1.save(os.path.join(r_slide_save_dir, '{}_att.tiff'.format(slide_id)), format='TIFF')


        heatmap_1 = drawCatHeatmap(labels, coords,
                                label2color_dict,
                                slide_path,
                                wsi_object = wsi_object,
                                alpha=heatmap_args.alpha,
                                use_holes=True,
                                vis_level=-1,
                                blur = False,
                                blank_canvas= True,
                                patch_size = vis_patch_size)

        heatmap_1.save(os.path.join(r_slide_save_dir, '{}_proto.tiff'.format(slide_id)), format='TIFF')



        label2color_dict = get_default_cmap(15)
        num_prototypes = probs.shape[1]
        top_k = 100



        for proto_id in range(num_prototypes):
            proto_indices = np.where(labels == proto_id)[0]

            proto_probs = probs[proto_indices, proto_id]
            num_to_select = min(top_k, len(proto_indices))

            top_indices = proto_indices[np.argsort(proto_probs)[::-1][:num_to_select]]

            for idx, patch_idx in enumerate(top_indices):
                    output_data = []
                    s_coord = coords[patch_idx]
                    s_prob = probs[patch_idx, proto_id]

                    proto_indices = np.where(labels == proto_id)[0]

                    patch = wsi_object.wsi.read_region(tuple(s_coord), patch_args.patch_level, (patch_args.patch_size,
                                                                                                patch_args.patch_size)).convert('RGB')


                    proto_slide_save_dir = os.path.join(r_slide_save_dir, f'prototype_{proto_id}')
                    os.makedirs(proto_slide_save_dir, exist_ok=True)
                    patch.save(os.path.join(proto_slide_save_dir ,'{}_x_{}_y_{}.png'.format(idx, s_coord[0], s_coord[1])))


        average_contributions = np.zeros(num_prototypes)

        for i in range(num_prototypes):

            assigned_tiles = labels == i
            if np.sum(assigned_tiles) > 0:

                average_contributions[i] = np.mean(scores[assigned_tiles])

        min_contrib = 0.01  # Minimum value for inactive prototypes
        max_contrib = 1.0  # Maximum value for the most active prototype
        active_mask = average_contributions > 0  # Identify active prototypes


        if np.any(active_mask):  # Check if there are any active prototypes
            active_contributions = average_contributions[active_mask]

            scaled_contributions = min_contrib + (active_contributions - active_contributions.min()) * \
                                   (max_contrib - min_contrib) / (
                                               active_contributions.max() - active_contributions.min())
            average_contributions[active_mask] = scaled_contributions


        epsilon = 1e-6
        average_contributions = np.where(average_contributions == 0, epsilon, average_contributions)

        normalized_colors = {k: tuple([v_i / 255.0 for v_i in v] + [0.4]) for k, v in label2color_dict.items()}
        with plt.style.context(['science']):
            plt.figure(figsize=(4, 6))
            bars = plt.barh(range(len(average_contributions)), average_contributions,
                            color=[normalized_colors[i] for i in range(len(average_contributions))])

            plt.xlabel('Contributions')
            plt.ylabel('Prototypes')
            plt.yticks(range(len(average_contributions)), [f'P{i}' for i in range(len(average_contributions))])

            # Save the plot
            barplot_filename = os.path.join(r_slide_save_dir, 'prototype_probabilities.png')
            plt.savefig(barplot_filename, dpi=300)

        X = torch.load(os.path.join(vis_folder,
                                    '{}_feats.pt'.format(slide_id))).squeeze(0)
        X = X.cpu()

        probs = np.load(os.path.join(vis_folder, '{}_distances.npy'.format(slide_id)))
        labels = np.argmax(probs, axis=1)

        num_prototypes = probs.shape[1]
        top_k = 50

        all_features = []
        all_prototype_labels = []
        tsne_colors = [normalized_colors[label] for label in all_prototype_labels]

        for proto_id in range(num_prototypes):
            proto_indices = np.where(labels == proto_id)[0]

            proto_probs = probs[proto_indices, proto_id]
            num_to_select = min(top_k, len(proto_indices))

            top_indices = proto_indices[np.argsort(proto_probs)[::-1][:num_to_select]]
            top_k_features = X[top_indices].numpy()

            all_features.append(top_k_features)
            all_prototype_labels.extend([proto_id] * len(top_indices))

        all_features = np.vstack(all_features)
        tsne = TSNE(n_components=2, perplexity=30, learning_rate=300)
        X_tsne = tsne.fit_transform(all_features)

        all_prototype_labels = np.array(all_prototype_labels)

        tsne_colors = [normalized_colors[label] for label in all_prototype_labels]
        #
        with plt.style.context(['science']):
            # Plotting t-SNE with matching colors
            plt.figure(figsize=(8, 6))
            scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=tsne_colors, edgecolors='none')  # Use color mapping from normalized_colors

            unique_labels = np.unique(all_prototype_labels)
            handles = [plt.Line2D([0], [0], marker='o', color='w', label=f'P{label}',
                                  markerfacecolor=normalized_colors[label], markersize=10) for label in unique_labels]
            plt.legend(handles=handles, title="Prototypes", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')

            plt.xticks([])
            plt.yticks([])

        tsne_filename = os.path.join(r_slide_save_dir, 'tsne_matching_colors.png')
        plt.savefig(tsne_filename, bbox_inches='tight')

