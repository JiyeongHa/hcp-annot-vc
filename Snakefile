import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
configfile:
    "config.json"

# The cache directories:
cache_path = config['cache_path']
trace_save_path = config['trace_save_path']
image_cache_path = f'{cache_path}/annot-images'
v123_cache_path = f'{cache_path}/annot-v123'
csulc_cache_path = f'{cache_path}/annot-csulc'
labels_path = f'{cache_path}/results/labels'
os.environ['HCPANNOT_LOAD_PATH'] = cache_path
from visualization import *

rater_list = ['BrendaQiu',
              'bogengsong',
              'JiyeongHa',
              'lindazelinzhao',
              'nourahboujaber',
              'jennifertepan']

rule get_subj_ids_for_rater:
    output:
        subj_ids = os.path.join(config['cache_path'], 'subj_ids', "rater-{rater}_hemi-{hemi}_roi-{roi}.txt")
    params:
        trace_save_path = config['trace_save_path']
    run:
        subj_ids = viscontours.get_contour_list_drawn_by_rater(params.trace_save_path, wildcards.hemi, wildcards.roi, wildcards.rater)
        with open(output.subj_ids,'w') as fp:
            for item in subj_ids:
                # write each item on a new line
                fp.write("%s," % item)

rule save_trace_cache:
    output:
        cache_file = os.path.join(config['cache_path'], 'traces', "contour-path_space-fsaverage_rater-{rater}_sid-{sid}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.mgz")
    params:
        trace_save_path = config['trace_save_path']
    run:

        #subj_ids = viscontours.get_contour_list_drawn_by_rater(params.trace_save_path, wildcards.hemi, wildcards.roi, wildcards.rater)
        x, y = viscontours.make_trace(wildcards.rater,
                                      wildcards.hemi,
                                      wildcards.roi,
                                      int(wildcards.sid),
                                      int(wildcards.n_points),
                                      False,
                                      output.cache_file,
                                      params.trace_save_path)

def convert_txt_to_list(wildcards):
    text_file = os.path.join(config['cache_path'], 'subj_ids', f"rater-{wildcards.rater}_hemi-{wildcards.hemi}_roi-{wildcards.roi}.txt")
    with open(text_file,'r') as fp:
        fd_list = fp.read().split(',')
    return fd_list[:-1]

def get_trace_file_names(wildcards):
    ids_list = convert_txt_to_list(wildcards)
    all_files = []
    for sid in ids_list:
        f = os.path.join(config['cache_path'],'traces',f"contour-path_space-fsaverage_rater-{wildcards.rater}_sid-{sid}_hemi-{wildcards.hemi}_roi-{wildcards.roi}_npoints-{wildcards.n_points}.mgz")
        all_files.append(f)
    return all_files

rule all_sids:
    input:
        all_files = lambda wildcards: get_trace_file_names(wildcards)
    output:
        os.path.join(config['cache_path'], 'traces', "allsids_contour-path_space-fsaverage_rater-{rater}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.txt")
    shell:
        "touch {output}"


rule all_sids_all_raters:
    input:
        expand(os.path.join(config['cache_path'], 'traces', "allsids_contour-path_space-fsaverage_rater-{rater}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.txt"), rater=['JiyeongHa'], hemi=['lh'], roi=['hV4'], n_points=[3])