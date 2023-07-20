import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
configfile:
    "config.json"

# The cache directories:
cache_path = config['CACHE_DIR']
# image_cache_path = f'{cache_path}/annot-images'
# v123_cache_path = f'{cache_path}/annot-v123'
# csulc_cache_path = f'{cache_path}/annot-csulc'
# labels_path = f'{cache_path}/results/labels'
os.environ['HCPANNOT_LOAD_PATH'] = cache_path
from visualization import *

RATERS = ['BrendaQiu', 'bogengsong', 'JiyeongHa', 'lindazelinzhao', 'nourahboujaber', 'jennifertepan']
ROIS = ['hV4', 'VO1', 'VO2']
HEMIS = ['lh','rh']


rule get_subj_ids_for_rater:
    output:
        subj_ids = os.path.join(config['CACHE_DIR'], 'subj_ids', "rater-{rater}_hemi-{hemi}_roi-{roi}.txt")
    params:
        trace_save_path = config['TRACE_DIR']
    run:
        subj_ids = viscontours.get_trace_list_drawn_by_rater(params.trace_save_path, wildcards.hemi, wildcards.roi, wildcards.rater)
        with open(output.subj_ids,'w') as fp:
            for item in subj_ids:
                # write each item on a new line
                fp.write("%s," % item)

rule save_trace_cache:
    output:
        path_cache_file = os.path.join(config['PATH_SAVE_DIR'], "contour-path_space-fsaverage_rater-{rater}_sid-{sid}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.mgz")
    params:
        trace_save_path = config['TRACE_DIR']
    run:
        x, y = viscontours.make_path(wildcards.rater,
                                     wildcards.hemi,
                                     wildcards.roi,
                                     int(wildcards.sid),
                                     int(wildcards.n_points),
                                     trace_dir=params.trace_save_path,
                                     save_path=output.path_cache_file,
                                     verbose=False)


def convert_txt_to_list(wildcards):
    text_file = os.path.join(config['PATH_SAVE_DIR'], f"rater-{wildcards.rater}_hemi-{wildcards.hemi}_roi-{wildcards.roi}.txt")
    with open(text_file,'r') as fp:
        fd_list = fp.read().split(',')
    return fd_list[:-1]

def get_subj_ids(wildcards, all=False):
    if all is False:
        subj_ids = viscontours.get_trace_list_drawn_by_rater(config['TRACE_DIR'],
                                                             wildcards.hemi,
                                                             wildcards.roi,
                                                             wildcards.rater)
    else:
        import neuropythy as ny
        subj_ids = ny.data['hcp_lines'].subject_list
    return subj_ids

def get_trace_file_names(wildcards, all):
    ids_list = get_subj_ids(wildcards, all)
    all_files = []
    for sid in ids_list:
        f = os.path.join(config['PATH_SAVE_DIR'], f"contour-path_space-fsaverage_rater-{wildcards.rater}_sid-{sid}_hemi-{wildcards.hemi}_roi-{wildcards.roi}_npoints-{wildcards.n_points}.mgz")
        all_files.append(f)
    return all_files

def all_subj_ids(all_subjects=True):
    import neuropythy as ny
    return ny.data['hcp_lines'].subject_list

rule all:
    input:
        expand(os.path.join(config['PATH_SAVE_DIR'],"contour-path_space-fsaverage_rater-{rater}_sid-{sid}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.mgz"), rater=RATERS, hemi=HEMIS, roi=ROIS, sid=all_subj_ids(all_subjects=True), n_points=[500])

rule all_sids:
    input:
        all_files = lambda wildcards: get_trace_file_names(wildcards, all=True)
    output:
        os.path.join(config['PATH_SAVE_DIR'], "allsids_contour-path_space-fsaverage_rater-{rater}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.txt")
    shell:
        "touch {output}"

rule all_sids_all_raters:
    input:
        expand(os.path.join(config['PATH_SAVE_DIR'],  "allsids_contour-path_space-fsaverage_rater-{rater}_hemi-{hemi}_roi-{roi}_npoints-{n_points}.txt"), rater=RATERS, hemi=HEMIS, roi=ROIS, n_points=[500])