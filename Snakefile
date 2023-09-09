import os
import sys
import numpy as np
import pandas as pd
import matplotlib as mpl
configfile:
    "config.json"
os.environ['HCPANNOT_LOAD_PATH'] = config['CACHE_DIR']

# The cache directories:
from visualization import *
RATERS = ['BrendaQiu', 'bogengsong', 'JiyeongHa', 'lindazelinzhao', 'nourahboujaber', 'jennifertepan']
ROIS = ['hV4', 'VO1', 'VO2']
HEMIS = ['lh','rh']


rule get_subj_ids_for_rater:
    output:
        subj_ids = os.path.join(config['CACHE_DIR'], 'subj_ids', "rater-{rater}_hemi-{hemi}_roi-{roi}.txt")
    params:
        trace_save_path = config['OUTPUT_DIR']
    run:
        subj_ids = viscontours.get_trace_list_drawn_by_rater(params.trace_save_path, wildcards.hemi, wildcards.roi, wildcards.rater)
        with open(output.subj_ids,'w') as fp:
            for item in subj_ids:
                # write each item on a new line
                fp.write("%s," % item)

rule save_trace_cache:
    output:
        os.path.join(config['PROC_DIR'],'fsaverage', "{rater}", "{sid}", "{hemi}.roi-{roi}_space-fsaverage_npoints-{n_points}.mgz")
    params:
        data_dir = config['DATA_DIR'],
        proc_dir = config['PROC_DIR']
    run:
        x, y = viscontours.make_fsaverage_coords(rater=wildcards.rater,
                                                 hemi=wildcards.hemi,
                                                 roi=wildcards.roi,
                                                 subject=int(wildcards.sid),
                                                 n_points=int(wildcards.n_points),
                                                 data_dir=params.data_dir,
                                                 proc_dir=params.proc_dir,
                                                 save_path=output[0])


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


rule all:
    input:
        lambda wildcards: expand(os.path.join(config['PROC_DIR'],'fsaverage', "{rater}", "{sid}", "{hemi}.roi-{roi}_space-fsaverage_npoints-{n_points}.mgz"), rater=['JiyeongHa', 'BrendaQiu'], hemi=HEMIS, roi=ROIS, sid=get_subj_ids(wildcards, all=True), n_points=[500])


def convert_txt_to_list(wildcards):
    text_file = os.path.join(config['PATH_SAVE_DIR'], f"rater-{wildcards.rater}_hemi-{wildcards.hemi}_roi-{wildcards.roi}.txt")
    with open(text_file,'r') as fp:
        fd_list = fp.read().split(',')
    return fd_list[:-1]

def get_trace_file_names(wildcards, all):
    ids_list = get_subj_ids(wildcards, all)
    all_files = []
    for sid in ids_list:
        f = os.path.join(config['PATH_SAVE_DIR'], f"contour-path_space-fsaverage_rater-{wildcards.rater}_sid-{sid}_hemi-{wildcards.hemi}_roi-{wildcards.roi}_npoints-{wildcards.n_points}.mgz")
        all_files.append(f)
    return all_files


rule save_hcp_subject_list:
    input:
        all_files = lambda wildcards: get_trace_file_names(wildcards, all=True)
    output:
        os.path.join(config['OUTPUT_DIR'], "subject_list", "hcp_subject_list.txt")
    run:
        import neuropythy as ny
        test = ny.data['hcp_lines'].subject_list
        with open(output[0],'w') as f:
            f.write('\n'.join(str(i) for i in test))
