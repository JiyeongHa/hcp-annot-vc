import os
import numpy as np
import neuropythy as ny
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import time
from glob import glob
from hcpannot import (vc_plan)


def load_cortex(subject, hemi):
    cortex = ny.data['hcp_lines'].subjects[subject].hemis[hemi]  # cortex object
    return cortex


def roi_trace_to_path(dat, cortex, roi):
    # grab a trace
    trace = dat['traces'][roi]
    # get the subject's hemisphere mesh
    path = trace.to_path(cortex)
    return path


def get_path_coords_in_fsaverage(path, cortex):
    fmap = ny.to_flatmap('occipital_pole', cortex)
    fsa_coords = fmap.unaddress(path.addresses)
    return fsa_coords


def normalize_coords(fsa_coords, n_points=800):
    curve = ny.util.CurveSpline(fsa_coords)
    normed_fsa_coords = curve.linspace(n_points)
    return normed_fsa_coords


def arrange_multiple_coords_into_dict(normed_x_list,
                                      normed_y_list,
                                      normed_fsa_coords_keys=None,
                                      average=True):
    if normed_fsa_coords_keys is None:
        normed_fsa_coords_keys = np.range(0, len(normed_x_list))
    normed_fsa_coords_vals = [np.asarray((x, y)) for x, y in zip(normed_x_list, normed_y_list)]
    normed_fsa_coords_dict = dict(zip(normed_fsa_coords_keys, normed_fsa_coords_vals))
    if average is True:
        avg_x = np.mean(normed_x_list, axis=0)
        avg_y = np.mean(normed_y_list, axis=0)
        normed_fsa_coords_dict['avg'] = np.asarray((avg_x, avg_y))
    return normed_fsa_coords_dict


def _display_msg(message, verbose):
    if verbose:
        print(message)
    else:
        pass


def make_trace(rater,
               hemi,
               roi,
               subject,
               n_points,
               verbose=True,
               cache_path=None,
               trace_save_path=None):
    _display_msg(f'---------------------', True)
    _display_msg(f'subject no.{subject}', True)
    total_start = time.time()
    try:
        if os.path.isfile(cache_path):
            _display_msg(f'found cache data!', True)
            (x, y) = ny.load(cache_path)
        else:
            annot = vc_plan(rater=rater, sid=subject, hemisphere=hemi, save_path=trace_save_path)
            _display_msg(f'loading surface mesh....', verbose)
            start = time.time()
            cortex = load_cortex(subject, hemi)
            end = time.time()
            _display_msg(f'done! elpased time is {np.round(end - start, 3)} sec.\nNow converting trace to path...', verbose)
            start = time.time()
            path = roi_trace_to_path(annot, cortex, roi)
            end = time.time()
            _display_msg(f'done! elpased time is {np.round(end - start, 3)} sec', verbose)
            _display_msg('Now transforming the path to fsaverage space...', verbose)
            start = time.time()
            fsa_coords = get_path_coords_in_fsaverage(path, cortex)
            end = time.time()
            _display_msg(f'done! elpased time is {np.round(end - start, 3)} sec.\nnow interpolating the coordinates..',
                         verbose)
            (x, y) = normalize_coords(fsa_coords, n_points)
            ny.save(cache_path, [x, y])
        total_end = time.time()
        _display_msg(f'subject no. {subject} is finished! Elapsed time: {np.round(total_end - total_start, 2)} sec',
                     True)
        return x, y
    except:
        print(f'{subject} has an error!')



def main(rater,
         hemi,
         roi,
         subject_list,
         n_points=800,
         verbose=False,
         cache_file_list=None,
         trace_save_path=None,
         average=True):
    x_list, y_list = [], []
    _display_msg(f'**number of subjects: {len(subject_list)}\n**nuber of points:{n_points}', verbose)

    for sid, cache_file in tqdm(zip(subject_list, cache_file_list)):
        x, y = make_trace(rater,
                          hemi,
                          roi,
                          sid,
                          n_points,
                          verbose,
                          cache_file,
                          trace_save_path)
        x_list.append(x)
        y_list.append(y)
        total_end = time.time()
    normed_fsa_coords_dict = arrange_multiple_coords_into_dict(x_list, y_list, subject_list, average)
    return normed_fsa_coords_dict

def plot_multiple_traces(normed_fsa_coords_dict, legend=True, average_only=False, **kwargs):
    title = kwargs.pop('title', None)
    if average_only:
        normed_fsa_coords_dict = {'avg': normed_fsa_coords_dict['avg']}
    n_lines = len(normed_fsa_coords_dict.keys())
    fig, ax = plt.subplots(1,1)
    pal = plt.cm.gist_rainbow(np.linspace(0, 1, n_lines))
    color = iter(pal)
    for k, fsa_coords in normed_fsa_coords_dict.items():
        (x,y) = fsa_coords
        c=next(color)
        if k == 'avg':
            c = 'k'
        ax.plot(x,y, label=str(k), color=c, **kwargs)
    ax.set_title(title)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncols=2)
    ax.get_legend().set_visible(legend)
    plt.tight_layout()

def get_contour_list_drawn_by_rater(trace_save_path, hemi, roi, rater, return_full_path=False):

    path_list = glob(os.path.join(trace_save_path, rater, f'*/{hemi}.{roi}.json'))
    if return_full_path is True:
        return path_list
    else:
        sids_list = [int(k.split('/')[-2]) for k in path_list]
        return sids_list