################################################################################
# config.py
#
# Global configuration data and constants go in this file.
# This is primarily for defining things like the names of the default raters.

import os
import numpy as np


# Annotation Configuration #####################################################
# These configuration items concern how the annotation tool is/was run during
# the annotation of contours.

# The default image size; this is the assumed size of the images that are
# downloaded from the OSF and displayed, in pixels.
default_imshape = (864*2, 864*2)
# The default x- and y-value limits of the flatmaps that were used for
# generating the images.
default_xlim = (-100, 100)
default_ylim = (-100, 100)
# The default grid used for the display. The None is a stand-in for the image of
# the current contour's highlight.
default_grid = (
    (None,        'polar_angle'),
    ('curvature', 'eccentricity'))
# The path that we load images from by default. We want to use an environment
# variable if possible, otherwise we default to /data.
default_load_path = os.environ.get('HCPANNOT_LOAD_PATH', '/data')
default_osf_url   = os.environ.get('HCPANNOT_OSF_URL', 'osf://tery8/')
# The HCP Retinotopy subjects:
subject_list = np.array(
    [100610, 102311, 102816, 104416, 105923, 108323, 109123, 111312, 111514,
     114823, 115017, 115825, 116726, 118225, 125525, 126426, 128935, 130114,
     130518, 131217, 131722, 132118, 134627, 134829, 135124, 137128, 140117,
     144226, 145834, 146129, 146432, 146735, 146937, 148133, 150423, 155938,
     156334, 157336, 158035, 158136, 159239, 162935, 164131, 164636, 165436,
     167036, 167440, 169040, 169343, 169444, 169747, 171633, 172130, 173334,
     175237, 176542, 177140, 177645, 177746, 178142, 178243, 178647, 180533,
     181232, 181636, 182436, 182739, 185442, 186949, 187345, 191033, 191336,
     191841, 192439, 192641, 193845, 195041, 196144, 197348, 198653, 199655,
     200210, 200311, 200614, 201515, 203418, 204521, 205220, 209228, 212419,
     214019, 214524, 221319, 233326, 239136, 246133, 249947, 251833, 257845,
     263436, 283543, 318637, 320826, 330324, 346137, 352738, 360030, 365343,
     380036, 381038, 385046, 389357, 393247, 395756, 397760, 401422, 406836,
     412528, 429040, 436845, 463040, 467351, 525541, 536647, 541943, 547046,
     550439, 552241, 562345, 572045, 573249, 581450, 585256, 601127, 617748,
     627549, 638049, 644246, 654552, 671855, 680957, 690152, 706040, 724446,
     725751, 732243, 751550, 757764, 765864, 770352, 771354, 782561, 783462,
     789373, 814649, 818859, 825048, 826353, 833249, 859671, 861456, 871762,
     872764, 878776, 878877, 898176, 899885, 901139, 901442, 905147, 910241,
     926862, 927359, 942658, 943862, 951457, 958976, 966975, 971160, 973770,
     995174],
    dtype=int)
# Here we have the subject lists in the order we assigned them to the
# project's raters.
subject_list_1 = np.array(
    [100610, 102311, 102816, 104416, 105923, 108323, 109123, 111312, 111514,
     114823, 115017, 115825, 116726, 118225, 125525, 126426, 128935, 130114,
     130518, 131217, 132118, 145834, 146735, 157336, 158136, 164131, 167036,
     169747, 173334, 175237, 182436, 192439, 198653, 201515, 203418, 214019,
     221319, 318637, 320826, 346137, 360030, 365343, 385046, 393247, 401422,
     406836, 467351, 525541, 573249, 581450, 627549, 644246, 671855, 690152,
     732243, 783462, 814649, 878776, 898176, 958976],
    dtype=int)
subject_list_2 = np.array(
    [134627, 140117, 146129, 148133, 155938, 158035, 159239, 164636,
     165436, 167440, 169040, 169343, 171633, 176542, 177140, 178647,
     181636, 182739, 187345, 191336, 191841, 192641, 195041, 199655,
     204521, 205220, 212419, 233326, 239136, 246133, 251833, 263436,
     283543, 389357, 395756, 429040, 436845, 541943, 550439, 552241,
     601127, 638049, 724446, 751550, 757764, 765864, 770352, 782561,
     818859, 825048, 859671, 871762, 878877, 899885, 910241, 927359,
     942658, 951457, 971160, 973770],
    dtype=int)
subject_list_3 = ~(np.isin(subject_list, subject_list_1) |
                   np.isin(subject_list, subject_list_2))
subject_list_3 = np.sort(subject_list[subject_list_3])
# The sid that we use as a stand-in for the mean subject.
meansid = 999999


# Processing Configuration #####################################################
# This section defines variables that are needed for processing. This includes
# promarily the ids for the raters involved in the different annotation regions.

# Labels -----------------------------------------------------------------------
# The key for the labels that we use.
labelkey_early   = {'V1':  1, 'V2':  2, 'V3':   3}
labelkey_ventral = {'hV4': 4, 'VO1': 5, 'VO2':  6}
labelkey_dorsal  = {'V3a': 7, 'V3b': 8, 'IPS0': 9, 'LO1': 10}
labelkey_by_region = {
    'early': labelkey_early,
    'ventral': labelkey_ventral,
    'dorsal': labelkey_dorsal}
labelkey = {k:v for d in labelkey_by_region.values() for (k,v) in d.items()}
# Make lists of region names also:
region_areas = {k: tuple(v.keys()) for (k,v) in labelkey_by_region.items()}

# Raters -----------------------------------------------------------------------
# The name used for the mean rater (the average of other raters).
meanrater = 'mean'
# The github ids for the raters involved in the project (these are the default
# processing targets).
ventral_raters = [
    'R1',
    'R2',
    'R3',
    'R4',
    'R5']
dorsal_raters = [
    'R6',
    'R1',
    'R7',
    'R8']
raters_by_region = {
    'ventral': ventral_raters,
    'dorsal': dorsal_raters,
    'ventral_meanrater': ventral_raters,
    'dorsal_meanrater': dorsal_raters,
    'ventral_meansub': ventral_raters,
    'dorsal_meansub': dorsal_raters}

# Contours ---------------------------------------------------------------------
# Data about the visual cortex contours we are analyzing.
contours_ventral = {
    'hV4_VO1': '{hemisphere}.hV4_VO1.json',
    'VO1_VO2': '{hemisphere}.VO1_VO2.json',
    'hV4_outer': '{hemisphere}.hV4.json',
    'VO_outer': '{hemisphere}.VO_outer.json'}
contours_dorsal = {
    'V3ab_outer': '{hemisphere}.V3ab_outer.json',
    'V3ab_inner': '{hemisphere}.V3ab_inner.json',
    'IPS0_outer': '{hemisphere}.IPS0_outer.json',
    'LO1_outer':  '{hemisphere}.LO1_outer.json'}

# Traces -----------------------------------------------------------------------
# A dict of the important traces currently generated.
traces_ventral = {
    'hV4_VO1': '{hemisphere}.hV4_VO1_trace.json.gz',
    'VO1_VO2': '{hemisphere}.VO1_VO2_trace.json.gz',
    'hV4_outer': '{hemisphere}.hV4_outer_trace.json.gz',
    'VO_outer': '{hemisphere}.VO_outer_trace.json.gz',
    'outer': '{hemisphere}.outer_trace.json.gz',
    'V3v': '{hemisphere}.V3v_trace.json.gz',
    'hV4': '{hemisphere}.hV4_trace.json.gz',
    'VO1': '{hemisphere}.VO1_trace.json.gz',
    'VO2': '{hemisphere}.VO2_trace.json.gz'}
traces_dorsal = {
    'V3ab_outer': '{hemisphere}.V3ab_outer_trace.json.gz',
    'V3ab_inner': '{hemisphere}.V3ab_inner_trace.json.gz',
    'IPS0_outer': '{hemisphere}.IPS0_outer_trace.json.gz',
    'LO1_outer': '{hemisphere}.LO1_outer_trace.json.gz',
    'V3a': '{hemisphere}.V3a_trace.json.gz',
    'V3b': '{hemisphere}.V3b_trace.json.gz',
    'IPS0': '{hemisphere}.IPS0_trace.json.gz',
    'LO1': '{hemisphere}.LO1_trace.json.gz'}

# Paths and Boundaries ---------------------------------------------------------
# And the important paths (which include boundaries).
boundaries_ventral = {
    'hV4': '{hemisphere}.hV4_path.json.gz',
    'VO1': '{hemisphere}.VO1_path.json.gz',
    'VO2': '{hemisphere}.VO2_path.json.gz'}
boundaries_dorsal = {
    'V3a': '{hemisphere}.V3a_path.json.gz',
    'V3b': '{hemisphere}.V3b_path.json.gz',
    'IPS0': '{hemisphere}.IPS0_path.json.gz',
    'LO1': '{hemisphere}.LO1_path.json.gz'}
paths_ventral = dict(
    boundaries_ventral,
    hV4_VO1='{hemisphere}.hV4_VO1_path.json.gz',
    VO1_VO2='{hemisphere}.VO1_VO2_path.json.gz',
    outer='{hemisphere}.outer_path.json.gz',
    V3v='{hemisphere}.V3v_path.json.gz',
    hV4_outer='{hemisphere}.hV4_outer_path.json.gz',
    VO_outer='{hemisphere}.VO_outer_path.json.gz')
paths_dorsal = dict(
    boundaries_dorsal,
    V3ab_outer='{hemisphere}.V3ab_outer_path.json.gz',
    V3ab_inner='{hemisphere}.V3ab_inner_path.json.gz',
    IPS0_outer='{hemisphere}.IPS0_outer_path.json.gz',
    LO1_outer='{hemisphere}.LO1_outer_path.json.gz')

# Labels -----------------------------------------------------------------------
labels_ventral = {
    'labels': '{hemisphere}.ventral_label.mgz',
    'weights': '{hemisphere}.ventral_weight.mgz'}
labels_dorsal = {
    'labels': '{hemisphere}.dorsal_label.mgz',
    'weights': '{hemisphere}.dorsal_weight.mgz'}

# Reports ----------------------------------------------------------------------
reports_ventral = {
    'surface_area': '{hemisphere}.ventral_sareas.json'}
reports_dorsal = {
    'surface_area': '{hemisphere}.dorsal_sareas.json'}

# Means ------------------------------------------------------------------------
# When we create the mean contours, we operate on the processed traces of the
# individual raters. The traces that form the inputs to the meancontours are the
# meansources. The meansources must get processed into the meancontours, which
# continue forward with the meantraces, meanpaths, etc.
sources_ventral_meanrater = {
    'hV4_VO1': '{hemisphere}.hV4_VO1_trace.json.gz',
    'VO1_VO2': '{hemisphere}.VO1_VO2_trace.json.gz',
    'hV4_outer': '{hemisphere}.hV4_outer_trace.json.gz',
    'VO_outer': '{hemisphere}.VO_outer_trace.json.gz'}
contours_ventral_meanrater = {
    'hV4_VO1': '{hemisphere}.hV4_VO1.json',
    'VO1_VO2': '{hemisphere}.VO1_VO2.json',
    'hV4_outer': '{hemisphere}.hV4_outer.json',
    'VO_outer': '{hemisphere}.VO_outer.json'}
traces_ventral_meanrater = dict(traces_ventral)
boundaries_ventral_meanrater = dict(boundaries_ventral)
paths_ventral_meanrater = dict(paths_ventral)
sources_ventral_meansub = dict(
    sources_ventral_meanrater,
    V3v='{hemisphere}.V3v_trace.json.gz')
contours_ventral_meansub = dict(
    contours_ventral_meanrater,
    V3v='{hemisphere}.V3v.json')
traces_ventral_meansub = traces_ventral_meanrater
boundaries_ventral_meansub = boundaries_ventral_meanrater
paths_ventral_meansub = paths_ventral_meanrater
sources_dorsal_meanrater = {
    'V3ab_outer': '{hemisphere}.V3ab_outer_trace.json.gz',
    'V3ab_inner': '{hemisphere}.V3ab_inner_trace.json.gz',
    'IPS0_outer': '{hemisphere}.IPS0_outer_trace.json.gz',
    'LO1_outer': '{hemisphere}.LO1_outer_trace.json.gz'}
contours_dorsal_meanrater = {
    'V3ab_outer': '{hemisphere}.V3ab_outer.json',
    'V3ab_inner': '{hemisphere}.V3ab_inner.json',
    'IPS0_outer': '{hemisphere}.IPS0_outer.json',
    'LO1_outer':  '{hemisphere}.LO1_outer.json'}
traces_dorsal_meanrater = dict(traces_dorsal)
boundaries_dorsal_meanrater = dict(boundaries_dorsal)
paths_dorsal_meanrater = dict(paths_dorsal)
sources_dorsal_meansub = sources_dorsal_meanrater
contours_dorsal_meansub = contours_dorsal_meanrater
traces_dorsal_meansub = traces_dorsal_meanrater
boundaries_dorsal_meansub = boundaries_dorsal_meanrater
paths_dorsal_meansub = paths_dorsal_meanrater

# Processing Data by Group -----------------------------------------------------
# Mean items 
contours_by_region = {
    'ventral': contours_ventral,
    'dorsal': contours_dorsal}
traces_by_region = {
    'ventral': traces_ventral,
    'dorsal': traces_dorsal}
boundaries_by_region = {
    'ventral': boundaries_ventral,
    'dorsal': boundaries_dorsal}
paths_by_region = {
    'ventral': paths_ventral,
    'dorsal': paths_dorsal}
labels_by_region = {
    'ventral': labels_ventral,
    'dorsal': labels_dorsal}
reports_by_region = {
    'ventral': reports_ventral,
    'dorsal': reports_dorsal}
sources_by_region_meanrater = {
    'ventral': sources_ventral_meanrater,
    'dorsal': sources_dorsal_meanrater}
contours_by_region_meanrater = {
    'ventral': contours_ventral_meanrater,
    'dorsal': contours_dorsal_meanrater}
traces_by_region_meanrater = {
    'ventral': traces_ventral_meanrater,
    'dorsal': traces_dorsal_meanrater}
boundaries_by_region_meanrater = {
    'ventral': boundaries_ventral_meanrater,
    'dorsal': boundaries_dorsal_meanrater}
paths_by_region_meanrater = {
    'ventral': paths_ventral_meanrater,
    'dorsal': paths_dorsal_meanrater}
sources_by_region_meansub = {
    'ventral': sources_ventral_meansub,
    'dorsal': sources_dorsal_meansub}
contours_by_region_meansub = {
    'ventral': contours_ventral_meansub,
    'dorsal': contours_dorsal_meansub}
traces_by_region_meansub = {
    'ventral': traces_ventral_meansub,
    'dorsal': traces_dorsal_meansub}
boundaries_by_region_meansub = {
    'ventral': boundaries_ventral_meansub,
    'dorsal': boundaries_dorsal_meansub}
paths_by_region_meansub = {
    'ventral': paths_ventral_meansub,
    'dorsal': paths_dorsal_meansub}
region_procdata = {
    k: {
        'raters':     raters_by_region.get(k),
        'contours':   contours_by_region.get(k),
        'traces':     traces_by_region.get(k),
        'boundaries': boundaries_by_region.get(k),
        'paths':      paths_by_region.get(k),
        'labels':     labels_by_region.get(k),
        'reports':    reports_by_region.get(k)}
    for k in ('ventral', 'dorsal')}
region_procdata.update(
    {f'{k}_meanrater': {
        'raters':     raters_by_region.get(k),
        'sources':    sources_by_region_meanrater.get(k),
        'contours':   contours_by_region_meanrater.get(k),
        'traces':     traces_by_region_meanrater.get(k),
        'boundaries': boundaries_by_region_meanrater.get(k),
        'paths':      paths_by_region_meanrater.get(k),
        'labels':     labels_by_region.get(k),
        'reports':    reports_by_region.get(k)}
     for k in ('ventral', 'dorsal')})
region_procdata.update(
    {f'{k}_meansub': {
        'raters':     raters_by_region.get(k),
        'sources':    sources_by_region_meansub.get(k),
        'contours':   contours_by_region_meansub.get(k),
        'traces':     traces_by_region_meansub.get(k),
        'boundaries': boundaries_by_region_meansub.get(k),
        'paths':      paths_by_region_meansub.get(k),
        'labels':     labels_by_region.get(k),
        'reports':    reports_by_region.get(k)}
     for k in ('ventral', 'dorsal')})
# We define this simple function for looking up processing data.
def procdata(region, step):
    """Returns the processing data for the requested processing region and step.

    The `procdata` function returns a dictionary of the processing data
    associated with the requested processing region and step. If no such region
    or step is found, then an exception is raised.

    Parameters
    ----------
    region : str
        The cortical regions; may be `'ventral'` or `'dorsal'`.
    step : str
        The processing step; may be `'contours'`, `'traces'`, `'boundaries'`,
        `'paths'`, `'sources_meanrater'`, `'contours_meanrater'`,
        `'traces_meanrater'`, `'boundaries_meanrater'`, or `'paths_meanrater'`.

    Returns
    -------
    dict
        A dictionary of data associated with the given region and step.

    """
    if not isinstance(region, str):
        raise ValueError("procdata argument region must be a string")
    elif region not in region_procdata:
        raise ValueError(f'requested region not in processing data: {region}')
    else:
        regdata = region_procdata[region]
    if not isinstance(step, str):
        raise ValueError("procdata argument step must be a string")
    res = regdata.get(step, None)
    if res is None:
        raise ValueError(
            f"step not in region '{region}' processing data: {step}")
    else:
        return res


################################################################################
# Input/Output Configuration

def to_data_path(rater, sid, save_path,
                 mkdir=False, mkdir_mode=0o775,
                 expanduser=True, expandvars=True):
    """Returns a save path for the given rater, subject ID, and save path.

    `to_data_path(rater, sid, save_path)` appends directories for the rater and
    subject id (`sid`) to the given `save_path` and returns it. This is roughly
    equivalent to `os.path.join(save_path, rater, str(sid))` but it returns a
    pathlib Path object instead of a string.

    If either of `rater` or `sid` is `None` then that directory is excluded from
    the path. If `save_path` is `None`, then the current working directory is
    used in its place.

    In addition to joining the path, `to_data_path` expands variables and user
    components of the path, and the `mkdir` and `mkdir_mode` options may be
    passed to ensure that the directories are created.
    """
    import os
    from pathlib import Path
    from numbers import Integral
    if isinstance(sid, Integral):
        sid = str(sid)
    elif sid is not None and not isinstance(sid, str):
        raise ValueError(
            f"to_data_path argument sid ({sid}) must be an int, str, or None")
    if rater is not None and not isinstance(rater, str):
        raise ValueError(
            f"to_data_path argument rater ({rater}) must be a str or None")
    if not (save_path is None or
            isinstance(save_path, str) or
            isinstance(save_path, Path)):
        t = type(save_path)
        raise ValueError(
            f"to_data_path argument save_path has invalid type: {t}")
    # See if we are missing pieces.
    if sid is None:
        if rater is None:
            if save_path is None:
                # Nothing provided: just return the current directory.
                path = Path()
            else:
                path = Path(save_path)
        else:
            if save_path is None:
                path = Path(rater)
            else:
                path = Path(save_path) / rater
    else:
        if rater is None:
            if save_path is None:
                path = Path(sid)
            else:
                path = Path(save_path) / sid
        else:
            if save_path is None:
                path = Path(rater) / sid
            else:
                path = Path(save_path) / rater / sid
    if expandvars:
        path = Path(os.path.expandvars(path))
    if expanduser:
        path = path.expanduser()
    if mkdir and not path.exists():
        path.mkdir(mode=mkdir_mode, parents=True, exist_ok=True)
    return path
def cortex(sid, h):
    """Returns the cortex object for an HCP subject.

    This function is the API entry point for obtaining the hemisphere of a
    subject from the HCP.
    """
    import neuropythy as ny
    sid = int(sid)
    # If this is the mean subject, we return the fsaverage hemisphere.
    if sid == meansid:
        sub = ny.freesurfer_subject('fsaverage')
    else:
        sub = ny.data['hcp_lines'].subjects[sid]
    hem = sub.hemis[h]
    return hem


# Clean up the config namespace.
del np, os
