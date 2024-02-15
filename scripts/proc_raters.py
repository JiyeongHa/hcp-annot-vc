#! /usr/bin/env python
################################################################################

import os, sys
import pandas as pd

import hcpannot
import hcpannot.cmd as hcpa_cmd

from hcpannot.mp import (makejobs, mprun)
from hcpannot.proc import allproc

hcpa_conf = hcpa_cmd.ConfigInOut(
    prog='proc_raters.py',
    description='Processes the individual raters one at a time..')
hcpannot.interface.default_load_path = hcpa_conf.opts['cache_path']

raters = hcpa_conf.raters
if raters is None:
    raters = hcpa_cmd.default_raters[region]
sids = hcpa_conf.sids
hemis = hcpa_conf.hemis
opts = hcpa_conf.opts
save_path = hcpa_conf.opts['save_path']
load_path = hcpa_conf.opts['load_path']
overwrite = hcpa_conf.opts['overwrite']
if overwrite is False:
    overwrite = None
nproc = hcpa_conf.opts['nproc']
region = hcpa_conf.region
if region not in ('ventral', 'dorsal'):
    raise ValueError(f"region must be ventral or dorsal; got {region}")

# Running the Jobs #############################################################

# Make the job list.
opts = dict(save_path=save_path, load_path=load_path, overwrite=overwrite)
def call_allproc(sid, h):
    return allproc(region, rater=raters, sid=sid, hemisphere=h, **opts)
def firstarg(a, b):
    return a
jobs = makejobs(sids, hemis)
# Run this step in the processing.
dfs = proc_traces_results = mprun(
    call_allproc, jobs, region,
    nproc=nproc,
    onfail=firstarg,
    onokay=firstarg)
df = pd.concat(dfs)
df.to_csv(os.path.join(save_path, f'proc_{region}.tsv'), sep='\t', index=False)

