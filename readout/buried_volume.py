#! /hits/fast/mbm/treydewk/.venv/venv_up/bin/python
from dbstep.Dbstep import dbstep
from pathlib import Path
import pandas as pd
import numpy as np
from contextlib import contextmanager
import os
from openbabel import openbabel as ob
import subprocess
import multiprocessing as mp
from concurrent import futures
import re

### some function definitions ###
@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)

def get_sterics_default(file, center):
    sterics_default = dbstep('%s.pdb' % file.stem, volume=True, atom1=center, quiet=True, commandline=True)
    return sterics_default.bur_vol
def get_sterics_custom(file, center, radius):
    sterics_custom = dbstep('%s.pdb' % file.stem, volume=True, atom1=center, r=radius, quiet=True, commandline=True)
    return sterics_custom.bur_vol
def get_sterics_iso(file, center, radius, isoval):
    sterics_iso = dbstep('%s.cube' % file.stem, volume=True, atom1=center, r=radius, surface='density', isoval=isoval, quiet=True, commandline=True)
    return sterics_iso.bur_vol

path = Path.cwd()

logfiles = [f for f in path.glob('**/*.log') if not re.search('reactant', str(f.resolve()))] #### 

### convert log to pdb ###
obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("log", "pdb")
mol = ob.OBMol()

names = []
volumes_default = []; volumes_3A = []; volumes_25A = []; volumes_2A = []; volumes_15A = []
volumes_iso35 = []; volumes_iso3 = []; volumes_iso25 = []; volumes_iso2 = []; volumes_iso15 = [] 
max_spins = []; charges = []

for file in logfiles:

    names.append(file.stem)
    pdb_path = str(file.parent.resolve()) + '/%s.pdb' % file.stem
    cube_path = str(file.parent.resolve()) + '/%s.cube' % file.stem

    with working_directory(file.parent):
        ### convert log to pdb ###
        if not os.path.exists(pdb_path):
           obConversion.ReadFile(mol, file.name)
           obConversion.WriteFile(mol, '%s.pdb' % file.stem)
           obConversion.CloseOutFile()

        ### create cube file ###
        if not os.path.exists(cube_path):
            formchk_str = 'formchk %s.chk' % file.stem
            print('Executing command:', formchk_str)
            formchk = subprocess.run(formchk_str, shell=True)
            cubegen_str = 'cubegen 4 density %s.fchk %s.cube 80 h' % (file.stem, file.stem)
            print('Executing command:', cubegen_str)
            cubegen = subprocess.run(cubegen_str, shell=True)

        ### to get maximum fractional spin density ###
        with open('%s.log' % file.stem, 'r') as f:
            log_lines = f.readlines()

    ### get spin densities ###
    stage = 0; idx = []; eles = []; spins = []; c =[]

    for line in log_lines[::-1]:
        if re.search('Mulliken\scharges\sand\sspin\sdensities', line):
            index = log_lines[::-1].index(line)
            start = len(log_lines) - index - 1

    for line in log_lines[start:]:
        if stage == 0:
            if re.search('Mulliken\scharges\sand\sspin\sdensities', line):
                stage += 1
        
        if stage == 1:
            if re.search('Sum\sof\sMulliken\scharges', line):
                stage += 1
            elif m := re.match('\s+(\d+)\s+([A-Z][a-z]*)\s+([-\s]\d\.\d+)\s+([-\s]\d\.\d+)', line):
                idx.append(m.group(1))
                eles.append(m.group(2))
                c.append(m.group(3))
                spins.append(float(m.group(4)))

        if stage == 2: break

    ### radical center has highest spin density ###
    spins = list(np.abs(spins))
    center = int(idx[spins.index(max(spins))])
    max_spins.append(max(spins)/np.sum(spins))
    charges.append(c[spins.index(max(spins))])

    ### compute buried volumes ###
    with working_directory(file.parent):

        with futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as executor:
            print('Computing buried volumes of %s using 4 CPUS...' % file.stem)
            job = executor.submit(get_sterics_default, file, center,)
            volumes_default.append(job.result())

            job = executor.submit(get_sterics_custom, file, center, radius=3,)
            volumes_3A.append(job.result())

            job = executor.submit(get_sterics_custom, file, center, radius=2.5,)
            volumes_25A.append(job.result())

            job = executor.submit(get_sterics_custom, file, center, radius=2,)
            volumes_2A.append(job.result())            

            job = executor.submit(get_sterics_custom, file, center, radius=1.5,)
            volumes_15A.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=3.5, isoval=0.002,)
            volumes_iso35.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=3, isoval=0.002,)
            volumes_iso3.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=2.5, isoval=0.002,)
            volumes_iso25.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=2, isoval=0.002,)
            volumes_iso2.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=1.5, isoval=0.002,)
            volumes_iso15.append(job.result())

results = pd.DataFrame(
    zip(
        names,max_spins,charges,volumes_default,volumes_3A,volumes_25A,volumes_2A,volumes_15A,
        volumes_iso35, volumes_iso3, volumes_iso25, volumes_iso2, volumes_iso15
    ),
    columns = [
        'names', 'max_spin', 'mull_charge', 'V_bur_35A', 'V_bur_3A', 'V_bur_25A', 'V_bur_2A', 'V_bur_15A',
        'V_iso_35A', 'V_iso_3A', 'V_iso_25A', 'V_iso_2A', 'V_iso_15A'
    ]    
)

results.to_csv('bur_vols.csv')