#! /hits/fast/mbm/treydewk/.venv/venv_up/bin/python
from pathlib import Path
import cclib as cc
from contextlib import contextmanager
import os
import re
from openbabel import openbabel as ob

@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)

path = Path.cwd()

logfiles = [f for f in path.glob('**/*.log')]

drop = []

### convert log to pdb ###
obConversion = ob.OBConversion()
obConversion.SetInAndOutFormats("log", "pdb")
mol = ob.OBMol()

for file in logfiles:

    name = file.name
    parent = file.parent
    stem = file.stem
    print(name)

    with open(file) as f:
        content = f.readlines()

    # check for normal termination
    z = 0 # counter to check for convergence on minimum

    for line in content:
        if re.search(r'\*+\s+\d\simaginary\sfrequencies.+', line): # True if not a minimum
            z += 1
 
    if z == 1:
        drop.append(file)
        continue

    with working_directory(parent):

        ### read in data for cclib ###
        data = cc.io.ccread(name)

        ### convert log to pdb ###
        obConversion.ReadFile(mol, '%s.log' % stem)
        obConversion.WriteFile(mol, '%s.pdb' % stem)
        obConversion.CloseOutFile()
        pdb_path = str(parent.resolve()) + '/%s.pdb' % stem

        ### to get spin contamination ###
        with open('%s.log' % stem, 'r') as f:
            log_lines = f.readlines()

    ### collect spin contamination ###
    stage = 0
    S2 = 0
    for line in log_lines[::-1]:
        if stage == 0:
            if re.search('S\*\*2\sbefore\sannihilation', line):
                index = log_lines[::-1].index(line)
                start = len(log_lines) - index - 1
                S2 = log_lines[start].split(' ')[-1]
                stage += 1
        if stage == 1:
            break

    ### collect other data from output file ###
    optdone = data.optdone
    enthalpy = data.enthalpy 
    freeenergy = data.freeenergy
    charge = data.charge

    with open('checks.txt', 'a') as f:
        f.write('%s\t%s\t%s' % (stem, optdone, S2))
    
    with open('energies.txt', 'a') as f:
        f.write('%s\t%s\t%s\t%s\t%s\n' % (stem, enthalpy, freeenergy, charge, pdb_path))

print('Optimizations that converged on a maximum:', drop)
confirm = input('Delete these files? [y/n]:')

if confirm == 'y':
    for file in drop:
        parent = file.parent
        name = file.name        
        with working_directory(parent):
            os.remove(name)