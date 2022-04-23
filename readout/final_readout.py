#! /hits/fast/mbm/treydewk/.venv/venv_up/bin/python
from pathlib import Path
import cclib as cc
import numpy as np
from contextlib import contextmanager
import os
from dbstep.Dbstep import dbstep
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, rdmolops
import re
from mordred import Calculator, descriptors
from dscribe.descriptors import ACSF, SOAP, LMBTR
from ase.io import read
import pandas as pd
import subprocess
import multiprocessing as mp
from concurrent import futures

@contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(prev_cwd)

### some function definitions for computing the buried volume
def get_sterics_default(file, center):
    sterics_default = dbstep('%s.pdb' % file.stem, volume=True, atom1=center, quiet=True, commandline=True)
    return sterics_default.bur_vol
def get_sterics_custom(file, center, radius):
    sterics_custom = dbstep('%s.pdb' % file.stem, volume=True, atom1=center, r=radius, quiet=True, commandline=True)
    return sterics_custom.bur_vol
def get_sterics_iso(file, center, isoval, radius):
    sterics_iso = dbstep('%s.cube' % file.stem, volume=True, atom1=center, r=radius, surface='density', isoval=isoval, quiet=True, commandline=True)
    return sterics_iso.bur_vol

path = Path.cwd()

logfiles = [f for f in path.glob('**/*.log') if not re.search('reactant', str(f.resolve()))] #### 

### get mordred descriptors ###
### Mordred is much faster than PaDEL: https://github.com/ecrl/padelpy
# create descriptor calculator with all descriptors
calc = Calculator(descriptors, ignore_3D=False)
molecules = []

names = []
pdbs = []
max_spins_all = []
charges_all = []
volumes_default = []
volumes_2A = []
volumes_iso = []
smiles_all = []
morgan_all = []
acsf_all = []
soap_all = []
lmbtr_all = []

for file in logfiles:

    name = file.name
    parent = file.parent
    stem = file.stem
    names.append(stem)
    pdb_path = str(parent.resolve()) + '/%s.pdb' % stem # path to pdb file created by final_energies.py
    cube_path = str(parent.resolve()) + '/%s.cube' % stem
    pdbs.append(pdb_path)

    with working_directory(parent):

        ### create cube file ###
        if not os.path.exists(cube_path):
            formchk_str = 'formchk %s.chk' % file.stem
            print('Executing command:', formchk_str)
            formchk = subprocess.run(formchk_str, shell=True)
            cubegen_str = 'cubegen 4 density %s.fchk %s.cube 80 h' % (file.stem,file.stem)
            print('Executing command:', cubegen_str)
            cubegen = subprocess.run(cubegen_str, shell=True)

        ### for mordred descriptors ###
        m = Chem.MolFromPDBFile('%s.pdb' % stem)
        molecules.append(m)

        ### read in data for ase ###
        ase_mol = read(name)

        ### to get maximum fractional spin density ###
        with open('%s.log' % stem, 'r') as f:
            log_lines = f.readlines()

    ### get spins densities ###
    stage = 0; idx = []; eles = []; spins = []; c = []

    ### find final spin densities ###
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
            elif l := re.match('\s+(\d+)\s+([A-Z][a-z]*)\s+([-\s]\d\.\d+)\s+([-\s]\d\.\d+)', line):
                idx.append(l.group(1))
                eles.append(l.group(2))
                c.append(l.group(3))
                spins.append(float(l.group(4)))

        if stage == 2: break

    ### radical center has highest spin density ###
    spins = list(np.abs(spins))
    center = int(idx[spins.index(max(spins))]) 
    max_spins_all.append(max(spins)/np.sum(spins))
    charges_all.append(c[spins.index(max(spins))])


    ### compute buried volume ###
    with working_directory(parent):
        with futures.ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context('fork')) as executor:
            job = executor.submit(get_sterics_default, file, center,)
            volumes_default.append(job.result())

            job = executor.submit(get_sterics_custom, file, center, radius=2,)
            volumes_2A.append(job.result())

            job = executor.submit(get_sterics_iso, file, center, radius=2, isoval=0.002,)
            volumes_iso.append(job.result())

    ### collect SMILES ###
    smiles = Chem.MolToSmiles(m)
    smiles_all.append(smiles)

    ### Morgan fingerprint
    morgan = np.asarray(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=1024))
    morgan_all.append(morgan)

    ### DScribe descriptors ###
    species=["H", "O", "C", "N", "S"]
    # Atom-centered symmetry functions
    acsf = ACSF(
        species=species,
        rcut=6.0,
        g2_params=[[1, 1], [1, 2], [1, 3]],
        g4_params=[[1, 1, 1], [1, 2, 1], [1, 1, -1], [1, 2, -1]],)
    acsf_mol = acsf.create(ase_mol, positions=[center-1,])
    acsf_all.append(acsf_mol)
    # Smooth overlap of atomic positions, Kai's parameters
    soap = SOAP(
        species=species,
        periodic=False,
        rcut=6.0,
        nmax=8,
        lmax=6,
        sigma=1.0)
    soap_mol = soap.create(ase_mol, positions=[center-1,])
    soap_all.append(soap_mol)
    # Local many-body tensor representation, Kai's parameters
    lmbtr = LMBTR(
        species=species,
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 1, "max": 5.8, "n": 50, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.8, "cutoff": 1e-2, "threshold": 1e-3},
        },
        k3={
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "n": 100, "sigma": 2},
            "weighting": {"function": "exp", "scale": 0.3, "cutoff": 1e-2, "threshold": 1e-3},
        },
        periodic=False,
        normalization="l2_each",
        flatten=True
    )
    lmbtr_mol = lmbtr.create(ase_mol, positions=[center-1,])
    lmbtr_all.append(lmbtr_mol)

mordred_descriptors = calc.pandas(molecules)

results = pd.DataFrame(
    zip(names, pdbs, smiles_all, max_spins_all, charges_all, volumes_default, volumes_2A,
    volumes_iso, morgan_all, acsf_all, soap_all, lmbtr_all),
    columns = ['names', 'pdb','SMILES', 'max_spin', 'mull_charge', 'bur_vol_default', 'bur_vol_2A', 'bur_vol_iso',
    'morgan', 'ACSF', 'SOAP', 'LMBTR']
)
results = results.join(mordred_descriptors)

results.to_csv('descriptors.csv')