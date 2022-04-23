#! /hits/fast/mbm/treydewk/.venv/venv_up/bin/python
import numpy as np
from pathlib import Path
from dscribe.descriptors import LMBTR, SOAP
from ase.io import read
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
import re
import pandas as pd

# Dscribe descriptors
species=["H", "O", "C", "N", "S"] # maybe add 'Y'
# Smooth overlap of atomic positions, Kai's parameters
soap = SOAP(
    species=species,
    periodic=False,
    rcut=6.0,
    nmax=8,
    lmax=6,
    sigma=1.0)
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

data = pd.read_pickle('/hits/fast/mbm/treydewk/documentation/docs/data_complete_w_descriptors')

root = Path('/hits/basement/mbm/riedmiki/structures/KR0008/')

se_folders = list((root/'traj').glob('batch*/se'))

se_folders = se_folders + [root/'start_end_prod_1', root/'start_end_prod_2', root/'start_end_prod_3', root/'start_end_prod_4', root/'start_end_prod_6',
root/'start_end_prod_7', root/'start_end_prod_8', root/'start_end_prod_9', root/'start_end_prod_10', root/'start_end_prod_11', root/'start_end_prod_intra_2']

pdb_files_all = [f for folder in se_folders for f in folder.glob('**/*.pdb') if (re.search('_1.pdb', f.name) or re.search('_2.pdb', f.name))]

hashes_se = []
for hash1, hash2 in zip(data['hash_u1'], data['hash_u2']):
    hashes_se.append(str(hash1) + '_' + str(hash2) + '_1.pdb')
    hashes_se.append(str(hash1) + '_' + str(hash2) + '_2.pdb')

pdb_files = [f for f in pdb_files_all if f.name in hashes_se]

del hashes_se, data, pdb_files_all, root, se_folders

RDLogger.DisableLog('rdApp.*')
hash_u1 = []
hash_u2 = []
morgan = []
soap_rad = []
soap_H = []
lmbtr_rad = []
lmbtr_H = []


for file in pdb_files:
    print('Computing descriptors for {}'.format(file.name))
    split_hash = file.name.split('_')
    hash_u1.append(split_hash[0])
    hash_u2.append(split_hash[1])

    mol_ase = read(str(file.resolve()))
    mol_rdkit = Chem.MolFromPDBFile(str(file.resolve()))
    H_pos = mol_ase[0].position
    rad_idx = mol_ase.get_distance(0, slice(1, None)).argmin()+1 # find radical center
    rad_pos = mol_ase[rad_idx].position

    # Morgan fingerprint
    morgan_fp = np.asarray(AllChem.GetMorganFingerprintAsBitVect(mol_rdkit, radius=2, nBits=1024))
    morgan.append(morgan_fp)

    ### DScribe descriptors ###
    soap_on_rad = soap.create(mol_ase, [rad_idx,], verbose=False)
    soap_rad.append(soap_on_rad)
    soap_on_H = soap.create(mol_ase, [0,], verbose=False)
    soap_H.append(soap_on_H)

    lmbtr_on_rad = lmbtr.create(mol_ase, [rad_idx,], verbose=False)
    lmbtr_rad.append(lmbtr_on_rad)
    lmbtr_on_H = lmbtr.create(mol_ase, [0,], verbose=False)
    lmbtr_H.append(lmbtr_on_H)

results = pd.DataFrame(
    zip(
        pdb_files, hash_u1, hash_u2, morgan, soap_rad, soap_H, lmbtr_rad, lmbtr_H
    ),
    columns = [
        'pdb_file', 'hash_u1', 'hash_u2', 'morgan', 'soap_rad', 'soap_H', 'lmbtr_rad', 'lmbtr_H'
    ]
)

results.to_pickle('/hits/fast/mbm/treydewk/documentation/docs/Kais_descriptors')