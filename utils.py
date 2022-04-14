from pathlib import Path
import numpy as np
from tempfile import TemporaryDirectory
import pandas as pd
from rdkit import Chem
from ase.io import read, write
from tempfile import TemporaryDirectory
from rdkit.Chem import rdFMCS
import re
from kgcnn.utils.data import ragged_tensor_from_nested_numpy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import tensorflow as tf
from kgcnn.utils.adj import coordinates_to_distancematrix, define_adjacency_from_distance, sort_edge_indices
from kgcnn.layers.conv.painn_conv import PAiNNUpdate, EquivariantInitialize
from kgcnn.layers.conv.painn_conv import PAiNNconv
from kgcnn.layers.geom import NodeDistanceEuclidean, BesselBasisLayer, EdgeDirectionNormalized, CosCutOffEnvelope, NodePosition
from kgcnn.layers.modules import LazyAdd, OptionalInputEmbedding
from kgcnn.layers.mlp import MLP
from kgcnn.layers.pooling import PoolingNodes
from kgcnn.layers.gather import GatherNodes, GatherNodesIngoing

### Data preparation

ref_dir = Path(
    "/hits/basement/mbm/riedmiki/structures/KR0008/reference_structures"
)

def get_charge_from_name(name):
    if m := re.match(r".*_[cx](-?\d+).*", name):
        return int(m.group(1))
    return 0

def mol_remove_implicit_H(mol):
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", atom.GetSymbol())
    return mol

def get_difference(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1, mol2]) # find maximum common substructure, converts it to a SMART string
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol) # Returns the indices of the molecule’s atoms that match a substructure query
    target_atm1 = []
    for atom in mol1.GetAtoms():
        if atom.GetIdx() not in match1:
            target_atm1.append(atom.GetIdx())
    match2 = mol2.GetSubstructMatch(mcs_mol)
    target_atm2 = []
    for atom in mol2.GetAtoms():
        if atom.GetIdx() not in match2:
            target_atm2.append(atom.GetIdx())
    return target_atm1, target_atm2

def sanitize_pdb(pdb):
    with TemporaryDirectory() as tmpdir:
        tmp = tmpdir + "tmp.pdb"
        write(tmp, read(pdb))
        mol = Chem.MolFromPDBFile(tmp, removeHs=False, sanitize=False)
        mol_remove_implicit_H(mol)
    return mol

def get_bond_idx(pdb, pdb_H, ref_dir, name):
    """Returns a unique index for an abstracted H atom.
    Input
    ----------
    PDB files of the reactant and the product radical, name of the amino acid,
    and a directory with reference PDBs.  

    Parameters
    ----------
    pdb : str
        Path to pdb of product radical.
    pdb_H : str
        Path to pdb of reactant.
    ref_dir : Path
        Directory containing reference structure PDBs that the indices are taken from.
    name : str
        Name of the amino acid.

    Returns
    -------
    list[(Path, int), (Path, int)]
        contains the path to the reference PDB, the zero-indexed index of the abstracted H atom.
        Note that PDB files are one-indexed.
    """

    # find reference pdb
    reference_pdbs = np.array(list(ref_dir.glob("*.pdb")))
    reference_names = np.array([p.stem[:-2] for p in reference_pdbs])
    reference_lvls = np.array([p.stem[-1:] for p in reference_pdbs])
    reference_charges = np.array(list(map(get_charge_from_name, reference_names)))
    match_idxs = [ref[:3].lower() == name[:3].lower() for ref in reference_names]
    # Refs w/ more atoms need to come first, w/in charge: sort levels
    order = np.lexsort((reference_lvls[match_idxs], reference_charges[match_idxs]))

    print("Structure: ", name, end="")
    pdb_H = sanitize_pdb(pdb_H)
    pdb = sanitize_pdb(pdb)
    for ref_pdb in reference_pdbs[match_idxs][order[::-1]]:

        # match non-radical to reference to find correct reference
        ref = sanitize_pdb(ref_pdb)
        diff = get_difference(ref, pdb_H)
        if len(diff[0]) > 0:
            continue

        diff = get_difference(ref, pdb)
        if len(diff[0]) != 1:
            print(
                "\nWarning: Could not identify radical in correct reference!"
            )
            continue

        print("-->", ref_pdb.name)
        idx = diff[0][0]
        break

    else:
        print("\nWarning: No matching reference was found for", name)

    return ref_pdb, idx

def add_ref_idx(row):
    ref_pdb, idx = get_bond_idx(row.pdb, row.pdb_H, ref_dir, row.names) ####
    return {'ref': ref_pdb, 'ref_idx': idx}

### Training and evaluating final model
def K_fold_cross_validation(
  inputs, targets, num_folds, model, no_data_points=6150, loss='mae', batch_size = 32, no_epochs = 100, verbose = 1,
  callbacks=tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20)
):
    """Perform K fold cross validation with a train:val:test split.
    Data will be split into K folds, one of which will be used as the test data,
    the other (K-1) will be split into training and validation data using a split
    of 8:2.
    
    Input
    ----------
    Input data, targets, number of folds to use, model to use, total number of data
    points, loss, batch size, number of epochs, level of verbosity, callbacks.  

    Parameters
    ----------
    inputs : list of numpy arrays
        List of numpy arrays to be passed to the model. Each array will be used as
        :obj:`tf.keras.layers.Input`. 
    targets : numpy array
        Array of target barriers.
    num_folds: int
        Number of folds to use.
    model: tf.keras.models.Model
        Model to train and evaluate.
    no_data_points: int, optional
        Total number of data points, defaults to 6150.
    loss: str, optional
        Loss function to use, by default, 'mae'.
    batch_size: int, optional
        Batch size to use, default is 32.
    no_epochs: int, optional
        Number of epochs to train the model for, default is 100.
    verbose: int, optional
        Level of verbosity. Default is 1.
    callbacks: tf.keras.callbacks, optional
        Callbacks to use during training. Default is
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20).

    Returns
    -------
    list
        MAEs on training data sets.
    list
        MAEs on validation data sets.
    list
        MAEs on test data sets.
    """
    train_mae_per_fold = []
    val_mae_per_fold = []
    test_mae_per_fold = []
  
    lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
        0.001, decay_steps=no_data_points*0.8/batch_size*1000, decay_rate=1, staircase=False
    )
    optimizer=tf.keras.optimizers.Adam(lr_schedule)
  
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state = 1)
    
    fold_no = 1
    for train, test in kfold.split(np.ones(no_data_points), targets):
  
        model.compile(loss=loss, optimizer=optimizer)

        targets_train, targets_val, nodes_train, nodes_val, pos_train, pos_val, \
            edge_idx_train, edge_idx_val, eri_n_train, eri_n_val, \
                eri_e_train, eri_e_val, descriptors_train, descriptors_val = train_test_split(
                    targets[train], inputs[0][train], inputs[1][train], inputs[2][train],
                    inputs[3][train], inputs[4][train], inputs[5][train], test_size = 0.2,
                    random_state=1
                )

        # normalize descriptors
        means = descriptors_train.mean(axis=0)
        stds = descriptors_train.std(axis=0)
        descriptors_train = (descriptors_train - means) / stds
        descriptors_val = (descriptors_val - means) / stds

        nodes_train, nodes_val = ragged_tensor_from_nested_numpy(nodes_train), ragged_tensor_from_nested_numpy(nodes_val)
        pos_train, pos_val = ragged_tensor_from_nested_numpy(pos_train), ragged_tensor_from_nested_numpy(pos_val)
        edge_idx_train, edge_idx_val = ragged_tensor_from_nested_numpy(edge_idx_train), ragged_tensor_from_nested_numpy(edge_idx_val)
        eri_n_train, eri_n_val = ragged_tensor_from_nested_numpy(eri_n_train), ragged_tensor_from_nested_numpy(eri_n_val)
        eri_e_train, eri_e_val = ragged_tensor_from_nested_numpy(eri_e_train), ragged_tensor_from_nested_numpy(eri_e_val)

        data_train = nodes_train, pos_train, edge_idx_train, eri_n_train, eri_e_train, descriptors_train
        data_val = nodes_val, pos_val, edge_idx_val, eri_n_val, eri_e_val, descriptors_val

        history = model.fit(data_train, targets_train,
                    batch_size=batch_size,
                    epochs=no_epochs,
                    verbose=verbose,
                    validation_data=(data_val, targets_val),
                    callbacks=callbacks)

        train_scores = model.evaluate(data_train, targets_train, verbose=0, batch_size=batch_size)
        val_scores = model.evaluate(data_val, targets_val, verbose=0, batch_size=batch_size)

        nodes_test = ragged_tensor_from_nested_numpy(inputs[0][test])
        pos_test = ragged_tensor_from_nested_numpy(inputs[1][test])
        edge_idx_test = ragged_tensor_from_nested_numpy(inputs[2][test])
        eri_n_test = ragged_tensor_from_nested_numpy(inputs[3][test])
        eri_e_test = ragged_tensor_from_nested_numpy(inputs[4][test])

        descriptors_test = inputs[5][test]
        descriptors_test = (descriptors_test - means) / stds

        data_test = nodes_test, pos_test, edge_idx_test, eri_n_test, eri_e_test, descriptors_test

        test_scores = model.evaluate(data_test, targets[test], verbose=0, batch_size=batch_size)
        
        train_mae_per_fold.append(train_scores)
        val_mae_per_fold.append(val_scores)
        test_mae_per_fold.append(test_scores)
      
        fold_no = fold_no + 1
  
    return train_mae_per_fold, val_mae_per_fold, test_mae_per_fold

def painn(inputs=[{"shape": (None,), "name": "node_attributes", "dtype": "float32", "ragged": True},
                            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32", "ragged": True},
                            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64", "ragged": True},
                            {'shape': (None, 2), 'name': 'node_radical_input', 'dtype': 'int64', 'ragged': True},
                            {'shape': (None, 1), 'name': 'edge_radical_input', 'dtype': 'int64', 'ragged': True},
                            {'shape': (25,), 'name': 'input_desc', 'dtype': 'float32', 'ragged': True}],
               input_embedding={"node": {"input_dim": 95, "output_dim": 128}},
               bessel_basis={"num_radial": 20, "cutoff": 5.0, "envelope_exponent": 5},
               depth=2,
               pooling_args={"pooling_method": "sum"},
               conv_args=None,
               update_args=None,
               output_mlp={"use_bias": [True, True, True, True, True],
                                "units": [512, 256, 128, 64, 1], "activation": ["swish", "swish", "swish", "swish", "linear"]}
               ):
    """Make PAiNN graph network via functional API.

    Args:
        inputs : list
            List of dictionaries unpacked in :obj:`tf.keras.layers.Input`. Order must match model definition.
        input_embedding : dict
            Dictionary of embedding arguments for nodes etc. unpacked in `Embedding` layers.
        bessel_basis : dict
            Dictionary of layer arguments unpacked in final `BesselBasisLayer` layer.
        depth : int
            Number of graph embedding units or depth of the network.
        pooling_args : dict
            Dictionary of layer arguments unpacked in `PoolingNodes` layer.
        conv_args : dict
            Dictionary of layer arguments unpacked in `PAiNNconv` layer.
        update_args : dict
            Dictionary of layer arguments unpacked in `PAiNNUpdate` layer.
        output_mlp : dict
            Dictionary of layer arguments unpacked in the final classification `MLP` layer block.
            Defines number of model outputs and activation.

    Returns:
        tf.keras.models.Model
    """

    # Make input
    node_input = tf.keras.layers.Input(**inputs[0])
    xyz_input = tf.keras.layers.Input(**inputs[1])
    bond_index_input = tf.keras.layers.Input(**inputs[2])
    node_radical_index = tf.keras.layers.Input(**inputs[3])
    edge_radical_index = tf.keras.layers.Input(**inputs[4])
    eri_n = node_radical_index
    eri_e = edge_radical_index
    descriptors = tf.keras.layers.Input(**inputs[5])
    z = OptionalInputEmbedding(**input_embedding['node'],
                               use_embedding=len(inputs[0]['shape']) < 2)(node_input)

    equiv_input = EquivariantInitialize(dim=3)(z)

    edi = bond_index_input
    x = xyz_input
    v = equiv_input

    pos1, pos2 = NodePosition()([x, edi])
    rij = EdgeDirectionNormalized()([pos1, pos2])
    d = NodeDistanceEuclidean()([pos1, pos2])
    env = CosCutOffEnvelope(conv_args["cutoff"])(d)
    rbf = BesselBasisLayer(**bessel_basis)(d)

    for i in range(depth):
        # Message
        ds, dv = PAiNNconv(**conv_args)([z, v, rbf, env, rij, edi])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
        # Update
        ds, dv = PAiNNUpdate(**update_args)([z, v])
        z = LazyAdd()([z, ds])
        v = LazyAdd()([v, dv])
    n = z

    n_radical = GatherNodes()([n, eri_n])
    e_radical = GatherNodesIngoing()([rbf, eri_e])

    rad_embedd = tf.keras.layers.Concatenate(axis=-1)([n_radical, e_radical])
    rad_embedd = PoolingNodes(**pooling_args)(rad_embedd)

    out = tf.keras.layers.Concatenate()([rad_embedd, descriptors])

    initial_output = MLP(**output_mlp)(out)
    concat_input = tf.keras.layers.Concatenate(axis=-1)([out, initial_output])
    main_output = MLP(**output_mlp)(concat_input)

    model = tf.keras.models.Model(inputs=[
        node_input, xyz_input, bond_index_input, node_radical_index, edge_radical_index, descriptors
    ], outputs=main_output)
    
    return model