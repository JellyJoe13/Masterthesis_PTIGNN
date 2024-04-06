import os
import pickle

import pandas as pd
import torch
import torch_geometric as pyg
import typing

from multiprocess.pool import Pool
from tqdm import tqdm

from ptgnn.dataset.utils import dict_to_storage_path
from ptgnn.dataset.utils_chienn import download_url_to_path
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


class EZDataset(pyg.data.InMemoryDataset):
    """
    Transforms RS dataset to only contain one E/Z with 4 constituents (could thus have more than one E/Z)
    """
    def __init__(
            self,
            root: str = "src/ez",
            single_conformer: bool = True,
            mask_chiral_tags: bool = True,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
            **kwargs
    ):
        """
        Init of the EZ dataset class. Starts the downloading and processing if the corresponding files are not present.
        Uses multiprocess library to speed up computation by using all CPU cores.

        :param root: Path to which the dataset should be saved
        :param single_conformer: Filtering option removing duplicate entries using ID column of data
        :param mask_chiral_tags: If true, removes part of features of data related to chiral tags. Affects node features
            and edge attributes
        :param split: Determines whether test, train or val dataset should be loaded
        :param graph_mode: Determines whether an edge graph or vertex graph should be produced
        :param transformation_mode: Specifies the mode of transformation. Currently available: ``default``=``chienn``,
            ``permutationTree``.
        :param transformation_parameters: Configurable options of the transformation.
        :param max_atoms: Max number of atoms - required for creating embedding
        :param max_attempts: Maximal attempts of creating a 3d version of the molecule. Warning: Do not set too high
            as some molecules cannot be 'rendered' in reasonable time.
        """
        # link storage - where to fetch the data from - source of the RS dataset
        self.link_storage = {
            'train': 'https://figshare.com/ndownloader/files/30975694?private_link=e23be65a884ce7fc8543',
            'val': 'https://figshare.com/ndownloader/files/30975703?private_link=e23be65a884ce7fc8543',
            'test': 'https://figshare.com/ndownloader/files/30975679?private_link=e23be65a884ce7fc8543'
        }

        # set internal parameters
        self.single_conformer = single_conformer
        self.mask_chiral_tags = mask_chiral_tags
        self.split = split
        self.graph_mode = graph_mode
        self.transformation_mode = transformation_mode
        self.transformation_parameters = transformation_parameters
        self.pre_transform = PRE_TRANSFORM_MAPPING.get(self.graph_mode)
        self.masking = MASKING_MAPPING.get(self.graph_mode)
        self.max_atoms = max_atoms
        self.max_attempts = max_attempts

        # starts procedure of downloading and processing
        super().__init__(
            root=root,
            transform=None,
            pre_transform=self.pre_transform,
            pre_filter=None,
        )

        # part that actually loads the data into the class
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f"{split}.pt"))
        self.dataframe = pd.read_csv(os.path.join(self.processed_dir, f'{split}.csv'))

    @property
    def raw_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return f'{self.split}.csv'

    @property
    def processed_dir(self) -> str:
        name = 'single_conformer' if self.single_conformer else 'all_conformers'
        graph_mode = self.graph_mode if self.graph_mode else ''
        graph_mode += "+" + self.transformation_mode if self.transformation_mode else ''
        return os.path.join(
            self.root,
            name,
            graph_mode,
            dict_to_storage_path(self.transformation_parameters),
            'processed'
        )

    @property
    def processed_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return [f'{self.split}.pt', f'{self.split}.csv']

    def download(self) -> None:
        print(f"Starting download and dataset generation for split: {self.split}")
        # download rs dataset
        split_pickle_path = os.path.join(self.raw_dir, f'{self.split}.pickle')
        download_url_to_path(self.link_storage[self.split], split_pickle_path)

        # generate the transformation to the EZ dataset
        self.transform_to_ez()
        print("Done!")

    def transform_to_ez(self):

        # load rs dataset
        with open(os.path.join(self.raw_dir, f'{self.split}.pickle'), 'rb') as f:
            split_df = pickle.load(f)

        # should always be true
        if self.single_conformer:
            split_df = split_df.drop_duplicates(subset="ID")

        # define first worker
        def worker_filter_one_four_const_stereo(df_entry) -> bool:
            from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
            from rdkit import Chem

            # unpack passed object
            idx, entry = df_entry

            # define allowed stereo types
            allowed_stereo = [
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE
            ]

            # extract smiles string
            smiles = entry['ID']

            # get molecule
            molecule = smiles_to_3d_mol(
                smiles=smiles,
                max_number_of_atoms=100,
                max_number_of_attempts=100
            )
            if molecule is None:
                return idx, df_entry, -1

            # determine whether molecule has exactly one double bond with marked E/Z
            from rdkit import Chem
            bonds = [
                bond
                for bond in molecule.GetBonds()
                if (bond.GetBondType() == Chem.rdchem.BondType.DOUBLE) and
                   (bond.GetStereo() in allowed_stereo) and
                   (bond.GetBeginAtom().GetDegree() == 3) and
                   (bond.GetEndAtom().GetDegree() == 3)
            ]

            return idx, entry, len(bonds)

        # run first worker
        count = max(os.cpu_count(), 30)
        with Pool(processes=count) as p:
            df_collection = list(p.imap(
                worker_filter_one_four_const_stereo,
                tqdm(split_df.iterrows(), total=len(split_df), desc=f"{self.split}: filter")
            ))

        # filter and extract elements from df_collection
        df_collection = pd.DataFrame([
            elem[1]
            for elem in df_collection
            if elem[2] == 1
        ])
        df_collection.reset_index(drop='index', inplace=True)

        # second worker for generating all
        def worker_create_all_ez_isomers(df_entry):
            idx, entry = df_entry

            # fetch smiles string
            smiles = entry.ID

            # transform all to one sign
            smiles = smiles.replace("\\", "/")

            # split string
            split_smiles = smiles.split("/")

            # init new smiles
            new_smiles = [split_smiles[0]]

            for i in range(1, len(split_smiles)):
                # duplicate elements
                new_smiles = new_smiles * 2

                # append new content
                new_smiles[:int(len(new_smiles)/2)] = [
                    elem + "/" + split_smiles[i]
                    for elem in new_smiles[:int(len(new_smiles)/2)]
                ]
                new_smiles[int(len(new_smiles)/2):] = [
                    elem + "\\" + split_smiles[i]
                    for elem in new_smiles[int(len(new_smiles)/2):]
                ]

            return pd.DataFrame({
                "ID": new_smiles,
                "SMILES_nostereo": [entry['SMILES_nostereo']]*len(new_smiles)
            })

        # generate all e/z isomers of each element (then merge and drop duplicates)
        with Pool(processes=count) as p:
            df_collection = pd.concat(
                list(p.imap(
                    worker_create_all_ez_isomers,
                    tqdm(df_collection.iterrows(), total=len(df_collection), desc=f"{self.split}: isomer creation")
                ))
            )

        df_collection.reset_index(inplace=True, drop='index')
        df_collection.drop_duplicates(subset=['ID', 'SMILES_nostereo'], inplace=True)

        # third worker for generating labels
        def worker_gen_labels(df_entry):
            from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
            from rdkit import Chem

            idx, entry = df_entry

            # render molecule
            molecule = smiles_to_3d_mol(
                entry.ID,
                max_number_of_atoms=self.max_atoms,
                max_number_of_attempts=self.max_attempts
            )

            if molecule is None:
                return None

            # iterate over label until the E/Z double bond is found.
            for bond in molecule.GetBonds():
                if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
                    if bond.GetStereo() == Chem.rdchem.BondStereo.STEREOE:
                        entry['EZ_label'] = "E"
                        entry['EZ_label_binary'] = 1
                    elif bond.GetStereo() == Chem.rdchem.BondStereo.STEREOZ:
                        entry['EZ_label'] = "Z"
                        entry['EZ_label_binary'] = 0
                    else:
                        continue

            return entry

        # execute worker
        with Pool(processes=count) as p:
            df_collection = list(p.imap(
                worker_gen_labels,
                tqdm(df_collection.iterrows(), total=len(df_collection), desc=f"{self.split}: label generation")
            ))

        # filter out None elements (if molecule could not be rendered)
        df_collection = pd.DataFrame(
            [
                elem for elem in df_collection
                if elem is not None
            ]
        )

        # save dataframe
        df_collection.to_csv(os.path.join(self.raw_dir, f'{self.split}.csv'), index=None)

    def process(self) -> None:
        # load downloaded data
        split_df = pd.read_csv(os.path.join(self.raw_dir, f'{self.split}.csv'))

        if self.single_conformer:
            split_df = split_df.drop_duplicates(subset="ID")

        # iterate over dataframe (multiprocessing)
        def worker(entry):
            # required for Windows to make the threads have the proper modules
            from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
            from ptgnn.dataset.utils_chienn import get_chiro_data_from_mol
            import logging
            import torch

            index, row = entry

            # get nonstereo smiles string
            smiles_nonstereo = row["SMILES_nostereo"]

            # get the normal smiles
            smiles = row['ID']
            # get the molecule
            mol = smiles_to_3d_mol(
                smiles,
                max_number_of_attempts=self.max_attempts,
                max_number_of_atoms=self.max_atoms
            )

            # check if mol present
            if mol is None:
                return index, smiles_nonstereo, None

            # attempt to generate data object (raw)
            try:
                data = get_chiro_data_from_mol(mol)
            except Exception as e:
                logging.warning(f"Omitting molecule {smiles} as cannot be properly embedded. The original error message"
                                f" was: {e}.")
                return index, smiles_nonstereo, None

            # do transformation
            if self.pre_transform is not None:
                data = self.pre_transform(
                    data,
                    self.transformation_mode,
                    self.transformation_parameters,
                    mol=mol
                )

            # set label and append
            data.y = torch.tensor(row['EZ_label_binary']).long()

            return index, smiles_nonstereo, data

        with Pool(processes=max(os.cpu_count(), 32)) as p:
            data_list = list(p.imap(
                worker,
                tqdm(
                    split_df.iterrows(),
                    total=len(split_df),
                    desc=f"Split: {self.split}"
                )
            ))

        # re-create ordering before multiprocessing
        data_list = sorted(data_list, key=lambda x: x[0])

        # extract elements to remove
        to_remove = set([
            smiles_entry
            for _, smiles_entry, indicator in data_list
            if indicator is None
        ])

        # extract data list and remove elements to remove (from previous list)
        data_list = [
            data_object
            for _, smiles, data_object in data_list
            if data_object is not None and smiles not in to_remove
        ]

        # get data object to have the same number of layers
        if 'num_layer' in data_list[0]:
            # get number of layers
            n_layers = max([
                data.num_layer
                for data in data_list
            ])
            for data in tqdm(data_list, desc="Postprocessing matrices"):
                if data.num_layer == n_layers:
                    continue
                for idx in range(data.num_layer, n_layers):
                    for matrix in ['type_mask', 'order_matrix', 'pooling']:
                        name = f"layer{idx}_{matrix}"
                        if name not in data:
                            data[name] = [[None]]

        # save processed data
        torch.save(
            self.collate(data_list),
            os.path.join(self.processed_dir, f"{self.split}.pt")
        )
        # split_df = split_df.drop(columns="rdkit_mol_cistrans_stereo")
        split_df = split_df[~split_df['SMILES_nostereo'].isin(to_remove)]
        split_df.to_csv(os.path.join(self.processed_dir, f"{self.split}.csv"), index=None)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if isinstance(data, pyg.data.Data):
            if self.mask_chiral_tags:
                data = self.masking(data)
        return data
