import os
import pickle
import typing

import torch
import torch_geometric as pyg
from multiprocess.pool import Pool
from tqdm import tqdm

from ptgnn.dataset.utils_chienn import download_url_to_path
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


class RSDataset(pyg.data.InMemoryDataset):
    """
    Heavily based and modified(optimized) from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/rs_dataset.py
    """
    def __init__(
            self,
            root: str = "src/rs",
            single_conformer: bool = True,
            mask_chiral_tags: bool = False,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
    ):
        """
        Init of the RS dataset class. Starts the downloading and processing if the corresponding files are not present.
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
        # link storage - where to fetch the data from
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
            pre_filter=None
        )

        # part that actually loads the data into the class
        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f"{split}.pt"))

    @property
    def raw_file_names(self):
        return [f'{self.split}.pickle']

    @property
    def processed_dir(self) -> str:
        name = 'single_conformer' if self.single_conformer else 'all_conformers'
        graph_mode = self.graph_mode if self.graph_mode else ''
        return os.path.join(self.root, name, graph_mode, 'processed')

    @property
    def processed_file_names(self):
        return [f'{self.split}.pt', f'{self.split}.csv']

    def download(self):
        for split, link in self.link_storage.items():
            split_pickle_path = os.path.join(self.raw_dir, f'{split}.pickle')
            download_url_to_path(link, split_pickle_path)

    def process(self):
        # load downloaded data
        with open(os.path.join(self.raw_dir, f'{self.split}.pickle'), 'rb') as f:
            split_df = pickle.load(f)

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
                return smiles_nonstereo, None

            # attempt to generate data object (raw)
            try:
                data = get_chiro_data_from_mol(mol)
            except Exception as e:
                logging.warning(f"Omitting molecule {smiles} as cannot be properly embedded. The original error message"
                                f" was: {e}.")
                return smiles_nonstereo, None

            # do transformation
            if self.pre_transform is not None:
                data = self.pre_transform(data, self.transformation_mode, self.transformation_parameters)

            # set label and append
            data.y = torch.tensor(row['RS_label_binary']).long()

            return smiles_nonstereo, data

        with Pool(processes=os.cpu_count()) as p:
            data_list = list(p.imap(worker, tqdm(split_df.iterrows())))

        # extract elements to remove
        to_remove = set([
            smiles_entry
            for smiles_entry, indicator in data_list
            if indicator is None
        ])

        # extract data list and remove elements to remove (from previous list)
        data_list = [
            data_object
            for smiles, data_object in data_list
            if data_object is not None and smiles not in to_remove
        ]

        # save processed data
        torch.save(
            self.collate(data_list),
            os.path.join(self.processed_dir, f"{self.split}.pt")
        )
        split_df = split_df.drop(columns="rdkit_mol_cistrans_stereo")
        split_df[~split_df['SMILES_nostereo'].isin(to_remove)]
        split_df.to_csv(os.path.join(self.processed_dir, f"{self.split}.csv"), index=None)