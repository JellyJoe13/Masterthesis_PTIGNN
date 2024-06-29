import os
import pickle
import typing

import pandas as pd
import torch
import torch_geometric
from multiprocess.pool import Pool
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ptgnn.dataset.utils import dict_to_storage_path
from ptgnn.dataset.utils_chienn import download_url_to_path
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


class BindingAffinityDataset(InMemoryDataset):
    """
    Heavily based and modified(optimized) from
    https://github.com/gmum/ChiENN/blob/master/experiments/graphgps/dataset/binding_affinity_dataset.py
    """
    def __init__(
            self,
            root: str = "src/bindingaffinity",
            single_conformer: bool = True,
            single_enantiomer: bool = False,
            mask_chiral_tags: bool = True,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
            use_multiprocess: bool = True,
            **kwargs
    ):
        """
        Init of Bindingaffinity/Bindingrank dataset. If not downloaded already in specified root file it is downloaded
        and processed. Uses multiprocess library to speed up computation which works perfectly for Windows but
        apparently causes some issues on linux (thus can be disabled).

        :param root: Path to which the dataset should be downloaded/saved.
        :type root: str
        :param single_conformer: Filtering option removing duplicate entries using stereo-SMILES entry in data.
        :type single_conformer: bool
        :param single_enantiomer: Controls whether only one enantiomer should be in dataset. This means that the
            non-stereo SMILES string is used to sample one element. Counterproductive for the use in this project, so
            always ``False``.
        :type single_enantiomer: bool
        :param mask_chiral_tags: Whether or not to mask the chiral tags in the nodes and edges. Default ``True`` as
            it is of interest whether or not the model can learn the stereo-properties without having direct access
            to them. Should infer them using neighbor order.
        :type mask_chiral_tags: bool
        :param split: Specifies which dataset split to load for this class. Options: ``"train"``, ``"val"`` and ``test``
        :type split: str
        :param graph_mode: Mode of the graph. Either ``"edge"`` or ``"vertex"``. Controls whether an edge graph
            transformation should take place or not. For PTGNN both options are possible, for ChiENN only edge graph
            is possible or else their 'circle_index' will not work.
        :type graph_mode: str
        :param transformation_mode: Specifies the mode of transformation. Currently available: ``"default"``=``chienn``,
            ``"permutation_tree"``.
        :type transformation_mode: str
        :param transformation_parameters: Configurable options of the transformation.
        :type transformation_parameters: typing.Dict[str, typing.Any]
        :param max_atoms: Max number of atoms - required for creating embedding
        :type max_atoms: int
        :param max_attempts: Maximal attempts of creating a 3d version of the molecule. Warning: Do not set too high
            as some molecules cannot be 'rendered' in reasonable time.
        :type max_attempts: int
        :param use_multiprocess: Whether or not the generation should be parallelized using the multiprocess framework.
            Use with caution on Linux - for some reason it uses significantly more cores on Linux, maybe some reason
            the default process uses multiple CPUs already...
        :type use_multiprocess: bool
        :param kwargs: Catches access arguments which are not specified for all datasets. CAUTION: make sure the
            parameters have the correct name as a mis-spelling will not cause an error.
        """
        self.link_storage = {
            'train': 'https://figshare.com/ndownloader/files/30975697?private_link=e23be65a884ce7fc8543',
            'val': 'https://figshare.com/ndownloader/files/30975706?private_link=e23be65a884ce7fc8543',
            'test': 'https://figshare.com/ndownloader/files/30975682?private_link=e23be65a884ce7fc8543'
        }

        # set internal parameters
        self.single_conformer = single_conformer
        self.single_enantiomer = single_enantiomer
        self.mask_chiral_tags = mask_chiral_tags
        self.split = split
        self.graph_mode = graph_mode
        self.transformation_mode = transformation_mode
        self.transformation_parameters = transformation_parameters
        self.pre_transform = PRE_TRANSFORM_MAPPING.get(self.graph_mode)
        self.masking = MASKING_MAPPING.get(self.graph_mode)
        self.max_atoms = max_atoms
        self.max_attempts = max_attempts
        self.use_multiprocess = use_multiprocess

        # starts procedure of downloading and processing
        super().__init__(
            root=root,
            transform=None,
            pre_transform=self.pre_transform,
            pre_filter=None
        )
        # part that actually loads the data into the class

        self.data, self.slices = torch.load(os.path.join(self.processed_dir, f"{split}.pt"))
        self.dataframe = pd.read_csv(os.path.join(self.processed_dir, f"{split}.csv"))

    @property
    def raw_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return f"{self.split}.pickle"

    @property
    def processed_dir(self) -> str:
        directory = 'single_conformer' if self.single_conformer else 'all_conformers'
        if self.single_enantiomer:
            directory = directory + "+single_enantiomer"
        graph_mode = self.graph_mode if self.graph_mode else ''
        graph_mode += "+" + self.transformation_mode if self.transformation_mode else ''
        return os.path.join(
            self.root,
            directory,
            graph_mode,
            dict_to_storage_path(self.transformation_parameters),
            'processed'
        )

    @property
    def processed_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return [f'{self.split}.pt', f'{self.split}.csv']

    def download(self) -> None:
        for split, link in self.link_storage.items():
            split_pickle_path = os.path.join(self.raw_dir, f'{split}.pickle')
            download_url_to_path(link, split_pickle_path)

    def process(self) -> None:
        # load downloaded data
        with open(os.path.join(self.raw_dir, f'{self.split}.pickle'), 'rb') as f:
            split_df = pickle.load(f)

        if self.single_conformer:
            split_df = split_df.drop_duplicates(subset="ID")

        if self.single_enantiomer:
            split_df = split_df.group_by('SMILES_nostereo').sample(n=1, random_state=0)

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
            data.y = torch.tensor(row['top_score']).float()

            return index, smiles_nonstereo, data

        if self.use_multiprocess:
            with Pool(processes=min(os.cpu_count(), 24)) as p:
                data_list = list(p.imap(
                    worker,
                    tqdm(
                        split_df.iterrows(),
                        total=len(split_df),
                        desc=f"Split: {self.split}"
                    )
                ))
        else:
            data_list = [
                worker(entry)
                for entry in tqdm(
                    split_df.iterrows(),
                    total=len(split_df),
                    desc=f"Split: {self.split}"
                )
            ]

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
        split_df = split_df.drop(columns="rdkit_mol_cistrans_stereo")
        split_df = split_df[~split_df['SMILES_nostereo'].isin(to_remove)]
        split_df.to_csv(os.path.join(self.processed_dir, f"{self.split}.csv"), index=None)

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if isinstance(data, torch_geometric.data.Data):
            if self.mask_chiral_tags:
                data = self.masking(data)
        return data
