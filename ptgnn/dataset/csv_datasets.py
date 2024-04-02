import os
import typing

import pandas as pd
import torch
import torch_geometric
from chainer_chemistry.dataset.splitters.random_splitter import RandomSplitter
from chainer_chemistry.dataset.splitters.scaffold_splitter import ScaffoldSplitter
from multiprocess.pool import Pool
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ptgnn.dataset.chainer_chemistry_split import scaffold_split
from ptgnn.dataset.utils import dict_to_storage_path
from ptgnn.dataset.utils_chienn import download_url_to_path, convert_target_for_task
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


class BaceDataset(InMemoryDataset):
    """
    Heavily based and modified(optimized) from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/csv_dataset.py
    """
    def __init__(
            self,
            root: str = "src/bace",
            task_type: str = 'regression',  # or 'classification_multilabel'
            mask_chiral_tags: bool = False,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
            **kwargs
    ):
        # connection to dataset download
        self.link_storage = {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
            'raw_name': 'bace.csv',
            'data_column': 'mol',
            'split_type': 'scaffold',
            'target_column': 'Class'
        }

        # set internal parameters
        self.task_type = task_type
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
    def raw_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return self.link_storage['raw_name']

    @property
    def processed_dir(self) -> str:
        name = self.graph_mode if self.graph_mode else ''
        name += "+" + self.transformation_mode if self.transformation_mode else ''
        return os.path.join(
            self.root,
            name,
            self.task_type,
            dict_to_storage_path(self.transformation_parameters),
            'processed'
        )

    @property
    def processed_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return ['train.pt', 'test.pt', 'val.pt']

    def download(self) -> None:
        path = os.path.join(self.raw_dir, self.raw_file_names)
        download_url_to_path(self.link_storage['url'], path)

    def split_fn(
            self,
            df: pd.DataFrame,
            data_col: str
    ) -> pd.DataFrame:
        # fetch split type from the dict
        split_type = self.link_storage['split_type']

        if split_type == 'random':
            splitter = RandomSplitter()
            train_idx, valid_idx, test_idx = splitter.train_valid_test_split(
                df,
                smiles_list=df[data_col],
                frac_train=0.7,
                frac_valid=0.1,
                frac_test=0.2,
                seed=0,
                include_chirality=True,
                return_index=True
            )
        elif split_type == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(
                df,
                smiles_list=df[data_col],
                frac_train=0.7,
                frac_valid=0.1,
                frac_test=0.2,
                seed=0,
                include_chirality=True
            )
        else:
            raise NotImplementedError(f"Split type {split_type} is not allowed, use random or scaffold instead.")

        # split
        df['split'] = None
        df.loc[train_idx, 'split'] = 'train'
        df.loc[valid_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'
        return df

    def process(self) -> None:
        # load dataframe
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names))

        # get special columns
        data_col = self.link_storage['data_column']
        target_col = self.link_storage['target_column']

        # execute split
        df = self.split_fn(df, data_col)

        # for each split type
        for split in ['train', 'val', 'test']:
            # get split of dataframe
            split_df = df[df['split'] == split]

            # do multiprocess processing of data
            def worker(entry):
                # required for Windows to make the threads have the proper modules
                from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
                from ptgnn.dataset.utils_chienn import get_chiro_data_from_mol, convert_target_for_task
                from ptgnn.transform.edge_graph import to_edge_graph
                import logging
                import torch

                index, row = entry

                # get smiles
                smiles = row[data_col]

                # get the molecule
                mol = smiles_to_3d_mol(
                    smiles,
                    max_number_of_attempts=self.max_attempts,
                    max_number_of_atoms=self.max_atoms
                )

                # check if mol present
                if mol is None:
                    return smiles, None

                # attempt to generate data object (raw)
                try:
                    data = get_chiro_data_from_mol(mol)
                except Exception as e:
                    logging.warning(f"Omitting molecule {smiles} as cannot be properly embedded. The original error "
                                    f"message was: {e}.")
                    return smiles, None

                # do transformation
                if self.pre_transform is not None:
                    data = self.pre_transform(
                        data,
                        self.transformation_mode,
                        self.transformation_parameters,
                        mol=mol
                    )

                # create label
                data.y = torch.tensor(row[target_col])
                data.y = convert_target_for_task(data.y, self.task_type)
                return smiles, data

            with Pool(processes=max(os.cpu_count(), 32)) as p:
                data_list = list(p.imap(
                    worker,
                    tqdm(
                        split_df.iterrows(),
                        total=len(split_df),
                        desc=f"Split: {split}"
                    )
                ))

            # extract data list and remove elements to remove (from previous list)
            data_list = [
                data_object
                for _, data_object in data_list
                if data_object is not None
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
                os.path.join(self.processed_dir, f"{split}.pt")
            )

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if isinstance(data, torch_geometric.data.Data):
            if self.mask_chiral_tags:
                data = self.masking(data)
        return data


class Tox21Dataset(InMemoryDataset):
    """
    Heavily based and modified(optimized) from
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/csv_dataset.py
    """
    def __init__(
            self,
            root: str = "src/tox21",
            task_type: str = 'regression',  # or 'classification_multilabel'
            mask_chiral_tags: bool = False,
            split: str = "train",
            graph_mode: str = "edge",
            transformation_mode: str = "default",
            transformation_parameters: typing.Dict[str, typing.Any] = {},
            max_atoms: int = 100,
            max_attempts: int = 100,  # significantly decreased - 5000 is way too much!
            **kwargs
    ):
        # connection to dataset download
        self.link_storage = {
            'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
            'raw_name': 'tox21.csv.gz',
            'data_column': 'smiles',
            'split_type': 'random',
            'target_column': ['NR-AR', 'NR-AR-LBD', 'NR-AhR', 'NR-Aromatase', 'NR-ER', 'NR-ER-LBD', 'NR-PPAR-gamma',
                              'SR-ARE', 'SR-ATAD5', 'SR-HSE', 'SR-MMP', 'SR-p53']
        }

        # set internal parameters
        self.task_type = task_type
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
    def raw_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return self.link_storage['raw_name']

    @property
    def processed_dir(self) -> str:
        name = self.graph_mode if self.graph_mode else ''
        name += "+" + self.transformation_mode if self.transformation_mode else ''
        return os.path.join(self.root, name, self.task_type, 'processed')

    @property
    def processed_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return ['train.pt', 'test.pt', 'val.pt']

    def download(self) -> None:
        path = os.path.join(self.raw_dir, self.raw_file_names)
        download_url_to_path(self.link_storage['url'], path)

    def split_fn(
            self,
            df: pd.DataFrame,
            data_col: str
    ) -> pd.DataFrame:
        # fetch split type from the dict
        split_type = self.link_storage['split_type']

        if split_type == 'random':
            splitter = RandomSplitter()
            train_idx, valid_idx, test_idx = splitter.train_valid_test_split(
                df,
                smiles_list=df[data_col],
                frac_train=0.7,
                frac_valid=0.1,
                frac_test=0.2,
                seed=0,
                include_chirality=True,
                return_index=True
            )
        elif split_type == 'scaffold':
            train_idx, valid_idx, test_idx = scaffold_split(
                df,
                smiles_list=df[data_col],
                frac_train=0.7,
                frac_valid=0.1,
                frac_test=0.2,
                seed=0,
                include_chirality=True
            )
        else:
            raise NotImplementedError(f"Split type {split_type} is not allowed, use random or scaffold instead.")

        # split
        df['split'] = None
        df.loc[train_idx, 'split'] = 'train'
        df.loc[valid_idx, 'split'] = 'val'
        df.loc[test_idx, 'split'] = 'test'
        return df

    def process(self) -> None:
        # load dataframe
        df = pd.read_csv(os.path.join(self.raw_dir, self.raw_file_names))

        # get special columns
        data_col = self.link_storage['data_column']
        target_col = self.link_storage['target_column']

        # execute split
        df = self.split_fn(df, data_col)

        # for each split type
        for split in ['train', 'val', 'test']:
            # get split of dataframe
            split_df = df[df['split'] == split]

            # do multiprocess processing of data
            def worker(entry):
                # required for Windows to make the threads have the proper modules
                from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
                from ptgnn.dataset.utils_chienn import get_chiro_data_from_mol, convert_target_for_task
                import logging
                import torch

                index, row = entry

                # get smiles
                smiles = row[data_col]

                # get the molecule
                mol = smiles_to_3d_mol(
                    smiles,
                    max_number_of_attempts=self.max_attempts,
                    max_number_of_atoms=self.max_atoms
                )

                # check if mol present
                if mol is None:
                    return smiles, None

                # attempt to generate data object (raw)
                try:
                    data = get_chiro_data_from_mol(mol)
                except Exception as e:
                    logging.warning(f"Omitting molecule {smiles} as cannot be properly embedded. The original error "
                                    f"message was: {e}.")
                    return smiles, None

                # do transformation
                if self.pre_transform is not None:
                    data = self.pre_transform(
                        data,
                        self.transformation_mode,
                        self.transformation_parameters,
                        mol=mol
                    )

                # create label
                data.y = torch.tensor(row[target_col])
                data.y = convert_target_for_task(data.y, self.task_type)
                return smiles, data

            with Pool(processes=max(os.cpu_count(), 32)) as p:
                data_list = list(p.imap(
                    worker,
                    tqdm(
                        split_df.iterrows(),
                        total=len(split_df),
                        desc=f"Split: {split}"
                    )
                ))

            # extract data list and remove elements to remove (from previous list)
            data_list = [
                data_object
                for _, data_object in data_list
                if data_object is not None
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
                os.path.join(self.processed_dir, f"{split}.pt")
            )

    def __getitem__(self, item):
        data = super().__getitem__(item)
        if isinstance(data, torch_geometric.data.Data):
            if self.mask_chiral_tags:
                data = self.masking(data)
        return data
