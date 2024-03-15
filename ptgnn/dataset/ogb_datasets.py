import os
import typing
import zipfile

import pandas as pd
import torch
import torch_geometric
from multiprocess.pool import Pool
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ptgnn.dataset.utils import dict_to_storage_path
from ptgnn.dataset.utils_chienn import download_url_to_path, convert_target_for_task
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


class OGBDataset(InMemoryDataset):
    """
    Adapted from:
    https://github.com/gmum/ChiENN/blob/ee3185b39e8469a8caacf3d6d45a04c4a1cfff5b/experiments/graphgps/dataset/ogb_dataset.py
    """
    def __init__(
            self,
            ds_name: str = "hiv",
            root: str = "src/ogb",
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
        self.link_storage = {
            "hiv": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/hiv.zip",
            "pcba": "http://snap.stanford.edu/ogb/data/graphproppred/csv_mol_download/pcba.zip",
        }

        if ds_name not in self.link_storage.keys():
            raise NotImplementedError(f"Expected ogb_dataset_name in {set(self.url_dict.keys())}, got '{ds_name}'.")

        # set internal parameters
        self.ds_name = ds_name
        self.mask_chiral_tags = mask_chiral_tags
        self.split = split
        self.graph_mode = graph_mode
        self.transformation_mode = transformation_mode
        self.transformation_parameters = transformation_parameters
        self.pre_transform = PRE_TRANSFORM_MAPPING.get(self.graph_mode)
        self.masking = MASKING_MAPPING.get(self.graph_mode)
        self.max_atoms = max_atoms
        self.max_attempts = max_attempts
        self.task_type = task_type

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
    def processed_dir(self):
        graph_mode = self.graph_mode if self.graph_mode else ''
        graph_mode += "+" + self.transformation_mode if self.transformation_mode else ''
        return os.path.join(
            self.root,
            self.ds_name,
            graph_mode,
            self.task_type,
            dict_to_storage_path(self.transformation_parameters),
            'processed'
        )

    @property
    def raw_file_names(self):
        return [
            self.link_storage[self.ds_name].split("/")[-1],
            self.link_storage[self.ds_name].split("/")[-1].replace(".zip", ""),
        ]

    @property
    def processed_file_names(self) -> typing.Union[str, typing.List[str], typing.Tuple[str, ...]]:
        return ["train.pt", "val.pt", "test.pt"]

    def download(self):
        zip_path = os.path.join(self.raw_dir, self.raw_file_names[0])
        ext_path = os.path.join(self.raw_dir, self.raw_file_names[1])

        download_url_to_path(self.link_storage[self.ds_name], zip_path)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(ext_path)

    def process(self):
        dataset_dir = os.path.join(
            self.raw_dir, self.raw_file_names[1], self.ds_name
        )
        data_path = os.path.join(dataset_dir, "mapping", "mol.csv.gz")
        split_path = os.path.join(dataset_dir, "split", "scaffold")

        df = pd.read_csv(data_path).iloc[:, :-1]
        split_dict = {}

        for split in ["train", "valid", "test"]:
            idx_df = pd.read_csv(os.path.join(split_path, f"{split}.csv.gz"), header=None)
            split_dict[split] = df.iloc[idx_df[0]]

        for split, split_df in split_dict.items():
            split = "val" if split == "valid" else split

            def worker(entry):
                # required for Windows to make the threads have the proper modules
                from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
                from ptgnn.dataset.utils_chienn import get_chiro_data_from_mol
                import logging
                import torch

                index, row = entry

                # get nonstereo smiles string
                smiles = row.smiles

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

                # set label and append
                data.y = torch.tensor(row.iloc[:-1].to_list())
                data.y = convert_target_for_task(data.y, self.task_type)

                return smiles, data

            with Pool(processes=os.cpu_count()) as p:
                data_list = list(p.imap(worker, tqdm(split_df.iterrows())))

            # extract data list and remove elements to remove (from previous list)
            data_list = [
                data_object
                for _, data_object in data_list
                if data_object is not None
            ]

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
