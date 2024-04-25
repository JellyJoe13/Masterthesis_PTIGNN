import os
import typing

import pandas as pd
import torch
import torch_geometric
from chainer_chemistry.dataset.splitters import RandomSplitter
from multiprocess.pool import Pool
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm

from ptgnn.dataset.chainer_chemistry_split import scaffold_split
from ptgnn.dataset.utils import dict_to_storage_path
from ptgnn.dataset.utils_chienn import download_url_to_path
from ptgnn.masking import MASKING_MAPPING
from ptgnn.transform import PRE_TRANSFORM_MAPPING


def filter_mc_tetra_mol(
        df_entry,
        smiles_name: str = "smiles"
):
    """
    Returns true if molecule has multiple chiral centers.

    :param df_entry: pandas row object
    :param smiles_name: name of the columns under which the stereochemical SMILES can be queried from the
        df_entry
    :return: whether or not the molecule has multiple chiral centers (plus the idx and the entry before that)
    """
    from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
    from rdkit.Chem import AllChem

    # unpack passed object
    idx, entry = df_entry

    # extract smiles string
    smiles = entry[smiles_name]

    # get molecule
    molecule = smiles_to_3d_mol(
        smiles=smiles,
        max_number_of_atoms=100,
        max_number_of_attempts=100
    )

    # if molecule cannot be rendered (i.e. is None) return appropriate information
    if molecule is None:
        return idx, df_entry, False

    # does molecule have multiple marked chiral centers
    return idx, entry, len(AllChem.FindMolChiralCenters(molecule)) == 2


def worker_create_all_combinations(
        df_entry,
        smiles_name: str = 'smiles'
):
    # unpack passed object
    idx, entry = df_entry

    # extract smiles string
    smiles = entry[smiles_name]

    # replace all double @ with normal @... wait but different forms? urgh the smiles may be different...
    # what in the first place is the label?
    # if I make it such that EXACTLY 2, then u/l prediction is possible. or distribution of max(|R|, |S|) in general

    # get clearned smiles version with only one @
    cleaned_smiles = smiles.replace("@@", "@")

    # split it
    split_smiles = cleaned_smiles.split("@")
    if len(split_smiles) != 3:
        print(f"SMILES {smiles} dropped as too many @ in smiles")
        return pd.DataFrame()

    # it is basically guaranteed that this results in 3 parts as there must be 2 centers in there
    return pd.DataFrame({
        "smiles": [
            split_smiles[0] + "@" + split_smiles[1] + "@" + split_smiles[2],
            split_smiles[0] + "@" + split_smiles[1] + "@@" + split_smiles[2],
            split_smiles[0] + "@@" + split_smiles[1] + "@" + split_smiles[2],
            split_smiles[0] + "@@" + split_smiles[1] + "@@" + split_smiles[2],
            ],
    })


def assign_lu_label(df_entry):
    # unpack passed object
    idx, entry = df_entry

    # extract smiles string
    smiles = entry['smiles']

    # render molecule
    from ptgnn.features.chienn.molecule3d import smiles_to_3d_mol
    molecule = smiles_to_3d_mol(smiles, max_number_of_atoms=100, max_number_of_attempts=100)

    if molecule is None:
        return None

    # create the label, thus first fetch stereo centers
    from rdkit import Chem
    center_one, center_two = Chem.FindMolChiralCenters(molecule)

    binary_label = center_one[1] == center_two[1]

    entry['MC_label'] = "L" if binary_label else "U"
    entry['MC_label_binary'] = 1 if binary_label else 0

    return entry


class MCDataset(InMemoryDataset):
    """
    Dataset based on Tox21 dataset and only contains molecules with 2 stereo centers. From this the tetrahedral
    stereoisomers were created and labeled using the L/U logic.
    """
    def __init__(
            self,
            root: str = "src/mc",
            task_type: str = 'classification',  # or 'classification_multilabel'
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
        Init of the MC dataset class. Starts the downloading and processing if the corresponding files are not
        present. Uses multiprocess library to speed up computation by using all CPU cores.

        :param root: Path to which the dataset should be downloaded/saved.
        :type root: str
        :param mask_chiral_tags: Whether or not to mask the chiral tags in the nodes and edges. Default ``True`` as
            it is of interest whether or not the model can learn the stereo-properties without having direct access
            to them. Should infer them using neighbor order.
        :type mask_chiral_tags: bool
        :param split: Specifies which dataset split to load for this class. Options: ``"train"``, ``"val"`` and ``test``
        :type split: str
        :param graph_mode: Mode of the graph. Either ``"edge"`` or ``"vertex"``. Controls whether an edge graph
            transformation should take place or not. For PTGNN both options are possible, for ChiENN only edge graph
            is possible or else their 'cycle_index' will not work.
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
        target_col = "MC_label_binary"

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

            if self.use_multiprocess:
                with Pool(processes=min(os.cpu_count(), 24)) as p:
                    # first process step: filtering molecules with exactly 2 stereo centers
                    df_collection = list(p.imap(
                        filter_mc_tetra_mol,
                        tqdm(split_df.iterrows(), total=len(split_df))
                    ))
                    df_collection = pd.DataFrame([
                        elem[1]
                        for elem in df_collection
                        if elem[2] == 1
                    ]).reset_index(drop='index')

                    # creating all combinations
                    df_collection = pd.concat(
                        list(map(
                            worker_create_all_combinations,
                            tqdm(df_collection.iterrows(), total=len(df_collection))
                        ))
                    ).reset_index(drop='index')
                    df_collection.drop_duplicates(subset=['smiles'], inplace=True)

                    # generate label
                    df_collection = list(p.imap(
                        assign_lu_label,
                        tqdm(df_collection.iterrows(), total=len(df_collection))
                    ))
                    df_collection = pd.DataFrame([elem for elem in df_collection if elem is not None])

                    data_list = [
                        worker(elem)
                        for elem in tqdm(df_collection.iterrows(), total=len(df_collection))
                    ]

                    data_list = list(p.map(
                        worker,
                        tqdm(
                            df_collection.iterrows(),
                            total=len(df_collection),
                            desc=f"Split: {split}"
                        )
                    ))
            else:
                # first process step: filtering molecules with exactly 2 stereo centers
                df_collection = list(map(
                    filter_mc_tetra_mol,
                    tqdm(split_df.iterrows(), total=len(split_df))
                ))
                df_collection = pd.DataFrame([
                    elem[1]
                    for elem in df_collection
                    if elem[2] == 1
                ]).reset_index(drop='index')

                # creating all combinations
                df_collection = pd.concat(
                    list(map(
                        worker_create_all_combinations,
                        tqdm(df_collection.iterrows(), total=len(df_collection))
                    ))
                ).reset_index(drop='index')
                df_collection.drop_duplicates(subset=['smiles'], inplace=True)

                # generate label
                df_collection = list(map(
                    assign_lu_label,
                    tqdm(df_collection.iterrows(), total=len(df_collection))
                ))
                df_collection = pd.DataFrame([elem for elem in df_collection if elem is not None])

                data_list = list(map(
                    worker,
                    tqdm(
                        df_collection.iterrows(),
                        total=len(df_collection),
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
