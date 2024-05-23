import csv
import json
import pathlib
from ast import literal_eval
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

_MAX_DIMS = 2


class _NumpyDataset(Dataset):
    def __init__(self, X, y=None) -> None:
        self.X = X
        self.y = None
        if y is not None:
            self.y = y

    def __getitem__(self, index):
        if self.y is not None:
            return self.X[index], self.y[index]
        return self.X[index]

    def __len__(self):
        return len(self.X)


def tensor_to_files(data_types: List[str], data: Union[torch.Tensor, np.ndarray], names: List[str], folder_path: str):
    """
    Function that stores a tabular array/tensor with
    its associated data types into a folder

    Args:
        data_types (List[str]): The list of data types associated with each covariate
        data (torch.Tensor | np.ndarray): a 2D tensor to be stored into a csv file
        names (List[str]): the header of the csv identifying covariate names
        folder_path (str): the folder path in which

    Raises:
        ValueError: if data rank is different to 2D
        ValueError: if the number of columns and data width are not equal
    """
    if isinstance(data, torch.Tensor):
        data = data.numpy()
    if d := data.ndim != _MAX_DIMS:
        msg = f"data must have {_MAX_DIMS} dimensions but found {d}"
        raise ValueError(msg)
    if len(names) != data.shape[-1]:
        msg = f"data's column size and names' lenght must be equal, found {len(names)} and {d}"
        raise ValueError(msg)

    path = pathlib.Path(folder_path)
    path.mkdir()

    with open(path.joinpath("data.csv"), "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(names)
        writer.writerows(data)

    with open(path.joinpath("types.json"), "w") as file:
        json.dump(data_types, file)


def files_to_memory(folder_path: str) -> Tuple[List[str], List[str], torch.Tensor]:
    """
    Function that retrieves a triplet from a given folder (see return hints).
    Expecting data types inside 'types.json' and data itself inside 'data.csv'.

    Args:
        folder_path (str): directory containing the two mentioned files

    Returns:
        Tuple[List[str], List[str], torch.Tensor]: A list of data types (str),
        a list of headers from its first row, data as a tensor.
    """
    path = pathlib.Path(folder_path)

    with open(path.joinpath("types.json")) as file:
        data_types = json.load(file)

    def evaluate(expression):
        try:
            return literal_eval(expression)
        except ValueError:
            return str(expression)

    with open(path.joinpath("data.csv"), newline="") as file:
        reader = csv.reader(file)
        data = []
        it = iter(reader)
        names = next(it)
        for line in it:
            line_ = [evaluate(elem) for elem in line]
            data.append(line_)
    return data_types, names, torch.tensor(data)


def tensor_to_dataloader(
    X: Union[torch.Tensor, np.ndarray], y: Union[torch.Tensor, np.ndarray, None] = None, batch_size=1000000, shuffle=False
):
    ds = _NumpyDataset(X, y)
    dl = DataLoader(ds, batch_size, shuffle=shuffle)
    return dl