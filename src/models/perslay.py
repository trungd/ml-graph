from dlex import MainConfig
from dlex.datasets.torch import Dataset
from dlex.torch.models import BaseModel


class PersLayModel(BaseModel):
    def __init__(self, params: MainConfig, dataset: Dataset):
        super().__init__(params, dataset)