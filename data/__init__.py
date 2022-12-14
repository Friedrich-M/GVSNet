__all__ = ['get_dataset']

from .carla_samples_loader import LoadSamples
from .carla_loader import Carla
from .lhq_loader import LHQ
def get_dataset(dataset):
    data_map = {}
    data_map['carla'] = Carla
    data_map['lhq'] = LHQ
    data_map['carla_samples'] = LoadSamples

    return data_map[dataset]
