# Copyright (c) OpenMMLab. All rights reserved.
import os
import pandas as pd
import glob
import re

from .base_dataset import BaseDataset
from .builder import DATASETS


@DATASETS.register_module()
class DatabaseRegularDataset(BaseDataset):

    def __init__(self,
                 data_base,
                 data_regular,
                 pipeline,
                 test_mode=False):
        super().__init__(pipeline, test_mode)
        self.data_base = data_base
        self.data_regular = data_regular
        self.data_infos = self.load_datalist()

    def load_datalist(self):
        """
        Returns:
            list[dict]: A list of dicts for paired paths of LQ and GT.
        """
        data_infos = []

        data_base_root = self.data_base.get('data_root', None)
        datalist_file = self.data_base.get('datalist_file', None)
        data_keyword = self.data_base.get('data_keyword', None)
        data_info_name = self.data_base.get('data_info_name', None)
        data_name_pick = self.data_base.get('data_name_pick', None)

        data_list = pd.read_csv(datalist_file, dtype=str, header=None, comment='#')
        data_list = data_list[0]

        for data_name in data_list:
            data_name = data_name.replace('\\', '/')
            if data_name_pick is not None:
                if data_name not in data_name_pick:
                    continue
            data_paths = []
            data_folder = os.path.join(data_base_root, data_name)
            for k in data_keyword:
                pth = glob.glob(os.path.join(data_folder, k))
                pth = [x.replace('\\', '/') for x in pth]
                data_paths.extend(pth)
            data_paths = sorted(data_paths)
            if len(data_paths) == 0:
                continue

            for pth in data_paths:
                base_name = pth.split(data_name)[1][1:]
                data_pair = dict()
                data_pair[data_info_name] = pth
                if self.data_regular is not None:
                    for key, val in self.data_regular.items():
                        data_root = val.get('data_root', None)
                        if data_root is None:
                            data_root = data_base_root
                        data_re = val.get('data_re', None)
                        assert data_re is not None, f'regular not found...'
                        name = base_name
                        for val1, val2 in data_re:
                            name = re.sub(val1, val2, name)
                        path = os.path.join(data_root, data_name, name)
                        assert os.path.exists(path), f'{path} not existing...'
                        data_pair[key] = path
                data_infos.append(data_pair)
        assert len(data_infos), f'no data existing...'
        return data_infos
