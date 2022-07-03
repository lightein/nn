import os
import pandas as pd
from mmcv.utils import scandir
from dnn.datasets import BaseDataset
from dnn.datasets import DATASETS


@DATASETS.register_module()
class SRGTDataset(BaseDataset):

    def __init__(self,
                 data_root,
                 datalist_file,
                 pipeline,
                 test_mode=False):
        super(SRGTDataset, self).__init__(pipeline, test_mode)
        self.data_root = str(data_root)
        self.datalist_file = str(datalist_file)
        self.data_infos = self.load_datalist()

    def load_datalist(self):
        data_infos = []
        data_list = pd.read_csv(self.datalist_file, header=None, comment='#')
        data_list = data_list[0]
        gt_paths = []
        for data_name in data_list:
            data_folder = os.path.join(self.data_root, data_name)
            image = list(scandir(data_folder, suffix=('.png', '.jpg'), recursive=True))
            gt_path = [os.path.join(data_folder, v) for v in image]
            gt_paths.extend(gt_path)
        for pth in gt_paths:
            data_infos.append(dict(gt_path=pth))
        return data_infos

