# Copyright (c) OpenMMLab. All rights reserved.
import copy
from abc import ABCMeta
from collections import defaultdict
from torch.utils.data import Dataset

from .pipelines.compose import Compose


class BaseDataset(Dataset, metaclass=ABCMeta):
    """Base class for datasets.

    All datasets should subclass it.
    Args:
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): If True, the dataset will work in test mode.
            Otherwise, in train mode.
    """

    def __init__(self, pipeline, test_mode=False):
        super().__init__()
        self.test_mode = test_mode
        self.pipeline = Compose(pipeline)

    def prepare_train_data(self, idx):
        """Prepare training data.

        Args:
            idx (int): Index of the training batch data.

        Returns:
            dict: Returned training batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        """Prepare testing data.

        Args:
            idx (int): Index for getting each testing batch.

        Returns:
            Tensor: Returned testing batch.
        """
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        """Length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_infos)

    def __getitem__(self, idx):
        """Get item at each call.

        Args:
            idx (int): Index for getting each item.
        """
        if self.test_mode:
            return self.prepare_test_data(idx)

        return self.prepare_train_data(idx)

    def evaluate(self, results, logger=None):
        """Evaluate with different metrics.
        Args:
            results (list[tuple]): The output of forward_test() of the model.
        Return:
            dict: Evaluation results dict.
        """
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        results = [res['eval_result'] for res in results]  # a list of dict
        eval_result = defaultdict(list)  # a dict of list

        for res in results:
            for metric, val in res.items():
                eval_result[metric].append(val)
        for metric, val_list in eval_result.items():
            assert len(val_list) == len(self), (
                f'Length of evaluation result of {metric} is {len(val_list)}, '
                f'should be {len(self)}')

        # average the results
        eval_result = {
            metric: sum(values) / len(self)
            for metric, values in eval_result.items()
        }

        return eval_result
