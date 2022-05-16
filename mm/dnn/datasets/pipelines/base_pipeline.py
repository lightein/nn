from abc import ABCMeta


class BasePipeline(metaclass=ABCMeta):

    def __call__(self, results):
        """Call function.
        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation.
        Returns:
            dict: A dict containing the processed data and information.
        """

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str
