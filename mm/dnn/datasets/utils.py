from pathlib import Path
import os.path as osp
from mmcv import scandir


def scan_folder(path, img_extensions):
    """Obtain image path list (including sub-folders) from a given folder.

    Args:
        path (str | :obj:`Path`): Folder path.
        img_extensions(tuple): image extensions

    Returns:
        list[str]: image list obtained form given folder.
    """

    if isinstance(path, (str, Path)):
        path = str(path)
    else:
        raise TypeError("'path' must be a str or a Path object, "
                        f'but received {type(path)}.')

    images = list(scandir(path, suffix=img_extensions, recursive=True))
    images = [osp.join(path, v) for v in images]
    assert images, f'{path} has no valid image file.'
    return images
