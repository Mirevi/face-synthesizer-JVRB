import os
import sys


def get_latest_checkpoint_epoch(checkpoints_location, max_epoch=sys.maxsize):
    checkpoints = [i for i in os.listdir(checkpoints_location) if i.endswith('.pth')]
    epochs = list(set([int(i.split('_')[0]) for i in checkpoints]))
    epochs.sort(reverse=True)
    for epoch in epochs:
        if epoch <= max_epoch:
            return epoch

    raise RuntimeError('Checkpoint with max epoch {} does not exist! Only found checkpoints from epochs {}'.format(
        max_epoch, epochs))


def count_images(image_location):
    return len(get_image_list(image_location))


def get_image_list(image_location):
    return [i for i in os.listdir(image_location) if i.lower().endswith('.png') or i.lower().endswith('.jpg')]


def create_text_file(file_path, content):
    if type(content) is not list:
        content = [content]

    writer = open(file_path, 'w')
    writer.writelines(content)
    writer.close()


def read_text_file(file_path):
    reader = open(file_path, 'r')
    content = reader.readlines()
    reader.close()
    return content


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
