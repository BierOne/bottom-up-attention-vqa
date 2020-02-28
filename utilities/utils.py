import errno
import os
import numpy as np
import json
from utilities import config

EPS = 1e-7


def assert_eq(real, expected):
    assert real == expected, '%s (true) vs %s (expected)' % (real, expected)


def get_file(split, question=False, answer=False, target=False):
    """ load the questions or annotations through the given split """

    def _load_file(file_name):
        print(file_name)
        with open(os.path.join(config.qa_path, file_name), 'r') as fd:
            _object = json.load(fd)
        if config.type != 'cp':
            _object = _object['annotations'] if answer else _object['questions']
        return _object

    assert question + answer + target == 1
    assert split in ['train', 'val', 'trainval', 'test']
    add_val_later = False
    if split == 'trainval':
        add_val_later = True

    if question:
        fmt = '{0}_{1}_{2}_questions.json'
    elif answer:
        fmt = '{0}_{1}_{2}_annotations.json' if config.type == 'cp' \
            else '{1}_{2}_annotations.json'

    if config.type == 'cp':
        # question file of VQA-CP only contains two splits: ['train', 'test']
        split = 'test' if split != 'train' else split
        s = fmt.format('vqacp', config.version, split)
    else:
        if split == 'val':
            split = 'val2014'
        elif split == 'test':
            split = config.test_split
        else:
            split = 'train2014'
        if config.version == 'v2':
            fmt = 'v2_' + fmt
        s = fmt.format(config.task, config.dataset, split)
    object = _load_file(s)
    if add_val_later:
        s = fmt.format(config.task, config.dataset, 'val2014')
        object += _load_file(s)
    return object


def assert_array_eq(real, expected):
    assert (np.abs(real - expected) < EPS).all(), \
        '%s (true) vs %s (expected)' % (real, expected)


def load_folder(folder, suffix):
    imgs = []
    for f in sorted(os.listdir(folder)):
        if f.endswith(suffix):
            imgs.append(os.path.join(folder, f))
    return imgs


def load_imageid(folder):
    images = load_folder(folder, 'jpg')
    img_ids = set()
    for img in images:
        img_id = int(img.split('/')[-1].split('.')[0].split('_')[-1])
        img_ids.add(img_id)
    return img_ids


def create_dir(path):
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


class Tracker:
    """
        Keep track of results over time, while having access to
        monitors to display information about them.
    """

    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors
            to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}

    class ListStorage:
        """ Storage of data points that updates the given monitors """

        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
