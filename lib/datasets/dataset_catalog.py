import os
BASE_DIR = 'datasets/'

class DatasetCatalog(object):
    dataset_attrs = {
        'VegasTrain': {
            'id': 'vegas',
            'png_dir': os.path.join(BASE_DIR, 'vegas/train/images/'),
            'txt_dir': os.path.join(BASE_DIR, 'vegas/train/anns/'),
            'split': 'train'
        },
        'VegasVal': {
            'id': 'vegas',
            'png_dir': os.path.join(BASE_DIR, 'vegas/test/images/'),
            'txt_dir': os.path.join(BASE_DIR, 'vegas/test/anns/'),
            'split': 'test'
        }
    }

    @staticmethod
    def get(name):
        attrs = DatasetCatalog.dataset_attrs[name]
        return attrs.copy()

