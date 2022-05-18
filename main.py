from config import get_config, print_usage
from network import Network

from data.data import load_data
import pickle

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
from data.my_data import load_data


def main(config):


    # Initialize Network
    net = Network(config)

    data = {}

    if config.run_mode == 'train':
        data['train'] = load_data(config, 'train')
        data['valid'] = load_data(config, 'valid')

        net.train(data)

    if config.run_mode == 'test':
        data['test'] = load_data(config, 'test')

        net.test(data)

    return 0

if __name__ == "__main__":

    config, unparsed = get_config()

    if len(unparsed) > 0:
        # print_usage()
        exit(1)

    main(config)
