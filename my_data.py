import os
import pickle
from glob import glob
import numpy as np
from transformations import rotation_matrix
from ops import np_matrix4_vector_mul
from config import get_config, print_usage
import random



def my_load_data(config, mode):
    data = {}
    if mode == "train":
        pickle_dir = config.my_data_tr
        pickle_name = os.path.join(pickle_dir, "train.pickle")
        file = open(pickle_name, 'rb')
        data = pickle.load(file)
        file.close()
    elif mode == "valid":
        pickle_dir = config.my_data_tr
        pickle_name = os.path.join(pickle_dir, "valid.pickle")
        file = open(pickle_name, 'rb')
        data = pickle.load(file)
        file.close()
    elif mode == "test":
        pickle_dir = config.my_data_tr
        pickle_name = os.path.join(pickle_dir, "test.pickle")
        file = open(pickle_name, 'rb')
        data = pickle.load(file)
        file.close()
    elif mode == "test_unknown":
        pickle_dir = config.my_unknown_data
        pickle_name = os.path.join(pickle_dir, "test.pickle")
        file = open(pickle_name, 'rb')
        data = pickle.load(file)
        file.close()
    else:
        data = {}
    return data


def separateBatch(x, b):
    return [x[i] for i in b]

def myPrepareBatch(data, batch_size):

    batches = np.arange(0, len(data['x1']))
    np.random.shuffle(batches)
    max_steps = len(data['x1']) // batch_size
    batches = np.array_split(batches[:int(batch_size*max_steps)], max_steps)

    x1 = data['x1']
    x2 = data['x2']
    Rs = data['R']
    ts = data['t']
    fs = data['flag']

    x1_b = []
    x2_b = []
    Rs_b = []
    ts_b = []
    fs_b = []

    for b in batches:

        x1_b.append(separateBatch(x1, b))
        x2_b.append(separateBatch(x2, b))
        Rs_b.append(separateBatch(Rs, b))
        fs_b.append(separateBatch(fs, b))
        ts_b.append(separateBatch(ts, b))


    return x1_b, x2_b, Rs_b, ts_b, fs_b, max_steps

def setup_data(config, mode):
    data_dir = "sun3d"
    pickle_dir = "train_pickle"
    pickle_dir = os.path.join(data_dir, pickle_dir)

    picklelist = os.listdir(pickle_dir)
    data = {}


    for name in picklelist:

        # name += '_3views'
        filepath = os.path.join(pickle_dir, name)

        # files = glob(filepath)
        # if not files:
        #     # data_gen_lock.unlock()
        #     raise RuntimeError("Data is not prepared!")


        # for filename in files:

            # pos_point = [i for i, letter in enumerate(filename) if letter == '.'][-1]
            # type = filename[pos_point + 1:]

        # if type != 'pickle':
        #     continue

        file = open(filepath, 'rb')
        data_temp = pickle.load(file)
        file.close()

        if not data:
            data = data_temp

        else:
            for key in data:
                data[key] += data_temp[key]

            # else:
            #     for key in data:
            #         if key != 'trios':
            #             for k in data[key]:
            #                 data[key][k] += data_temp[key][k]
            #         else:
            #             data[key] += data_temp[key]


    x1 = data["x1"]
    x2 = data["x2"]
    Rs = data["Rs"]
    ts = data["ts"]
    fs = data["fs"]

    shuffle_list = list(zip(x1, x2, Rs, ts, fs))
    random.shuffle(shuffle_list)

    x1, x2, Rs, ts, fs = zip(*shuffle_list)

    train_data_path = os.path.join(data_dir, "train_data")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    data = {}
    data["x1"] = x1[:int(0.8 * len(x1))]
    data["x2"] = x2[:int(0.8 * len(x1))]
    data["R"] = Rs[:int(0.8 * len(x1))]
    data["t"] = ts[:int(0.8 * len(x1))]
    data["flag"] = fs[:int(0.8 * len(x1))]

    print('Size of training data', len(data["x1"]))

    train_file_name = os.path.join(train_data_path, "train") + ".pickle"
    with open(train_file_name, "wb") as ofp:
        pickle.dump(data, ofp)

    data = {}
    data["x1"] = x1[int(0.8 * len(x1)) : int(0.9 * len(x1))]
    data["x2"] = x2[int(0.8 * len(x1)) : int(0.9 * len(x1))]
    data["R"] = Rs[int(0.8 * len(x1)) : int(0.9 * len(x1))]
    data["t"] = ts[int(0.8 * len(x1)) : int(0.9 * len(x1))]
    data["flag"] = fs[int(0.8 * len(x1)) : int(0.9 * len(x1))]

    valid_file_name = os.path.join(train_data_path, "valid") + ".pickle"
    with open(valid_file_name, "wb") as ofp:
        pickle.dump(data, ofp)

    data = {}
    data["x1"] = x1[int(0.9 * len(x1)) : len(x1)]
    data["x2"] = x2[int(0.9 * len(x1)) : len(x1)]
    data["R"] = Rs[int(0.9 * len(x1)) : len(x1)]
    data["t"] = ts[int(0.9 * len(x1)) : len(x1)]
    data["flag"] = fs[int(0.9 * len(x1)) : len(x1)]

    test_file_name = os.path.join(train_data_path, "test") + ".pickle"
    with open(test_file_name, "wb") as ofp:
        pickle.dump(data, ofp)



    return data


def main(config):

    setup_data(config, 'train')

if __name__ == '__main__':

    config, unparsed = get_config()

    if len(unparsed) > 0:
        # print_usage()
        exit(1)

    main(config)