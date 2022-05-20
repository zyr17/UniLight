import torch
import json
import numpy as np
import gzip
import shutil
import os
import wandb


last_use_cuda = True


def cuda(tensor, use_cuda = None):
    """
    A cuda wrapper
    """
    global last_use_cuda
    if use_cuda is None:
        use_cuda = last_use_cuda
    last_use_cuda = use_cuda
    if not use_cuda:
        return tensor
    if tensor is None:
        return None
    if torch.cuda.is_available():
        return tensor.cuda()
    else:
        return tensor


class Fake_TXSW:
    def __init__(self):
        pass

    def add_scalar(self, *x):
        pass

    def add_image(self, *x):
        pass

    def add_graph(self, *x):
        pass

    def close(self):
        pass


class WanDB_TXSW:
    def __init__(self, wandb_api_key, wandb_entity_name, wandb_project_name, 
                 wandb_sync_mode, tensorboardx_comment, **kwargs):
        os.environ['WANDB_API_KEY'] = wandb_api_key
        wandb.init(
            entity = wandb_entity_name,
            project = wandb_project_name,
            mode = wandb_sync_mode,
            name = tensorboardx_comment,
            config = kwargs
        )

    def log_one(self, name, data, step):
        wandb.log({name: data}, step)

    def add_scalar(self, *x):
        self.log_one(*x)

    def add_image(self, *x):
        self.log_one(*x)

    def add_graph(self, *x):
        raise NotImplementedError

    def close(self):
        pass


def showarray(arr):
    arr = np.array(arr)
    print('max: %.2f, min: %.2f' % (arr.max(), arr.min()))
    # plt.imshow(arr)
    # plt.show()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def _real_gzip_file(filename):
    with open(filename, 'rb') as f_in:
        with gzip.open(filename + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filename)


def gzip_file(filename):
    # _gzip_processor(filename)
    _real_gzip_file(filename)


# CityFlow utils


def floyd(adj_mat):
    # input: adjacent np.array, disconnect is assigned a big number
    assert len(adj_mat.shape) == 2 and adj_mat.shape[0] == adj_mat.shape[1]
    n = adj_mat.shape[0]
    res = adj_mat.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if res[i][k] + res[k][j] < res[i][j]:
                    res[i][j] = res[i][k] + res[k][j]
    return res


def get_length(points):
    res = 0
    for p1, p2 in zip(points[:-1], points[1:]):
        dx = p1['x'] - p2['x']
        dy = p1['y'] - p2['y']
        res += (dx ** 2 + dy ** 2) ** 0.5
    return res


def parse_roadnet(filename):
    roadnet = json.load(open(filename))
    inters = roadnet['intersections']
    roads = {}
    for road in roadnet['roads']:
        roads[road['id']] = road
    res = {}
    virtual_res = {}
    for inter in inters:
        one = {}
        one['roadlinks'] = inter['roadLinks']
        for i in one['roadlinks']:
            lanes = set()
            detail_lanes = []
            for j in i['laneLinks']:
                lanes.add(j['startLaneIndex'])
                detail_lanes.append([
                    i['startRoad'] + '_' + str(j['startLaneIndex']), 
                    i['endRoad'] + '_' + str(j['endLaneIndex'])
                ])
            i['lanenumber'] = len(lanes)
            i['lanelinks'] = detail_lanes
            del i['laneLinks']
        one['connection'] = {}  # save edges to other intersections
        for road in inter['roads']:
            if roads[road]['startIntersection'] == inter['id']:
                one['connection'][road] = [
                    roads[road]['endIntersection'], 
                    get_length(roads[road]['points'])]
        if inter['virtual']:
            virtual_res[inter['id']] = one
        else:
            phase = inter['trafficLight']['lightphases']
            phase = [x['availableRoadLinks'] for x in phase]
            one['phases'] = phase
            res[inter['id']] = one
    return res, virtual_res


def flatten_data(datatype, data):
    if datatype == 'array':
        data = list(zip(*data))
        data = list(map(lambda x: np.stack(x), data))
        return data[0]
    elif datatype == 'dict':
        dic = data
        if len(dic) == 0:
            return {}
        res = {}
        for key in dic[0].keys():
            res[key] = np.stack([x[key] for x in dic])
        return res
    else:
        raise NotImplementedError('unknown flatten type ' + datatype)


def unpack_flattened_data(datatype, data):
    if datatype == 'array':
        raise NotImplementedError()
    elif datatype == 'dict':
        res = []
        keys = list(data.keys())
        size = len(data[keys[0]])
        for i in range(size):
            one = {}
            for key in keys:
                one[key] = data[key][i]
            res.append(one)
        return res
    else:
        raise NotImplementedError('unknown flatten type ' + datatype)


def get_intersection_info(filename, intername):
    j = json.load(open(filename))
    j = j['intersections']
    for i in j:
        if i['id'] == intername:
            return i
