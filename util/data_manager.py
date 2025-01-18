import glob
import os
import re
from os import path as osp
from IPython import embed


class Market1501(object):
    dataset_dir = "Market-1501"

    def __init__(self, root="../resource", **kwargs):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, "bounding_box_train")
        self.test_dir = osp.join(self.dataset_dir, "bounding_box_test")
        self.query_dir = osp.join(self.dataset_dir, "query")

        self.check_before_run()
        train, num_train_pids, num_train_imgs = self.process_dir(self.train_dir, relabel=True)
        query, num_query_pids, num_query_imgs = self.process_dir(self.query_dir)
        test, num_test_pids, num_test_imgs = self.process_dir(self.test_dir)

        num_total_pids = num_train_pids + num_query_pids + num_test_pids
        num_total_imgs = num_train_imgs + num_query_imgs + num_test_imgs

        print("=> Market1501 loaded")
        print("Dataset statistics:")
        print("  ------------------------------")
        print("  subset   | # ids | # images")
        print("  ------------------------------")
        print("  train    | {:5d} | {:8d}".format(num_train_pids, num_train_imgs))
        print("  query    | {:5d} | {:8d}".format(num_query_pids, num_query_imgs))
        print("  test     | {:5d} | {:8d}".format(num_test_pids, num_test_imgs))
        print("  ------------------------------")
        print("  total    | {:5d} | {:8d}".format(num_total_pids, num_total_imgs))
        print("  ------------------------------")

        self.train = train
        self.query = query
        self.gallery = test

        self.num_train_pids = num_train_pids
        self.num_query_pids = num_query_pids
        self.num_test_pids = num_test_pids

    # 检查文件夹是否存在
    def check_before_run(self):
        if not osp.exists(self.dataset_dir):
            raise RuntimeError(f"{self.dataset_dir} is not available")
        if not osp.exists(self.train_dir):
            raise RuntimeError(f"{self.train_dir} is not available")
        if not osp.exists(self.test_dir):
            raise RuntimeError(f"{self.test_dir} is not available")
        if not osp.exists(self.query_dir):
            raise RuntimeError(f"{self.query_dir} is not available")

    """
        处理指定目录下的所有 .jpg 图像文件。
    
        参数:
            dir_path (str): 需要处理的目录路径。
            relabel (bool, 可选): 是否重新标记人物ID。默认为 False。
    
        返回:
            dataset (list): 包含人物ID、摄像头ID和图像路径的元组列表。
            num_pids (int): 不同人物ID的数量。
            num_imgs (int): 图像文件的数量。
    """

    def process_dir(self, dir_path, relabel=False):
        # 读取所有jpg文件
        img_paths = glob.glob(osp.join(dir_path, "*.jpg"))
        pattern = re.compile(r"(\d+)_c(\d)")
        pid_container = set()

        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            assert 0 <= pid <= 1501
            assert 1 <= camid <= 6
            camid -= 1
            if relabel:
                pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
        num_pids = len(pid_container)
        num_imgs = len(img_paths)
        return dataset, num_pids, num_imgs


img_factory = {
    "market1501": Market1501
}

def init_img_dataset(name, **kwargs):
    if name not in img_factory.keys():
        raise KeyError(f"Invalid dataset, got '{name}', but excepted to be one of {img_factory.keys()}")
    return img_factory[name](**kwargs)
