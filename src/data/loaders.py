# torch.utils.data.Dataset stores the samples and their corresponding labels.
# torch.utils.data.DataLoader wraps an iterable around the Dataset.
import os
import cv2
import torch
import threading
import queue as Queue
import mxnet as mx
import numpy as np
import numbers
from torch.utils.data import DataLoader, Dataset

from utils.utils import sort_directories_by_file_count


class IdifFace(Dataset):
    def __init__(self, root_dir, local_rank, transform, num_classes):
        super(IdifFace, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels = self.scan(root_dir, num_classes)

    def scan(self, root, num_classes):
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        for l in list_dir[:num_classes]:
            images = os.listdir(os.path.join(root,l))
            lb += 1
            for img in images:
                imgidex.append(os.path.join(l,img))
                labels.append(lb)

        return imgidex,labels
        
    def read_image(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label

    def __len__(self):
        return len(self.imgidx)


class CasiaWebFace(Dataset):
    def __init__(self, root_dir, local_rank, transform, num_classes, selective):
        super(CasiaWebFace, self).__init__()
        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        self.imgidx, self.labels = self.scan(root_dir, num_classes, selective)

    def scan(self, root, num_classes, selective):
        imgidex = []
        labels = []
        lb = -1
        list_dir = os.listdir(root)
        list_dir.sort()

        if selective:
            sorted_directories = sort_directories_by_file_count(root)
            for l, file_count in sorted_directories[:num_classes]:
                images = os.listdir(os.path.join(root, l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l, img))
                    labels.append(lb)
        else:
            for l in list_dir[:num_classes]:
                images = os.listdir(os.path.join(root,l))
                lb += 1
                for img in images:
                    imgidex.append(os.path.join(l,img))
                    labels.append(lb)

        return imgidex,labels
        
    def read_image(self,path):
        return cv2.imread(os.path.join(self.root_dir,path))

    def __getitem__(self, index):
        path = self.imgidx[index]
        img = self.read_image(path)
        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)
        sample = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class MS1MV2(Dataset):
    def __init__(self, root_dir, local_rank, img_size, transform):
        super(MS1MV2, self).__init__()

        self.transform = transform
        self.root_dir = root_dir
        self.local_rank = local_rank
        path_imgrec = os.path.join(root_dir, 'train.rec')
        path_imgidx = os.path.join(root_dir, 'train.idx')
        self.imgrec = mx.recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
        s = self.imgrec.read_idx(0)
        header, _ = mx.recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = np.array(range(1, int(header.label[0])))
        else:
            self.imgidx = np.array(list(self.imgrec.keys))

    def __getitem__(self, index):
        idx = self.imgidx[index]
        s = self.imgrec.read_idx(idx)
        header, img = mx.recordio.unpack(s)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        label = torch.tensor(label, dtype=torch.long)
        sample = mx.image.imdecode(img).asnumpy()
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, label

    def __len__(self):
        return len(self.imgidx)

class BackgroundGenerator(threading.Thread):
    def __init__(self, generator, local_rank, max_prefetch=6):
        super(BackgroundGenerator, self).__init__()
        self.queue = Queue.Queue(max_prefetch)
        self.generator = generator
        self.local_rank = local_rank
        self.daemon = True
        self.start()

    def run(self):
        torch.cuda.set_device(self.local_rank)
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)

    def next(self):
        next_item = self.queue.get()
        if next_item is None:
            raise StopIteration
        return next_item

    def __next__(self):
        return self.next()

    def __iter__(self):
        return self


class DataLoaderX(DataLoader):
    def __init__(self, local_rank, **kwargs):
        super(DataLoaderX, self).__init__(**kwargs)
        self.stream = torch.cuda.Stream(local_rank)
        self.local_rank = local_rank

    def __iter__(self):
        self.iter = super(DataLoaderX, self).__iter__()
        self.iter = BackgroundGenerator(self.iter, self.local_rank)
        self.preload()
        return self

    def preload(self):
        self.batch = next(self.iter, None)
        if self.batch is None:
            return None
        with torch.cuda.stream(self.stream):
            for k in range(len(self.batch)):
                self.batch[k] = self.batch[k].to(device=self.local_rank,
                                                 non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        if batch is None:
            raise StopIteration
        self.preload()
        return batch
