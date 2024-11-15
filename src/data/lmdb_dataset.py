"""
LMBD dataset for Pytorch.
Based on Jonas Geiping's torchlmdb (https://github.com/JonasGeiping/torchlmdb), licensed under MIT.
"""

# Imports: Python
import io

import os
import pickle
import warnings
import time

from pathlib import Path

# Imports: Module dependencies
import torch
import torchvision
import torchvision.transforms.functional as TF

import lmdb

from PIL import Image
from typing import Union


class LmdbHandler:

    def __init__(
        self,
        database_path: Union[str, os.PathLike],
        readonly: bool = True,
        map_size: int = 1099511627776 * 2,
        write_frequency: int = 4096,
        db_channels_first: bool = True,
        num_db_attempts: int = 10,
        max_readers: int = 128,
        readahead: bool = False,
        meminit: bool = True,
        max_spare_txns: int = 128,
    ):
        self.database_path = database_path
        self.readonly = readonly
        self.db: None | lmdb.Environment
        self.db = None

        # Writing:
        self.map_size = map_size  # Linux can grow memory as needed.
        self.write_frequency = write_frequency
        self.db_channels_first = db_channels_first

        # reading:
        self.max_readers = max_readers
        self.readahead = readahead
        self.meminit = meminit
        self.max_spare_txns = max_spare_txns

        self.open_database(readonly=self.readonly)

    def _assert_open_database(self):
        if self.db is None:
            raise ValueError(
                "LMDB Database is not opened! Call 'open_database(readonly : bool)'."
            )
        if not isinstance(self.db, lmdb.Environment):
            raise ValueError("The db attribute is not an instance of lmdb.Environment!")

    def open_database(self, readonly: bool = True):
        self.db = lmdb.open(
            self.database_path,
            subdir=False,
            max_readers=self.max_readers,
            map_size=self.map_size,
            readonly=readonly,
            lock=not readonly,
            readahead=self.readahead,
            meminit=self.meminit,
            max_spare_txns=self.max_spare_txns,
            writemap=False if readonly else True,
            map_async=False if readonly else True,
        )

    def close_database(self):
        self._assert_open_database()
        try:
            self.db.close()
        except AttributeError:
            pass

    def __getitem__(self, key: bytes) -> bytes:
        self._assert_open_database()

        with self.db.begin(write=False) as txn:
            data = txn.get(key)

        if not isinstance(data, bytes):
            data = bytes(*data)

        return data

    def get(self, key: bytes) -> bytes:
        return self.__getitem__(key)

    def write_buffer(self, buffer: list[tuple[bytes, bytes]]) -> bool:
        self._assert_open_database()
        if self.readonly:
            raise ValueError(
                "LmdbHandler was initialized with readonly=True. No write operatoins are allowed."
            )

        try:
            with self.db.begin(write=True) as txn:
                for key, value in buffer:
                    txn.put(key, value)
            return True

        except (lmdb.MapFullError, lmdb.MapResizedError) as e:
            self.close_database()
            required_bytes = sum([len(t[1]) for t in buffer])
            # Required bytes with additional buffer (*2)
            self.map_size += required_bytes * 2
            self.open_database(readonly=False)
            # Retry
            return self.write_buffer(buffer)

        except (lmdb.Error, lmdb.MemoryError, lmdb.DiskError) as e:
            return False

    def write(self, key: bytes, value: bytes) -> bool:
        return self.write_buffer([(key, value)])


class LmdbDataset(torch.utils.data.Dataset):
    """
    Implements Pytorch Dataset using Lightning Memory-Mapped Database (LMDB).
    LMDB offers fast IO operations with fast access.
    Images are pre-stored in an LMDB database using lossy WebP format on maximuim quality settings.
    To create an LMBD database from a folder with structure <dataset>/<class-folder>/<image-of-class>.{jpg|jpeg|png},
    use the static method '_create_database_from_image_folder' of this class.

    Args:
        lmdb_file : Path to pre-stored LMDB database
        transforms : torchvision transform operation(s) (Optional) that are applied on a torch.Tensor image (CxHxW).
                     No conversion from PIL/numpy etc. are needed, the image used in transforms is already a torch.Tensor.
        lmdb_config : Configuration for the LMDB dabase. If None is given, default value of LmdbConfig are used.

    """
    def __init__(
        self,
        lmdb_file: Union[str, os.PathLike],
        transforms: Union[torch.nn.Module, torchvision.transforms.Compose, None] = None,
        lmdb_handler: Union[LmdbHandler, None] = None,
    ):
        self._lmdb_file = lmdb_file
        self._transforms = transforms
        self.handler = LmdbHandler(lmdb_file) if lmdb_handler is None else lmdb_handler

        self.length = int(pickle.loads(self.handler.get(b"__len__")))
        self.keys = pickle.loads(self.handler.get(b"__keys__"))
        self.labels = pickle.loads(self.handler.get(b"__labels__"))


    def __len__(self) -> int:
        """
        Returns the length of the dataset as int (amount of samples).
        """
        return self.length

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns for a given index the respective
        tuple of (image : torch.Tensor, label : torch.Tensor)
        that is stored in the LMDB database.

        Args:
            index : int: The index to be returned

        Returns:
            image_tensor : torch.Tensor: The stored image as torch Tensor,
                           with transforms applied if transforms were
                           given during class creation.
            label : torch.Tensor: The class label of the image.

        """
        image_bytes = self.handler.get(self.keys[index])

        image_bytes = io.BytesIO(image_bytes)
        image_tensor = TF.pil_to_tensor(Image.open(image_bytes))

        if self._transforms:
            image_tensor = self._transforms(image_tensor)

        label = self.labels[index]
        label = torch.tensor(label, dtype=torch.long)

        return image_tensor, label

    @staticmethod
    def _create_database_from_image_folder(
        folder_path: Union[str, os.PathLike],
        database_path: Union[str, os.PathLike],
        lmdb_handler: Union[LmdbHandler, None] = None,
        guess_database_size: bool = True,
    ):
        folder_to_label = {}
        current_label = 0

        files = sorted(
            [
                f.resolve()
                for f in Path(folder_path).rglob("*")
                if f.suffix in [".png", ".jpg", ".jpeg"]
            ]
        )

        labels = torch.zeros((len(files),), dtype=torch.long)

        if guess_database_size:
            map_size = int(len(files) * 6_000)
        else:
            map_size = lmdb_handler.map_size if lmdb_handler else 1099511627776 * 2

        lmdb_handler = (
            LmdbHandler(database_path, readonly=False, map_size=map_size)
            if lmdb_handler is None
            else lmdb_handler
        )

        buffer = []

        from tqdm import tqdm

        for idx, file in tqdm(enumerate(files), total=len(files)):
            image = Image.open(file)
            image_bytes = io.BytesIO()
            image.save(image_bytes, "webp", lossless=False, quality=100)
            image_bytes = image_bytes.getvalue()

            folder_name = file.parent.name
            if folder_name not in folder_to_label.keys():
                folder_to_label[folder_name] = current_label
                current_label += 1

            labels[idx] = folder_to_label[folder_name]

            # Write to memory buffer instead of database
            buffer.append(
                (
                    "{}".format(idx).encode("ascii"),
                    image_bytes,
                )
            )
            # Continue if no database write is required
            if idx % lmdb_handler.write_frequency != 0:
                continue

            ret = lmdb_handler.write_buffer(buffer)

            if ret:
                buffer = []
            else:
                raise ValueError("Could not write buffer to database.")

        ret = lmdb_handler.write_buffer(buffer)
        if not ret:
            raise ValueError("Could not write buffer to database.")

        keys = ["{}".format(k).encode("ascii") for k in range(len(files))]
        ret = lmdb_handler.write(b"__keys__", pickle.dumps(keys))
        if not ret:
            raise ValueError("Could not write keys to database.")

        ret = lmdb_handler.write(b"__labels__", pickle.dumps([int(l) for l in labels]))
        if not ret:
            raise ValueError("Could not write labels to database.")

        ret = lmdb_handler.write(b"__len__", pickle.dumps(len(keys)))
        if not ret:
            raise ValueError("Could not write length to database.")

    @staticmethod
    def _create_database_from_rec(
        rec_path: Union[str, os.PathLike],
        idx_path: Union[str, os.PathLike],
        database_path: Union[str, os.PathLike],
        lmdb_handler: Union[LmdbHandler, None] = None,
        guess_database_size: bool = True,
    ):
        import mxnet as mx
        import numbers
        from tqdm import tqdm

        imgrec = mx.recordio.MXIndexedRecordIO(idx_path, rec_path, "r")
        header, _ = mx.recordio.unpack(imgrec.read_idx(0))
        if header.flag > 0:
            imgidx = list(range(1, int(header.label[0])))
        else:
            imgidx = list(imgrec.keys)

        if guess_database_size:
            map_size = int(len(imgidx) * 6_000)
        else:
            map_size = lmdb_handler.map_size if lmdb_handler else 1099511627776 * 2

        lmdb_handler = (
            LmdbHandler(database_path, readonly=False, map_size=map_size)
            if lmdb_handler is None
            else lmdb_handler
        )

        buffer = []
        labels = []

        for idx in tqdm(range(len(imgidx)), total=len(imgidx)):

            rec_index = imgidx[idx]
            s = imgrec.read_idx(rec_index)
            header, img = mx.recordio.unpack(s)
            label = header.label
            if not isinstance(label, numbers.Number):
                label = label[0]
            label = int(label)
            sample = mx.image.imdecode(img).asnumpy()

            image = Image.fromarray(sample)
            image_bytes = io.BytesIO()
            image.save(image_bytes, "webp", lossless=False, quality=100)
            image_bytes = image_bytes.getvalue()

            labels.append(label)

            # Write to memory buffer instead of database
            buffer.append(
                (
                    "{}".format(idx).encode("ascii"),
                    image_bytes,
                )
            )
            # Continue if no database write is required
            if idx % lmdb_handler.write_frequency != 0:
                continue

            ret = lmdb_handler.write_buffer(buffer)

            if ret:
                buffer = []
            else:
                raise ValueError("Could not write buffer to database.")

        ret = lmdb_handler.write_buffer(buffer)
        if not ret:
            raise ValueError("Could not write buffer to database.")

        keys = ["{}".format(k).encode("ascii") for k in range(len(files))]
        ret = lmdb_handler.write(b"__keys__", pickle.dumps(keys))
        if not ret:
            raise ValueError("Could not write keys to database.")

        ret = lmdb_handler.write(b"__labels__", pickle.dumps([int(l) for l in labels]))
        if not ret:
            raise ValueError("Could not write labels to database.")

        ret = lmdb_handler.write(b"__len__", pickle.dumps(len(keys)))
        if not ret:
            raise ValueError("Could not write length to database.")
