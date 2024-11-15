from .loaders import CasiaWebFace, MS1MV2, IdifFace
from .lmdb_dataset import LmdbDataset


def get_dataset(rank, transform, **kwargs):
    dataset_name = kwargs["dataset_name"]

    if dataset_name == "casia_webface":
        trainset = CasiaWebFace(
            root_dir=kwargs["dataset_path"], 
            local_rank=rank, 
            transform=transform,
            num_classes=kwargs["num_classes"], 
            selective=kwargs["selective_dataset"]
        )

    elif dataset_name == "WEBFACE4M":
        trainset = LmdbDataset(
            lmdb_file=kwargs["dataset_path"], 
            transforms=transform
        )

    elif dataset_name == "MS1MV2":
        trainset = MS1MV2(
            root_dir=kwargs["dataset_path"], 
            local_rank=rank, 
            img_size=kwargs["image_size"],
            transform=transform,
        )

    elif dataset_name == "Idifface":
        trainset = IdifFace(
            root_dir=kwargs["dataset_path"], 
            local_rank=rank, 
            transform=transform,
            num_classes=kwargs["num_classes"]
        )
        
    else:
        raise ValueError()

    return trainset