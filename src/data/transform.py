from torchvision.transforms import ToTensor
from torchvision import transforms

from .rand_augment import RandAugment

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

def transform_image(image_size, normalize_type, horizontal_flip=False, rand_augment=False, interpolation_type="bicubic"):
    if normalize_type == "imagenet":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif normalize_type == "clip":
        mean = CLIP_MEAN 
        std = CLIP_STD
    else:
        raise ValueError()

    if interpolation_type == "bicubic":
        interpolation = transforms.InterpolationMode.BICUBIC

    if rand_augment:
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size ,interpolation=interpolation, antialias=True),
                RandAugment(num_ops=4, magnitude=16),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean , std=std),
        ])
    elif horizontal_flip:
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size ,interpolation=interpolation, antialias=True),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean , std=std),
        ])
    else:
        transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(image_size ,interpolation=interpolation, antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean , std=std),
        ])

    return transform

def normalize_image(images, image_size, normalize_type):
    if normalize_type == "imagenet":
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
    elif normalize_type == "clip":
        mean = CLIP_MEAN 
        std = CLIP_STD
    else:
        raise ValueError()

    transform = transforms.Compose([
            transforms.Normalize(mean=mean , std=std),
    ])

    return transform(images)
