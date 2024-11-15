import os
import sys

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    result = f"all params: {all_param} || trainable params: {trainable_params} || trainable%: {100 * trainable_params / all_param}"    
    return result


def get_file_count(directory):
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        file_count += len(filenames)
    return file_count


def sort_directories_by_file_count(base_path):
    directories = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    directories_file_counts = [(d, get_file_count(os.path.join(base_path, d))) for d in directories]
    directories_file_counts.sort(key=lambda x: x[1], reverse=True)
    return directories_file_counts
