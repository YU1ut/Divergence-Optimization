from .mydataset import ImageFolder
from collections import Counter
import os
import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler


def get_loader_with_noise(source_path, target_path, evaluation_path, transforms,  noise_func, nb_classes, noise_type=None, noise_rate=0, 
               batch_size=32, return_id=False, balanced=False, return_paths=True):
    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path],
                                return_id=return_id)
    target_folder_train = ImageFolder(os.path.join(target_path),
                                      transform=transforms[target_path],
                                      return_paths=False, return_id=return_id)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                   transform=transforms[evaluation_path],
                                   return_paths=return_paths)
    if noise_type != 'clean':
        source_labels = np.asarray([[source_folder.labels[i]] for i in range(len(source_folder.labels))])
        train_noisy_labels, actual_noise_rate = noise_func(train_labels=source_labels, noise_type=noise_type, noise_rate=noise_rate, nb_classes=nb_classes)
        source_folder.labels = [i[0] for i in train_noisy_labels]

    if balanced:
        freq = Counter(source_folder.labels)
        class_weight = {x: 1.0 / freq[x] for x in freq}
        source_weights = [class_weight[x] for x in source_folder.labels]
        sampler = WeightedRandomSampler(source_weights,
                                        len(source_folder.labels))
        print("use balanced loader")
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=True,
            num_workers=4)
    else:
        source_loader = torch.utils.data.DataLoader(
            source_folder,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=4)

    target_loader = torch.utils.data.DataLoader(
        target_folder_train,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)

    return source_loader, target_loader, test_loader, target_folder_train