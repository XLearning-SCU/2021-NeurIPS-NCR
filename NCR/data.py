"""Dataloader"""

import os
import copy
import csv
import nltk
import numpy as np

import torch
import torch.utils.data as data


class PrecompDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        captions,
        images,
        data_split,
        noise_ratio=0,
        noise_file="",
        mode="",
        pred=[],
        probability=[],
    ):
        assert 0 <= noise_ratio < 1

        self.captions = captions
        self.images = images
        self.noise_ratio = noise_ratio
        self.data_split = data_split
        self.mode = mode

        self.length = len(self.captions)

        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't.
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1

        # the development set for coco is large and so validation would be slow
        if data_split == "dev":
            self.length = 1000 * self.im_div

        # one image has five captions
        self.t2i_index = np.arange(0, self.length) // self.im_div

        # Noisy label
        if data_split == "train":
            split_idx = None
            self._t2i_index = copy.deepcopy(self.t2i_index)
            if noise_ratio:
                if os.path.exists(noise_file):
                    print("=> load noisy index from {}".format(noise_file))
                    self.t2i_index = np.load(noise_file)
                else:
                    idx = np.arange(self.length)
                    np.random.shuffle(idx)
                    noise_length = int(noise_ratio * self.length)

                    shuffle_index = self.t2i_index[idx[:noise_length]]
                    np.random.shuffle(shuffle_index)
                    self.t2i_index[idx[:noise_length]] = shuffle_index

                    np.save(noise_file, self.t2i_index)
                    print("=> save noisy index to {}".format(noise_file))

            # save clean labels
            self._labels = np.ones((self.length), dtype="int")
            self._labels[self._t2i_index != self.t2i_index] = 0

            noise_label = np.ones_like(self._labels)
            if self.mode == "labeled":
                split_idx = pred.nonzero()[0]
                self.probability = [probability[i] for i in split_idx]

            elif self.mode == "unlabeled":
                split_idx = (1 - pred).nonzero()[0]

            if split_idx is not None:
                # self.images = self.images[split_idx]
                self.captions = [self.captions[i] for i in split_idx]
                self.t2i_index = [self.t2i_index[i] for i in split_idx]
                self._t2i_index = [self._t2i_index[i] for i in split_idx]
                self._labels = [self._labels[i] for i in split_idx]
                self.length = len(self.captions)

        print("{} {} data has a size of {}".format(data_split, self.mode, self.length))

    def __getitem__(self, index):
        image = torch.Tensor(self.images[self.t2i_index[index]])
        text = torch.Tensor(self.captions[index])

        if self.data_split == "train":
            if self.mode == "labeled":
                return (
                    image,
                    text,
                    index,
                    torch.Tensor([1]),
                    torch.Tensor([self.probability[index]]),
                    self._labels[index],
                )
            elif self.mode == "unlabeled":
                return image, text, index, self._labels[index], 0
            else:
                return image, text, index, self.t2i_index[index]
        else:
            return image, text, index, self.t2i_index[index]

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        text: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    labels = None
    if len(data[0]) == 6:
        images, captions, ids, labels, prob, _labels = zip(*data)
        # Merge
        labels = torch.stack(labels, 0).long()
        # Merge
        prob = torch.stack(prob, 0)

    elif len(data[0]) == 5:
        images, captions, ids, _labels, _ = zip(*data)

    else:
        images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    text = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        text[i, :end] = cap[:end]

    if len(data[0]) == 6:
        return images, text, lengths, ids, labels, prob, _labels
    elif len(data[0]) == 5:
        return images, text, lengths, ids, _labels
    else:
        return images, text, lengths, ids


def get_dataset(data_path, data_name, data_split, vocab, return_id_caps=False):
    data_path = os.path.join(data_path, data_name)

    # Captions
    captions = []
    if data_name == "cc152k_precomp":
        img_ids = []
        with open(os.path.join(data_path, "%s_caps.tsv" % data_split)) as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for line in tsvreader:
                captions.append(line[1].strip())
                img_ids.append(line[0])

    elif data_name in ["coco_precomp", "f30k_precomp"]:
        with open(os.path.join(data_path, "%s_caps.txt" % data_split), "r") as f:
            for line in f:
                captions.append(line.strip())

    else:
        raise NotImplementedError("Unsupported dataset!")

    # caption tokens
    captions_token = []
    for index in range(len(captions)):
        caption = captions[index]
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        caption = []
        caption.append(vocab("<start>"))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab("<end>"))
        captions_token.append(caption)

    # images
    images = np.load(os.path.join(data_path, "%s_ims.npy" % data_split))
    print(
        "load {} / {} data: {} images, {} captions".format(
            data_path, data_split, images.shape[0], len(captions)
        )
    )
    if return_id_caps:
        return captions_token, images, img_ids, captions
    else:
        return captions_token, images


def get_loader(
    captions,
    images,
    data_split,
    batch_size,
    workers,
    noise_ratio=0,
    noise_file="",
    pred=[],
    prob=[],
):
    if data_split == "warmup":
        dset = PrecompDataset(captions, images, "train", noise_ratio, noise_file)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
        return data_loader, dset.length, dset._labels

    elif data_split == "train":
        labeled_dataset = PrecompDataset(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
            mode="labeled",
            pred=pred,
            probability=prob,
        )
        labeled_trainloader = torch.utils.data.DataLoader(
            dataset=labeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

        unlabeled_dataset = PrecompDataset(
            captions,
            images,
            "train",
            noise_ratio,
            noise_file,
            mode="unlabeled",
            pred=pred,
            probability=prob,
        )
        unlabeled_trainloader = torch.utils.data.DataLoader(
            dataset=unlabeled_dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )

        return labeled_trainloader, unlabeled_trainloader

    elif data_split == "dev":
        dset = PrecompDataset(captions, images, data_split)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    elif data_split in ["test", "testall", "test5k"]:
        dset = PrecompDataset(captions, images, data_split)
        data_loader = torch.utils.data.DataLoader(
            dataset=dset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            num_workers=workers,
        )
    else:
        raise NotImplementedError("Not support data split!")
    return data_loader
