"""Training script"""

import os
import time
import copy
import shutil
import random

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model import SGRAF
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
)


def main(opt):

    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    best_rsum = 0
    start_epoch = 0

    # save the history of losses from two networks
    all_loss = [[], []]

    # Warmup
    print("\n* Warmup")
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            print(
                "=> load warmup checkpoint '{}' (epoch {})".format(
                    opt.warmup_model_path, checkpoint["epoch"]
                )
            )
            print("\nValidattion ...")
            validate(opt, val_loader, [model_A, model_B])
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        epoch = 0
        for epoch in range(0, opt.warmup_epoch):
            print("[{}/{}] Warmup model_A".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_A, epoch)
            print("[{}/{}] Warmup model_B".format(epoch + 1, opt.warmup_epoch))
            warmup(opt, noisy_trainloader, model_B, epoch)

        save_checkpoint(
            {
                "epoch": epoch,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "opt": opt,
            },
            is_best=False,
            filename="warmup_model_{}.pth.tar".format(epoch),
            prefix=opt.output_dir + "/",
        )

        # evaluate on validation set
        print("\nValidattion ...")
        validate(opt, val_loader, [model_A, model_B])

    # save the history of losses from two networks
    all_loss = [[], []]
    print("\n* Co-training")

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (labeled, unlabeled)
        print("Split dataset ...")
        prob_A, prob_B, all_loss = eval_train(
            opt,
            model_A,
            model_B,
            noisy_trainloader,
            data_size,
            all_loss,
            clean_labels,
            epoch,
        )

        pred_A = split_prob(prob_A, opt.p_threshold)
        pred_B = split_prob(prob_B, opt.p_threshold)

        print("\nModel A training ...")
        # train model_A
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_B,
            prob=prob_B,
        )
        train(opt, model_A, model_B, labeled_trainloader, unlabeled_trainloader, epoch)

        print("\nModel B training ...")
        # train model_B
        labeled_trainloader, unlabeled_trainloader = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_A,
            prob=prob_A,
        )
        train(opt, model_B, model_A, labeled_trainloader, unlabeled_trainloader, epoch)

        print("\nValidattion ...")
        # evaluate on validation set
        rsum = validate(opt, val_loader, [model_A, model_B])

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader=None, epoch=None):
    """
    One epoch training.
    """
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labels_l = []
    pred_labels_l = []
    labels_u = []
    pred_labels_u = []
    end = time.time()
    for i, batch_train_data in enumerate(labeled_trainloader):
        (
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            _,
            batch_labels_l,
            batch_prob_l,
            batch_clean_labels_l,
        ) = batch_train_data
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)

        # unlabeled data
        try:
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                _,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        labels_u.append(batch_clean_labels_u)

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda()
            batch_labels_l = batch_labels_l.cuda()

        # label refinement
        with torch.no_grad():
            net.val_start()
            # labeled data
            pl = net.predict(batch_images_l, batch_text_l, batch_lengths_l)
            ptl = batch_prob_l * batch_labels_l + (1 - batch_prob_l) * pl
            targets_l = ptl.detach()
            pred_labels_l.append(ptl.cpu().numpy())

            # unlabeled data
            pu1 = net.predict(batch_images_u, batch_text_u, batch_lengths_u)
            pu2 = net2.predict(batch_images_u, batch_text_u, batch_lengths_u)
            ptu = (pu1 + pu2) / 2
            targets_u = ptu.detach()
            targets_u = targets_u.view(-1, 1)
            pred_labels_u.append(ptu.cpu().numpy())

        # drop last batch if only one sample (batch normalization require)
        if batch_images_l.size(0) == 1 or batch_images_u.size(0) == 1:
            break

        net.train_start()
        # train with labeled + unlabeled data  exponential or linear
        loss_l = net.train(
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            labels=targets_l,
            hard_negative=True,
            soft_margin=opt.soft_margin,
            mode="train",
        )
        if epoch < (opt.num_epochs // 2):
            loss_u = 0
        else:
            loss_u = net.train(
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                labels=targets_u,
                hard_negative=True,
                soft_margin=opt.soft_margin,
                mode="train",
            )

        loss = loss_l + loss_u
        losses.update(loss, batch_images_l.size(0) + batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss = model.train(images, captions, lengths, mode="warmup")
        losses.update(loss, images.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=100
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    print("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    return pred
