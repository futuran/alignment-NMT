#!/usr/bin/env python
"""Training on a single process."""
import torch

from onmt.inputters.inputter import IterOnDevice
from onmt.model_builder import build_model
from onmt.utils.optimizers import Optimizer
from onmt.utils.misc import set_random_seed
from onmt.trainer import build_trainer
from onmt.models import build_model_saver
from onmt.utils.logging import init_logger, logger
from onmt.utils.parse import ArgumentParser

from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter


def configure_process(opt, device_id):
    if device_id >= 0:
        torch.cuda.set_device(device_id)
    set_random_seed(opt.seed, device_id >= 0)


def _get_model_opts(opt, checkpoint=None):
    """Get `model_opt` to build model, may load from `checkpoint` if any."""
    if checkpoint is not None:
        model_opt = ArgumentParser.ckpt_model_opts(checkpoint["opt"])
        ArgumentParser.update_model_opts(model_opt)
        ArgumentParser.validate_model_opts(model_opt)
        if (opt.tensorboard_log_dir == model_opt.tensorboard_log_dir and
                hasattr(model_opt, 'tensorboard_log_dir_dated')):
            # ensure tensorboard output is written in the directory
            # of previous checkpoints
            opt.tensorboard_log_dir_dated = model_opt.tensorboard_log_dir_dated
        # Override checkpoint's update_embeddings as it defaults to false
        model_opt.update_vocab = opt.update_vocab
    else:
        model_opt = opt
    return model_opt


def _build_valid_iter(opt, fields, transforms_cls):
    """Build iterator used for validation."""
    valid_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=False)
    return valid_iter


def _build_train_iter(opt, fields, transforms_cls, stride=1, offset=0):
    """Build training iterator."""
    train_iter = build_dynamic_dataset_iter(
        fields, transforms_cls, opt, is_train=True,
        stride=stride, offset=offset)
    return train_iter


def main(opt, fields, transforms_cls, checkpoint, device_id,
         batch_queue=None, semaphore=None):
    """Start training on `device_id`."""
    # NOTE: It's important that ``opt`` has been validated and updated
    # at this point.
    configure_process(opt, device_id)
    init_logger(opt.log_file)

    model_opt = _get_model_opts(opt, checkpoint=checkpoint)

    # Build model.
    model = build_model(model_opt, opt, fields, checkpoint)
    model.count_parameters(log=logger.info)

    # Build optimizer.
    optim = Optimizer.from_opt(model, opt, checkpoint=checkpoint)

    # Build model saver
    model_saver = build_model_saver(model_opt, opt, model, fields, optim)

    trainer = build_trainer(
        opt, device_id, model, fields, optim, model_saver=model_saver)

    if batch_queue is None:
        _train_iter = _build_train_iter(opt, fields, transforms_cls)
        train_iter = IterOnDevice(_train_iter, device_id)
    else:
        assert semaphore is not None, \
            "Using batch_queue requires semaphore as well"

        def _train_iter():
            while True:
                batch = batch_queue.get()
                semaphore.release()
                # Move batch to specified device
                IterOnDevice.batch_to_device(batch, device_id)
                yield batch

        train_iter = _train_iter()

    valid_iter = _build_valid_iter(opt, fields, transforms_cls)
    if valid_iter is not None:
        valid_iter = IterOnDevice(valid_iter, device_id)

    if len(opt.gpu_ranks):
        logger.info('Starting training on GPU: %s' % opt.gpu_ranks)
    else:
        logger.info('Starting training on CPU, could be very slow')
    train_steps = opt.train_steps
    if opt.single_pass and train_steps > 0:
        logger.warning("Option single_pass is enabled, ignoring train_steps.")
        train_steps = 0

    """
    tmp=torch.tensor([[[ 1606,  1310,  2280,    22,   140,   201,  1092,    12,     3,  4758,
           4713,    22,   140,   784,  1092,     6,     9,  7160, 11554,    15,
             47,   893,  2992,    14,     4,  2954,  5220,    15,    64,   893,
           2992,    19,    31,    11],
         [   95,    22,   498,    48,  4310,  1105,  4815,    22,  2227,    29,
            785,   699,   126,  9018, 11131,     6,     9,   572,  1770,    15,
            848,  2566,  1003,  8230,   721,    82,   276,    44,    52,    27,
             55,    19,   258,    10],
         [ 2184,  6966,     5,   879,  3574,  1676,    12,  4742,     5,     3,
           4709,  2338,  3518,    38,   133,     6,     9,  7452,    14,     4,
            465, 12504,     2,   433,     4,    86,  2061,  5322,     2,  7689,
             43,    88,    10,    11],
         [  628,  8891,  4396,     5,    21,  1986,   743,  2402,    84,    23,
           3346,    32,    75,    22,   978,     6,     9,   289,    14,   488,
             13,    10,  1039, 14243,  2530,     4,  3882,  4015,   361,   117,
              8,   146,    13,    10],
         [  628,  8891,  4396,     5,    21,  1986,   743,  2402,    84,    23,
           3346,    32,    75,    22,   978,     6,     9,   289,    14,   488,
             13,    10,  1039, 14243,  2530,     4,  3882,  4015,   361,   117,
              8,   146,    13,    10],
         [ 5671,     5,  2695,    12,  2315,  1908, 10572,  1498,  1873,     7,
             12,  2766,   950,     3,  6316,     6,     9, 12814,   155,    19,
          13695,   846,    17, 12184,  3054,    14,  1153,    13,    27,  1542,
             19,   153,    10,    11],
         [  398,  6105,  4432,    22,  3367,   336,  5883,   961,  1645,  1650,
            207,    12,   862, 10796,   207,     6,     9,   909,    15,  1230,
           1348,   318,     2,   909,    19,  3855,  1348,   318,     2,   909,
             14,  3002,   389,    11],
         [  398,  6105,  4432,    22,  3367,   336,  5883,   961,  1645,  1650,
            207,    12,   862, 10796,   207,     6,     9,   909,    15,  1230,
           1348,   318,     2,   909,    19,  3855,  1348,   318,     2,   909,
             14,  3002,   389,    11]]])

    tmp2=torch.tensor([[[  938,  2265,  2214,  1492,    23,    50,   406,    28,     3,   680,
              5,  9163,  9933,  2672,   111,  6479,     6,     9, 13595,  3056,
              2,  5349,   368,    14,    15,   713,  2534,    17,   177,    16,
             37,    19,   258,   116,    10,    11],
         [   25,  3649,    79,    49,   356,  2876,  3467,    20,   681,    28,
           2826,     5,  1728,  4538,  1981,  4542,     6,     9,  2031,  1008,
           4998,  1750,  2942,     2,   251,    14,  3171,  2600,  7220,  3742,
             15,   379,    16,    97,    10,    11],
         [   25,  4373,   357,   338,     5,   195,   974,   654,  2699,    18,
            181,  2226,   506,  1195,    20,   107,     6,     9,  1036,     8,
           3810,  7051,  7820,   123,   434,  2049,  5113,   721,     2,  1229,
          12448,     8,  1252,    13,    10,    11],
         [   25,  4373,   357,   338,     5,   195,   974,   654,  2699,    18,
            181,  2226,   506,  1195,    20,   107,     6,     9,  1036,     8,
           3810,  7051,  7820,   123,   434,  2049,  5113,   721,     2,  1229,
          12448,     8,  1252,    13,    10,    11],
         [  938,  2265,  2214,  1492,    23,    50,   406,    28,     3,   680,
              5,  9163,  9933,  2672,   111,  6479,     6,     9, 13595,  3056,
              2,  5349,   368,    14,    15,   713,  2534,    17,   177,    16,
             37,    19,   258,   116,    10,    11],
         [ 1870,  3573,    23,     3,  1612,  9505,   151,    35,  4135,  2879,
           6438,     7,  9954,   524,    39,  2240,     6,     9,  3179,  2953,
           1136,    14,  1124,  1844,     2, 12719,   172,     2,  1390,    54,
          12138,   172,     8,  2577,    10,    11],
         [  938,  2265,  2214,  1492,    23,    50,   406,    28,     3,   680,
              5,  9163,  9933,  2672,   111,  6479,     6,     9, 13595,  3056,
              2,  5349,   368,    14,    15,   713,  2534,    17,   177,    16,
             37,    19,   258,   116,    10,    11],
         [   77,   340,   107,     3,   951,  6549,  1108,   338,    18,  6783,
           1300,    40,    85,    34,    32,  4175,     6,     9,  5527,    82,
          13438,   158,    41,   150,    36,   123,  1254,     2,  1250,  1770,
            573,     8,   100,    13,    10,    11]]])

    for sentence in tmp[0]:
        decoded = []
        for i, vocab in enumerate(sentence):
            dd = fields['src'].base_field.vocab.itos[vocab.item()]
            if dd == '<sep>':
                print(i)
            decoded.append(dd)
        print(' '.join(decoded))

    for sentence in tmp2[0]:
        decoded = []
        for i, vocab in enumerate(sentence):
            dd = fields['src'].base_field.vocab.itos[vocab.item()]
            if dd == '<sep>':
                print(i)
            decoded.append(dd)
        print(' '.join(decoded))
    """

    trainer.train(
        train_iter,
        train_steps,
        save_checkpoint_steps=opt.save_checkpoint_steps,
        valid_iter=valid_iter,
        valid_steps=opt.valid_steps)

    if trainer.report_manager.tensorboard_writer is not None:
        trainer.report_manager.tensorboard_writer.close()
