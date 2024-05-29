import copy
import logging
import os
import pathlib
import threading
import time
import torch

from torch.multiprocessing import Process, Queue, Event
from tqdm.auto import tqdm

import pprint
import sys
import gc

# get the current directory
current_directory = os.path.dirname(os.path.realpath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from torch.multiprocessing import set_start_method, set_sharing_strategy
import torch.multiprocessing as mp
from .utils import init_logging, load_model, load_dataset

from .multiqueue_worker import multiqueue_worker
from .utils import init_torch, humanize_units
from torch.nn.utils.rnn import pad_sequence

local = threading.local()


def cleanup():
    for thread in threading.enumerate():
        if hasattr(thread, 'grad'):
            del thread.grad
        if hasattr(thread, 'active_params'):
            del thread.active_params
        if hasattr(thread, 'params_slice'):
            del thread.params_slice
        if hasattr(thread, 'buffer'):
            del thread.buffer
        if hasattr(thread, 'model'):
            del thread.model
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()

def _init_compute_gradients(model, params_slice, buffer_size):
    if not "model" in local.__dict__:
        local.model = model.cuda()

    if not "grad" in local.__dict__:
        param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        grad_size = param_count + (-param_count % buffer_size[1])
        local.grad = torch.zeros(grad_size, dtype=torch.float, device="cuda")

    slot_start = 0
    local.active_params = []
    for param, local_param in zip(model.parameters(), local.model.parameters()):
        has_grad_item = False
        if not param.requires_grad:
            continue
        local_param.requires_grad = False 
        slot_stop = slot_start + param.numel()
        if not (slot_stop <= params_slice.start or slot_start >= params_slice.stop): # select slot_stop > params_slice.start and slot_start < params_slice.stop
            local.active_params.append((slice(slot_start, slot_stop), local_param)) # NOTE: it's possible that the same slices may be recomputed twice
            local_param.requires_grad = True
            has_grad_item = True
        if not has_grad_item:
            logging.info(f"Warning: no grad item selected for params_slice {params_slice.start}, {params_slice.stop}")
        slot_start = slot_stop

    local.params_slice = params_slice


def _compute_gradients(model, params_slice, buffer_size, data, batch_info):
    if not "params_slice" in local.__dict__ or local.params_slice != params_slice:
        _init_compute_gradients(model, params_slice, buffer_size)

    local.grad.zero_()
    gpu_buffer = torch.zeros(buffer_size, device="cuda")

    data = data.cuda()
    for i in range(0, data.size()[0]):
        local.model.zero_grad(set_to_none=True)
        # TODO: check if there is something wrong here with the implementation (why is there a 0?)
        # TODO: improve this by using the stateless version of the model
        local.model.forward(data[i : i + 1])[0].backward()
        for slot, param in local.active_params:
            local.grad[slot] = param.grad.flatten()
        gpu_buffer[i] = local.grad[local.params_slice]

    del data
    torch.cuda.empty_cache()

    return gpu_buffer, batch_info


def _compute_gradients_nlp(model, params_slice, buffer_size, data, batch_info, class_i=0, signgd=False):
    if not "params_slice" in local.__dict__ or local.params_slice != params_slice:
        _init_compute_gradients(model, params_slice, buffer_size)

    local.grad.zero_()
    gpu_buffer = torch.zeros(buffer_size, device="cuda")
    del data['label']

    data = to_cuda(data)
    #TODO: update this to perform per_sample_gradients
    for i in range(0, data['input_ids'].size()[0]):
        # input_ids = data['input_ids']
        # attention_masks = data['attention_mask']
        local.model.zero_grad(set_to_none=True)
        local.model.forward(slice_data(data, i, i+1))[0][class_i].backward()
        for slot, param in local.active_params:
            try:
                if signgd:
                    local.grad[slot] = torch.sign(param.grad.flatten())
                else:
                    local.grad[slot] = param.grad.flatten()
            except Exception as e:
                print(e)
                print(i)
                print(slice_data(data, i, i+1))
                print(param.size())
                print(slot)
                exit()
        gpu_buffer[i] = local.grad[local.params_slice]

    del data
    torch.cuda.empty_cache()

    return gpu_buffer, batch_info


def slice_data(data, begin, end):
    if isinstance(data, dict):
        new_data = {}
        for key in data.keys():
            new_data[key] = data[key][begin:end]
    else:
        new_data = data[begin:end]
    return new_data
    

def _clone_data(data):
    if isinstance(data, dict):
        data_clone = {}
        for key in data.keys():
            data_clone[key] = data[key].clone()
    else:
        data_clone = data.clone()
    return data_clone


def to_cuda(data):
    if isinstance(data, dict):
        for key in data.keys():
            data[key] = data[key].cuda()
    else:
        data = data.cuda()
    return data


def wrap_loader(loader):
    batch_start = 0
    for batch in loader:
        if isinstance(batch, list):
            batch_len = batch[0].size()[0]
        elif isinstance(batch, dict):
            batch_len = batch['input_ids'].size()[0]
        else:
            raise Exception("Unknown data type for the batch")
        batch_stop = batch_start + batch_len
        yield (batch, (slice(batch_start, batch_stop), batch_len))
        batch_start = batch_stop


def compute_gradients(in_queue, out_queue, model, params_slice, loader, out, class_i=0, signgd=False):
    buffer_size = (loader.batch_size, out.size()[1])

    in_flight = 0
    pbar = tqdm(total=len(loader.dataset))

    for batch, batch_info in wrap_loader(loader):
        if isinstance(batch, dict): # For BERT
            data = batch
            args = (model, params_slice, buffer_size, _clone_data(data), batch_info, class_i, signgd)
            in_queue.put((_compute_gradients_nlp, args))
        else:
            data, _ = batch
            args = (model, params_slice, buffer_size, _clone_data(data), batch_info)
            in_queue.put((_compute_gradients, args))
        in_flight += 1

        if in_flight >= 36:
            gpu_buffer, batch_info = out_queue.get()
            gpu_buffer_clone = gpu_buffer.clone()
            batch_slice, batch_len = batch_info
            del gpu_buffer

            out[batch_slice].copy_(gpu_buffer_clone[:batch_len])
            del gpu_buffer_clone
            in_flight -= 1
            pbar.update(batch_len)

    while in_flight > 0:
        gpu_buffer, batch_info = out_queue.get()
        gpu_buffer_clone = gpu_buffer.clone()
        batch_slice, batch_len = batch_info
        del gpu_buffer

        out[batch_slice].copy_(gpu_buffer_clone[:batch_len])
        del gpu_buffer_clone
        in_flight -= 1
        pbar.update(batch_len)

    pbar.close()


def _compute_XXt(chunk, buffer_size, buffer_dtype, train_slice, test_slice):
    if not "buffer" in local.__dict__:
        local.buffer = torch.zeros(buffer_size, dtype=buffer_dtype, device="cuda")

    chunk = chunk.to(local.buffer)
    local.buffer.addmm_(chunk[test_slice], chunk[train_slice].T)


def _return_XXt_buffer():
    assert "buffer" in local.__dict__
    return local.buffer


def _clear_XXt_buffer():
    assert "buffer" in local.__dict__
    del local.buffer
    torch.cuda.empty_cache()


def compute_XXt(
    in_queue_XXt, in_queues_devices, out_queue, X, out, row_chunksize, col_chunksize
):
    in_flight = 0
    train_slice = slice(0, out.size()[1])
    for i in range(0, X.size()[0], row_chunksize):
        test_slice = slice(i, i + row_chunksize)
        for j in tqdm(range(0, X.size()[1], col_chunksize)):
            chunk = X[:, j : j + col_chunksize].clone()
            args = (
                chunk,
                out[test_slice].shape,
                out.dtype,
                train_slice,
                test_slice,
            )
            in_queue_XXt.put((_compute_XXt, args))
            in_flight += 1

            if in_flight >= 3 * len(in_queues_devices):
                _ = out_queue.get()
                in_flight -= 1

        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_return_XXt_buffer, ()))
            in_flight += 1

        while in_flight > 0:
            gpu_buffer = out_queue.get()
            out[test_slice].add_(gpu_buffer.cpu())
            gpu_buffer.zero_()
            del gpu_buffer
            in_flight -= 1

        for in_queue in in_queues_devices:
            in_queue.put((_clear_XXt_buffer, ()))
            in_flight += 1

        while in_flight > 0:
            _ = out_queue.get()
            in_flight -= 1


def compute_ntk(
        model,
        train_set,
        test_set,
        num_class=1,
        num_devices=None,
        workers_per_device=1,
        grad_chunksize=None,
        mm_col_chunksize=None,
        mm_row_chunksize=None,
        loader_kwargs={},
        pin_memory=True,
        ntk_dtype=torch.double,
        signgd=False,
        init_torch_kwargs={},
):
    if num_devices is None:
        num_devices = torch.cuda.device_count()
    if grad_chunksize is None:
        assert False  # TODO: Tune automatically?
    if mm_col_chunksize is None:
        assert False  # TODO: Tune automatically?
    if mm_row_chunksize is None:
        mm_row_chunksize = 1000000000  # Don't chunk rows by default
    if signgd is not None:
        signgd = signgd
    if not "persistent_workers" in loader_kwargs:
        loader_kwargs["persistent_workers"] = True

    logging.info(f"Executing on {num_devices} device(s)")

    num_workers = num_devices * workers_per_device
    in_queue_grad = Queue()
    in_queue_XXt = Queue()
    in_queues_devices = [Queue() for _ in range(num_devices)]
    out_queue = Queue()
    processes = []
    stop_event = Event()
    for i in range(num_workers):
        device = i % num_devices
        i_in_queues = [in_queue_grad]
        if i < num_devices:
            i_in_queues.append(in_queue_XXt)
            i_in_queues.append(in_queues_devices[i])
        args = (device, init_torch_kwargs, i_in_queues, out_queue, stop_event)
        p = Process(target=multiqueue_worker, args=args)
        p.start()
        processes.append(p)


    # try:
    model.zero_grad(set_to_none=True)
    model.eval()

    if test_set is not None: # for sst2, the test_set is constructed from val partition
        train_test_sets = torch.utils.data.ConcatDataset([train_set, test_set])
    else:
        train_test_sets = train_set
    loader = torch.utils.data.DataLoader(train_test_sets, **loader_kwargs)

    if pin_memory:
        grads_bytes = 4 * len(loader.dataset) * grad_chunksize
        grad_buffer_bytes = 4 * loader.batch_size * grad_chunksize
        mm_buffer_bytes = 4 * len(loader.dataset) * mm_col_chunksize
        logging.info(f"Pinning gradient Tensor of size {humanize_units(grads_bytes)}")
        logging.info(f"Using gradient buffers of size {humanize_units(grad_buffer_bytes)}")
        logging.info(f"Using matmul buffers of size {humanize_units(mm_buffer_bytes)}")
    else:
        logging.info("Not pinning memory")


    pin_begin = time.time()
    grads_size = (len(loader.dataset), grad_chunksize)
    grads = torch.zeros(grads_size, dtype=torch.float, pin_memory=pin_memory)
    pin_end = time.time()
    logging.info(f"Allocated grads in {int(pin_end - pin_begin)}s")

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_batches = (param_count - 1) // grad_chunksize + 1
    logging.info(f"Total parameter count: {int(param_count)}")

    ntk_list = []
    for class_i in range(num_class):
        ntk = torch.zeros((len(train_test_sets), len(train_set)), dtype=ntk_dtype)


        for i, params_start in enumerate(range(0, param_count, grad_chunksize)):
            logging.info(f"Starting batch {i + 1}/{param_batches}")

            params_stop = params_start + grad_chunksize
            params_slice = slice(params_start, params_stop)
            # logging.info(f"param slices: {params_start}, {params_stop}")

            grads_begin = time.time()
            compute_gradients(in_queue_grad, out_queue, model, params_slice, loader, grads, class_i, signgd)
            grads_end = time.time()
            # logging.info(f"Computed partial Jacobian in {int(grads_end - grads_begin)}s")
            torch.cuda.empty_cache()

            ntk_begin = time.time()
            compute_XXt(
                in_queue_XXt,
                in_queues_devices,
                out_queue,
                grads,
                ntk,
                mm_row_chunksize,
                mm_col_chunksize,
            )
            ntk_end = time.time()
            # logging.info(f"Computed partial NTK in {int(ntk_end - ntk_begin)}s")
            torch.cuda.empty_cache()
        # except KeyboardInterrupt:
            # logging.info("Caught KeyboardInterrupt, terminating workers...")
            # for p in mp.active_children():
            #     p.terminate()
            # stop_event.set()
            # for p in processes:
            #     p.join()
            # logging.info("Cleaning up...")
            # cleanup()
            # logging.info("Clean up completed.")
            # raise KeyboardInterrupt

        ntk_list.append(ntk)

    for i in range(num_workers):
        logging.info(f"Terminating worker {i}")
        in_queue_grad.put(None)

    stop_event.set()
    for p in processes:
        p.join()

    ntk_list = torch.stack(ntk_list, dim=0)
    return ntk_list


def save_ntk(ntk, savedir, handle):
    savedir = pathlib.Path(savedir)
    savedir.mkdir(parents=True, exist_ok=True)

    timestamp = int(time.time())
    path = savedir / f"{handle}_ntk-v2_{timestamp}.pt"
    torch.save(ntk, path)
    logging.info(f"Saved NTK to {path}")


def load_ntk(savedir, handle, map_location=None):
    savedir = pathlib.Path(savedir).resolve()
    files = list(savedir.glob(f"{handle}_ntk-v2_*.pt"))

    assert len(files) > 0, f"No matching files for {handle}_ntk-v2_*.pt in {savedir}!"
    if len(files) > 1:
        logging.warning(f"Multiple matching NTKs found!")

    files = sorted(files)
    logging.info(f"Loading NTK from {files[-1]}")
    ntk = torch.load(files[-1], map_location=map_location)
    return ntk


def plm_collate(batch):
    ret = {}
    ret['input_ids'] = pad_sequence([item['input_ids'] for item in batch], batch_first=True)
    ret['attention_mask'] = pad_sequence([item['attention_mask'] for item in batch], batch_first=True)
    ret['label'] = torch.stack([item['label'] for item in batch])
    if 'token_type_ids' in batch[0].keys():
        ret['token_type_ids'] = pad_sequence([item['token_type_ids'] for item in batch], batch_first=True)
    if 'mask_pos' in batch[0].keys():
        ret['mask_pos'] = torch.stack([item['mask_pos'] for item in batch])
    return ret


def process_args(args):
    # Initialize torch
    init_torch_kwargs = {
        "allow_tf32": args.allow_tf32,
        "benchmark": args.benchmark,
        "deterministic": args.deterministic,
    }
    init_torch(**init_torch_kwargs, verbose=False)

    loader_kwargs = {
        "batch_size": args.loader_batch_size,
        "num_workers": args.loader_num_workers,
        "persistent_workers": False if args.loader_num_workers == 0 else None,
        "shuffle": False,
    }
    loader_kwargs = {k: v for k, v in loader_kwargs.items() if v is not None}
    # if args.dataset.lower().startswith('sst2'):
    #     loader_kwargs['collate_fn'] = plm_collate
    kwargs = {
        "workers_per_device": args.workers_per_device,
        "grad_chunksize": args.grad_chunksize,
        "mm_col_chunksize": args.mm_col_chunksize,
        "loader_kwargs": loader_kwargs,
        "pin_memory": args.pin_memory,
        "init_torch_kwargs": init_torch_kwargs,
        "ntk_dtype": torch.float32 if args.ntk_dtype == "float32" else torch.float64,
        "signgd": args.signgd,
    }

    return kwargs


def main(args):

    # Set up
    set_start_method("spawn")
    set_sharing_strategy("file_system")

    init_logging("ntk", args.logdir)
    logging.info(f"args =\n{pprint.pformat(vars(args))}")

    kwargs = process_args(args)
    init_torch_kwargs = kwargs["init_torch_kwargs"]

    # init_torch(**init_torch_kwargs, verbose=True)

    # Initialize model
    model = load_model(args.model, args)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_batches = (param_count - 1) // args.grad_chunksize + 1
    logging.info(f"Splitting {param_count} parameters into {param_batches} batches")

    # Initialize datasets
    datadir = pathlib.Path(args.datadir)
    train_set = load_dataset(datadir, args, "train")
    if args.dataset.lower().startswith('sst2'):
        test_set = load_dataset(datadir, args, "validation")
        # test_set = None
    else:
        test_set = load_dataset(datadir, args, "test")

    try: 
        ntk = compute_ntk(model, train_set, test_set, **kwargs)
    except Exception as e:
        print(e)
        exit()
    # ntk = compute_ntk(model, train_set, test_set, **kwargs)

    # Save NTK
    save_ntk(ntk, args.savedir, f"{args.dataset}_{args.model}_{args.num_frozen_layers}")

    logging.info(f"{ntk.size() = }")
    logging.info(f"ntk =\n{ntk}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str)
    parser.add_argument("model", type=str)
    parser.add_argument("--datadir", type=str, default="./datasets")
    parser.add_argument("--savedir", type=str, default="./ntks")
    parser.add_argument("--logdir", type=str)
    parser.add_argument("--workers-per-device", type=int, default=1)
    parser.add_argument("--grad-chunksize", type=int)
    parser.add_argument("--mm-col-chunksize", type=int)
    parser.add_argument("--ntk-dtype", type=str, default="float32")
    parser.add_argument("--loader-batch-size", type=int)
    parser.add_argument("--loader-num-workers", type=int)
    parser.add_argument("--no-pinned-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--subset", action="store_true")
    parser.add_argument("--num_dp", type=int, default=999)
    parser.add_argument("--num_frozen_layers", type=int, default=0)
    parser.add_argument(
        "--non-deterministic", dest="deterministic", action="store_false"
    )
    args = parser.parse_args()

    main(args)
