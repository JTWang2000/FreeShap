import os
import torch

from queue import Empty
from multiprocessing import connection

from .utils import init_torch
import logging


def multiqueue_worker(device, init_torch_kwargs, in_queues, out_queue, stop_event):
    logging.info(f"Initializing process {os.getpid()} on device {device}")
    torch.cuda.set_device(device)
    init_torch(**init_torch_kwargs)

    readers = [q._reader for q in in_queues]
    queue_of = {q._reader: q for q in in_queues}
    while not stop_event.is_set():
        r = connection.wait(readers)[0]
        q = queue_of[r]
        try:
            if (res := q.get_nowait()) is not None:
                f, args = res
                out_queue.put(f(*args))
            else:
                break
        except Empty as e:
            logging.info(e)
            # break # NOTE: changed to break
            pass
