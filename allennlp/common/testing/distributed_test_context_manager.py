from typing import List
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from allennlp.common.checks import check_for_gpu


class DistributedTestContextManager:
    """
    This context manager is for simulating a distributed environment.

    # Parameters

    device_ids: `List[int]`
        List of cuda devices. There need to be at least 2 devices. Default is [-1, -1].

    # Usage

    ```
    python
    with DistributedTestContextManager(devices_id) as test_this_function:
        test_this_function(your_distributed_func, name_of_variable_to_check, value_to_check_against)
    ```

    """

    def __init__(self, device_ids: List[int] = [-1, -1]):
        self.device_ids = device_ids
        self.nprocs = self.world_size = len(self.device_ids)

    def init_process(
        self,
        process_rank: int,
        distributed_device_ids: List[int] = None,
        world_size: int = 1,
        master_addr: str = "127.0.0.1",
        master_port: int = 29500,
    ):
        assert world_size > 1

        global_rank = process_rank

        gpu_id = distributed_device_ids[process_rank]  # type: ignore

        if gpu_id >= 0:
            torch.cuda.set_device(int(gpu_id))
            dist.init_process_group(
                backend="nccl",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=global_rank,
            )
        else:
            dist.init_process_group(
                backend="gloo",
                init_method=f"tcp://{master_addr}:{master_port}",
                world_size=world_size,
                rank=global_rank,
            )

        self.func(global_rank, world_size, gpu_id, *self.args, **self.kwargs)

        dist.barrier()

    def test_this(self, func, *args, **kwargs):
        """
        `func` needs to be global for spawning the processes, so that it can be pickled.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs
        mp.start_processes(
            self.init_process,
            args=(self.device_ids, self.world_size),
            nprocs=self.nprocs,
            start_method="fork",
        )

    def __enter__(self):
        check_for_gpu(self.device_ids)
        return self.test_this

    def __exit__(self, type, value, traceback):
        pass
