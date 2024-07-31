# not for land

import csv
import gc
import os
import time
from typing import Callable

import pandas as pd

import torch
import torch.utils.benchmark as benchmark

from .vasiliy_debug_extract_subgraphs import summary_headers

# don't truncate long fields
# pd.set_option('display.max_colwidth', None)
# pd.set_option('display.max_columns', None)  

bytes_in_gb = 1024 * 1024 * 1024

def benchmark_torch_function_in_microseconds(
    func: Callable,
    *args,
    **kwargs,
) -> float:
    if True:
        t0 = benchmark.Timer(
            stmt="func(*args, **kwargs)",
            globals={"args": args, "kwargs": kwargs, "func": func},
        )
        return t0.blocked_autorange().median * 1e6

    if False:
        n_warmup = 3
        n_iter = 10

        for _ in range(n_warmup):
            func(*args, **kwargs)

        t0 = time.time()
        for _ in range(n_iter):
            func(*args, **kwargs)
        t1 = time.time()
        return (t1 - t0) / n_iter
    

def fwd_and_bwd(m, args):
    outs = m(*args)
    outs = torch.cat([outs], dim=0)
    outs.sum().backward()

def analyze_subgraphs(
    target_folder: str,
    extracted_bsz: int,
    target_bsz: int,
) -> None:
    """
    Assumes folder structure:

        target_folder/
          debug_logs.txt
          summary.csv
          subgraph_with_inputs_0.pt
          ...
          subgraph_with_inputs_(n-1).pt

    Does the following:
    * load each subgraph in bf16
    * increase batch size to target_batch_size
    * benchmark fw+bw for each and record the runtime, display a table comparing
      the relative runtime of each in bf16
    """
    summary_df = pd.read_csv(os.path.join(target_folder, 'summary.csv'))
    print()
    print(summary_df)

    # updating pandas rows inplace is annoying with iterrows, so just do a batch column
    # append.  There is a definitely a better way to do this.
    time_results = []

    for index, row in summary_df.iterrows():
        subgraph_fname = f'subgraph_with_inputs_{row["subgraph_idx"]}.pt'
        print(f'benchmarking {subgraph_fname}')
        subgraph_fname = os.path.join(target_folder, subgraph_fname)

        m, inputs = torch.load(subgraph_fname, weights_only=False)

        # adjust each input's bsz to target_bsz
        # enable grad
        def resize_input_and_enable_grad(t):
            if len(t.shape) > 1:
                old_first_dim, old_rest = t.size()[0], t.size()[1:]
                new_first_dim = old_first_dim // extracted_bsz * target_bsz
                new_shape = (new_first_dim, *old_rest)
                t = torch.randn(*new_shape, dtype=t.dtype, device=t.device, requires_grad=True)
                # t.resize_(new_first_dim, *old_rest).random_(-1000, 1000).div_(1000.0)
            else:
                # assume that rank 1 tensors do not depend on batch size
                t.requires_grad_(True)
                pass
            return t

        inputs = [resize_input_and_enable_grad(t) for t in inputs]

        # estimate memory used by inputs, params, grads
        input_gb = 0
        for inp in inputs:
            input_gb += (inp.numel() * inp.element_size()) / bytes_in_gb
        model_gb = sum(p.numel() * p.element_size() / bytes_in_gb for p in m.parameters())
        grad_gb = input_gb + model_gb
        total_gb = input_gb + model_gb + grad_gb
        # print(f'param mem estimate (GB): input {input_gb} model {model_gb} grad {grad_gb} total {total_gb}')

        # TODO(before land): enable this
        # m = torch.compile(m)

        time_us = benchmark_torch_function_in_microseconds(m, *inputs)
        print(f'time_us: {time_us}')
        time_results.append([time_us])

        # need to manually delete grad from inputs, otherwise it would survive
        # and eventually OOM for large problem sizes
        for inp in inputs:
            del inp.grad
        del m, inputs
        gc.collect()
        torch.cuda.empty_cache()

    time_df = pd.DataFrame(time_results)
    summary_df = summary_df.join(time_df)
    print(summary_df)

    print('done')
