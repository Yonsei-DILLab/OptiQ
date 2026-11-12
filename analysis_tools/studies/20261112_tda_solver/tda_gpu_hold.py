"""User-authorized TDA GPU reservation workload; stops promptly on SIGTERM."""
import argparse
import os
import signal
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',required=True);a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=True
    running=True
    def stop(signum,frame):
        nonlocal running
        running=False
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    x=torch.randn((8192,8192),device='cuda',dtype=torch.float32)
    y=torch.randn_like(x);z=torch.empty_like(x)
    print(f'TDA GPU reservation active: physical GPU {a.gpu}, PID {os.getpid()}',flush=True)
    start=time.monotonic();iterations=0
    with torch.no_grad():
        while running:
            torch.mm(x,y,out=z);torch.cuda.synchronize();iterations+=1
    print(f'TDA GPU reservation stopped: {iterations} iterations, {time.monotonic()-start:.1f}s',flush=True)


if __name__=='__main__':main()
