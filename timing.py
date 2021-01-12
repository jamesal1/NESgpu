import time
import torch as t
from contextlib import contextmanager

@contextmanager
def timer(message=""):
    t.cuda.synchronize()
    start = time.time()
    yield None
    t.cuda.synchronize()
    print(message, time.time() - start)
