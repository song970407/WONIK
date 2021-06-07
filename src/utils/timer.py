import functools
import time


def timeit(func):
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = time.perf_counter()
        out = func(*args, **kwargs)
        end = time.perf_counter()
        print("Finished {} in {:.4f} seconds".format(func.__name__, end - start))
        return out

    return wrapper_timer
