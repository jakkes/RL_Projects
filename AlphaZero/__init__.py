from multiprocessing import set_start_method, get_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    if get_start_method().lower() != "spawn":
        raise RuntimeError("Start method is not spawn")
