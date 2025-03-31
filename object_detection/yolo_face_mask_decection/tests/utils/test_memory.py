# memory_test.py
import psutil
import os
import pandas as pd
import gc
import time

process = psutil.Process(os.getpid())

def print_memory(label):
    mem = process.memory_info().rss / (1024 ** 2)  # MB
    print(f"[{label}] Memory usage: {mem:.2f} MB")

print_memory("Before")

df = pd.DataFrame({
    'a': range(10**7),
    'b': ['x'] * 10**7
})

print_memory("After DataFrame creation")

del df
gc.collect()
time.sleep(20)  # Give it time

print_memory("After deleting and collecting")
