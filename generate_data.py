import numpy as np
import pandas as pd

def generate_oom_dataset(n_samples=2000, seed=42):
    rng = np.random.default_rng(seed)

    rss = rng.uniform(200, 2000, n_samples)        
    heap = rss * rng.uniform(0.5, 0.9, n_samples)  
    gc = rng.uniform(1, 150, n_samples)            
    threads = rng.integers(10, 500, n_samples)     
    fds = rng.integers(50, 5000, n_samples)        


    score = (0.004 * rss + 0.01 * gc + 0.002 * threads
             + 0.0005 * fds + rng.normal(0, 0.8, n_samples))
    y = (score > np.quantile(score, 0.85)).astype(int)  

    return pd.DataFrame({
        "RSS_RAM_MB": rss, "Heap_Alloc": heap, "GC_Pause_ms": gc,
        "ActiveThreads": threads, "FileDescriptors": fds, "OOM_Risk": y,
    })

df = generate_oom_dataset()
df.to_csv("data/telemetry_dataset.csv", index=False)