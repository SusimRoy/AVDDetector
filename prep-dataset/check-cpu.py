import multiprocessing
import psutil
import time

def get_free_cores(threshold=90.0, check_interval=1.0):
    """
    Get the number of CPU cores that are relatively free (below threshold usage)
    Args:
        threshold: CPU usage percentage threshold (default 50%)
        check_interval: How long to monitor CPU usage (default 1 second)
    Returns:
        Number of cores that are relatively free
    """
    # Get initial CPU usage
    cpu_percent = psutil.cpu_percent(interval=check_interval, percpu=True)
    
    # Count cores that are below threshold
    free_cores = sum(1 for core_usage in cpu_percent if core_usage < threshold)
    
    return free_cores

def main():
    # ... (previous code remains the same until ThreadPoolExecutor)
    
    # Get the number of free CPU cores
    print(multiprocessing.cpu_count())
    free_cores = get_free_cores()
    # Use 75% of free cores
    cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
    for i, percentage in enumerate(cpu_percent):
        print(f"CPU {i}: {percentage}%")
    # max_workers = max(1, int(free_cores * 0.75))
    # print(f"Found {free_cores} free cores, using {max_workers} workers")
    
    # Process each identity with progress trackin

if __name__ == "__main__":
    main()