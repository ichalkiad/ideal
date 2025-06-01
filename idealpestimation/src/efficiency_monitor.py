import time
import os
import psutil
import threading


class Monitor:
    def __init__(self, interval=0.5):
        self.process = psutil.Process(os.getpid())
        self.interval = interval
        self._running = False

        self.cpu_samples = []
        self.mem_samples = []
        self.thread_counts = []
        self.process_counts = []

    def _sample(self):
        while self._running:
            cpu = self._get_total_cpu_percent()
            mem = self._get_total_memory_rss()
            threads = self._get_total_threads()
            procs = len(self.process.children(recursive=True)) + 1

            self.cpu_samples.append(cpu)
            self.mem_samples.append(mem)
            self.thread_counts.append(threads)
            self.process_counts.append(procs)

            time.sleep(self.interval)

    def _get_total_cpu_percent(self):
        total = self.process.cpu_percent(interval=None)
        for child in self.process.children(recursive=True):
            try:
                total += child.cpu_percent(interval=None)
            except psutil.NoSuchProcess:
                continue
        return total

    def _get_total_memory_rss(self):
        total = self.process.memory_info().rss
        for child in self.process.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.NoSuchProcess:
                continue
        return total

    def _get_total_threads(self):
        count = self.process.num_threads()
        for child in self.process.children(recursive=True):
            try:
                count += child.num_threads()
            except psutil.NoSuchProcess:
                continue
        return count

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        self._thread.join()

    def report(self, wall_duration):
        def avg(lst): return sum(lst) / len(lst) if lst else 0
        def mb(x): return x / (1024 ** 2)

        avg_total_cpu_util = avg(self.cpu_samples)
        max_total_cpu_util = max(self.cpu_samples)
        avg_total_ram_residentsetsize_MB = mb(avg(self.mem_samples)) # in MB
        max_total_ram_residentsetsize_MB = mb(max(self.mem_samples)) # in MB
        avg_threads = avg(self.thread_counts)
        max_threads = max(self.thread_counts)
        avg_processes = avg(self.process_counts)
        max_processes = max(self.process_counts)

        print("\n✅ Execution Summary:")
        print(f"Wall Time: {wall_duration:.2f} seconds")
        print(f"CPU Utilization (main + children):")
        print(f"  - Avg: {avg_total_cpu_util:.2f}%")
        print(f"  - Max: {max_total_cpu_util:.2f}%")
        print(f"RAM Usage (RSS total):")
        print(f"  - Avg: {avg_total_ram_residentsetsize_MB:.2f} MB")
        print(f"  - Max: {max_total_ram_residentsetsize_MB:.2f} MB")
        print(f"Threads:")
        print(f"  - Avg: {avg_threads:.1f}")
        print(f"  - Max: {max_threads}")
        print(f"Processes:")
        print(f"  - Avg: {avg_processes:.1f}")
        print(f"  - Max: {max_processes}")

        return wall_duration, avg_total_cpu_util, max_total_cpu_util, avg_total_ram_residentsetsize_MB, max_total_ram_residentsetsize_MB,\
                avg_threads, max_threads, avg_processes, max_processes