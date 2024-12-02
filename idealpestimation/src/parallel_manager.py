import time
import os
import multiprocessing
from multiprocessing import Process, Manager, Lock
import fcntl
import jsonlines 

class ProcessManager:

    def __init__(self, max_processes):
        self.max_processes = max_processes
        self.running_processes = []
        self.manager = Manager()        
        self.execution_counter = multiprocessing.Value('i', 0)        
        self.file_lock = Lock()

    def append_to_json_file(self, message, output_file):
        
        with self.file_lock:                        
            with open(output_file, 'a') as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    writer = jsonlines.Writer(f)
                    writer.write(message)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def worker_process(self, args):
        current_pid = os.getpid()
        with self.execution_counter.get_lock():
            self.execution_counter.value += 1
            self.shared_dict[current_pid] = self.execution_counter.value
        
        # Simulate some work
        time.sleep(5)

        completion_message = {
            "pid": os.getpid(),
            "execution_order": self.shared_dict[current_pid],
            "status": "completed",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self.append_to_json_file(completion_message)
    
    def create_results_dict(self, optim_target="all"):

        self.shared_dict = self.manager.dict()
        self.optim_target = optim_target
    
    def spawn_process(self, args):
        process = Process(target=self.worker_process, args=args)
        process.start()
        self.running_processes.append(process)
        print(f"Spawned process with PID: {process.pid}")
    
    def cleanup_finished_processes(self):
        still_running = []
        for process in self.running_processes:
            if not process.is_alive():
                print(f"Process {process.pid} has finished")
                process.join()
            else:
                still_running.append(process)
        self.running_processes = still_running
    
    def current_process_count(self):
        return len(self.running_processes)
    
    def cleanup_all_processes(self):
        for process in self.running_processes:
            process.terminate()
            process.join()
        self.running_processes = []
    
    def print_shared_dict(self):
        print("Shared Dictionary Contents:")
        for pid, order in self.shared_dict.items():
            print(f"Process ID: {pid}, Execution Order: {order}")