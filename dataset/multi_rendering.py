import time
from multiprocessing import Process
import os


    
def call2(process_idx, num_processes):
    os.system(f"blenderproc run renderNormal_training.py {process_idx} {num_processes}")


start_processes = 0
end_processes = 8
total_processes = 8
workers = [Process(target=call2, args = (i,total_processes)) for i in range(start_processes,end_processes)]

a = time.time()
for i,p in enumerate(workers):
    print("launching worker",i)
    p.start()

while True:
    time.sleep(60)
    allExited = True
    for p in workers:
        if p.exitcode is None:
            allExited = False
            break
    if allExited:
        break

print("exit,total time",time.time()-a)
