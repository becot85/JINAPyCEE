# Import needed stuff
from JINAPyCEE import omega_plus
import multiprocessing as mp

class Wrapper_omega_plus():
    '''Class to return the omega_plus instance'''
    
    def __init__(self, args):
        self.args = args
    
    def launch_omega_plus(self, q):
        q.put(omega_plus.omega_plus(**self.args))

def launch_parallel(arguments, n_proc = 1):

    # Get all runs
    n_runs = len(arguments)

    # Define return list
    omega_objs = []

    # If only one object or process, do serial instead
    if n_runs == 1 or n_proc == 1:
        for argument in arguments:
            omega_objs.append(omega_plus.omega_plus(**argument))

        # All done, just return the omega_plus instances
        return omega_objs

    # Change in_parallel value for each argument and create objects
    omega_objs = []
    for arg in arguments:
        omega_objs.append(Wrapper_omega_plus(arg))

    # Generate the queue
    queue = mp.Queue()

    # Initiate all processes
    i_run = 0; objs = []
    while i_run < n_runs:

        # Define use_proc
        use_proc = min(n_proc, n_runs - i_run)

        # Launch each process
        processes = []
        for ii in range(use_proc):
            process = mp.Process(target = lambda q, obj: \
                    obj.launch_omega_plus(q), \
                    args = (queue, omega_objs[i_run + ii]))
            process.start()

        # Join the processes
        for ii in range(use_proc):
            objs.append(queue.get())

        # Increase i_run:
        i_run += use_proc
    
    #return omega_objs
    return objs
