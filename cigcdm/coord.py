import atexit
import subprocess
import sys


script = sys.argv[1]
n_procs = int(sys.argv[2])
ps = []
for i in range(n_procs):
    ps.append(subprocess.Popen(['python', script, str(i), str(n_procs)]))

def cleanup():
    for p in ps:
        p.kill()

atexit.register(cleanup)
for p in ps:
    p.wait()
atexit.unregister(cleanup)
