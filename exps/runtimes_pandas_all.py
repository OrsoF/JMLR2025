import subprocess
import threading


def run_command(command):
    subprocess.run(command, shell=True)


models = [
    "barto.py",
    "block.py",
    "garnet.py",
    "impatience.py",
    "maze_backtrack.py",
    "maze_prims.py",
    "maze_wilson.py",
    "mountain.py",
    "rooms.py",
    "sutton.py",
    "tandem.py",
    "toy.py",
]


threads = []
for model in models:
    command = f"python test/runtimes_pandas.py --model {model}"
    thread = threading.Thread(target=run_command, args=(command,))
    thread.start()
    threads.append(thread)

# Wait for all threads to complete
for thread in threads:
    thread.join()

print("All experiments have completed.")
