import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
from simulation.store_simulator import simulate
if __name__ == "__main__":
    simulate(20, scenario="random")
