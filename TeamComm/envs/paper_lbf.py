"""Paper protocol backed by the workspace's frozen scenario registry."""
import importlib.util
from pathlib import Path
import numpy as np

_workspace = Path(__file__).resolve().parents[1]
_source = _workspace / 'shared_lbf_env.py'
if str(_workspace) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(_workspace))
_spec = importlib.util.spec_from_file_location('_paper_shared_lbf', _source)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

class PaperForagingEnv(_module.ForagingEnv):
    # Legacy runner API names: in this protocol "food value" means item count.
    # Capacity-weighted workload must be reported separately, never mixed.
    def get_initial_food_value(self):
        return self.initial_food

    def get_collected_initial_food(self):
        return int(self._completed_tasks)
