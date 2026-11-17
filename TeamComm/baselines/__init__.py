from .tiecomm import TieCommAgent
from .commnet import CommNetAgent
from .tarmac import TarCommAgent
from .magic import MAGICAgent
from .teamcomm import TeamCommAgent

REGISTRY = {
    "commnet": CommNetAgent,
    "ic3net": CommNetAgent,
    "magic": MAGICAgent,
    "tarmac": TarCommAgent,
    "teamcomm": TeamCommAgent,
    "tiecomm": TieCommAgent,
}
