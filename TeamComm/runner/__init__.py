from .runner import Runner
from .runner_magic import RunnerMagic
from .runner_baselines import RunnerBaseline
from .runner_ic3net import RunnerIcnet
from .runner_tiecomm import RunnerTiecomm
from .runner_teamcomm import RunnerTeamComm

REGISTRY = {
    "commnet": RunnerBaseline,
    "ic3net": RunnerIcnet,
    "magic": RunnerMagic,
    "tarmac": RunnerBaseline,
    "teamcomm": RunnerTeamComm,
    "tiecomm": RunnerTiecomm,
}
