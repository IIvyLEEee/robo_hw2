from .basic import BasicEnv
from .manip_pap import ManipPAPScene

TASK_ENV_LIST = {
    "basic": BasicEnv,
    "manipulation_pap": ManipPAPScene,
}
