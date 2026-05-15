from .randomizations import *
from .observations import *
from .terminations import *
from .action import *

def get_obj_by_class(mapping, obj_class):
    return {
        k: v for k, v in mapping.items() 
        if isinstance(v, type) and issubclass(v, obj_class)
    }

OBS_FUNCS = get_obj_by_class(vars(observations), observations.Observation)
TERM_FUNCS = get_obj_by_class(vars(terminations), terminations.Termination)
RAND_FUNCS = get_obj_by_class(vars(randomizations), randomizations.Randomization)
