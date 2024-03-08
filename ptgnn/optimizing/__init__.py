from .optimizer import adam_w_optimizer
from .scheduler import cosine_with_warmup_scheduler

OPTIMIZER_DICT = {
    "adam_w": adam_w_optimizer
}

SCHEDULER_DICT = {
    "cosine_with_warmup": cosine_with_warmup_scheduler
}
