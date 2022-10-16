from .tasks.single_kitchen import SingleKitchenDomain, get_single_kitchen_exp_tasks
from . tasks.sort_two_objects import SortTwoObjectsDomain, get_sort_two_objects_exp_tasks
from .tasks.stack_two_types import StackTwoTypesDomain, get_stack_two_types_exp_tasks


TASK_MAPPING = {
    "single-kitchen": get_single_kitchen_exp_tasks,
    "sort-two-objects": get_sort_two_objects_exp_tasks,
    "stack-two-types": get_stack_two_types_exp_tasks
}


