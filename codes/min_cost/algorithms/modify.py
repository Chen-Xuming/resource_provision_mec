from codes.min_cost.algorithms.base import BaseAlgorithm


class ModifyAssignmentReallocation(BaseAlgorithm):
    """
        在传入函数之前，env已经做了某种关联、分配方案
    """
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_modify" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]





    def run(self):
        pass
