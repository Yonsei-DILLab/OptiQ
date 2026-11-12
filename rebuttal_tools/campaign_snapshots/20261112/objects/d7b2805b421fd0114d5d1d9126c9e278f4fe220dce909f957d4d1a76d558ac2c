"""Use the existing native evaluation policies on every Fetch task."""


def evaluate(agent, task, episodes, seed, mode):
    if task == 'pick-and-place':
        from rebuttal_tools.fetch_grasp_evaluation import evaluate as grasp
        return grasp(agent, episodes, seed, mode)
    from fetch_rebuttal.run import evaluate as native
    return native(agent, task, episodes, seed, mode)
