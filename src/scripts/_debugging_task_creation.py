# %%
from pathlib import Path
from iblphotometry.tasks import get_photometry_tasks
from ibllib.io.session_params import read_params
from one.api import ONE

one = ONE()
eid = "5539a38a-eda5-4833-bb3d-a38b102ab967"
session_path = one.eid2path(eid)
experiment_description = read_params(session_path)
# tasks = get_photometry_tasks(experiment_description, session_path=session_path, location='local')

# %%
# task = tasks[next(iter(tasks.keys()))]
# task.setUp()
# %%
from ibllib.pipes.dynamic_pipeline import get_audio_tasks
tasks = get_audio_tasks(experiment_description, session_path=session_path)

# %%
for task_name in tasks:
    tasks[task_name].setUp()
# %%
