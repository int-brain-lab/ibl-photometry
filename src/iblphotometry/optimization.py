import inspect
from typing import Literal, get_type_hints, get_origin, get_args
from copy import deepcopy
from iblphotometry.pipelines import run_pipeline


def introspect_processing_function(
    func: callable,
) -> list[dict]:
    # for introspection of a processing function
    sig = inspect.signature(func)
    hints = get_type_hints(func)

    args = []
    param_names = list(sig.parameters.keys())
    for param_name in param_names:
        # position = list(sig.parameters.keys()).index(param_name)
        param_type = sig.parameters[param_name].annotation
        if get_origin(param_type) is Literal:
            options = get_args(hints[param_name])
            param_type = Literal
        else:
            options = []
        args.append(dict(name=param_name, type=param_type, options=options))
    return args


def analyze_pipeline(pipeline: dict) -> list[list]:
    # get optimizable arguments for a pipeline
    optimizable_args = []
    for step in pipeline:
        args = introspect_processing_function(step['function'])
        # probably here: filter out categorical kwargs like substract/divide etc
        optimizable_args.append([arg for arg in args if arg['type'] in [int, float]])
    return optimizable_args


def get_param_map_for_pipeline(pipeline: dict):
    # get the pos in p -> kwarg mapping
    apipe = analyze_pipeline(pipeline)
    param_map = []
    for i in range(len(apipe)):
        for j in range(len(apipe[i])):
            param_map.append((i, apipe[i][j]['name'], apipe[i][j]['type']))
    return param_map


def apply_p(p, pipeline, param_map):
    # apply parameters p to a pipeline
    pipeline = deepcopy(pipeline)
    # map params
    for i in range(len(p)):
        j, name, _ = param_map[i]  # j is the step index
        pipeline[j]['parameters'][name] = p[i]
    return pipeline


def loss(p, pipeline, param_map, signal, metric):
    # map params into pipeline dicts
    pipeline = apply_p(p, pipeline, param_map)
    # run pipeline
    signal_proc = run_pipeline(pipeline, signal)
    # eval
    res = metric(signal_proc)
    return -res


# %%
# from one.api import ONE
# from brainbox.io.one import PhotometrySessionLoader

# one = ONE()

# eid = '58861dac-4b4c-4f82-83fb-33d98d67df3a'
# eid = '34f55b3a-725e-4cc7-aed3-6e6338f573bf'
# psl = PhotometrySessionLoader(eid=eid, one=one)
# psl.load_photometry()
# brain_region = psl.photometry['GCaMP'].columns[0]
# signal = psl.photometry['GCaMP'][brain_region]

# signal_opt = run_pipeline(sliding_mad_pipeline, signal=signal)
# from iblphotometry.plotters import plot_photometry_trace

# plot_photometry_trace(signal_opt)

# # %% optimization playgroud
# # find a way to get the bounds
# bounds = ((1, 5), (1e-3, 1), (100, 150), (50, 90))

# param_map = get_param_map_for_pipeline(sliding_mad_pipeline)
# integrality = [True if p[2] is int else False for p in param_map]

# pipe = deepcopy(sliding_mad_pipeline)
# metric = metrics.ar_score
# res = differential_evolution(
#     loss,
#     bounds=bounds,
#     args=(pipe, param_map, signal, metric),
#     integrality=integrality,
# )

# # %%
# from iblphotometry.plotters import plot_photometry_trace

# pipe_opt = apply_p(res.x, sliding_mad_pipeline, param_map)
# signal_opt = run_pipeline(pipe_opt, signal)
# ax = plot_photometry_trace(signal_opt, lw=1, alpha=1, color='r')

# signal_orig = run_pipeline(sliding_mad_pipeline, signal=signal)
# ax = plot_photometry_trace(signal_orig, axes=ax, color='k')
