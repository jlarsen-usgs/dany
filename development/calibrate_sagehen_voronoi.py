import os
import gsflow
import numpy as np
from flopy.plot import styles
from hyperopt import hp, tpe, fmin
from hyperopt.fmin import generate_trials_to_calculate
import matplotlib.pyplot as plt
import pandas as pd
pd.options.mode.chained_assignment = None


trials = generate_trials_to_calculate(
    [
        {
            'jh_coef': 0.03665286093082785,  # x
            'slowcoef_lin_adj': 3.362573799283341,  # x
            'slowcoef_sq_adj': 1.8551608564990651,  # x
            'soil_moist_max_adj': 0.90084405657269,  # x
            'ssr2gw_rate_adj': 0.7795299319190928,  # x
            'tmax_allrain_offset': 3.9111063566957407,  # x 0.7503
            "sat_threshold": 578.5778450056432,  # x
        }

    ]
)


def nash_sutcliffe_efficiency(qsim, qobs, flg=False, nnse=False):
    if flg:
        qsim[qsim == 0] = 1e-06
        qobs[qobs == 0] = 1e-06
        qsim = np.log(qsim)
        qobs = np.log(qobs)
    qsim[np.isinf(qsim)] = np.nan
    qobs[np.isinf(qobs)] = np.nan
    numerator = np.nansum((qobs - qsim) ** 2)
    denominator = np.nansum((qobs - np.nanmean(qobs)) ** 2)
    nse = 1 - (numerator / denominator)
    if nnse:
        nse = 1 / (2 - nse)
    return nse


def tune_sagehen(args):
    import gsflow
    import os

    ws = os.path.abspath(os.path.dirname(__file__))
    model_ws = os.path.join(ws, "..", "data", "sagehen_voronoi")
    output_ws = os.path.join(ws, "..", "data", "sagehen_vcal")
    gsf = gsflow.GsflowModel.load_from_file(
        os.path.join(model_ws, "sagehen_voronoi_cont.control"),
    )

    gsf.prms.parameters.tmax_allsnow[:] =  0.7
    gsf.prms.parameters.radmax[:] = 0.80
    gsf.prms.parameters.soil_moist_max *= args["soil_moist_max_adj"]
    gsf.prms.parameters.slowcoef_sq *= args["slowcoef_sq_adj"]
    gsf.prms.parameters.slowcoef_lin *= args["slowcoef_lin_adj"]
    gsf.prms.parameters.ssr2gw_rate[:] *= args["ssr2gw_rate_adj"]
    gsf.prms.parameters.sat_threshold[:] = args["sat_threshold"]
    gsf.prms.parameters.gwflow_coef[:] = 0.04
    gsf.prms.parameters.gwsink_coef[:] = 0.03

    # snow adjustments to match sagehen 50m
    gsf.prms.parameters.tmax_allrain_offset[:] = args["tmax_allrain_offset"]
    gsf.prms.parameters.rad_trncf[:] = 0.8 * gsf.prms.parameters.covden_win.values
    gsf.prms.parameters.snowinfil_max[:] = 20

    # soil adjustments
    gsf.prms.parameters.jh_coef[:] = args["jh_coef"]

    gsf.write_input(basename="sagehen_vcal", workspace=str(output_ws))

    gsf = gsflow.GsflowModel.load_from_file(
        os.path.join(output_ws, "sagehen_vcal_cont.control")
    )
    success, buff = gsf.run_model(gsflow_exe="gsflow.exe")

    #
    try:
        stats = gsf.prms.get_StatVar()
        if len(stats) != 5115:
            raise AssertionError

        stats = stats[1096:]
        stats.reset_index(inplace=True, drop=True)

        nse = nash_sutcliffe_efficiency(
            stats.basin_cfs_1, stats.runoff_1,
        )

        qsim = np.array(stats.basin_cfs_1.values)
        qobs = np.array(stats.runoff_1.values)
        print(np.min(qobs), np.max(qobs))
        qsim[qsim == 0] = 1e-06
        qsim[np.isnan(qsim)] = 1e-06
        qsim[np.isinf(qsim)] = 1e-06
        qobs[qobs == 0] = 1e-06
        qobs[np.isnan(qobs)] = 1e-06
        qobs[np.isinf(qobs)] = 1e-06
        qsim = np.log10(qsim)
        qobs = np.log10(qobs)
        qsim[np.isinf(qsim)] = -6
        qobs[np.isinf(qobs)] = -6
        mae = np.nansum(np.abs(qsim - qobs))
        print(f"@@@@@@@@ LOG10(MAE): {mae} @@@@@@@@@\n"
              f"--------------------------------------")
        print(f"@@@@@@@@@@@@ NSE: {nse} @@@@@@@@@@@@@\n"
              f"--------------------------------------")
    except (ValueError, AssertionError):
        nse = -10000000000
        mae = 1e+10

    objective = (1 - nse) * 1000
    # objective = mae
    return objective


if __name__ == "__main__":
    calibrate = True
    debug = False
    space = {
        'soil_moist_max_adj': hp.uniform('soil_moist_max_adj', 0.0001, 1.3333),
        'slowcoef_sq_adj': hp.uniform('slowcoef_sq_adj', 0.0001, 2),
        'slowcoef_lin_adj': hp.uniform('slowcoef_lin_adj', 0.0001, 5),
        'ssr2gw_rate_adj': hp.uniform('ssr2gw_rate_adj', 0.00001, 1.000001),
        'sat_threshold': hp.uniform('sat_threshold', 0.00001, 999),
        "jh_coef": hp.uniform("jh_coef", 0, 1),
        "tmax_allrain_offset": hp.uniform("tmax_allrain_offset", 0, 5),

    }

    if calibrate:
        args = fmin(
            fn=tune_sagehen,
            space=space,
            algo=tpe.suggest,
            max_evals=10,
            trials=trials
        )

    else:
        args = {
            'jh_coef': 0.03665286093082785,
            'sat_threshold': 578.5778450056432,
            'slowcoef_lin_adj': 3.362573799283341,
            'slowcoef_sq_adj': 1.8551608564990651,
            'soil_moist_max_adj': 0.90084405657269,
            'ssr2gw_rate_adj': 0.7795299319190928,
            'tmax_allrain_offset': 3.9111063566957407
        }


        if debug:
            objective = tune_sagehen(args)

    model_ws = os.path.join("..", "data", "sagehen_voronoi")
    cal_ws = os.path.join("..", "data", "sagehen_vcal")
    gsf = gsflow.GsflowModel.load_from_file(
        os.path.join(model_ws, "sagehen_voronoi_cont.control")
    )

    gsf.prms.parameters.tmax_allsnow[:] = 0.7
    gsf.prms.parameters.radmax[:] = 0.80
    gsf.prms.parameters.soil_moist_max *= args["soil_moist_max_adj"]
    gsf.prms.parameters.slowcoef_sq *= args["slowcoef_sq_adj"]
    gsf.prms.parameters.slowcoef_lin *= args["slowcoef_lin_adj"]
    gsf.prms.parameters.ssr2gw_rate[:] *= args["ssr2gw_rate_adj"]
    gsf.prms.parameters.sat_threshold[:] = args["sat_threshold"]
    gsf.prms.parameters.gwflow_coef[:] = 0.04
    gsf.prms.parameters.gwsink_coef[:] = 0.03

    # snow adjustments to match sagehen 50m
    gsf.prms.parameters.tmax_allrain_offset[:] = args["tmax_allrain_offset"]
    gsf.prms.parameters.rad_trncf[:] = 0.8 * gsf.prms.parameters.covden_win.values
    gsf.prms.parameters.snowinfil_max[:] = 20

    # soil adjustments
    gsf.prms.parameters.jh_coef[:] = args["jh_coef"]

    gsf.write_input(basename="sagehen_vcal", workspace=str(cal_ws))

    gsf = gsflow.GsflowModel.load_from_file(
        os.path.join(cal_ws, "sagehen_vcal_cont.control")
    )
    success, buff = gsf.run_model(gsflow_exe="gsflow.exe")

    stats = gsf.prms.get_StatVar()

    stats = stats[1096:]
    stats.reset_index(inplace=True, drop=True)
    print(nash_sutcliffe_efficiency(
            stats.basin_cfs_1.values, stats.runoff_1.values
        )
    )
    print(args)
    gw_seepage = stats.basin_cfs_1.values.copy() - (
            stats.basin_ssflow_cfs_1.values.copy() +
            stats.basin_sroff_cfs_1.values.copy() +
            stats.basin_dunnian_1.values.copy()
    )

    with styles.USGSMap():
        fig, axis = plt.subplots(2, 1, figsize=(10, 6))
        plt.rcParams.update({'font.size': 100})
        axis[0].plot(stats.Date.values, stats.basin_cfs_1.values, color='r',
                     linewidth=2.2, label='simulated voronoi model')
        axis[0].plot(stats.Date.values, stats.runoff_1.values, '--', color='b',
                     linewidth=1.5, label='measured')
        handles, labels = axis[0].get_legend_handles_labels()
        axis[0].legend(handles, labels, bbox_to_anchor=(0.25, 0.65))
        axis[0].set_xlabel("Date")
        axis[0].set_ylabel("Streamflow, in cfs")
        axis[0].set_ylim(0, 300)

        plt.xlabel("Date")
        plt.ylabel("Streamflow, in cfs")
        plt.ylim(0, 300)

    with styles.USGSMap():

        axis[1].set_xlabel("Date")
        axis[1].set_ylabel("Flow Components, in cfs")
        axis[1].set_yscale("log")
        plt.xlabel("Date")
        plt.ylabel("Flow Components, in cfs")
        plt.yscale("log")
        plt.ylim(1.0e-3, 1.0e4)
        axis[1].plot(stats.Date.values, stats.basin_ssflow_cfs_1.values,
                     color='r', linewidth=1.5, label='Interflow')
        axis[1].plot(stats.Date.values, gw_seepage, color='purple',
                     linewidth=1.5, label='Groundwater seepage')
        axis[1].plot(stats.Date.values, stats.basin_sroff_cfs_1.values,
                     color='y', linewidth=1.5, label='Hortonian runoff')
        axis[1].plot(stats.Date.values, stats.basin_dunnian_1.values,
                     color='b', linewidth=1.5, label='Dunnian runoff')
        handles, labels = axis[1].get_legend_handles_labels()
        axis[1].legend(handles, labels, bbox_to_anchor=(0.25, 0.65))
        plt.tight_layout()
        plt.show()