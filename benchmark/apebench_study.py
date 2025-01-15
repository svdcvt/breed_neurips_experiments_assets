"""
Trains nonlinear emulators for the (linear) 1D advection equation under varying
difficulty in terms of the `advction_gamma` (=CFL).
"""


CONFIGS = [
    {
        "scenario_name": "diff_adv",
        "network_config": net,
        "advection_gamma": advection_gamma,
    }
    for s in [0, 10, 20, 30, 40]
    for net in [
        *[f"Conv;34;{depth};relu" for depth in [0, 2]],  # , 1, 2, 10]],
        "UNet;12;2;relu",  # 27'193 params, 29 receptive field per direction
        "Res;26;8;relu",  # 32'943 params, 16 receptive field per direction
        "FNO;12;18;4;gelu",  # 32'527 params, inf receptive field per direction
        "Dil;2;32;2;relu",  # 31'777 params, 20 receptive field per direction
    ]
    for advection_gamma in [
        0.5,
        2.5,
        10.5,
    ]
]
