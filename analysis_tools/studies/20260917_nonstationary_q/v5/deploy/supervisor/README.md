# Managed experiment workers

## Current checked-K64 v2

On the `/root/OptiQ` instance, the canonical configuration is
[`optiq-v2-checked64.conf`](optiq-v2-checked64.conf). It starts one Humanoid-v4
seed per GPU (0..3), using the unchanged successful numerical configuration
through [`scripts/run_v2.sh`](../../scripts/run_v2.sh).

Its wrapper source is
[`scripts/supervisor_v2_checked64.sh`](../../scripts/supervisor_v2_checked64.sh),
installed at `/opt/supervisor-scripts/optiq-v2-checked64.sh`. The configuration
is installed at `/etc/supervisor/conf.d/optiq-v2-checked64.conf`.
These two files contain this instance's absolute paths. On another machine,
adapt paths and credentials explicitly, or use the portable
[`scripts/supervisor_v2.sh`](../../scripts/supervisor_v2.sh) wrapper with
`OPTIQ_ROOT`, `OPTIQ_PYTHON`, `OPTIQ_ENV_FILE` and `CUDA_VISIBLE_DEVICES` set.

Both `autostart` and `autorestart` are false. Installing or updating the workers
does not request a new experiment. Do not start them until training is requested.
The four seeds each have a 1M-step budget without performance-based early stopping.
Logs use the existing `OptiQ/optiq_mujoco_v2_confirmation` project and fresh run
IDs. Outputs live outside the repository. See the
[instance guide](../../docs/v2/INSTANCE.md) for paths and explicit start commands.

## Historical configurations

All other configuration files here describe prior experiment deployments,
including original/calibration, screen, conditional, guarded, finite, proximal,
confirmation and scalar MuJoCo experiments. Preserve them for historical
manifests and services, but do not install the directory with a wildcard.
Several use paths from another server (`/workspace/OptiQ-v2`) or different
algorithm variants and stopping rules. They are not the current default.

Only install/update the intended named service. Do not restart unrelated
services or reapply archived local code when deploying checked-K64 v2.
