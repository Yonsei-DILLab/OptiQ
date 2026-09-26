# Third-party environment source

The required PointMaze environment files are included unchanged from
PneuC/DrAC, commit `4e718983ea29aa3a856955f553a99795fcb4e94d`:
https://github.com/PneuC/DrAC

The source headers identify Apache License 2.0 and acknowledge the D4RL and
Gymnasium-Robotics ancestry. Preserve those headers and the included LICENSE.
Only PointMaze, its maps, MuJoCo utilities and XML asset are included; no DrAC
learner or baseline is imported. The iBOLT adapter lives outside this folder.
