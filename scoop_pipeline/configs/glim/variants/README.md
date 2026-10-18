# GLIM config variants (mobile_2), to compare against the reference

Each folder is `../mobile_2` with a few settings changed; `processing/glim_sweep.sh`
runs them on one bag and scores each against the reference trajectory.

| folder | changed | question |
|---|---|---|
| `mobile_2` (in `..`) | -- (CPU modules: submaps of 6 keyframes 0.5 m apart, downsampled to 0.25 m; global registration on 20 % of the points) | the current baseline |
| `mobile_2_gpu` | `config.json` selects the GPU modules, GLIM's own defaults for them (needs `koide3/glim_ros2:humble_cuda12.2-mcap`) | the setup GLIM is published with |
| `mobile_2_cpu_fine` | sub-mapping: all points, submaps downsampled to 0.1 m; global: all points | is the coarse global registration the limit? |
| `mobile_2_short_submaps` | sub-mapping: 3 keyframes per submap, 0.25 m / 0.15 rad apart | does a shorter submap shorten the heading error? |
