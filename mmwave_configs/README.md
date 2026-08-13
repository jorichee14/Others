# IWR6843 3D People-Counting configs — dual-sensor band plan

Two radars running the original config swept the **same** 61.75–63.68 GHz band,
so their chirps could cross through each other's receive band and inject
interference spikes. These configs fix that with **frequency division**: each
sensor gets its own non-overlapping slice of the 60–64 GHz band, so both can
chirp at the same time without ever hearing each other.

| file | sensor | `profileCfg` startFreq | occupied band |
|---|---|---|---|
| `people_counting_radar1_60G.cfg` | radar 1 (cal set A) | 60.05 GHz | 60.05 – 61.98 GHz |
| `people_counting_radar2_62G.cfg` | radar 2 (cal set B) | 62.05 GHz | 62.05 – 63.98 GHz |

- Ramp bandwidth per sensor: 32.71 MHz/µs × 59.10 µs ≈ **1933 MHz**
- Guard band between the two: **~67 MHz** — orders of magnitude wider than the
  receiver IF bandwidth, and TX is off during idle time, so no leakage path.
- The **only** RF change vs the original is `startFreq`; slope, ADC samples and
  sample rate are untouched, so per-sensor performance is unchanged:
  range resolution ≈ 0.16 m, max range ≈ 10.1 m, same frame timing and
  radar-cube memory.

Each file keeps its sensor's own `compRangeBiasAndRxChanPhase` line — keep the
file → physical-sensor pairing consistent.

## Caveats

1. **Re-calibrate after flashing.** The `compRangeBiasAndRxChanPhase` values
   were measured on the old 61.75–63.68 GHz band. RX phase/bias compensation is
   frequency-dependent, so re-run the TI range-bias & phase calibration
   procedure per sensor at its new band, then paste the new line into its file.
2. Velocity scaling shifts by ~±1.5 % because the carrier wavelength changed
   slightly per sensor; the SDK derives this from `startFreq` automatically, so
   nothing to tune.
3. If you ever need both sensors on the *same* band (e.g. for identical
   Doppler behavior), the alternative is time-division — hardware-synced frame
   triggers with staggered `frameCfg` trigger delays — but frequency division
   is simpler and has no sync requirement.
