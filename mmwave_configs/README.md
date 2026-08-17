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

## Range-bias / RX-phase calibration

**The calibration is per-BOARD, not per-band.** The mmWave SDK user guide is
explicit: `measureRangeBiasAndRxChanPhase` "should be enabled only using the
`profile_calibration.cfg` profile in the mmW demo profiles directory", and the
resulting string is saved "for that particular mmWave sensor" and then used to
"tune any running profile in real time". One measurement per physical sensor
covers every profile and chirp config it will ever run.

So the supported procedure is:

1. Flash the **mmWave SDK out-of-box (mmw) demo** — the measurement lives in the
   OOB demo, not in the people-counting binary.
2. Trihedral corner reflector at boresight, apex level with the antenna array
   centre, area otherwise clear — it must be the strongest return in the search
   window.
3. Edit `targetDistance` in
   `mmwave_sdk_<ver>/packages/ti/demo/xwr68xx/mmw/profiles/profile_calibration.cfg`
   to the tape-measured reflector range and send that file.
4. Copy the `compRangeBiasAndRxChanPhase` line the console prints into the
   matching `people_counting_radar*.cfg`. **The values belong to that physical
   board** — never share them between sensors.
5. Verify: re-send with compensation applied and measurement disabled; the
   reflector should report ~0° azimuth at boresight and its true range.

### The two `calibration_radar*.cfg` files here

They do the same measurement but with RF lines identical to the matching
operating config, so the sensor is measured on the band it actually runs on.
That is an **optional refinement that deviates from TI's documented procedure**
— worth at most ~0.5° azimuth on radar 1 and ~0.15° on radar 2 (see below).
Prefer TI's shipped `profile_calibration.cfg`; reach for these only if you want
that last fraction of a degree.

### Is a re-calibration needed at all after the band move?

Per TI, no — the same string is meant to serve any profile. The physics agrees:
only the *relative* phase between
the 12 virtual channels affects angle, and decoding the existing coefficients
that spread is ~90°. Phase scales with frequency, so the band moves cost:

| sensor | band shift | phase error if not recalibrated | azimuth error |
|---|---|---|---|
| radar 1 | −2.75 % | ~0.95° rms, 1.7° max | ~0.5° (≈5 cm at 5 m) |
| radar 2 | +0.49 % | ~0.30° rms, 0.5° max | ~0.15° (negligible) |

Both are well inside the fusion validation thresholds. The caveat is that the
stored coefficients are phase **modulo 360°**, so a channel delay spanning extra
wavelengths would make the real error larger (~10° phase per extra wrap on radar
1). Settle it empirically: place the reflector at boresight and ±20°/±40° and
compare reported vs true azimuth. Under a degree ⇒ keep the old line.

Note also that a *constant* azimuth bias is absorbed into the radar↔camera
extrinsic when it is solved, so only angle-**dependent** error actually survives
into the fused output.

## Caveats

1. **Consider re-calibrating after flashing** — see the section above for the
   procedure, the measured size of the error, and how to decide.
2. Velocity scaling shifts by ~±1.5 % because the carrier wavelength changed
   slightly per sensor; the SDK derives this from `startFreq` automatically, so
   nothing to tune.
3. If you ever need both sensors on the *same* band (e.g. for identical
   Doppler behavior), the alternative is time-division — hardware-synced frame
   triggers with staggered `frameCfg` trigger delays — but frequency division
   is simpler and has no sync requirement.
