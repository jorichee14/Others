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

`calibration_radar1_60G.cfg` and `calibration_radar2_62G.cfg` measure the
`compRangeBiasAndRxChanPhase` line for their sensor. Their RF lines
(`channelCfg` / `profileCfg` / `chirpCfg` / `frameCfg`) are identical to the
matching operating config, so the measurement is taken **at the band the sensor
actually runs on** and with the **same TX firing order** — a different TX order
would apply the 12 measured coefficients to the wrong virtual antennas.

1. Flash the **mmWave SDK out-of-box (mmw) demo** — the measurement is an OOB
   demo feature, not part of the people-counting binary.
2. Trihedral corner reflector at boresight, apex level with the antenna array
   centre, area otherwise clear — it must be the strongest return in the search
   window.
3. Edit `targetDistance` on the `measureRangeBiasAndRxChanPhase` line to the
   tape-measured reflector range, then send the cfg.
4. Copy the `compRangeBiasAndRxChanPhase` line the demo prints into the matching
   `people_counting_radar*.cfg`, replacing the existing one. **The values are
   unique to that physical board** — never share them between sensors.
5. Verify: re-send with `measureRangeBiasAndRxChanPhase 0 …` and the new
   compensation applied; the reflector should report ~0° azimuth at boresight
   and its true range.

### Is it actually needed?

Optional, and cheaper to check than to redo. Only the *relative* phase between
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
