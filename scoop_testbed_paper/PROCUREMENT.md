# SCooP-iT hardware procurement

Budget NT$52,000; no single line item over NT$10,000. Total below: **NT$49,650**,
leaving NT$2,350 for shipping and 5% VAT (the TI and Arducam orders are the
exposure).

The **must-have spec** column is what makes the part work in this rig. Where it
says *do not substitute*, a different part breaks either comparability with an
existing agent or a software assumption already in the repo.

---

## 1. Storage — NT$14,800

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| Samsung T7 Shield 2 TB | 2 | 9,000 | USB 3.2 Gen 2; **≥500 MB/s sustained** write (not burst); TLC; IP65 / shock-rated; USB-C | `mobile_2` writes 141 MB/s continuously on a moving pushcart. Burst numbers on the box are irrelevant — what matters is the rate after the SLC cache fills. Take 2 TB over 1 TB: larger cache, better sustained figure |
| 8 TB external desktop HDD (WD Elements / Seagate Expansion) | 1 | 5,500 | 3.5", mains-powered, USB 3.0+, **CMR** | Lab archive, never leaves the desk. Pure NT$/TB: 690 vs 2,250 for SSD. Avoid SMR — it collapses on sustained sequential writes |
| Right-angle USB-C cable, 30 cm | 2 | 300 | Right-angle, USB 3.2 Gen 2 rated (10 Gbps), ≤50 cm | A straight plug on a cart is what unseats mid-pass. Short runs protect signal integrity at Gen 2 |

**Acceptable substitutes for the T7 Shield:** Crucial X9 Pro, Samsung T9.
**Avoid:** any USB *flash drive* (20–40 MB/s sustained), DRAM-less QLC portables.

**Capacity check:** 141 MB/s = 8.5 GB/min. A 5-minute pass is 42 GB; 2 TB holds
~47 passes. One drive covers a field day — the second exists so your only copy
isn't on one device between recording and offload.

---

## 2. Timing — NT$3,600

Closes the open `\todo{}` in `tables/spec.tex`: *sniffer clock bias, worst case
7.8 ms*. Every clock number in the paper is currently measured against
`mobile_1`'s own chrony, with no independent reference anywhere in the rig.

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| u-blox GNSS timing module | 2 | 2,000 | **PPS output broken out to a pin.** NEO-M8T preferred (timing-grade, ~5 ns RMS); NEO-M9N/M8N acceptable if PPS is exposed. USB or UART | PPS is the whole point and most GPS breakouts don't expose it. M8T holds time on fewer satellites, which matters for an indoor window mount |
| Active GNSS antenna + 5 m coax | 2 | 1,600 | **Active** (powered LNA, 3–5 V bias); L1; IP67; magnetic mount; connector matched to the module (usually SMA male) | Passive antennas won't lock through a window. L1-only is fine — you're buying time, not RTK position |

Two units so you can put a reference at *both* ends of the bias measurement
rather than inferring it.

---

## 3. `infra_2` — second infrastructure node — NT$12,200

Your spec table claims `infra_N`. N is currently 1.

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| Raspberry Pi 4B 8 GB | 1 | 2,200 | **Model 4B.** Not Pi 5, not CM4 | Matches `infra_1`; 8 GB for camera + radar + rosbag concurrently |
| Arducam B0495 | 1 | 2,500 | **Exact model B0495**, 1920×1200 @ 30 Hz. *Do not substitute* | Must match `infra_1` or the two infra nodes have different intrinsics and distortion models, and the calibration pipeline forks |
| TI IWR6843ISK | 1 | 7,500 | **IWR6843ISK** — not ISK-ODS, not AOP. *Do not substitute* | ODS and AOP have different antenna patterns and FOV. Matching the other three boards is what makes radar results comparable across agents |

**Before ordering:** confirm whether your B0495 is the USB3 or CSI variant — it
changes the cable (1 m FPC ribbon vs 2 m USB3). Also confirm whether your radar
setup needs the MMWAVEICBOOST carrier; if so **order it as a separate line item**
to stay under the NT$10,000 cap.

---

## 4. Second CSI sniffer — NT$2,200

`record.yaml` already notes: *"the namespace is the RECEIVER, so a second
sniffer would not collide on these names."* The code is written for a receiver
you don't own.

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| Raspberry Pi 4B 4 GB | 1 | 2,200 | **Model 4B, BCM43455c0 radio. Not a Pi 5.** *Do not substitute* | Your nexmon CSI build targets BCM43455c0, and `tables/sensors_compact.tex` names it. A different radio means no CSI at all |

---

## 5. Pi support kit (×4 Pis) — NT$6,000

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| High-endurance microSD 128 GB | 4 | 2,800 | **"High Endurance" / "Pro Endurance" product line** (SanDisk Max Endurance, Samsung PRO Endurance). UHS-I U3/V30, A2 | Consumer cards (Ultra, Evo Select) die under continuous rosbag + nexmon logging. The endurance lines are rated in TBW/recording-hours for exactly this |
| Official Raspberry Pi USB-C PSU | 4 | 2,000 | **5.1 V** / ≥3 A | The 5.1 V is the spec that matters — the extra 0.1 V covers cable drop. A generic 5.0 V charger browns out a Pi 4B carrying a Wi-Fi dongle, and the symptom is CPU throttle → CSI rate falling off 140 Hz, which looks like a nexmon bug |
| Aluminium case with fan | 4 | 1,200 | Active cooling; **GPIO header accessible**; microSD accessible | Fanless Pi 4B + 140 Hz nexmon + rosbag writes throttles. GPIO must stay exposed for the serial console below — many sealed cases block it |

---

## 6. Mast & mounting — NT$4,500

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| Heavy-duty light stand / tripod, 2 m | 2 | 3,000 | ≥2 m; **load ≥3 kg**; 1/4"-20 or 5/8" stud; weighted or sandbagged base | A mast that flexes invalidates extrinsics you measure to 10 mm. Rigidity is the spec, not height |
| Super clamps, magic arms, 1/4-20 adapters | — | 1,000 | Metal, not plastic | Same reason |
| Foam-board ChArUco reprints | — | 500 | **Rigid** board (foam core or PVC), matte finish, printed at verified scale | A warped or glossy board is a systematic error in the fiducial chain. Verify printed square size with calipers before use |

---

## 7. Cables, power, cooling — NT$6,350

| Item | Qty | NT$ | Must-have spec | Why this spec |
|---|---|---|---|---|
| Powered USB3 hub, 4-port | 1 | 800 | **External 12 V PSU** (not bus-powered); USB 3.0+ | `mobile_1` runs ZED 2i + 2× IWR6843 + Wi-Fi dongle. A bus-powered hub cannot feed that |
| Micro-USB cable | 6 | 600 | **Data-capable** (not charge-only); 24–28 AWG; ≤1 m | Charge-only cables are the single most common "radar won't enumerate" cause |
| Cat6 / Cat6a, 2 m + 5 m mix | 8 | 800 | Cat6+; **stranded** (not solid) core | Stranded survives flexing on a robot. Solid-core cracks |
| 5-port gigabit switch | 1 | 450 | Unmanaged gigabit is fine | One wired segment for the Ouster link-local net + NTP. A PTP transparent-clock switch is NT$10k+ and buys nothing at your 0.99 ms budget |
| USB3 extension, 1 m | 2 | 400 | **Shielded**, USB 3.0 rated | Gets the RTL8852BU dongle off the metal chassis. Shielding matters: USB3 is a known 2.4 GHz noise source and your sniffer is on a 2.4 GHz link |
| micro-HDMI→HDMI ×2, HDMI ×1, DP→HDMI ×1 | 4 | 700 | Pi 4B is **micro**-HDMI; MIC-713 is HDMI 2.0 type-A; AGX Orin is DisplayPort | Three different connectors across the rig |
| USB–TTL serial adapter | 2 | 300 | **3.3 V logic** (CP2102 or FT232RL jumpered to 3.3 V). **Not 5 V** | 5 V on the Pi GPIO UART damages it. This is the debug path that works with no network and no monitor — the most useful NT$150 here |
| 12 V pack + DC connector + inline fuse | 1 | 1,500 | MIC-713 accepts **9–36 VDC**; size for ~135 W peak (≈11 A at 12 V). LiFePO4 12 V ≥20 Ah, ≥15 A continuous. 15 A inline fuse | 9–36 V means a 12 V or 24 V pack drives it directly, no inverter. AT mode (jumper default) auto-powers-on when voltage is applied — no button press when you set the cart down. 24 V halves the current if you prefer |
| 12 V fan, 120 mm + guard + mount | 1 | 400 | 12 V DC, 2-pin, with finger guard | The MIC-713 is **fanless**, derated to ~50 °C ambient in MAXN_SUPER. You're about to run 141 MB/s of serialization alongside Isaac VSLAM in it. The cart already has 12 V |
| Velcro, cable ties, labels, spare fuses | — | 400 | — | Strain relief on the SSD cable is not optional on a cart |

---

## Platform reference (for the `bom.tex` `\todo{}` cells)

| Agent | Compute | Key specs |
|---|---|---|
| `mobile_1` | Jetson AGX Orin | DisplayPort output |
| `mobile_2` | **Advantech MIC-713-OX4B2** | Jetson Orin NX 16 GB, 16 GB LPDDR5, fanless, 128 GB NVMe pre-installed, **1× M.2 2280 (PCIe x1 Gen4)**, 1× M.2 3052 B-key (**USB signal only** — not storage), 1× mPCIe, microSD, 2× GbE, 6× USB 3.2 Gen 2, HDMI 2.0 (no DP), 9–36 VDC, AT/ATX jumper (AT default) |
| `infra_1`, `infra_2`, sniffers | Raspberry Pi 4B | BCM43455c0 radio (required for nexmon CSI), micro-HDMI |

Leave `mobile_2`'s mPCIe slot empty: an mPCIe Wi-Fi card would change the
transmitter chipset away from the RTL8852BU that `csi_params.sh tx` tunes
(HE off, 1 TX stream, VHT, `ONE_CHAIN`), breaking comparability with `infra_1`.
The B-key 3052 slot carries USB only and cannot take an NVMe.

---

## Config changes to make before the parts arrive

These cost nothing and are what make the purchase work.

1. **Mixed resolution** for `mobile_2` in `record.yaml`: color `1280x800`,
   depth and both infra `848x480`. Depth and the IR pair are the same physical
   imagers, so they must match each other; color is an independent sensor.
   Rationale: 1280 color helps the perception task; 1280 *depth* does not —
   Intel tunes D4xx stereo matching for 848x480, and higher resolution gives
   more per-pixel noise and a higher min-Z. Net 141 MB/s instead of 220.
2. **`max_cache_size` 1 GiB → 2 GiB.** At 141 MB/s the current buffer is ~7 s
   of stall tolerance; 16 GB of LPDDR5 can spare it.
3. **Label the T7 filesystems**, add fstab entries, point `out_dir` at the
   mount. An unmounted drive silently fills the 128 GB internal instead.
4. **`lsusb -t`** — keep the T7 and the D455 off the same USB controller. Orin
   NX provides fewer root ports than the MIC-713 exposes, so some sit behind a
   hub; ~440 MB/s through one controller is where frame drops come from.
5. **Two-minute test pass with VSLAM running**, then
   `scoop_pipeline/slam/monitor/summarize.py` for per-topic drops and
   `tegrastats` for throttling. The thermal answer cannot be derived on paper.
6. **Sniffer chrony over eth0**, not the 2.4 GHz link. `tables/ntp.tex` shows
   `mobile_2` and `infra_1` at ~4.1 ms path delay and the sniffer at 15.6 ms —
   that gap is the Wi-Fi link. One Cat6 cable from this order fixes it.

## Deferred

| Item | Why not now |
|---|---|
| RTK-GNSS (2 rovers + base, ~NT$23k) | Position only, no heading — you report 0.88°/1.38° rotation residuals. Degrades worst beside buildings, which is exactly the NLOS geometry driving your clearance↔RSSI result. `docs/BENCHMARK.md:342` already proposes the outdoor sequence *"with the same board anchoring"*, not GNSS |
| PTP (`linuxptp`) | Costs NT$0 — the Orin, Pi 4B and Ouster all do IEEE 1588 hardware timestamping. But you're at 10.9–42.5 µs against a 0.99 ms requirement, so it solves nothing. The one free win worth taking: switch the Ouster from `TIME_FROM_ROS_TIME` to `TIME_FROM_PTP_1588` so LiDAR stamps come from sensor hardware rather than driver receipt — and add a PTP-lock check to `record.sh`, because the failure is silent |
| 24" monitor (~NT$3,000) | `rig.py` opens 29 tabs across 4 windows, so it would help. Add it only if shipping and VAT come in under budget |

---

## Appendix: data volume per field day

Rates below assume the mixed profile (`mobile_2` color 1280x800, depth and both
infra 848x480).

| Machine | Dominant topics | MB/s | GB/min |
|---|---|---|---|
| `mobile_1` bag | Ouster OS1-64 RAW packets 8.5, ZED IMU 0.15, 2x radar 0.2, link/ntp/tf 0.1 | ~9 | 0.54 |
| `mobile_1` SVO | separate `.svo2` file — **see warning** | ~60 | 3.6 |
| `mobile_2` bag | color 92.2, depth 24.4, infra1 12.2, infra2 12.2, imu/vslam/link 0.3 | ~141 | 8.5 |
| `infra_1` bag | Arducam JPEG ~3.5, radar 0.1, link/ntp 0.1 | ~3.7 | 0.22 |
| sniffer bag | 2x CSI topics, 242 subcarriers @ ~140 Hz | ~0.3 | 0.018 |

Method check: `scoop_pipeline/README.md:852` measures 100 GB for an 18-minute
ZED decode at 960x540 uncompressed = 92 MB/s. The same arithmetic applied to
those streams gives ~92 MB/s, so the framework reproduces a measured figure.

### A representative day

`map_1` (10 min), `survey_1-2` (5 each), `coop_1-3` (5 each), `contention_1`
(5 min). The `profiles:` block in `record.yaml` decides who records: `infra_1`
and the sniffer sit out `map`, and `contention` is link/IMU only.

| Machine | Recording min | Day total |
|---|---|---|
| `mobile_1` bag | 35 | 19 GB |
| `mobile_1` SVO | 35 | 126 GB (or 7 GB — see warning) |
| `mobile_2` bag | 35 | 297 GB |
| `infra_1` bag | 25 | 5.5 GB |
| sniffer bag | 30 | 0.5 GB |
| **total** | | **~330-450 GB/day** |

### What this means for the drives

- `mobile_2` at 297 GB/day: ~6.7 field days on a 2 TB T7 Shield.
- `mobile_1` at ~145 GB/day (bag + lossless SVO) **also needs external
  storage**. So the two T7 Shields are one per mobile agent, both justified on
  capacity rather than redundancy. Keep the end-of-day rsync to the 8 TB for
  the second-copy problem.
- 8 TB archive holds ~20 field days of raw bags. Processed products are several
  times larger again (a ZED decode alone is ~100 GB per 18-minute recording,
  ~245 GB with the point cloud), so plan to top up the archive, or keep only
  raw bags plus the processing recipe.

### Two things to verify before trusting these numbers

1. **SVO compression mode.** `svo.sh` calls `StartSvoRec` with
   `compression_mode: 0`. In the ZED SDK that enum is `0 = LOSSLESS`,
   `1 = H264`, `2 = H265`. If 0 means lossless in your wrapper version, the SVO
   is ~60 MB/s and is the largest single data producer in the rig — 7x
   `mobile_1`'s whole bag. `compression_mode: 2` (H265) would cut it ~20x to
   ~3 MB/s. Whether that is acceptable depends on whether lossless was
   deliberate for the depth/photometric work. Check the service definition and
   the size of an existing `.svo`; this one number moves the day estimate by
   120 GB.
2. **`slam_path` / `vo_path` are `nav_msgs/Path`**, which accumulate poses. If
   Isaac VSLAM republishes the full path at 30 Hz, total bytes grow as O(t^2):
   ~2.3 GB over a 5-minute pass, ~9 GB over 10 minutes. `summarize.py` prints
   per-topic MB/s and will show it immediately. If confirmed, drop both —
   `tracking/odometry` carries the same information incrementally.
