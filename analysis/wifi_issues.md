# Wi-Fi link measurement: what the suite records, what `coop2` showed, and what needs fixing

Findings from the first Wi-Fi analysis of
`mirc_dataset_coop2_20260828_completed_0.mcap`, and the changes needed before the
next recording session.

> **Short version.** Three of the connectivity claims in `DATASET_NOVELTY.md` are not
> supported by this run. `mobile_1` carries **one** radio, not two, so the inter-link
> correlation ρ cannot be measured. The adapters' driver does not implement
> `iw survey dump`, so **channel occupancy is unavailable**. And the link never
> degraded — median RSSI −39 dBm, zero Bad samples — so `coop2` is a **good-link
> condition**, not a test of connectivity-aware behaviour.
>
> The measurement pipeline itself is sound — these are findings about the run and the
> hardware, not bugs in the analysis.

---

## 1. What the suite measures

### 1.1 Two different things called "rate"

- **PHY rate** (`tx_bitrate_mbps`) is the modulation the radio negotiated: an upper
  bound set by signal quality, before any protocol overhead, contention or retries.
- **Goodput** (`IperfResult.bitrate_mbps`) is what an application actually moved.

They differ by a large factor and are not interchangeable. The PHY rate says how good
the radio thinks the channel is; goodput says what the link delivered. Both are plotted
on one axis in `fig_wifi_link` panel (b) — lines for the first, markers for the second —
precisely so the gap between them is visible.

### 1.2 Cumulative counters, not rates

`tx_retries`, `tx_failed`, `sta_tx_packets`, `channel_busy_ms` and the interface traffic
counters are **cumulative totals**, and the per-association ones **reset to zero when the
station reassociates**. Reading them directly gives a run total; differencing them without
care gives large negative spikes at every reassociation. `wifi_analysis.py` differences
them and discards the negative steps, so what it reports are per-interval rates.

### 1.3 Field groups that can be absent

`WifiLinkStatus` documents whole groups as NaN (floats) or −1 (integers) when the
underlying query is denied or unsupported:

| Group | Source command | Contains |
|---|---|---|
| station dump | `iw dev <if> station dump` | `tx_retries`, `tx_failed`, `expected_mbps`, `connected_time_s`, `sta_*` |
| channel survey | `iw dev <if> survey dump` | `channel_active_ms`, `channel_busy_ms`, `channel_busy_ratio` |
| noise / SNR | driver-dependent | `noise_dbm`, `snr_db`, `snr_valid` |

**A statistic computed over an absent group looks like a measurement of zero.** The
analysis therefore audits every field against its documented sentinel first, writes
`wifi_field_availability.csv`, and omits both the statistics and the figure panels for
groups nobody reported.

---

## 2. What we measured in `coop2`

Both robots associate with the **same access point on channel 149 (5 GHz)**, so each
link is agent → AP. The robot-to-robot iperf traverses that AP in both directions and is
reported as the two-hop path it is.

| | `mobile_1` | `mobile_2` |
|---|---|---|
| radios | **1** | 1 |
| band / channel | 5 GHz / 149 | 5 GHz / 149 |
| status rate | 10.0 Hz | 4.9 Hz |
| median RSSI | −39 dBm | −39 dBm |
| minimum RSSI | −52 dBm | −51 dBm |
| median PHY rate | 1201 Mbit/s | 1081 Mbit/s |
| TX failure rate, p95 | 0.0 | 0.0 |
| Bad fraction | 0 | 0 |

Field availability: **station dump present**, so the retry and failure figures are real
measurements. **Channel survey absent** — the driver does not implement it.

---

## 3. Problems

### 3.1 One radio, not two — *blocks §2.4 and benchmark F2*

`DATASET_NOVELTY.md` §2.4 and §4.3 describe a **dual-radio mode** whose purpose is to
measure the inter-link correlation **ρ**, and benchmark **F2** is built entirely on it.
In this run `/mobile_1/wifi/status` carries a single `interface`, so:

- ρ, `p_joint` and the per-link Gilbert–Elliott fits **cannot be computed**,
- `wifi_rho.csv` and `fig_wifi_rho` are correctly not produced.

*A note on how this was nearly missed.* The topic advertises **two**
`offered_qos_profiles`, which looks like two publishers. It is not evidence of two
radios: `/mobile_1/radar1/radar/points_all` and `/tf` also show two profiles, and those
are one sensor and one shared topic respectively. Two profiles mean two publishers
**registered at some point**, which a node restart produces. The message count is also
misleading — 1520 versus `mobile_2`'s 717 looked like 2×, but it is simply a 10 Hz
publish rate against 4.9 Hz. **Only the `interface` and `mac_address` fields settle it**,
which is what the analysis uses.

### 3.2 Channel occupancy is unavailable — *blocks part of §2.4*

The adapters' driver does not implement `iw survey dump`. Confirmed by running it
manually; it is a driver limitation, not a permissions problem, so `CAP_NET_ADMIN`
will not fix it.

§2.4 states that "channel-occupancy and retry counters expose whether the links draw on
separate resources or merely contend on a shared medium". The **retry** half of that
sentence is supported; the **occupancy** half is not measurable with this hardware.

### 3.3 RSSI varies, but performance never does — *blocks F1; limits S2*

Two distinct facts, and conflating them would misstate what this run supports.

**RSSI does vary.** The traces swing roughly 20 dB over the run as the robots move, with
clear structure that tracks the path. So there is a real spatial signal here.

**Performance does not.** Every one of those samples is in the strong regime — nothing
below −52 dBm, 18 dB clear of even a generous −70 dBm Bad threshold. The PHY rate sits at
its ceiling throughout, no frame fails after retries, and iperf goodput is steady. The
link had roughly 20 dB of margin and the variation consumed none of it.

Consequences differ per benchmark:

- **F1**, connectivity-aware cooperative perception under a *measured* channel:
  **not demonstrable**. The channel never constrained anything, so a link-aware policy and
  a link-agnostic one behave identically. There is no accuracy-versus-cost curve to draw.
- **S2**, connectivity mapping fused with scene geometry: **partially demonstrable**. An
  RSSI coverage map over ~20 dB of dynamic range is buildable and can be validated on
  held-out poses. What cannot be validated is **dead-zone prediction**, since there are no
  dead zones, and that is the part of S2 that distinguishes it from ordinary radio mapping.

The finding worth reporting in its own right: **in this environment, at these distances,
20 dB of RSSI variation produced no measurable change in throughput or reliability.** That
is a statement about link margin, and it is the reason a degraded run is needed rather
than more runs like this one.

### 3.4 No pose join yet

Every number above is per-run, not per-location. The coverage/throughput/dead-zone maps
that §4.3 and S2 promise need the link samples time-joined to the certified trajectories.
That is the next analysis, not a defect.

---

## 4. Solutions

### 4.1 Record a run that actually stresses the link — *fixes §3.3*

This is the most important change, and it costs nothing but a different path. Drive the
robots **behind the metal shelving, around corners, and to the far end of the space**,
and dwell there long enough for several iperf tests to land. A run whose RSSI spans
−39 to −80 dBm with real dead zones is what F1 and S2 need. Keep `coop2` as the
good-link control — the contrast between them is itself a result.

### 4.2 Decide the dual-radio question — *fixes §3.1*

Either:

1. **Run two radios** on `mobile_1` in a future session (different bands, or two APs),
   confirm with `iw dev` that both interfaces publish, and ρ becomes measurable; or
2. **Reword §2.4, §4.3, §7 and F2** so the ρ axis is presented as the framework's
   parameter with the measurement deferred, rather than as something this dataset
   delivers.

Option 1 is a hardware change; option 2 is honest and cheap. Do not ship the current
text with single-radio data.

### 4.3 Channel occupancy: three options — *for §3.2*

1. **Measure it on the AP instead.** If the access point runs OpenWrt or similar,
   `iw dev <if> survey dump` on the AP gives occupancy for the whole cell — arguably a
   better measure of "do the links contend on a shared medium" than a per-station view.
2. **Change adapter** to one whose driver implements survey dump (`ath9k`/`ath10k`
   generally do; many USB Realtek and Mediatek parts do not).
3. **Drop the claim** and rely on the retry and failure counters, which are available and
   do respond to contention, stating the limitation explicitly.

Option 1 is the cheapest if the AP is manageable.

### 4.4 Record the adapter and driver in the metadata

Nothing in the bag says which adapter or driver produced these fields, so a reader cannot
tell why occupancy is missing. Add the adapter model and `ethtool -i <iface>` driver name
to the per-run metadata.

---

## 5. What to report in the paper

### 5.1 For `coop2`

> Both robots associated with the same access point on channel 149 (5 GHz). Received
> signal strength varied by about 20 dB along the trajectories, but remained in the strong
> regime throughout — median −39 dBm, minimum −52 dBm on both agents — so the negotiated
> PHY rate stayed at its ceiling (1.2 and 1.1 Gbit/s) and no frame failed after retries.
> This sequence therefore serves as a good-link control condition: it exhibits spatial RSSI
> structure without any accompanying degradation in link performance. The adapters' driver
> does not implement channel-survey reporting, so channel occupancy is not available.

Do **not** report a ρ, a joint-loss fraction, or a channel-occupancy figure for this run.

### 5.2 Across runs

Once a degraded run exists, report per run: RSSI median and 5th percentile, PHY rate,
goodput, retry and failure rates, and the fraction of the run in the Bad state, alongside
the trajectory paradigm. The contrast between a good-link and a degraded run is what
makes the connectivity modality a variable rather than a constant.

---

## 6. Reproducing this analysis

```bash
python analysis/wifi_analysis.py --bag <run>.mcap --run coop2
```

Outputs land in `results/coop2/wifi/` — see `analysis/README.md` for the file list.
Check `wifi_field_availability.csv` first: it says which fields the adapter actually
reported, and every statistic and figure panel downstream depends on it.
