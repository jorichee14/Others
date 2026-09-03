# NTP synchronization: how it works, what went wrong in `coop2`, and how to fix it

Findings from the first NTP analysis of
`mirc_dataset_coop2_20260828_completed_0.mcap`, and the changes needed before the
next recording session.

> **Short version.** The clocks were fine — sub-millisecond. The *measurement* of
> them was not. The NTP daemons poll every 256 s but a run lasts 156 s, so each
> agent contributes **one** offset measurement per run, republished ~1300 times by
> the status topic. Any mean or percentile computed over those messages describes
> the daemon's interpolation, not the clock. Pin the poll interval to 16 s and the
> problem disappears.

---

## 1. What NTP actually does

### 1.1 The measurement

Every poll is one packet exchange carrying four timestamps:

| | |
|---|---|
| t₁ | client sends the request (client clock) |
| t₂ | server receives it (server clock) |
| t₃ | server replies (server clock) |
| t₄ | client receives the reply (client clock) |

From these the client computes:

```
delay  = (t₄ − t₁) − (t₃ − t₂)         round trip, minus the server's processing time
offset = ((t₂ − t₁) + (t₃ − t₄)) / 2    how far the client's clock is from the server's
```

The offset formula assumes **the packet took the same time in each direction**. If
the outbound leg is slower than the return by Δ, the computed offset is wrong by
Δ/2, and averaging does not remove it because the error is systematic.

This gives a hard bound worth remembering: **the offset error can be as large as
half the round-trip delay.** On wired Ethernet, delay is a few hundred microseconds
and this is negligible. Over Wi-Fi it is not.

### 1.2 What the daemon does with it

The daemon does *not* jump the clock. It runs a feedback loop that adjusts the
clock's **frequency** — running it slightly fast or slow — so the offset converges
toward zero. This is *slewing*. It *steps* the clock only when the offset is
enormous, or once at startup.

Because it is a frequency loop, **between polls the daemon predicts where the clock
is rather than measuring it.** This matters for everything in §3.

### 1.3 The poll interval, and why it grows

Poll values in `chrony.conf` are **exponents of two, not seconds**:

| value | interval | |
|---|---|---|
| 4 | 16 s | |
| 5 | 32 s | |
| 6 | 64 s | chrony's default `minpoll` |
| 7 | 128 s | |
| 8 | **256 s** | **what this recording used** |
| 9 | 512 s | |
| 10 | 1024 s | chrony's default `maxpoll` |

The daemon **starts at `minpoll` and doubles** the interval as its estimate settles,
up to `maxpoll`. This is correct behaviour for a long-lived server: once the
frequency error is well characterized, the daemon predicts drift better than a
single noisy measurement corrects it, so polling more often only adds traffic.

It backs off *precisely when things are going well* — which is why it happened here.

### 1.4 The other fields

- **Stratum** is distance from a reference clock. Stratum 0 is a GPS or atomic
  source, stratum 1 is directly attached to one, each hop adds one.
- **Reach** is an 8-bit shift register, one bit per recent poll. `255` (octal 377)
  means the last eight polls all succeeded.
- **Root dispersion** is NTP's own estimate of its maximum error, and it grows
  between polls. Unlike a mean over republished messages, this *is* a legitimate
  uncertainty to quote.

---

## 2. What we measured in `coop2`

Topology: **`mobile_1` is the NTP server**; `mobile_2` and `infra_1` are clients at
stratum 5. Only the clients publish an NTP status topic, so `mobile_1`'s own clock
is the reference and has no offset of its own.

| Agent | Hostname | Offset to server | Poll | Reach | Messages |
|---|---|---|---|---|---|
| `infra_1` | `wicoms-robot2` | 0.22 ms | 256 s | 255 | 1234 |
| `mobile_2` | `ubuntu` | 0.67 ms (from t = 46 s) | 256 s | 255 | 1363 |

Delivery to the recorder (`header.stamp − log_time`, median per node):
`mobile_1` ≈ 0 ms, `infra_1` ≈ −6 ms, `mobile_2` ≈ −17 ms.

**The clocks themselves are good.** Sub-millisecond agreement is well inside one
frame at every sensor rate in the bag. Everything below is about the evidence, not
the synchronization.

---

## 3. Problems

### 3.1 One measurement per run, reported as a distribution — *blocking*

Poll interval 256 s, run length 156 s ⇒ **each client is polled at most once during
a run.** The status topic publishes at ~8–9 Hz regardless, so the bag contains ~1300
copies of that single result, each nudged slightly by the daemon's frequency
prediction (§1.2).

Computing statistics over those messages gives:

```
infra_1:  mean 0.220   median 0.220   p95 0.227   max 0.228     (ms)
```

Those four numbers are **one number**. The 0.008 ms of apparent spread is the slope
of the daemon's extrapolation, not variation in the clock.

Worse, it is **overconfident**. Between polls the clock drifts by the residual
frequency error times elapsed time; at even 1 ppm that is ±0.16 ms over 156 s, with
nothing in the run able to detect it. The reported p95 suggests the offset is pinned
to 8 microseconds when the honest uncertainty is comparable to the value itself.

### 3.2 `delay` and `jitter` are always zero — *blocking for an error bar*

**What the fields are.** `delay` is the round-trip time of the NTP packet exchange:
how long the request took to reach the server plus how long the reply took to come
back, minus the server's own processing time. Over Wi-Fi this is typically 2–20 ms.
`jitter` is how much that measurement scatters from poll to poll.

**Zero is impossible.** A packet cannot travel to another machine and back in no
time. Zero here is not a small measurement, it is the absence of one.

**Why it happens.** The fields the monitor *does* populate map cleanly onto two
specific chrony commands:

| Field in `NtpStatus` | Comes from |
|---|---|
| `offset_seconds`, `root_delay`, `root_dispersion`, `frequency_error_ppm`, `leap_indicator`, `reference_time`, `sync_source` | `chronyc tracking` |
| `stratum`, `poll_interval_seconds`, `reach_register` | `chronyc sources` |

Neither of those commands reports **per-source** round-trip delay or jitter.
`tracking` gives *root* delay — the accumulated delay up the whole chain to the
reference clock — not the delay of this client's link to `mobile_1`. `sources` gives
poll and reach but no timing. So the node declares `delay_seconds` and
`jitter_seconds`, never finds a value for them, and they stay at their default 0.0.

**It also violates the message's own convention.** `NtpStatus.msg` states that
fields which could not be read are reported as NaN for floats and 0 for integers.
`delay_seconds` and `jitter_seconds` are `float64`, so they should be **NaN**.
Publishing 0.0 is worse than NaN: NaN says "unknown", 0.0 says "measured, and it
was zero", and no downstream consumer can tell the difference. (`ntp_analysis.py`
already omits the delay sentence when it sees NaN; with 0.0 it cannot.)

**Why it matters.** By §1.1 the offset error is bounded by half the round-trip
delay. Delay is therefore the only thing that puts an error bar on the 0.22 ms
figure. Without it we can say the daemon *estimated* 0.22 ms, but not whether the
truth is 0.22 ± 0.01 ms or 0.22 ± 3 ms — and with 6–17 ms delivery times over
Wi-Fi (§2), the second is entirely plausible.

There is a secondary reason to want it: delay over time is itself a Wi-Fi link
measurement, and it will correlate with the RSSI and throughput analysis.

### 3.3 Topics stamped seconds behind the recorder — *open, needs identifying*

Most topics deliver within ~20 ms, but a few sit at **−3200 to −3400 ms**: header
stamps 3.3 seconds behind the recorder's receive time, on both `mobile_1` and
`mobile_2`. These have valid stamps, so they are not the unset-stamp case below.

This is larger than any NTP effect and would break cross-agent alignment on those
topics regardless of clock quality. **Which topics these are is not yet
established** — check `ntp_audit.csv`, sort by `stamp_minus_log_median_ms`.

### 3.4 Topics with unset header stamps

Some topics carry `header.stamp = 0` because the driver never set it. They are
listed in `ntp_unset_stamps.csv` and excluded from the delivery check; only the
recorder's receive time is available for them.

### 3.5 Two hosts both named `ubuntu`

`mobile_1` and `mobile_2` report the same hostname, so `sync_source: "ubuntu"` is
ambiguous. The analysis now labels agents by topic namespace instead, but this
should be fixed at the source.

### 3.6 The configured poll limits are not recorded

The message carries `poll_interval_seconds` (the *current* interval) but not the
configured `minpoll`/`maxpoll`. So a reader of the released dataset cannot tell
whether 256 s was chosen deliberately or drifted into. *(Which of the two it was
here is still unconfirmed — see §4.1.)*

---

## 4. Solutions

### 4.1 Pin the poll interval — fixes §3.1

First confirm which daemon is running and what it is configured with:

```bash
systemctl is-active chrony chronyd ntp systemd-timesyncd
grep -E "^(server|pool)" /etc/chrony/chrony.conf
chronyc sources -v          # the "Poll" column is log2 seconds, so 8 = 256 s
```

Then, on **each client** (`mobile_2`, `infra_1`), set the source line in
`/etc/chrony/chrony.conf` to:

```
server <mobile_1 address> iburst minpoll 4 maxpoll 4
```

| Option | Effect |
|---|---|
| `minpoll 4` | poll no faster than every 2⁴ = 16 s |
| `maxpoll 4` | poll no slower than every 16 s — setting both **pins** the interval and stops the doubling |
| `iburst` | send a rapid burst at startup, so the first result lands in seconds instead of after a full poll interval |

```bash
sudo systemctl restart chrony
chronyc sources -v          # "Poll" should now read 4
```

A 156 s run then contains ~10 genuine measurements per client, and mean and p95
become real. `iburst` also removes the 46 s dead period `mobile_2` showed at the
start of this run.

**Cost:** four packets per minute per client, ~90 bytes each — about 6 B/s, against
a 170 Hz CSI stream.

**Do not go below `minpoll 4`.** The daemon estimates clock *frequency* by comparing
measurements spaced apart in time; samples crammed together make that estimate
noisier, not better.

### 4.2 Populate `delay` and `jitter` — fixes §3.2

Per-source timing lives in a third command that the monitor is not reading:

```bash
chronyc -c ntpdata <mobile_1 address>   # "Peer delay" is the round-trip delay
chronyc -c sourcestats                  # "Std Dev" serves as jitter
```

The `-c` flag gives comma-separated output instead of the aligned table, which is
far easier to parse reliably than column positions.

Two changes to the monitor node:

1. Read `ntpdata` (and `sourcestats`) in addition to `tracking` and `sources`, and
   fill `delay_seconds` and `jitter_seconds` from them.
2. Change the fallback for **every float field** from `0.0` to `float("nan")`, so
   the message matches the convention its own definition states. Unknown must be
   distinguishable from zero.

With delay populated, the offset can be quoted with the ±delay/2 asymmetry bound
from §1.1 — which is what turns a single number into a measurement with an error
bar.

### 4.3 Record the configuration — fixes §3.6

Add the configured `minpoll` and `maxpoll` to `NtpStatus`, or write them into the
per-run metadata alongside the daemon name and version.

### 4.4 Give each machine a distinct hostname — fixes §3.5

`sudo hostnamectl set-hostname mobile-2` (and equivalents), before the next session.

### 4.5 Investigate the 3.3 s topics — §3.3

Sort `results/<run>/ntp/ntp_audit.csv` by `stamp_minus_log_median_ms` and identify
what is publishing that late. Likely candidates are buffered or post-processed
topics rather than live drivers.

---

## 5. What to report in the paper

### 5.1 With the current data

State it as what it is — a single measurement per agent per run:

> All agents synchronize over NTP to `mobile_1` on the shared wireless network, as
> stratum-5 clients. The clients poll every 256 s, longer than a run, so each run
> contributes one measurement per agent: over `coop2` these were 0.22 ms
> (`infra_1`) and 0.67 ms (`mobile_2`) from the server's clock, both far below one
> sample period of the fastest sensor. No sensor is hardware-triggered or
> hardware-timestamped; every message is stamped in software by its driver on
> arrival at the host, so these offsets bound clock disagreement between agents,
> not sensor exposure time.

Do **not** write mean, p95 and max as if they described variation over the run
(§3.1).

Across many runs, each run's single measurement is one legitimate draw, so a
distribution pooled **across sessions** is defensible and arguably the more useful
claim: it describes how reliably the agents synchronize session to session.

### 5.2 After the fix

With a 16 s poll, report per client, pooled across runs: median and p95 absolute
offset, worst-run maximum, number of runs containing a clock step, and the fraction
of runs whose offsets stayed below the shortest sensor period. Ship the per-run
table with the dataset.

---

## 6. Reproducing this analysis

```bash
python analysis/ntp_analysis.py --bag <run>.mcap --run coop2 --server mobile_1
```

Outputs land in `results/coop2/ntp/` — see `analysis/README.md` for the file list.
