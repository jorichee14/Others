# ros2_wifi_monitor — Block Diagram

Architecture of the Wi‑Fi monitoring stack: three independent ROS 2 nodes,
each wrapping OS/kernel data sources through pure parser modules and publishing
a stamped message. Nodes never depend on one another — a failed source degrades
to `NaN`/`-1` rather than crashing — and `survey.launch.py` starts all three
sharing one server/target address.

## System block diagram

```mermaid
flowchart LR
    %% ---------------- OS / kernel data sources ----------------
    subgraph SRC["OS / kernel data sources (read-only, no root)"]
        direction TB
        PW["/proc/net/wireless<br/>quality · signal · noise"]
        IW["iw dev &lt;iface&gt;<br/>link · station dump · survey dump"]
        IWC["iwconfig (legacy fallback)<br/>+ wireless error counters"]
        SYS["/sys/class/net/&lt;iface&gt;/*<br/>statistics · address · flags"]
        IPERF3["iperf3 -c (subprocess)"]
        SS["ss -ti (live TCP RTT)"]
        PING["ping (ICMP)"]
    end

    %% ---------------- Pure parser modules ----------------
    subgraph PARSE["Pure parsers (no ROS, unit-tested)"]
        direction TB
        WP["wifi_parsers.py<br/>collect_all()"]
        IP["iperf_parse.py<br/>parse_iperf_json · parse_interval_line · parse_ss_rtt"]
        PP["ping_parse.py<br/>parse_ping_line"]
    end

    %% ---------------- ROS 2 nodes ----------------
    subgraph NODES["ROS 2 nodes (wifi_monitor pkg)"]
        direction TB
        N1["wifi_monitor_node<br/>passive sampler @ ~5 Hz"]
        N2["iperf_runner_node<br/>active throughput (bg thread)"]
        N3["ping_monitor_node<br/>latency + loss (bg thread)"]
    end

    %% ---------------- Topics / messages ----------------
    subgraph TOPICS["Topics · wifi_monitor_msgs"]
        direction TB
        T1(["/wifi/status<br/>WifiLinkStatus"])
        TD(["/diagnostics<br/>DiagnosticArray"])
        T2(["/wifi/iperf<br/>IperfResult"])
        T3(["/wifi/ping<br/>PingStat"])
    end

    %% ---------------- Consumers ----------------
    subgraph SINK["Consumers"]
        direction TB
        RQT["rqt_robot_monitor"]
        ECHO["ros2 topic echo"]
        BAG["ros2 bag record<br/>+ /tf /odom → offline pose join"]
    end

    %% wires: sources -> parsers
    PW --> WP
    IW --> WP
    IWC --> WP
    SYS --> WP
    IPERF3 --> IP
    SS --> IP
    PING --> PP

    %% parsers -> nodes
    WP --> N1
    IP --> N2
    PP --> N3

    %% nodes -> topics
    N1 --> T1
    N1 --> TD
    N2 --> T2
    N3 --> T3

    %% topics -> consumers
    TD --> RQT
    T1 --> ECHO
    T2 --> ECHO
    T3 --> ECHO
    T1 --> BAG
    T2 --> BAG
    T3 --> BAG

    %% launch orchestration
    LAUNCH["survey.launch.py<br/>(shared server/target address)"]
    LAUNCH -.starts.-> N1
    LAUNCH -.starts.-> N2
    LAUNCH -.starts.-> N3
```

## Node reference

| Node | Reads (via parser) | Publishes | Default rate | Saturates link? |
| ---- | ------------------ | --------- | ------------ | --------------- |
| `wifi_monitor_node` | `/proc/net/wireless`, `iw link/station/survey`, `iwconfig`, `/sys/class/net/*` | `/wifi/status` (`WifiLinkStatus`), `/diagnostics` (`DiagnosticArray`) | 5 Hz | No (passive) |
| `iperf_runner_node` | `iperf3 -c` (+ `ss -ti` for RTT), link carrier for the failsafe | `/wifi/iperf` (`IperfResult`) | ~1 Hz (continuous) / burst | **Yes** — dedicated survey pass |
| `ping_monitor_node` | `ping` (ICMP) | `/wifi/ping` (`PingStat`) | 1/`interval_s` Hz | No (cheap) |

## Data-flow notes

- **Passive vs. active.** `wifi_monitor_node` only *reads* driver/kernel state
  (RSSI, MCS/NSS, retries, counters) and never touches the air. `iperf_runner_node`
  actively *pushes traffic* to measure real throughput, so it saturates the link
  and is meant for a dedicated survey pass. `ping_monitor_node` is the cheap
  middle ground for continuous latency/loss during real operation.
- **Graceful degradation.** Every parser is defensive: a missing tool, sysfs
  file, or denied `iw` sub-command yields absent keys, and the node fills the
  message with `NaN` (floats) / `-1` (ints) instead of failing. `wifi_monitor`
  keeps publishing right through a disconnect (`associated=false`).
- **Failsafes on the active nodes.** `iperf_runner_node` gates on link carrier
  (won't launch a hanging test while disassociated), bounds waits with
  `--connect-timeout`, and backs off on failure. Both `iperf_runner` and
  `ping_monitor` run their subprocess loop in a background thread so the ROS
  executor is never blocked, and both restart the subprocess if it exits.
- **RTT source in continuous iperf.** Continuous mode fills `rtt_ms_*` by
  sampling the *live* iperf connection's kernel TCP RTT via `ss -ti`, so
  throughput and RTT come from one consistent source — which is why the ping
  node is off by default in `survey.launch.py`.
- **Pose mapping.** Every message is stamped, so recording `/wifi/*` alongside
  `/tf` and `/odom` on the robot's local disk lets you time-join link quality
  to position offline.
