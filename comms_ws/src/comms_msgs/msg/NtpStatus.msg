# NTP Status message for both client and server metrics
std_msgs/Header header

# Node role: "client", "server", or "client_view_from_server"
string role

# Hostname of this node
string hostname

# Synchronization state
bool synchronized
string sync_source
string stratum_level
int32  stratum

# Timing metrics (all in seconds)
# NEGATIVE = local clock behind NTP time, positive = ahead. Chrony's steered
# estimate ("System time"): the error that applies to a sensor stamp taken now.
float64 offset_seconds
# The MEASURED offset at the most recent daemon poll ("Last offset"), signed
# the same way. This is the observation; offset_seconds is what the daemon
# made of it. Changes only when a poll lands, so it counts polls.
float64 last_offset_seconds
float64 delay_seconds
float64 jitter_seconds
float64 root_delay
float64 root_dispersion

# Frequency / drift
# NEGATIVE = clock runs slow. Sign preserved from "N ppm slow/fast".
float64 frequency_error_ppm
# Estimated error of the frequency estimate, from sourcestats.
float64 skew_ppm
# Samples in the daemon's current regression fit, from sourcestats.
int32   fit_samples

# Poll / reach
int32  poll_interval_seconds
int32  reach_register
int32  reachability_percent

# Reference time
builtin_interfaces/Time reference_time

# Server-only: connected client count (-1 when role == "client")
int32  connected_clients

# Diagnostics
string leap_indicator
string[] warnings

# Rosbag / distributed alignment fields
uint64  seq
float64 monotonic_seconds
bool    clock_stepped
float64 offset_delta_seconds

# --- Everything else that moves the clock, appended (13 Sep) ---
float64 rms_offset_seconds       # tracking "RMS offset": typical recent measured offset
float64 residual_freq_ppm        # tracking "Residual freq": what the last fit could not explain
float64 update_interval_seconds  # tracking "Update interval": the actual poll spacing
float64 fit_span_seconds         # sourcestats "Span": time covered by the fit's samples
float64 temperature_c            # SoC temperature; the crystal's rate follows it (~0.1-1 ppm/degree)
float64 cpu_load_1min            # /proc/loadavg, so a stress phase is visible in the same message
