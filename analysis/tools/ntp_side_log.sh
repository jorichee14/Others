#!/usr/bin/env bash
# ntp_side_log.sh <agent> <phase>
#   e.g.  ./ntp_side_log.sh mobile_2 thermal      (Ctrl-C to stop)
#
# One CSV line every 8 s with what the NtpStatus topic does not carry:
# SoC temperature, and chrony's fit statistics. Runs beside `ros2 bag record`,
# needs no ROS. Timestamps are UTC so ntp_stress.py can join the file to the
# bag by wall-clock time. Both robots are called "ubuntu", so the agent name
# is an argument rather than the hostname. The phase label names the stressor
# (thermal, coldstart, wifiload, motion, idle); start one file per phase.
agent=${1:?usage: ntp_side_log.sh <agent> <phase>}; phase=${2:-run}
out="ntp_side_${agent}_${phase}_$(date -u +%Y%m%d_%H%M%S).csv"
zone=""
for z in /sys/class/thermal/thermal_zone*; do
  if grep -qi cpu "$z/type" 2>/dev/null; then zone="$z/temp"; break; fi
done
zone=${zone:-/sys/class/thermal/thermal_zone0/temp}
echo "t_utc,agent,phase,temp_c,load1,system_offset_s,last_offset_s,rms_offset_s,freq_ppm,resid_ppm,skew_ppm,root_delay_s,interval_s,np,span_s,stddev_s" > "$out"
echo "logging to $out  (temperature from $zone)"
g() { awk -F: -v k="$1" 'index($0,k)==1{print $2}' <<<"$tr" | awk '{v=$1; if ($0 ~ /slow/) v=-v; print v}'; }
while true; do
  t=$(date -u -Is); tr=$(chronyc tracking)
  ss=$(chronyc -c sourcestats 2>/dev/null | head -1)
  temp=$(awk '{printf "%.2f",$1/1000}' "$zone" 2>/dev/null)
  load=$(cut -d' ' -f1 /proc/loadavg)
  echo "$t,$agent,$phase,$temp,$load,$(g 'System time'),$(g 'Last offset'),$(g 'RMS offset'),$(g 'Frequency'),$(g 'Residual freq'),$(g 'Skew'),$(g 'Root delay'),$(g 'Update interval'),$(cut -d, -f2 <<<"$ss"),$(cut -d, -f4 <<<"$ss"),$(cut -d, -f8 <<<"$ss")" >> "$out"
  sleep 8
done
