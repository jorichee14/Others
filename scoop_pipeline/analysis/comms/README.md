# analysis/comms

Wi-Fi / iperf / NTP / CSI analysis, on the tables processing/comms_tables.py writes
to data/processed/<date>/<pass>/comms/: ntp/ntp.csv, wifi/{wifi,ping,iperf}.csv,
csi/{csi,csi_status}.csv + csi_*.npz, summary.json; every row with its machine's
position in map.

    ntp_analysis.py <processed pass>   how well the clocks agreed (comms/ntp/ntp_report.json,
                                       ntp_offsets.png, ntp_frequency.png)
    wifi_analysis.py <processed pass>  the links: RSSI, PHY rates, ping per ICMP seq, iperf,
                                       contention (comms/wifi/wifi_report.json, wifi_timeline.png,
                                       wifi_map.png)
