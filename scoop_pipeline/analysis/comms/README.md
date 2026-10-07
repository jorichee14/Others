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
    comms_map.py <processed pass>      the RF map: NTP, RSSI, ping, iperf on the map's top view
                                       with the robot's path and the parked nodes (comms/maps/)
    csi_analysis.py <processed pass>   the CSI links: capture, frame types and the subcarriers
                                       they fill, path loss, delay spread, coherence vs speed
                                       (comms/csi/csi_report.json and plots)
    csi_view.py <processed pass>       pictures of the clean CSI per link: amplitude, phase and
                                       delay profile over time, snapshots (comms/csi/view/)
    link_clearance.py <processed pass> Fresnel clearance of each link's path through the anchored
      [--ap X Y Z]                     map against its RSSI / goodput, after the
    link_clearance.py --all <data/processed> [--ap-file F]
                                       distance trend (comms/clearance/; pooled over every
                                       pass in <data/processed>/clearance/)
