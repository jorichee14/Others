# ros_image_to_rtsp

Stream a ROS 2 `sensor_msgs/Image` topic (default `/mobile_2/color/image_raw`)
as H.264 over RTSP, so the robot camera can be opened like the ED305 room cameras
(VLC, ffplay, OpenCV `VideoCapture`).

Encodings: `rgb8`, `bgr8`, `rgba8`, `bgra8`, `mono8`, `yuv422_yuy2` / `yuyv`, `uyvy`.

## Push to an RTSP server (default)

ffmpeg encodes and publishes to an RTSP server. Run [MediaMTX](https://github.com/bluenviron/mediamtx)
on the robot (one binary, listens on 8554), or push to an existing one.

```bash
sudo apt install ffmpeg                      # if missing in the container
# MediaMTX, once (arm64v8 on a Jetson, amd64 on a PC):
ARCH=$( [ "$(uname -m)" = aarch64 ] && echo arm64v8 || echo amd64 )
mkdir -p ~/mediamtx && cd ~/mediamtx
wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_${ARCH}.tar.gz
tar xzf mediamtx_v1.9.3_linux_${ARCH}.tar.gz
./mediamtx &                                 # on the robot, or skip and use --url below
cd -
source /opt/ros/humble/setup.bash
python3 ros_image_to_rtsp.py                 # -> rtsp://<robot-ip>:8554/mobile_2/color
python3 ros_image_to_rtsp.py --url rtsp://<server>:8554/mobile_2_color --width 960 --bitrate 2000
```

If the server is not up yet, the script retries every 2 s.

## Serve RTSP directly (no MediaMTX)

```bash
sudo apt install python3-gi gir1.2-gst-rtsp-server-1.0 \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-plugins-ugly
python3 ros_image_to_rtsp.py --mode serve --port 8554 --path mobile_2/color
```

On a Jetson, `--gst-encoder "nvvidconv ! video/x-raw(memory:NVMM) ! nvv4l2h264enc bitrate=4000000 insert-sps-pps=true"`
uses the hardware encoder instead of x264.

## View

```bash
ffplay -rtsp_transport tcp rtsp://<robot-ip>:8554/mobile_2/color
```

## Options

| option | default | |
|---|---|---|
| `--topic` | `/mobile_2/color/image_raw` | any Image topic, e.g. `/mobile_2/infra1/image_rect_raw` |
| `--fps` | 30 | output rate; the latest frame is sent every tick |
| `--bitrate` | 4000 | kbit/s |
| `--width` | as published | scale before encoding (height keeps aspect) |
| `--encoder` | `libx264` | push mode ffmpeg encoder |

The subscription is best-effort, so it connects to both reliable and
best-effort publishers.

Tested: push mode with a stand-in publisher (640x480 `rgb8` with padded rows) into
MediaMTX, read back at 30 fps, and recovery when the server starts late. Serve mode
is not tested yet.
