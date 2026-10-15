#!/usr/bin/env python3
"""
Stream a ROS 2 sensor_msgs/Image topic as H.264 over RTSP.

Default topic: /mobile_2/color/image_raw (RealSense colour on the robot).

Two ways to get an RTSP stream out:

  --mode push   (default)  pipe the frames into ffmpeg, which publishes to an
                RTSP server you run next to it (MediaMTX, as on the ED305
                camera hosts).  Viewers read  rtsp://<server>:8554/<path>.
                Needs: ffmpeg.

  --mode serve  this script is the RTSP server itself (GStreamer
                gst-rtsp-server).  Viewers read  rtsp://<robot>:<port>/<path>.
                Needs: python3-gi, gir1.2-gst-rtsp-server-1.0,
                gstreamer1.0-plugins-{base,good,ugly} (x264enc).

Frames go out at a fixed --fps: the latest received frame is sent on every
tick (repeated if the topic is slower), so the RTSP clock stays steady.

Examples
  python3 ros_image_to_rtsp.py
  python3 ros_image_to_rtsp.py --url rtsp://192.168.1.56:8554/mobile_2_color
  python3 ros_image_to_rtsp.py --mode serve --port 8554 --path mobile_2/color
  python3 ros_image_to_rtsp.py --topic /mobile_2/infra1/image_rect_raw --fps 30
"""
import argparse
import shlex
import subprocess
import sys
import threading
import time

import numpy as np
import rclpy
from rclpy.node import Node
try:
    from rclpy.executors import ExternalShutdownException
except ImportError:                      # older rclpy
    ExternalShutdownException = KeyboardInterrupt
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

# ROS encoding -> (bytes per pixel, ffmpeg pix_fmt, GStreamer format)
ENCODINGS = {
    'rgb8': (3, 'rgb24', 'RGB'),
    'bgr8': (3, 'bgr24', 'BGR'),
    'rgba8': (4, 'rgba', 'RGBA'),
    'bgra8': (4, 'bgra', 'BGRA'),
    'mono8': (1, 'gray', 'GRAY8'),
    '8UC1': (1, 'gray', 'GRAY8'),
    'yuv422_yuy2': (2, 'yuyv422', 'YUY2'),
    'yuyv': (2, 'yuyv422', 'YUY2'),
    'uyvy': (2, 'uyvy422', 'UYVY'),
    'yuv422': (2, 'uyvy422', 'UYVY'),   # ROS 'yuv422' is UYVY byte order
}


def frame_bytes(msg):
    """Image message -> tightly packed pixel bytes (drops any row padding)."""
    bpp = ENCODINGS[msg.encoding][0]
    row = msg.width * bpp
    if msg.step == row:
        return bytes(msg.data)
    data = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    return data[:, :row].tobytes()


class FfmpegPush:
    """Raw frames on stdin -> ffmpeg -> H.264 -> RTSP server (push)."""

    def __init__(self, args, log):
        self.a, self.log = args, log
        self.proc = None
        self.fmt = None          # (width, height, encoding) ffmpeg was started for

    def _start(self, w, h, enc):
        a = self.a
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'warning',
               '-f', 'rawvideo', '-pix_fmt', ENCODINGS[enc][1],
               '-s', f'{w}x{h}', '-framerate', str(a.fps), '-i', '-']
        if a.width:
            cmd += ['-vf', f'scale={a.width}:-2']
        cmd += ['-c:v', a.encoder]
        if a.encoder == 'libx264':
            cmd += ['-preset', 'ultrafast', '-tune', 'zerolatency']
        cmd += ['-pix_fmt', 'yuv420p', '-g', str(a.fps), '-bf', '0',
                '-b:v', f'{a.bitrate}k', '-maxrate', f'{a.bitrate}k',
                '-bufsize', f'{a.bitrate}k',
                '-f', 'rtsp', '-rtsp_transport', 'tcp', a.url]
        self.log.info('starting: ' + ' '.join(shlex.quote(c) for c in cmd))
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        self.fmt = (w, h, enc)

    def _stop(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None

    def send(self, w, h, enc, data):
        if self.proc is not None and (self.fmt != (w, h, enc) or self.proc.poll() is not None):
            if self.proc.poll() is not None:
                self.log.warning(f'ffmpeg exited ({self.proc.returncode}); is the RTSP '
                              f'server at {self.a.url} running? retrying in 2 s')
                time.sleep(2)
            else:
                self.log.info(f'image format changed to {w}x{h} {enc}, restarting ffmpeg')
            self._stop()
        if self.proc is None:
            self._start(w, h, enc)
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, OSError):
            pass                 # picked up as an exited process on the next frame

    def close(self):
        self._stop()


class GstServe:
    """This process is the RTSP server: appsrc -> encoder -> rtph264pay."""

    def __init__(self, args, log):
        import gi
        gi.require_version('Gst', '1.0')
        gi.require_version('GstRtspServer', '1.0')
        from gi.repository import Gst, GstRtspServer, GLib
        Gst.init(None)
        self.Gst, self.GstRtspServer, self.GLib = Gst, GstRtspServer, GLib
        self.a, self.log = args, log
        self.fmt = None
        self.srcs = []
        self.lock = threading.Lock()
        self.loop = None

    def _start(self, w, h, enc):
        a, Gst, R = self.a, self.Gst, self.GstRtspServer
        scale = f'videoscale ! video/x-raw,width={a.width},pixel-aspect-ratio=1/1 ! ' if a.width else ''
        encoder = a.gst_encoder or (
            f'x264enc tune=zerolatency speed-preset=ultrafast bitrate={a.bitrate} '
            f'key-int-max={a.fps} bframes=0 ! video/x-h264,profile=baseline')
        launch = (f'( appsrc name=src is-live=true format=time do-timestamp=true '
                  f'caps=video/x-raw,format={ENCODINGS[enc][2]},width={w},height={h},'
                  f'framerate={a.fps}/1 ! videoconvert ! {scale}'
                  f'video/x-raw,format=I420 ! {encoder} ! '
                  f'rtph264pay name=pay0 pt=96 config-interval=1 )')
        self.log.info('pipeline: ' + launch)

        factory = R.RTSPMediaFactory()
        factory.set_launch(launch)
        factory.set_shared(True)
        factory.connect('media-configure', self._on_configure)
        server = R.RTSPServer()
        server.set_service(str(a.port))
        server.get_mount_points().add_factory('/' + a.path.strip('/'), factory)
        server.attach(None)
        self.loop = self.GLib.MainLoop()
        threading.Thread(target=self.loop.run, daemon=True).start()
        self.fmt = (w, h, enc)
        self.log.info(f'serving rtsp://<this-host>:{a.port}/{a.path.strip("/")}')

    def _on_configure(self, factory, media):
        src = media.get_element().get_by_name('src')
        with self.lock:
            self.srcs.append(src)
        media.connect('unprepared', lambda m: self._drop(src))
        self.log.info('client connected')

    def _drop(self, src):
        with self.lock:
            if src in self.srcs:
                self.srcs.remove(src)
        self.log.info('client gone')

    def send(self, w, h, enc, data):
        if self.fmt is None:
            self._start(w, h, enc)
        elif self.fmt != (w, h, enc):
            self.log.error(f'image format changed from {self.fmt} to {(w, h, enc)}; '
                           'restart the script', throttle_duration_sec=5.0)
            return
        with self.lock:
            srcs = list(self.srcs)
        for src in srcs:
            src.emit('push-buffer', self.Gst.Buffer.new_wrapped(data))

    def close(self):
        if self.loop is not None:
            self.loop.quit()


class ImageToRtsp(Node):
    def __init__(self, args):
        super().__init__('image_to_rtsp')
        self.a = args
        self.out = (GstServe if args.mode == 'serve' else FfmpegPush)(args, self.get_logger())
        self.latest = None       # (w, h, encoding, bytes, receive time)
        self.lock = threading.Lock()
        self.running = True
        self.create_subscription(Image, args.topic, self._on_image, qos_profile_sensor_data)
        self.get_logger().info(f'waiting for {args.topic}')
        self.thread = threading.Thread(target=self._writer, daemon=True)
        self.thread.start()

    def _on_image(self, msg):
        if msg.encoding not in ENCODINGS:
            self.get_logger().error(f'unsupported encoding {msg.encoding!r} '
                                    f'(supported: {", ".join(ENCODINGS)})',
                                    throttle_duration_sec=5.0)
            return
        frame = (msg.width, msg.height, msg.encoding, frame_bytes(msg), time.monotonic())
        with self.lock:
            first = self.latest is None
            self.latest = frame
        if first:
            self.get_logger().info(f'first frame: {msg.width}x{msg.height} {msg.encoding}')

    def _writer(self):
        period = 1.0 / self.a.fps
        nxt = time.monotonic()
        while self.running:
            with self.lock:
                frame = self.latest
            if frame is not None:
                w, h, enc, data, t = frame
                if time.monotonic() - t > 2.0:
                    self.get_logger().warning(f'no new image on {self.a.topic} for '
                                           f'{time.monotonic() - t:.0f} s, repeating the last one',
                                           throttle_duration_sec=5.0)
                self.out.send(w, h, enc, data)
            nxt += period
            delay = nxt - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                nxt = time.monotonic()   # fell behind (encoder or network): don't burst

    def close(self):
        self.running = False
        self.thread.join(timeout=2)
        self.out.close()


def parse(argv):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--topic', default='/mobile_2/color/image_raw')
    p.add_argument('--mode', choices=['push', 'serve'], default='push')
    p.add_argument('--url', default='rtsp://localhost:8554/mobile_2/color',
                   help='push: RTSP server URL to publish to')
    p.add_argument('--port', type=int, default=8554, help='serve: port to listen on')
    p.add_argument('--path', default='mobile_2/color', help='serve: stream path')
    p.add_argument('--fps', type=int, default=30, help='output frame rate')
    p.add_argument('--bitrate', type=int, default=4000, help='kbit/s')
    p.add_argument('--width', type=int, default=0,
                   help='scale to this width before encoding (0 = as published)')
    p.add_argument('--encoder', default='libx264', help='push: ffmpeg video encoder')
    p.add_argument('--gst-encoder', default='',
                   help='serve: GStreamer encoder chain from I420 to H.264, e.g. on a '
                        'Jetson "nvvidconv ! video/x-raw(memory:NVMM) ! '
                        'nvv4l2h264enc bitrate=4000000 insert-sps-pps=true"')
    return p.parse_args(argv)


def main():
    rclpy.init(args=sys.argv)
    args = parse(rclpy.utilities.remove_ros_args(sys.argv)[1:])
    node = ImageToRtsp(args)
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
