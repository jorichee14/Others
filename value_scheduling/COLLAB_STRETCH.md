# Short-term collaboration: a robot that streams to a server (e.g. Stretch)

Their setup: a mobile manipulator (Stretch, RealSense RGB-D on the
head and wrist) streams sensor data over Wi-Fi to a server that runs
the perception or policy model and sends commands back. Their
constraint: the loop sense → send → server compute → act has a
deadline, and the link and the server's queue both eat into it; under
lab Wi-Fi the stream is late or dropped and the robot stalls or acts
on stale frames.

What we bring is exactly the layer between their sensor and their
model: measuring where the time goes, sending only what the server
does not already have, choosing the representation the deadline
allows, and scoring at the moment the robot acts. Three packages, by
depth; each is self-contained and usable with their robot, their task
and their server unchanged.

## Package A: instrument and diagnose (two weeks)

- chrony on robot and server; stamp every message at send, at server
  receive, after model inference, at command receive on the robot.
- The robot's existing link monitor (RSSI, RTT, goodput) logged beside
  the stream on the same clock (our `link_trace` tooling).
- One figure: per frame, time in the link, time in the server queue,
  time in inference, time back; and the fraction of frames that
  arrived after the action they were meant for.
- Their stream replayed through our logged channel (`channel.py`) with
  the lab's measured link to show what a quieter or busier link would
  do, without re-running the robot.

Deliverable: the budget breakdown and the in-time fraction. Most
groups discover that the server queue, not the radio, is the larger
term; either way the next package is chosen from this figure.

## Package B: send what changed, in the form the deadline allows (one to two months)

- **Validity-based sending.** The server keeps the last RGB-D it
  processed (the "twin" of the workspace). The robot sends only the
  regions that changed against the server's copy, on change, not on a
  timer; a static scene costs almost nothing, an arm moving costs a
  crop. Same mechanism as our store with a lifetime, with the
  workspace as the map.
- **A representation ladder with a deadline.** Full RGB-D → a cropped
  region of interest → object keypoints or a mask → a one-line state
  ("grasp target unchanged"). The robot picks the rung whose transfer
  plus inference finishes before the action's deadline, from the
  measured link and the server's queue length. Later rungs cost more
  robot compute and less airtime: the slot-aware encode trade.
- **Deadline on the model side.** The server processes frames in
  deadline order rather than arrival order, and drops a frame whose
  action has already passed instead of queueing it.

Deliverable: their task's success rate and action latency with and
without, at equal link conditions; bytes on the air; the frames that
would have been wasted.

## Package C: score at the decision (a joint paper)

- Define, for their task, the places and times the robot's action
  depends on (the grasp target at the moment of closing; the path cell
  at the moment of committing speed) and score the stream by whether
  the server's picture of those was correct then (AoII at the
  decision), rather than by frame rate or latency.
- Show that this number predicts their task outcomes where latency and
  bandwidth do not, and that the ladder of package B keeps it under
  tolerance at a fraction of the bytes.
- If their server serves several robots, the fetch scheduler of our
  phase 3 applies unchanged: windows from each robot's deadline and the
  scene's change rate, one machine, EDF with downgrade.

Deliverable: a shared paper with their task, our metric and
mechanism, measured on their robot over real Wi-Fi.

## What we need from them

The robot and server for a few sessions; their stream's topics and
message sizes; their model's inference time; the task and its success
criterion; permission to put chrony and a logger on both ends. Nothing
in their model or policy changes in A or B.

## What they need to know

The gain comes from not sending what the server already has and from
matching the form of what is sent to the time left, not from a better
codec; and the right score is whether the robot's picture was correct
when it acted, not how many frames per second arrived.
