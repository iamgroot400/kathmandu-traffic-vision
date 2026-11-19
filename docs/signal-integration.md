# Signal state: camera, timing cycle or controller

`Monitor.update(t, light, tracks)` takes the signal state as an input, so its source can change
without touching the violation logic.

| Source | How | Accuracy | Status |
|---|---|---|---|
| Camera box | HSV colour share inside a box drawn around the signal head ([`light.py`](../light.py)) | fails with glare, at night, or when a bus blocks the view | implemented |
| Timing cycle | fixed green / amber / red durations and an offset (`"cycle"` in `config.json`) | only as good as the cycle's timing | implemented |
| Controller | read the controller's lamp outputs (isolated sensor on the red lamp line -> microcontroller -> network) or its data port | exact; usable as evidence | planned |

## Controller feed, if the signal authority agrees

- Signals belong to the traffic authorities; this route works only in partnership with them.
- Lamp circuits run at mains voltage: sensing must be isolated (optocoupler or current clamp) and
  installed by a qualified electrician. Nothing is connected to the controller's logic.
- Both the controller events and the video frames must be time-stamped on one clock (NTP). With the
  default 1 s grace period, an offset of 100 ms does not change any decision.
- Code change: a `--signal-feed` option reading timestamped states from a serial port or the network,
  in place of `light_colour()`.
