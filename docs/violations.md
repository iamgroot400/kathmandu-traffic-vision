# Violations

## Implemented: red-light running

| Element | Rule |
|---|---|
| Stop line | two points drawn once per camera (`watch.py --pick`) |
| Crossing | the vehicle's ground point (bottom centre of its box) moves from one side of the line segment to the other between frames; a point exactly on the line waits until it is clearly past |
| Direction | only crossings **into** the junction (`forward` in `config.json`) can be violations; vehicles leaving or reversing are counted but never flagged |
| Signal | debounced: a new colour must be seen in 3 consecutive frames |
| Grace | crossings in the first `--grace` seconds of red (default 1 s) are not flagged |
| Amber, unknown | never flagged |
| Evidence | the full frame with the vehicle boxed, plus time, track id, class, and seconds since red began |
| Once per vehicle | each track is counted once per direction |

Self-checks in [`test_logic.py`](../test_logic.py) cover a clear red-light run, green, the grace
period, reversing back over the line, a single misread frame, and on-line positions.

## Planned: restricted vehicles on inner roads

Trucks and tippers are not allowed on Kathmandu's inner roads during restricted hours (they use the
outer roads); buses and vans are allowed. A camera on an inner road gets a zone and banned hours in
its `config.json`, e.g. `"restricted": {"truck": ["07:00-20:00"]}`, and any banned class crossing its
line in that window is flagged. Needs: the official hours and the list of restricted roads. Because a
bus misread as a truck would be flagged wrongly, truck precision is evaluated on its own.

## Not planned yet

Helmet use, wrong-way driving, illegal turns and speeding are possible later; each needs its own
detector or calibration.
