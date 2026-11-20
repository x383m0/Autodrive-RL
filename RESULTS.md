# Results

All numbers below were measured in this repository on a 4-core CPU
(no GPU), Python 3.13, ~1,450 environment steps per second with 8 parallel
environments. Evaluation always uses cities the car has never seen (seeds
from 50,000,000; training draws seeds below 10,000,000) and a deterministic
policy.

## Update: training stability fix (`experiments/stable`)

`train.py` now uses a fixed reward scale instead of running reward
normalisation, learning rate 2e-4 and PPO's `target_kl` 0.02. One 45-minute
run (only ~1.9M steps: the CPU was shared with other work):

- reached **S5 pedestrians** (every earlier run collapsed at traffic) and kept
  36-66% success there without collapsing;
- **full city, 100 unseen cities: 23% parked** (95% CI 16-32%), up from 5%
  (CI 2-11%) for `balanced_45min`. Still 34% collisions and 30% pedestrian
  hits, so it is far from a good driver - but traffic is now learnable.

| Full city (100 unseen) | balanced_45min | stable |
|---|---|---|
| Parked | 5% | **23%** |
| Collision | 54% | 34% |
| Pedestrian hit | 37% | 30% |
| Average route completed | 43% | 51% |

`python watch.py --exp experiments/stable` to watch it.

## Summary (original 45-minute runs)

| | Honest verdict |
|---|---|
| Simulator, sensors, rewards, curriculum, renderer, sound, videos, evaluation, comparison | **Work.** Tested end to end. |
| Learning to **park** (reverse curriculum, 4 lessons) | **Works**: all 4 parking lessons pass in ~3 minutes of training. |
| Learning to **drive a route through empty streets and park** | **Works, roughly**: stage passed (70% success during training). |
| Learning **traffic lights + stop signs** | **Partly**: 37% of unseen cities parked, but ~2 red lights and ~4 rolled stop signs per km. |
| Learning **traffic and pedestrians** (the full city) | **Not achieved within the 45-minute budget**: 5% success in the full city. |

So: it's a real, working RL driving sandbox, and the shipped car really does
learn to drive to the lot and park, but it is **not** a competent city driver.
The full curriculum (traffic + pedestrians) did not train successfully in
45 minutes, and longer training made things worse (see below).

## Shipped model: `experiments/balanced_45min`

`python train.py --reward rewards.yaml --name balanced` (45 minutes, 3.9M steps).

Curriculum timeline (stage reached at step):

| Stage | Reached at |
|---|---|
| P1 park from 1-6 m | 0 |
| P2 park from 4-15 m | 52,568 |
| P3 park from 12-30 m | 82,680 |
| P4 park from 25-70 m | 126,568 |
| S1 empty streets | 210,272 |
| S2 traffic rules | 2,055,272 (~22 min) |
| S3 light traffic | not reached (S2 success hovered 25-50%, needs 65%) |

### Evaluation on 100 unseen cities at the level it trained for (S2: lights + stop signs, no other cars/people)

`experiments/balanced_45min/eval_traffic_rules/summary.md`

| Metric | Value |
|---|---|
| Parked successfully | **37%** (95% CI 28-47%) |
| Parked with zero rule violations | 4% |
| Collision (mostly parked cars / lot) | 31% |
| Left the route | 23% |
| Stuck | 9% |
| Average route completed | 95% |
| Red lights run per km | 2.16 |
| Stop signs rolled per km | 3.88 |
| Time spent speeding | 10.8% |
| Average speed | 15.0 km/h |
| Parking error (0 = perfect) | 0.38 |
| Time to park | 47.8 s |

### Evaluation on 100 unseen FULL cities (traffic, pedestrians, kids, school zones)

`experiments/balanced_45min/eval/summary.md`

| Metric | Value |
|---|---|
| Parked successfully | **5%** (95% CI 2-11%) |
| Collision | 54% |
| Pedestrian hit | 37% |
| Average route completed | 43% |
| Red lights / stop signs rolled per km | 2.82 / 7.44 |
| Crosswalk violations per km | 2.38 |

It has never trained with other cars or pedestrians, and it shows.

### Videos (mp4 with sound)

These were removed from the working tree to save disk space (they are in the
git history: `git checkout 9ce6079 -- experiments/balanced_45min/videos`).
Watch the same model live instead with `python watch.py`.

- `experiments/balanced_45min/videos/demo_traffic_rules_1_seed50000097.mp4` and `_2_seed50000073.mp4`:
  successful drives + parks with no red-light/stop-sign violations (picked from the
  37% that succeeded, so they show the best case)
- `experiments/balanced_45min/videos/demo_full_city_1_seed50000003.mp4`: the full city
  (ends in a pedestrian hit)
- `experiments/balanced_45min/eval/failures/`: collision, pedestrian hit, off route

## Comparison: `balanced` vs `cautious` (both 45 minutes)

`experiments/compare/balanced_45min_vs_cautious/report.md` (+ `curves.png`,
`metrics.png`, `side_by_side.mp4`).

The cautious preset (2-4x larger safety penalties, zero speed tolerance) did
**not** get out of the parking lessons: it stalled in "P4 park from 25-70 m"
at ~10-20% success. In the full city it parked 0%, left the route 63% of the
time, and completed 13% of routes on average. Its lower collision and
pedestrian-hit *rates* (25% / 12% vs 54% / 37%) are an artefact of driving
much less, which the report now points out automatically. Lesson: big
penalties early in training make learning harder (REWARD_GUIDE 4.2).

## What longer training did

Continuing the shipped run with `--resume` for another 45 minutes (90 minutes,
7.7M steps total) made it **worse**: S2 success fell from ~40% to ~3%, and
the replayed earlier stages collapsed too (parking ~13%). Two earlier runs
showed the same collapse in the traffic stage. I did not ship that model. I
tried re-injecting exploration noise and a gentler first traffic stage; these
did not fix it. It looks like a PPO stability problem on long multi-stage
runs (constant learning rate, reward normalisation drifting between stages)
rather than a single reward bug, and it is the main open issue.

## What it took to get here (all measured, details in REWARD_GUIDE.md)

| Attempt | Result | Fix |
|---|---|---|
| raw pedal action | car stood still 70% of the time | target-speed action (cruise controller) |
| comfort = jerk/10 | "don't touch the controls" | comfort normalised to 0-1 |
| first full 40-min run | parked **0 times in 3.7M steps** | reverse curriculum (start next to the slot) |
| overshooting the slot was free | stopped early or overshot | progress counts past-the-slot as backwards |
| wide start bands (1-16 m) | passed lessons on easy starts only | narrow, overlapping bands |
| 90 deg / 60 deg side slots | stalled at 30-45% after 45 min | end-of-aisle bays (passed in ~3 min) |
| lane_deviation off in the lot | under-steered into the bay | path-following penalty everywhere |
| stuck = timeout | "freeze when in trouble", S2 regressed 24% -> 0% | `stuck` term ends the episode |
| action noise collapsed to 0.05 | could not learn traffic | noise re-injected for new actors (did not suffice) |

## Assets

Drawn in code: cars (`autodrive/carart.py`) and pedestrians (`autodrive/pedart.py`).
Real (downloaded, CC0 Kenney): tree and skid sprites; engine,
crash (metal + glass), pedestrian thud, turn-signal tick, light-change ping,
parking chime, wind. Synthesised (`tools/make_sounds.py`): horn, brake squeal,
city ambience. kenney.nl, Freesound, Pixabay, Mixkit and OpenGameArt were
blocked by the network; the Kenney packs came from a GitHub mirror.

## Not done

- `SPEC.md` was not in the repository; everything was built from the summary
  in the request.
- The reward-hacking detector, the `broken_on_purpose` preset and the scripted
  scenario test suite were deferred at your request.
- Turning into side slots was replaced by end-of-aisle bays (see above).

## Training speed (profiling, Nov 2026)

Where the time goes (PPO, 8 workers on 4 CPU cores, rush-hour stage): ~86%
collecting experience, ~13% network updates. The simulation alone runs ~1,600
steps/s on 4 processes, but collection reaches only about half of that: the 8
environments step in lockstep (everyone waits for the slowest step or reset)
plus the main process's work between steps.

| Change | Effect on learning | Speed |
|---|---|---|
| `Path.pos` via `np.interp`, squared-distance threshold checks | none: same seeds give identical drives (max diff 1.8e-15) | env +13-20% |
| one OpenBLAS thread per worker | none | small (+~5%) |
| together, measured in real PPO training (rush hour) | none | **650 -> 796 steps/s (+22%)** |
| traffic cars see the agent's whole body (not only its centre) | fewer unfair crashes (22 -> 15 per 60 drives) - makes traffic learnable | - |
| PyTorch threads 1/2, 4-12 environments | none | within noise (690-720 steps/s) |
| PPO epochs 6 -> 10 (rejected) | **worse**: level 2, 8 min, 2 seeds: 919/874 -> 383/769 points | - |

With the faster simulation, level 2 (8 min) collected 557k-707k steps instead
of 464k and scored 919 and 874 (3 stars) vs 853 before.
