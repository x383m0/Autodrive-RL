# Reward design guide

The reward is the ONLY thing that tells the car what you want. It never reads
your intentions, only the numbers. This guide explains how the reward in this
project works, what each term does, typical weights, how rewards go wrong, and
experiments to try.

## 1. How the reward is computed

Every 0.1 s the simulator measures what happened (metres advanced, speed vs
limit, did we hit something, ...). Each measurement becomes a **signal**, and
your YAML gives each term an on/off switch and a **weight**:

```
reward this step = sum over switched-on terms of  (+ or -) weight x signal
```

- **Bonuses** (+) add, **penalties** (-) subtract. The sign is fixed per term;
  you always write positive weights.
- **Per-step terms** are paid every step while something is true (speeding,
  being off the road...). Their total grows with time.
- **Event terms** are paid once (a red light, a collision, parking).
- The agent maximises the **sum** of rewards over the episode, slightly
  favouring rewards that come sooner (discount 0.99 per step, a ~10 s horizon).

The HUD (right panel in the videos) and `episodes.csv` show each term's
running total, so you can always see *where* the reward came from.

## 2. Think in a common currency

The car drives about 1 m per step at normal speed, so with `progress: 0.1` one
metre is worth 0.1 points and a 300 m trip about 30 points. Price everything
else in "metres of driving":

| Weight | Means |
|---|---|
| `collision: 50` | a crash is as bad as losing 500 m of driving |
| `red_light: 20` | running a red is worth 200 m |
| `speeding: 0.05` per m/s | 10 km/h over (2.8 m/s) costs 0.14/step, which is more than the 0.03/step of extra progress it buys, so speeding doesn't pay |
| `stall: 0.03` per step | standing still for no reason costs 0.3 points per second |

If a penalty is cheaper than the progress it saves, the car **will** break the
rule. That is not a bug in the car; it's the deal you offered it.

## 3. The terms

| Term | Kind | What it does | Balanced | Typical range | If too low | If too high |
|---|---|---|---|---|---|---|
| `progress` | + step | metres along the GPS route (overshooting the slot counts as going back) | 0.1 | 0.05-0.3 | no reason to go anywhere | rules become "cheap" |
| `goal_reached` | + event | parked in the slot | 30 | 10-100 | may give up at the lot | fine, but sparse (see 4.4) |
| `parking_precision` | + event | bonus x (1 - error) when parked | 15 | 0-30 | sloppy parking | — |
| `collision` | - event | hit car/parked car/building, ends episode | 50 | 20-200 | crashes are "worth it" | too timid, freezes in tight spots |
| `pedestrian_hit` | - event | ends episode | 100 | 100-500 | — | (keep it the worst thing) |
| `pedestrian_proximity` | - step | fast near people | 0.1 | 0.05-0.5 | brushes past people | crawls through crowds |
| `crosswalk_yield` | - event | drove over an occupied crosswalk | 10 | 5-30 | doesn't yield | — |
| `speeding` | - step | m/s over limit + tolerance | 0.05 | 0.02-0.3 | speeds | drives well under the limit |
| `red_light` | - event | crossed stop line on red | 20 | 10-100 | runs reds | brakes hard at every yellow |
| `stop_sign` | - event | didn't stop (< 0.5 m/s) | 10 | 5-50 | rolls stops | — |
| `lane_deviation` | - step | metres from the lane centre / planned path beyond deadband | 0.03 | 0.01-0.1 | wobbles, cuts corners | twitchy steering |
| `off_road` | - step | share of the car on sidewalk/grass | 0.2 | 0.1-1 | sidewalk shortcuts | — |
| `comfort` | - step | jerk + steering speed (each 0-1) | 0.01 | 0-0.05 | jerky ride | **sits still** (see 4.1) |
| `stall` | - step | stopped without a reason | 0.03 | 0.01-0.1 | parks on the road | creeps through red lights?* |
| `time` | - step | flat cost of existing | 0.005 | 0-0.02 | dawdles | rushes, risks more |
| `off_route` | - event | > 6 m from the route, ends episode | 20 | 10-50 | wanders off | — |
| `stuck` | - event | idle 12 s with no reason, ends episode | 20 | 10-50 | **freezes when in trouble** (4.7) | — |
| `speed_bonus` | + step | raw speed anywhere | off | off! | — | drives fast circles (see 4.3) |

\* `stall` never punishes legitimate waiting (red light, stop sign, car or
pedestrian ahead), so it is safe to raise. The simulator uses ground truth to
decide what's legitimate; rewards may use privileged information, but the
car's observations never do.

## 4. How rewards go wrong (all of these happened while building this project)

### 4.1 The penalty that teaches "do nothing" (comfort)

The first version scaled `comfort` as jerk/10. A brand-new policy moves the
controls randomly, so jerk was huge and `comfort` cost about as much per step
as driving earned. The car learned the one sure way to avoid it: **don't touch
the controls**. It crept along at 1 m/s and every episode ended "stuck".

Fix: scale every per-step signal to a known range (comfort is now 0-1 per
component). Rule of thumb: a per-step penalty should cost much less than the
per-step progress at normal driving, unless you really mean "never".

### 4.2 Sitting still is always an option

Doing nothing is a guaranteed, risk-free return of roughly zero. Any time the
agent's *estimate* of trying is worse than zero, it will freeze. Typical
causes: big crash penalties early in training (when the agent is bad at
driving, crashing is likely, so trying looks terrible), and no cost for
waiting. Counter-measures: a `stall` penalty for idling without a reason, a
little `time` cost, and a curriculum so the agent can succeed early.

### 4.3 Reward the outcome, not a proxy (speed_bonus)

`speed_bonus` pays for speed, which *sounds* like "get there quickly". But
speed in a circle pays exactly as well as speed towards the goal, and circles
have no red lights, pedestrians or parking. Turn it on with a high weight and
watch the car invent the cheapest way to be fast. `progress` measures what you
actually want: distance along the route.

### 4.4 Sparse rewards: the bonus the car never finds

The first 40-minute run never parked once in 3.7 million steps. Parking pays
+45, but only if the car stops at exactly the right spot after a tight turn.
Random exploration never got there, so the car never learned that parking was
worth anything. Stopping early in the aisle looked safest.

A reward that is never experienced teaches nothing. Fixes used here:

- **Reverse curriculum**: the first lessons start the car right in front of
  its slot (1-6 m away, 100% success after ~2 minutes), then further back each
  lesson. Measured: starting 2-40 m away gave <5% success after 8 minutes.
  Each lesson's start band must overlap the previous one only a little: with
  a wide band (1-16 m) the car passed on the easy starts alone and never
  learned the hard part (0 of 38 from 15-25 m).
- **Make the task learnable within the budget**: turning into a 90 or 60
  degree side slot (with no reverse gear to fix a crooked approach) stalled
  at 30-45% success after 45 minutes. The shipped lot uses bays at the end
  of the aisle, reached with a gentle S-curve: all four parking lessons
  passed in ~3 minutes. Sometimes the right fix is the task, not the reward.
- **Make "almost right" better than "very wrong"**: progress now counts
  driving past the slot as going backwards, so the best place to be is
  exactly at the slot. That makes stopping there the natural thing to do.
- **Shape with information the agent can see**: the speed limit tapers to
  walking pace in the last metres before the slot, and the speeding term
  then teaches a smooth stop.

### 4.5 Every part of the task needs a guiding signal

`lane_deviation` used to be switched off inside the parking lot. On the
streets it constantly nudged the car back to its path; in the lot nothing
did, until the very end (park or not). The car under-steered into the
parking bay, arrived crooked, and the failure came far too late to teach it
anything about the steering two seconds earlier. Dense signals (a little
reward/penalty every step) teach much faster than one verdict at the end.

### 4.6 Penalties that can be dodged

If a penalty only fires on an exact event, the agent learns to avoid the
*event*, not the behaviour. Example: a "red light" penalty that only checks
the instant you cross the line encourages stopping a few cm past it. Write
signals so that the cheapest way to avoid the penalty is the behaviour you
want.

### 4.7 "Freeze when in trouble": how an episode ends matters

In the traffic-rules stage, training went BACKWARDS: success fell from 24% to
0%. The car had learned to stop dead whenever it was about to mess up a turn.
Why: going off route cost -20, but standing still cost only a little `stall`
penalty, after which the episode was simply cut off ("stuck") with no
further cost. Freezing was the cheapest way out of every mistake.

Fix: being stuck is now a real failure that ends the episode with the
`stuck` penalty (same size as `off_route`). Rule: every way an episode can
fail should cost at least as much as the honest attempt that failed.

### 4.8 Ending the episode is also a reward

A terminating penalty (`collision`, `off_route`) also takes away all future
rewards. If the future is very negative (e.g. a huge time penalty), dying
early can look *good*. Keep per-step penalties small enough that a normal
drive still has a positive total.

### 4.9 Bigger penalties are not "safer" (the cautious preset)

`rewards/cautious.yaml` multiplies every safety penalty by 2-4. Trained for
the same 45 minutes as `balanced`, it never got out of the parking lessons
(balanced reached the traffic-rules stage). Early in training every attempt
is clumsy, so huge penalties mostly teach "trying is expensive". In the full
city it "crashed less" only because it gave up early: compare crash rates
together with how much of the route was completed.

### 4.10 Not every failure is the reward's fault

Training the shipped car longer (90 instead of 45 minutes) made it worse,
across all stages. That is an optimisation problem (PPO instability on a
long multi-stage run), not a reward design problem. If a run gets steadily
worse while the reward has not changed, suspect the learning setup: action
noise collapsing (we saw it fall to 0.05), learning rate, reward
normalisation drifting between curriculum stages. Per-stage model snapshots
(`experiments/<name>/stages/`) let you keep the best model anyway.

## 5. A recipe for building a reward from scratch

1. Start with `progress` + `goal_reached` + `collision`. Train with `--quick`
   to check it runs, then 20-40 minutes.
2. Watch a video and read `plots/reward_terms.png`. Identify the worst
   behaviour.
3. Add **one** term that targets it. Retrain under a new `--name`.
4. `python compare.py old new`. Did the behaviour improve? What got worse?
5. Repeat. Change one thing at a time, otherwise you can't tell which change
   did what.

## 6. Experiments to try

| Experiment | Change | What to look for |
|---|---|---|
| Minimal | only `progress` + `collision` + `goal_reached` | does it run red lights? (it can't know they matter) |
| Fear | `collision: 500` | slower learning, freezing in the lot (4.2) |
| No stall | `stall` off | does it learn to idle on the road? |
| Cheap rules | `red_light: 1`, `stop_sign: 1` | violations per km in `eval/summary.md` |
| Speed demon | `speed_bonus: 0.2` on | circles / speeding (4.3) |
| Smooth operator | `comfort: 0.05` | smoother ride, or sitting still? |
| School zones | `speed_tolerance_kmh: 0`, `speeding: 0.3` | "worst school-zone overspeed" |
| Strict | `terminate_on_red_light: true` | fewer reds, but slower learning? |
| Presets | `compare.py cautious aggressive` | the safety vs time trade-off |

## 7. Reading the results honestly

- **Return is not comparable between reward files**: a different reward means
  a different scale. Compare success rate and the metrics in
  `eval/summary.md` instead.
- **Use enough cities**: `evaluate.py` reports a 95% confidence interval for
  the success rate. With 100 cities a 10-point difference can still be luck;
  `compare.py` warns you when intervals overlap.
- **Watch the failure videos**: a metric tells you *that* something fails,
  the video tells you *why*.
