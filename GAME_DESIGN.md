# Reward Designer - game design

A solo puzzle game in the browser. You never drive the car: you **design its
reward** (switch reward terms on, choose their weights), choose how long to
train, watch it learn live, and get a score from a hidden test on roads it
has never seen. Each level adds something new to the world and unlocks new
reward terms.

## Levels

| # | Level | World | Score |
|---|---|---|---|
| 1 | Park here | Just a parking pad on grass (no streets), start a few metres from the spot | parking precision |
| 2 | Straight road | A single empty straight road, parking spot at the end | speed |
| 3 | Bends | A single empty road with turns, spot at the end | speed |
| 4 | Into town | City streets with buildings and parked cars, no controls | speed |
| 5 | Rules of the road | + stop signs, traffic lights, school zones | speed + rule penalties |
| 6 | Rush hour | + normal pedestrians and normal traffic | speed + rule penalties |
| 7 | Chaos | + jaywalking pedestrians, reckless drivers | speed + rule penalties |
| 8 | Heavy traffic | + heavier traffic and trucks | speed + rule penalties |
| 9 | Night storm | Level 8's city at night in bad weather (shorter, noisier sensors). Only this level. | speed + rule penalties |
| 10 | Big roads | + roundabouts and highways | speed + rule penalties |
| 11 | Roadworks | + construction (cones, closed lanes, detours) | speed + rule penalties |

**Speed limits (game rule):** there are no speed limits except in school
zones. Going fast is the point; the challenge is going fast without crashing.

## Scoring (per test drive, averaged over the hidden test roads)

| Part | Points |
|---|---|
| Distance driven along the route | up to 400 |
| Parked in the spot | 400 |
| Quality, when parked: precision (level 1) or speed (levels 2+) | up to 200 |
| Rules (levels 5+): red light / rolled stop sign / crosswalk violation | -40 each |
| Rules (levels 5+): speeding in a school zone | -10 per km/h over |

Driving the whole way is worth about as much as parking itself. Stars (1-3)
for passing score thresholds; 1 star unlocks the next level.

## The reward workshop

- Every visit to a level starts with an **empty** reward list ("Load my best
  recipe" restores your best one).
- **+ Add reward** lists the level's built-in rewards, your custom rewards,
  and "Create custom reward...". No suggested numbers. 12 built-in rewards
  are available from level 1; rewards about things that appear later unlock
  in that level (with a "new" badge): speeding, red light, stop sign at
  level 5; pedestrian hit, pedestrian proximity, crosswalk yield at level 6.
- Number fields use styled -/+ buttons (step scales with the number).
- The number you type is **the points given** (any sign).
- **Custom rewards**: rule builder with AND-boxes joined by OR, a fixed or
  "x measurement" amount, paid every step or each time it becomes true;
  saved in a personal library (live link). Details:
  `.plans/reward-workshop-redesign-spec.md`.
- Before the first training of a level, a map of the level is shown next to
  the workshop (no button); after that, the live view takes its place. The
  live view has a fullscreen mode with progress, controls and the results popup.
- The level card shows the goal in large text and three SVG stars with the
  points each needs; stars turn gold when earned.
- Maps show no speed-limit signs (game rule: no limits except school zones).
- Zoom: the level map zooms (scroll, -/+ buttons, double-click; up to 300%)
  and pans by dragging. The live view zooms 50-300% (scroll on the picture or
  the -/+ buttons, also in fullscreen); the server renders the zoom, so it stays sharp.
- The live view is rendered at the viewer's real pixel size (fullscreen = the
  screen size, up to 2560x1440), so nothing is a stretched picture. While the
  zoom is changing the old frame is scaled for a moment, then the ground is
  redrawn at the new scale.
- "Stop and test now" asks training to stop through a `STOP` file in the run
  folder, so the car is saved on every OS (Windows included). If training was
  killed anyway, the last checkpoint is tested.

## Level page template (all current and future levels)

Every level page is built from the same template - new levels only need an
entry in `autodrive/levels.py`:

- the level goal in large text, then three SVG stars with the points each
  needs (empty outline until earned, gold once earned);
- no rule bullet points;
- the level's map shown right away next to the workshop (until the first
  training of that level, then the live view takes its place);
- the map shows **only what the level contains**: level 1 is just the
  parking pad, levels 2-3 are cropped to their road, city levels show the
  town because the town is part of the level. A future level that only uses
  part of the world must likewise show only that part;
- no speed-limit signs except where a limit really applies (school zones).

## Training

- The player chooses the training time.
- `python game.py --debug` unlocks every built level (for testing); progress is still saved.
- Single-road levels (2-3): every practice drive starts at the beginning of
  the road and drives all of it; the curriculum only makes the road longer
  (level 2: short -> medium -> full; level 3: short straight -> medium with
  1 bend -> full with 2 bends). Measured with the balanced reward: level 2
  reaches the full road in ~4.5 min and parks ~87% of practice drives by
  8 min; level 3 reaches the full road at ~6.5 min and parks ~77% by 12 min.
  Default training times: 8 and 10 min.

### Balance check (balanced reward, default time, the game's hidden 12-drive test, one run each)

| Level | Default time | Test score | Stars | Test drives |
|---|---|---|---|---|
| 1 Park here | 3 min | 908 | 3 | 11 parked, 1 off route |
| 2 Straight road | 8 min | 853 | 3 | 11 parked, 1 stuck |
| 3 Bends | 10 min | 813 | 2 | 10 parked, 2 collisions |
| 4 Into town | 14 min | 784 | 2 | 10 parked, 2 stuck (10 min gave 580 = barely 1 star) |
| 5 Rules of the road | 15 min | 668 | 2 | 9 parked, 1 stuck, 1 off route, 1 collision |

Every level is winnable in its default time with a sensible reward; a
well-designed reward should reach 3 stars.
- The level's curriculum runs automatically behind the scenes (e.g. starting
  close to the spot first); the player only designs the reward.
- **Start from**: a new car, *continue my last car here*, *continue my best car
  here*, or *my best car from the previous level*. After every training (good
  result or not) the results offer **Continue training this car**.
  Continuing keeps everything the car learned (weights, sensor normalisation
  and, on the same level, its curriculum stage); only the rewards (and time)
  can change. "use its rewards" loads the rewards that car was trained with.
  Each run stores `meta.json` (parent car, total minutes trained, score, recipe).

## Build order

1. Game shell + levels 1-5 (these train in minutes today).
2. Levels 6-8 once traffic training works (stability fix in progress).
3. Level 9 (night / weather), then 10 (roundabouts, highways), then 11 (construction).
