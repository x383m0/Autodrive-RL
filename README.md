# Autodrive-RL

A small 2D "Tesla-style" self-driving project: a car learns to drive across a
procedurally generated city with **reinforcement learning** (PPO), obey speed
limits, traffic lights and stop signs, avoid cars and pedestrians, and park in a
target slot. The car is **not omniscient**: it only sees through ray-cast
"cameras" with limited fields of view (things behind a parked car are
invisible), plus GPS route waypoints.

The point of the project is the **reward**: you design it yourself in a YAML
file, switching terms on and off and choosing weights, then train and compare
what different rewards teach the car.

Everything (city, physics, traffic, pedestrians, sensors, renderer, sound) is
written from scratch in Python with Gymnasium + Stable-Baselines3 + pygame.
CPU only.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # Python 3.10-3.14; on 3.14 this installs pygame-ce

# 2-minute smoke test (checks that everything runs)
python train.py --reward rewards.yaml --name smoke --quick

# real training run (45 minutes on a 4-core CPU)
python train.py --reward rewards.yaml --name balanced

# watch a trained car drive LIVE in your browser (http://localhost:8000)
python watch.py --exp experiments/balanced_45min

# test it on 100 cities it has never seen
python evaluate.py experiments/balanced

# compare two experiments side by side (report + charts)
python compare.py balanced cautious
```

### Watching the car: live, no files

`watch.py` runs the car in a fresh random city after each other and streams
it to a web page on your own computer, with sound (click **Sound on**),
pause / speed / next-city / stage buttons, sensor and HUD toggles, and live
statistics. Nothing is saved to disk.

```bash
python watch.py --exp experiments/mine --follow     # watch WHILE it trains
```

`--follow` reloads the newest checkpoint as training saves it (every 250k
steps), so you can see the car improve. `--host 0.0.0.0` lets a phone or
another computer on your network watch too. The stream runs at 30 fps by
default; `--fps 60` is smoother (the side panel shows the real frame rate).

### Saving videos (optional, off by default)

mp4 files are only written when you ask for them:

| Command | Flag | Saves |
|---|---|---|
| `train.py` | `--video` | demo drives into `experiments/<name>/videos/` |
| `evaluate.py` | `--failure-videos 3` | 3 failed drives into `eval/failures/` |
| `compare.py` | `--video` | `side_by_side.mp4` |

Continue an interrupted run with `--resume`:

```bash
python train.py --name balanced --resume --minutes 20
```

## Designing your own reward

Copy the fill-in template, switch on what you want, train:

```bash
cp rewards/template.yaml rewards/mine.yaml
# edit rewards/mine.yaml: change enabled: false -> true and pick a weight
python train.py --reward rewards/mine.yaml --name mine
```

Only the terms you switch on count. Training prints the list of active terms
when it starts. Typos and layout mistakes are caught with a plain-English
message. Presets to start from:

| File | Idea |
|---|---|
| `rewards.yaml` / `rewards/balanced.yaml` | sensible defaults |
| `rewards/cautious.yaml` | safety first: big rule/pedestrian penalties, zero speed tolerance |
| `rewards/aggressive.yaml` | get there fast: big progress reward, cheap rules |
| `rewards/template.yaml` | everything off - build from scratch |

The 17 terms: `progress`, `goal_reached`, `parking_precision`, `collision`,
`pedestrian_hit`, `pedestrian_proximity`, `crosswalk_yield`, `speeding`,
`red_light`, `stop_sign`, `lane_deviation`, `off_road`, `comfort`, `stall`,
`time`, `off_route`, `speed_bonus`. Each one is explained in the YAML files and
in `autodrive/rewards.py`.

## What an experiment produces

`experiments/<name>/`

| File | What |
|---|---|
| `reward_config.yaml`, `config.yaml` | exactly what was trained |
| `model.zip`, `vecnormalize.pkl` | the trained policy (both are needed to run it) |
| `episodes.csv` | one row per training episode: outcome, violations, reward per term |
| `progress.csv`, `curriculum.json` | learning progress, when each stage was reached |
| `plots/` | `training_curves.png`, `outcomes.png`, `reward_terms.png` |
| `videos/` | demo drives (mp4 with sound), only with `--video` |
| `eval/` | `summary.md`, `summary.json`, `episodes.csv` (+ `failures/*.mp4` with `--failure-videos`) |

## How it works

| Piece | File | Notes |
|---|---|---|
| City | `autodrive/city.py` | grid of streets with random block sizes; lanes, parking strips, sidewalks, crosswalks, traffic lights, all-way stops, speed zones incl. school zones, parked cars, one parking lot |
| Physics | `autodrive/vehicle.py` | kinematic bicycle model; actions are steering + target speed (a cruise controller tracks it) |
| Traffic | `autodrive/traffic.py` | rule-following NPC cars (IDM car-following) with personalities: calm, normal, rusher (runs yellows), timid, erratic (sudden hard braking) |
| Pedestrians | `autodrive/pedestrians.py` | crosswalk users (obey walk signals), jaywalkers (some careless), kids hidden between parked cars who run out |
| Sensors | `autodrive/sensors.py` | 5 cameras (front wide / narrow-long / left / right / rear) + 12-beam ultrasonic ring, real occlusion; last 4 frames stacked |
| Environment | `autodrive/env.py` | Gymnasium env: observation, rule checks, termination |
| Rewards | `autodrive/rewards.py` | the configurable reward, per-term breakdown |
| Curriculum | `autodrive/curriculum.py` | 10 stages that advance automatically: 4 parking lessons (reverse curriculum), empty streets, traffic rules, light traffic, traffic, pedestrians, full city |
| Render / sound / video | `autodrive/render.py`, `audio.py`, `video.py` | top-down pygame view with FOV cones and HUD; soundtrack mixed offline; mp4 export |

Sound is never played live (training is silent and the code runs headless);
videos get a soundtrack mixed from what happened in each frame.

## Assets

Sprites and most sounds are CC0 Kenney assets; horn, brake squeal and city
ambience are synthesised. See `assets/CREDITS.md`. Replace any file in
`assets/` with your own (same name) - missing files fall back to generated
shapes/sounds.

## Results

Short version: the car learns to drive to the lot and park (37% of unseen
cities with lights and stop signs) within the 45-minute budget, but it does
**not** learn to handle traffic and pedestrians in that time (5% success in
the full city). See `RESULTS.md` for all measured numbers, the comparison of
two reward presets, and what went wrong along the way.

Shipped, ready to watch or evaluate:

```bash
python watch.py
python evaluate.py experiments/balanced_45min --n 100
python compare.py balanced_45min cautious
```

Watch it: `python watch.py` (it uses `experiments/balanced_45min` by default).
