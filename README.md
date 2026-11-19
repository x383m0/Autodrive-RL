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
pip install -r requirements.txt

# 2-minute smoke test (checks that everything runs)
python train.py --reward rewards.yaml --name smoke --quick

# real training run (40 minutes on a 4-core CPU)
python train.py --reward rewards.yaml --name balanced

# test it on 100 cities it has never seen (+ failure videos)
python evaluate.py experiments/balanced

# compare two experiments side by side (report, charts, split-screen video)
python compare.py balanced cautious
```

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
| `videos/` | demo drives (mp4 with sound) |
| `eval/` | `summary.md`, `summary.json`, `episodes.csv`, `failures/*.mp4` |

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
| Curriculum | `autodrive/curriculum.py` | 11 stages that advance automatically (parking lessons, then streets, rules, traffic, pedestrians, full city) |
| Render / sound / video | `autodrive/render.py`, `audio.py`, `video.py` | top-down pygame view with FOV cones and HUD; soundtrack mixed offline; mp4 export |

Sound is never played live (training is silent and the code runs headless);
videos get a soundtrack mixed from what happened in each frame.

## Assets

Sprites and most sounds are CC0 Kenney assets; horn, brake squeal and city
ambience are synthesised. See `assets/CREDITS.md`. Replace any file in
`assets/` with your own (same name) - missing files fall back to generated
shapes/sounds.

## Results

See `RESULTS.md` for the measured numbers of the shipped models.
