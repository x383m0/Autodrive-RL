# Reward workshop redesign + custom rewards + fullscreen live view

Spec from the interview of 2026-11-19. Applies to the Reward Designer game
(`game.py`, `assets/game.html`) and, where noted, to the whole project.

## 1. Goals

- Every level starts with **no rewards**. The player builds the reward from
  nothing, adding one reward at a time, like assembling a toolkit.
- **No default or suggested values** anywhere in the game UI.
- Players can create **custom rewards** with a safe rule builder, keep them
  in a personal library, and use them in any level (and from YAML / train.py).
- The live view is **hidden until training starts** in a level, and can be
  shown **fullscreen** with all controls.

## 2. Reward list (workshop)

### 2.1 Starting state
- Opening a level (first visit **and** every return visit) shows an **empty
  reward list**.
- A **"Load my best recipe"** button (only if the level has a best result)
  fills the list with the rewards of the best-scoring attempt.
- The previous "restore last recipe automatically" behaviour is removed.

### 2.2 Adding rewards
- One **"+ Add reward"** button opens a picker (modal/dropdown list).
- The picker lists:
  1. all **built-in rewards** (all 18, available in every level - per-level
     unlocking is removed);
  2. the player's **custom rewards** from the library;
  3. a **"Create custom reward..."** entry that opens the rule builder.
- Rewards already in the list are shown as added (can't be added twice).
- Each built-in entry shows its **plain-English explanation** and how it is
  paid (*every step* or *once when it happens*), but **no number**.
- No limit on how many rewards a level can use.

### 2.3 A reward row in the list
- Name, explanation (collapsible), how it's paid, a **number field** (empty
  until the player types), and a remove (x) button.
- The number field accepts **any real number, positive or negative**.
- A row with an empty number is invalid; Train highlights it.

### 2.4 Meaning of the number ("points given")  -- applies EVERYWHERE
- The number is **exactly the points the car receives** when the reward
  fires: `collision: -50` punishes crashing, `collision: +50` rewards it.
- Per-step rewards: points per step x the measured amount (e.g. progress:
  points per metre; speeding: points per m/s over the limit).
- No hidden signs. The `TERMS` sign table in `autodrive/rewards.py` is
  removed from the computation (kept only to describe defaults/presets).
- **YAML format version 2:** `format: 2` at the top; weights are signed
  points. Presets (`rewards.yaml`, `rewards/*.yaml`), the template and
  `REWARD_GUIDE.md` are rewritten in this form (penalties become negative).
- **Backwards compatibility:** a YAML file without `format: 2` is treated as
  the old format (positive magnitudes, fixed directions) and converted on
  load (penalty terms negated). Shipped experiments
  (`experiments/balanced_45min`, `cautious`) keep working unchanged.
- The "negative weight" error is removed.

### 2.5 Training with nothing
- Pressing Train with **no rewards** shows a confirm: "No rewards: the car
  will learn nothing. Train anyway?" If confirmed, training runs (all
  rewards zero).
- `load_reward_config` must accept an empty reward (currently an error) when
  called from the game; the CLI keeps a warning instead of an error.

## 3. Custom rewards (rule builder)

### 3.1 Shape of a custom reward
```
NAME:   "slow near people"
WHEN:   any of these groups is true
          group 1: ALL of  [nearest pedestrian < 6 m] [speed > 15 km/h]
          group 2: ALL of  [crashed]
GIVE:   -0.2            (fixed points)
   or   -0.05 x speed   (points x a measurement)
PAID:   every step  |  each time it becomes true
```
- **Conditions:** `measurement  operator  value`, operators `<  <=  >  >=
  =  !=`; event/flag measurements use `is true / is false`.
- **AND/OR:** conditions are grouped. The reward fires if **any group** is
  true; a group is true if **all** its conditions are true. Shown as boxes
  ("All of" inside, "or" between boxes).
- A reward with **no conditions** fires always (useful with "x measurement").
- **Amount:** a fixed number (any sign), or a number **x one measurement**.
- **Timing (player chooses):**
  - *every step* - paid each 0.1 s while the condition holds;
  - *each time it becomes true* - paid on every false -> true transition.

### 3.2 Measurements (units the player sees)
| Group | Measurement | Unit / type |
|---|---|---|
| Driving | speed | km/h |
| | acceleration | m/s^2 (negative = braking) |
| | steering | -100..100 % (+ = right) |
| | distance driven (this drive) | m |
| | distance to the parking spot (along the route) | m |
| | time since start | s |
| Position | offset from lane centre / planned path | m (absolute) |
| | heading error vs the route | degrees (absolute) |
| | off road | true/false |
| | in parking lot | true/false |
| Surroundings | nearest car distance | m (from sensors; 999 if none seen) |
| | nearest pedestrian distance | m (from sensors; 999 if none seen) |
| | nearest obstacle straight ahead | m (front camera rays) |
| Events | crashed, hit pedestrian, parked, ran red light, rolled stop sign, went off route, got stuck | true on the step it happens |

Surroundings use the car's **sensor readings**, not ground truth, so a hidden
kid counts only once seen (honest to the "not omniscient" rule).

### 3.3 Library
- Custom rewards are saved in a **personal library** (`games/custom_rewards.json`).
- Library rewards appear in the "+ Add reward" picker of **every level**.
- Edit / rename / delete from the picker. **Live link:** editing a library
  reward changes it everywhere it is used, from the next training on.
- Reproducibility is kept anyway: each training run stores a full snapshot of
  the rewards it used (`experiments/game/<run>/my_reward.yaml`).

### 3.4 YAML / train.py
- A `custom:` list in the format-2 YAML, same structure as the builder:
```yaml
format: 2
terms:
  collision: {enabled: true, weight: -50}
custom:
  - name: slow near people
    when: [[{m: nearest_pedestrian_m, op: "<", v: 6}, {m: speed_kmh, op: ">", v: 15}]]
    give: -0.2
    times: null          # or a measurement name to multiply by
    paid: every_step     # or on_change
```
- Parsed and evaluated by a small safe interpreter (no `eval`, whitelisted
  measurement names and operators). Validation errors in plain English.
- Each custom reward gets its own line in the HUD breakdown, `episodes.csv`
  (`r_custom_<slug>`) and the game's results.

## 4. Live view

### 4.1 Visibility  (updated: map shown directly, no "Map & rules" button)
- Before the **first training of the level has started** (this session or
  earlier: any attempt on record), the live view is **not shown**. The
  workshop uses the **full width**.
- Instead, a button **"Map & rules"** opens a panel with:
  - a still **map preview** of the level's world (rendered test-like road,
    parking spot, obstacles);
  - a **short explanation** of the level's rules and scoring.
- When the player presses Train, the live view appears **during** that first
  training (it is the payoff of pressing Train).
- On later visits to a level that has attempts, the live view shows the best car.

### 4.2 Fullscreen
- A **fullscreen** button on the live view (browser Fullscreen API; Esc exits).
- Fullscreen shows:
  - the driving view with its in-video HUD (speed, reward breakdown, minimap);
  - a **training progress overlay** (time left, practice stage, % parked);
  - **controls**: speed, sound, next road, stop-and-test;
  - a **results popup** when the hidden test finishes (score, stars, hints).

## 5. Other changes implied
- Per-level reward unlocking is removed (`terms` lists in `levels.py` become
  unused for availability; hints can reference any reward).
- Hints are reworded to the signed-points model (e.g. "nothing gives points
  for parking", "crashing costs nothing").
- `REWARD_GUIDE.md` and `README.md` updated to signed points + custom rewards.

## 6. Acceptance checks
1. Opening any level shows an empty list; "+ Add reward" lists 18 built-ins,
   the library and "Create custom reward..."; no numbers are pre-filled anywhere.
2. `collision: -50` and `collision: +50` produce opposite behaviour signals
   (verified on a scripted crash: reward -50 vs +50).
3. Old-format YAML (no `format: 2`) loads and trains exactly as before
   (balanced_45min evaluation numbers unchanged).
4. A custom rule `WHEN speed > 30 km/h GIVE -1 every step` reduces reward by
   1 per step only while above 30 km/h (unit test with a scripted drive);
   `on_change` pays only on false -> true transitions.
5. AND/OR groups evaluate as "any group of all-conditions" (unit test).
6. A custom reward saved in the library appears in another level; editing it
   changes the next training there.
7. Live view hidden before the first Train of a level; appears immediately on
   Train; "Map & rules" shows a rendered preview.
8. Fullscreen shows HUD, progress overlay, controls and the results popup
   (browser screenshot test).
9. Train with zero rewards asks for confirmation, then trains.
