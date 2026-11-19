# Reward Designer - game design

A solo puzzle game in the browser. You never drive the car: you **design its
reward** (switch reward terms on, choose their weights), choose how long to
train, watch it learn live, and get a score from a hidden test on roads it
has never seen. Each level adds something new to the world and unlocks new
reward terms.

## Levels

| # | Level | World | Score |
|---|---|---|---|
| 1 | Park here | Parking lot, start a few metres from the spot | parking precision |
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

## Training

- The player chooses the training time.
- The level's curriculum runs automatically behind the scenes (e.g. starting
  close to the spot first); the player only designs the reward.
- Optional: start from your best car of the previous level.

## Build order

1. Game shell + levels 1-5 (these train in minutes today).
2. Levels 6-8 once traffic training works (stability fix in progress).
3. Level 9 (night / weather), then 10 (roundabouts, highways), then 11 (construction).
