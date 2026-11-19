# Assets

Everything here can be replaced: keep the file name (or the role name in
`autodrive/audio.py` / `autodrive/render.py`) and drop in your own file. Missing
files never crash anything - the renderer draws simple shapes and the audio
system synthesises a replacement.

## Sprites (`sprites/`) - recorded/drawn assets, not generated

| Files | Source | License |
|---|---|---|
| `cars/*.png`, `peds/*.png`, `objects/*.png` | Kenney "Racing Pack" (kenney.nl), via the `series-ai/jam-ready-assets` GitHub mirror | CC0 |

The agent car is `cars/car_blue_1.png` recoloured to white at load time.
Roads, buildings, markings, signs and traffic lights are drawn procedurally.

## Sounds (`sounds/`)

| File | Role | Real recording or synthesised? | Source / License |
|---|---|---|---|
| `engine_loop.ogg` | engine (pitch follows speed) | real | Kenney Sci-Fi Sounds `spaceEngineLow_000` - CC0 |
| `crash_metal.ogg` | crash | real | Kenney Impact Sounds `impactMetal_heavy_001` - CC0 |
| `crash_glass.ogg` | crash (layered) | real | Kenney Impact Sounds `impactGlass_heavy_001` - CC0 |
| `ped_hit.ogg` | pedestrian thud | real | Kenney Impact Sounds `impactSoft_heavy_000` - CC0 |
| `turn_tick.ogg` | turn-signal tick | real | Kenney Interface Sounds `tick_004` - CC0 |
| `light_change.ogg` | traffic-light ping | real | Kenney Interface Sounds `glass_001` - CC0 |
| `park_chime.ogg` | parking success | real | Kenney Interface Sounds `confirmation_004` - CC0 |
| `wind_loop.ogg` | ambience layer | real | "Ninja Adventure" pack `Wind2.wav` - CC0 |
| `horn.ogg` | NPC horn | **synthesised** (`tools/make_sounds.py`) | - |
| `brake_squeal.ogg` | brake squeal | **synthesised** | - |
| `ambient_city.ogg` | city hum | **synthesised** | - |

Why synthesised: the asset sites (Freesound, Pixabay, Mixkit, OpenGameArt,
kenney.nl) were blocked by this environment's network policy; only GitHub was
reachable, and no CC0 horn / brake / city recording was found there. Drop real
recordings in with the same names to replace them.

## Fonts (`fonts/`)

Inter (Regular, Bold) by Rasmus Andersson - SIL Open Font License 1.1.
