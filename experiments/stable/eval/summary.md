# Evaluation: stable

100 unseen random cities, stage **S6 full city**, deterministic policy.

| Metric | Value |
|---|---|
| Parked successfully | 23.0%  (95% CI 15.8%-32.2%) |
| Parked with zero rule violations | 1.0% |
| Collision (car/building) | 34.0% |
| Pedestrian hit | 30.0% |
| Left the route | 0.0% |
| Got stuck / timed out | 9.0% / 4.0% |
| Average route completed | 51.2% |
| Red lights run per km | 2.18 |
| Stop signs rolled per km | 6.54 |
| Crosswalk violations per km | 2.87 |
| Time spent speeding | 5.5% |
| Worst school-zone overspeed | 21.7 km/h |
| Average speed | 14.2 km/h |
| Average distance from lane centre | 0.98 m |
| Time with wheels off the road | 0.0% |
| Parking error (0 = perfect, 1 = at tolerance) | 0.43 |
| Time to park (successful trips) | 75.2 s |
| Total distance driven | 16.1 km |

Outcomes: collision 34, parked 23, pedestrian_hit 30, stuck 9, timeout 4
