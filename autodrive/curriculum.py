"""
curriculum.py - the automatic curriculum.

Learning to drive a whole city with traffic, pedestrians and parking from
scratch is too hard for a few million steps of PPO. So the world starts
easy and gets harder automatically: as soon as the agent's recent success
rate in a stage passes a threshold, every environment moves to the next
stage.

A small fraction of episodes is still sampled from earlier stages
("replay") so skills learned early are not forgotten.
"""
from __future__ import annotations

from collections import deque

STAGES = [
    dict(name="S0 empty streets", grid=(3, 3), hops=(0, 1), controls=False, n_cars=0,
         n_peds=0, parked=0.0, n_kids=0, jaywalk=0.0, school=0.0, lot_occ=0.25,
         hard_brake=0.0, advance=0.70),
    dict(name="S1 traffic rules", grid=(3, 3), hops=(1, 3), controls=True, n_cars=0,
         n_peds=0, parked=0.25, n_kids=0, jaywalk=0.0, school=0.15, lot_occ=0.4,
         hard_brake=0.0, advance=0.65),
    dict(name="S2 traffic", grid=(4, 4), hops=(1, 4), controls=True, n_cars=8,
         n_peds=0, parked=0.3, n_kids=0, jaywalk=0.0, school=0.15, lot_occ=0.5,
         hard_brake=0.5, advance=0.60),
    dict(name="S3 pedestrians", grid=(4, 4), hops=(1, 4), controls=True, n_cars=8,
         n_peds=14, parked=0.4, n_kids=1, jaywalk=0.15, school=0.15, lot_occ=0.6,
         hard_brake=1.0, advance=0.60),
    dict(name="S4 full city", grid=(4, 4), hops=(2, 6), controls=True, n_cars=10,
         n_peds=18, parked=0.5, n_kids=2, jaywalk=0.2, school=0.2, lot_occ=0.7,
         hard_brake=1.0, advance=1.01),   # final stage never "advances"
]
# Evaluation on unseen cities uses the full-city settings.
EVAL_STAGE = len(STAGES) - 1


class CurriculumTracker:
    """Keeps a rolling success rate for the current stage and decides when to advance."""

    def __init__(self, window=100, min_episodes=100, start_stage=0):
        self.stage = start_stage
        self.window = window
        self.min_episodes = min_episodes
        self.results = deque(maxlen=window)
        self.history = []          # (timestep, stage) whenever we advance

    def record(self, success: bool, stage_of_episode: int):
        if stage_of_episode == self.stage:     # ignore replayed easier episodes
            self.results.append(bool(success))

    @property
    def rate(self):
        return sum(self.results) / len(self.results) if self.results else 0.0

    def should_advance(self):
        return (self.stage < len(STAGES) - 1 and len(self.results) >= self.min_episodes
                and self.rate >= STAGES[self.stage]["advance"])

    def advance(self, timestep):
        self.stage += 1
        self.results.clear()
        self.history.append((int(timestep), self.stage))
        return self.stage
