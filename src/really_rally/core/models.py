from dataclasses import dataclass

@dataclass
class RallySegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float = 0.0

    @property
    def duration(self):
        return self.end_time - self.start_time


@dataclass
class ServeEvent:
    """A detected tennis serve with biomechanical event timestamps.

    Indices are into the (PROCESS_EVERY_N_FRAMES-downsampled) frame_data array;
    times are in seconds on the original video timeline.
    """
    start_idx: int
    end_idx: int
    start_time: float
    end_time: float
    score: float
    handedness: str                  # 'right' | 'left' | 'unknown' (hitting arm)
    toss_arm: str                    # 'left' | 'right' (arm that tosses)
    toss_idx: int                    # index of toss release / highest toss-arm point
    contact_idx: int                 # index of estimated ball contact
    toss_time: float = 0.0           # toss release time (seconds)
    contact_time: float = 0.0        # estimated contact time (seconds)
    toss_score: float = 0.0
    contact_score: float = 0.0
    follow_through_score: float = 0.0
    outcome: str = "unknown"         # 'in' | 'net' | 'long' | 'wide' | 'unknown'

    @property
    def duration(self):
        return self.end_time - self.start_time

# ==================================================================================
# CORE CLASSES
# ==================================================================================

