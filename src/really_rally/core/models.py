class RallySegment:
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    score: float = 0.0

    @property
    def duration(self):
        return self.end_time - self.start_time

# ==================================================================================
# CORE CLASSES
# ==================================================================================

