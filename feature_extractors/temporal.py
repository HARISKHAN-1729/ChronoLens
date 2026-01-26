"""
Temporal feature extractor
Extracts timing-based interaction features
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import TEMPORAL_CONFIG

class TemporalFeatureExtractor(FeatureExtractor):
    """Extract temporal features from video"""
    
    def extract(self):
        """Extract all temporal features"""
        interactions = []
        prev_frame = None
        last_interaction_time = None
        first_interaction_time = None
        
        pauses = []
        interaction_durations = []
        
        # Detection state
        in_interaction = False
        interaction_start_time = None
        
        for frame_num, timestamp, frame in self.iterate_frames():
            if prev_frame is not None:
                # Detect interaction through frame differences
                motion_score = self.calculate_frame_diff(frame, prev_frame)
                
                is_interacting = motion_score > TEMPORAL_CONFIG["interaction_threshold"] * 255
                
                if is_interacting:
                    # Record first interaction
                    if first_interaction_time is None:
                        first_interaction_time = timestamp
                    
                    # Start of new interaction
                    if not in_interaction:
                        interaction_start_time = timestamp
                        in_interaction = True
                        
                        # Calculate pause before this interaction
                        if last_interaction_time is not None:
                            pause_duration = timestamp - last_interaction_time
                            if pause_duration >= TEMPORAL_CONFIG["pause_threshold"]:
                                pauses.append({
                                    "start_time": last_interaction_time,
                                    "end_time": timestamp,
                                    "duration": pause_duration
                                })
                    
                    last_interaction_time = timestamp
                
                else:
                    # End of interaction
                    if in_interaction and interaction_start_time is not None:
                        duration = timestamp - interaction_start_time
                        if duration >= TEMPORAL_CONFIG["min_interaction_duration"]:
                            interactions.append({
                                "start_time": interaction_start_time,
                                "end_time": timestamp,
                                "duration": duration,
                                "frame_range": [self.timestamp_to_frame(interaction_start_time), frame_num]
                            })
                            interaction_durations.append(duration)
                        
                        in_interaction = False
                        interaction_start_time = None
            
            prev_frame = frame.copy()
        
        # Close any ongoing interaction at video end
        if in_interaction and interaction_start_time is not None:
            duration = self.duration - interaction_start_time
            if duration >= TEMPORAL_CONFIG["min_interaction_duration"]:
                interactions.append({
                    "start_time": interaction_start_time,
                    "end_time": self.duration,
                    "duration": duration,
                    "frame_range": [self.timestamp_to_frame(interaction_start_time), self.total_frames - 1]
                })
                interaction_durations.append(duration)
        
        # Calculate rhythm/cadence
        rhythm = self._calculate_rhythm(interactions)
        
        # Calculate aggregate metrics
        total_interactions = len(interactions)
        avg_interaction_duration = np.mean(interaction_durations) if interaction_durations else 0.0
        total_pause_time = sum([p["duration"] for p in pauses])
        avg_pause_duration = np.mean([p["duration"] for p in pauses]) if pauses else 0.0
        time_to_first = first_interaction_time if first_interaction_time else 0.0
        
        # Calculate reaction times (time between significant changes and interactions)
        reaction_times = self._estimate_reaction_times(interactions)
        
        return {
            "total_interactions": total_interactions,
            "average_interaction_duration": float(avg_interaction_duration),
            "total_pause_time": float(total_pause_time),
            "average_pause_duration": float(avg_pause_duration),
            "number_of_pauses": len(pauses),
            "task_completion_time": float(self.duration),
            "time_to_first_interaction": float(time_to_first),
            "interaction_rhythm": rhythm,
            "average_reaction_time": float(np.mean(reaction_times)) if reaction_times else 0.0,
            "detailed_interactions": interactions[:100],  # Limit to first 100 for JSON size
            "detailed_pauses": pauses[:100],
            "interaction_timeline": [i["start_time"] for i in interactions[:100]]
        }
    
    def _calculate_rhythm(self, interactions):
        """Calculate interaction rhythm/cadence"""
        if len(interactions) < 2:
            return "insufficient_data"
        
        # Calculate variance in interaction spacing
        intervals = []
        for i in range(1, len(interactions)):
            interval = interactions[i]["start_time"] - interactions[i-1]["end_time"]
            intervals.append(interval)
        
        if not intervals:
            return "steady"
        
        variance = np.var(intervals)
        mean_interval = np.mean(intervals)
        
        if variance < 1.0:
            return "steady"
        elif variance > mean_interval:
            return "bursty"
        else:
            return "moderate"
    
    def _estimate_reaction_times(self, interactions):
        """Estimate reaction times between UI changes and user actions"""
        reaction_times = []
        
        # Simple heuristic: look at gaps between interaction clusters
        for i in range(1, len(interactions)):
            gap = interactions[i]["start_time"] - interactions[i-1]["end_time"]
            if 0.1 < gap < 5.0:  # Reasonable reaction time range
                reaction_times.append(gap)
        
        return reaction_times
