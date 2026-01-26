"""
Segment matching for cross-video similarity detection
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from dtaidistance import dtw
import logging
from config import SIMILARITY_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SegmentMatcher:
    """Find similar segments across multiple videos"""
    
    def __init__(self, videos_features, similarity_threshold=None):
        """
        Initialize segment matcher
        
        Args:
            videos_features: Dict of {video_name: features_dict}
            similarity_threshold: Minimum similarity score (0-1)
        """
        self.videos_features = videos_features
        self.threshold = similarity_threshold or SIMILARITY_CONFIG["default_threshold"]
        
    def find_all_similarities(self, feature_categories):
        """
        Find similar segments across all videos for selected features
        
        Args:
            feature_categories: List of feature category names to analyze
            
        Returns:
            List of similar segment groups
        """
        all_similar_segments = []
        
        for feature_category in feature_categories:
            logger.info(f"Finding similarities for: {feature_category}")
            similar_segs = self.find_similar_segments(feature_category)
            all_similar_segments.extend(similar_segs)
        
        return all_similar_segments
    
    def find_similar_segments(self, feature_type):
        """Find similar segments for a specific feature type"""
        # Extract feature vectors for all videos
        video_segments = {}
        
        for video_name, features in self.videos_features.items():
            if feature_type not in features:
                continue
            
            segments = self._create_segments(features[feature_type], video_name)
            if segments:
                video_segments[video_name] = segments
        
        if len(video_segments) < 2:
            logger.warning(f"Not enough videos with {feature_type} data for comparison")
            return []
        
        # Compare all pairs of videos
        similar_groups = []
        video_names = list(video_segments.keys())
        
        for i in range(len(video_names)):
            for j in range(i + 1, len(video_names)):
                vid1, vid2 = video_names[i], video_names[j]
                
                matches = self._match_segments(
                    video_segments[vid1],
                    video_segments[vid2],
                    vid1,
                    vid2,
                    feature_type
                )
                
                similar_groups.extend(matches)
        
        return similar_groups
    
    def _create_segments(self, features, video_name):
        """Create temporal segments from feature data"""
        segments = []
        segment_duration = SIMILARITY_CONFIG["segment_duration"]
        overlap = SIMILARITY_CONFIG["overlap"]
        
        # Try to extract timeline data
        timeline_keys = [
            "interaction_timeline", "cursor_trajectory_sample",
            "click_locations", "detailed_interactions"
        ]
        
        timeline_data = None
        for key in timeline_keys:
            if key in features and features[key]:
                timeline_data = features[key]
                break
        
        if timeline_data is None:
            # Create single segment from aggregate features
            vector = self._features_to_vector(features)
            segments.append({
                "start_time": 0,
                "end_time": features.get("task_completion_time", 0),
                "video_name": video_name,
                "feature_vector": vector,
                "features": features
            })
            return segments
        
        # Create sliding window segments
        if isinstance(timeline_data, list) and len(timeline_data) > 0:
            # Extract time information
            times = []
            for item in timeline_data:
                if isinstance(item, dict) and "timestamp" in item:
                    times.append(item["timestamp"])
                elif isinstance(item, (int, float)):
                    times.append(item)
            
            if times:
                max_time = max(times)
                current_time = 0
                step = segment_duration * (1 - overlap)
                
                while current_time < max_time:
                    end_time = current_time + segment_duration
                    
                    # Extract features in this time window
                    segment_features = self._extract_segment_features(
                        features, timeline_data, current_time, end_time
                    )
                    
                    vector = self._features_to_vector(segment_features)
                    
                    segments.append({
                        "start_time": current_time,
                        "end_time": end_time,
                        "video_name": video_name,
                        "feature_vector": vector,
                        "features": segment_features
                    })
                    
                    current_time += step
        
        return segments
    
    def _extract_segment_features(self, full_features, timeline_data, start_time, end_time):
        """Extract features for a specific time segment"""
        segment_features = {}
        
        # Count events in time window
        if isinstance(timeline_data, list):
            events_in_window = []
            for item in timeline_data:
                timestamp = None
                if isinstance(item, dict):
                    timestamp = item.get("timestamp", item.get("start_time"))
                elif isinstance(item, (int, float)):
                    timestamp = item
                
                if timestamp and start_time <= timestamp < end_time:
                    events_in_window.append(item)
            
            segment_features["event_count"] = len(events_in_window)
            segment_features["event_density"] = len(events_in_window) / (end_time - start_time) if (end_time - start_time) > 0 else 0
        
        # Copy over relevant aggregate metrics
        numeric_keys = ["average_cursor_speed_pixels_per_sec", "total_clicks",
                       "typing_speed_wpm", "average_pause_duration"]
        for key in numeric_keys:
            if key in full_features:
                segment_features[key] = full_features[key]
        
        return segment_features
    
    def _features_to_vector(self, features):
        """Convert feature dict to numeric vector"""
        vector = []
        
        # Extract numeric values
        for key, value in features.items():
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, bool):
                vector.append(float(value))
            elif isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], (int, float)):
                    vector.append(float(len(value)))
        
        return np.array(vector) if vector else np.array([0.0])
    
    def _match_segments(self, segs1, segs2, vid1, vid2, feature_type):
        """Match segments between two videos"""
        matches = []
        
        for seg1 in segs1:
            for seg2 in segs2:
                # Calculate similarity
                vec1 = seg1["feature_vector"]
                vec2 = seg2["feature_vector"]
                
                # Ensure same dimensionality
                if len(vec1) != len(vec2):
                    min_len = min(len(vec1), len(vec2))
                    vec1 = vec1[:min_len]
                    vec2 = vec2[:min_len]
                
                if len(vec1) == 0:
                    continue
                
                # Calculate cosine similarity
                similarity = self._cosine_similarity(vec1, vec2)
                
                if similarity >= self.threshold:
                    matches.append({
                        "segment_id": f"seg_{len(matches):03d}",
                        "similarity_type": feature_type,
                        "similarity_score": float(similarity),
                        "videos": [
                            {
                                "video_name": vid1,
                                "time_interval": [seg1["start_time"], seg1["end_time"]],
                                "duration_seconds": seg1["end_time"] - seg1["start_time"],
                                "features": self._extract_key_features(seg1["features"])
                            },
                            {
                                "video_name": vid2,
                                "time_interval": [seg2["start_time"], seg2["end_time"]],
                                "duration_seconds": seg2["end_time"] - seg2["start_time"],
                                "features": self._extract_key_features(seg2["features"])
                            }
                        ],
                        "common_features": self._identify_common_features(seg1["features"], seg2["features"]),
                        "differences": self._identify_differences(seg1["features"], seg2["features"], vid1, vid2)
                    })
        
        return matches
    
    def _cosine_similarity(self, vec1, vec2):
        """Calculate cosine similarity between two vectors"""
        vec1 = vec1.reshape(1, -1)
        vec2 = vec2.reshape(1, -1)
        
        # Handle zero vectors
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0
        
        return float(cosine_similarity(vec1, vec2)[0][0])
    
    def _extract_key_features(self, features):
        """Extract key numeric features for comparison"""
        key_features = {}
        
        numeric_keys = ["event_count", "event_density", "average_cursor_speed_pixels_per_sec",
                       "total_clicks", "typing_speed_wpm", "errors"]
        
        for key in numeric_keys:
            if key in features:
                key_features[key] = features[key]
        
        return key_features
    
    def _identify_common_features(self, feat1, feat2):
        """Identify common features between segments"""
        common = {}
        
        for key in feat1.keys():
            if key in feat2:
                val1, val2 = feat1[key], feat2[key]
                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    if abs(val1 - val2) < 0.1 * max(abs(val1), abs(val2), 1):
                        common[key] = (val1 + val2) / 2
        
        return common
    
    def _identify_differences(self, feat1, feat2, vid1, vid2):
        """Identify differences between segments"""
        differences = {vid1: [], vid2: []}
        
        for key in set(feat1.keys()) | set(feat2.keys()):
            val1 = feat1.get(key, 0)
            val2 = feat2.get(key, 0)
            
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                if val1 > val2 * 1.2:
                    differences[vid1].append(f"higher {key}")
                    differences[vid2].append(f"lower {key}")
                elif val2 > val1 * 1.2:
                    differences[vid2].append(f"higher {key}")
                    differences[vid1].append(f"lower {key}")
        
        return {
            vid1: ", ".join(differences[vid1][:3]) if differences[vid1] else "similar pattern",
            vid2: ", ".join(differences[vid2][:3]) if differences[vid2] else "similar pattern"
        }
