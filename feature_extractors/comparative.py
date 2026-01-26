"""
Comparative feature extractor
Cross-video analysis, variance detection, pattern identification, outlier detection
"""
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class ComparativeExtractor(FeatureExtractor):
    """Extract comparative features across multiple videos or segments"""
    
    def __init__(self, video_path, model_loader, comparison_data=None):
        """
        Initialize with optional comparison data from other videos
        
        Args:
            video_path: Path to video file
            model_loader: Model loader instance
            comparison_data: Dict of features from other videos for comparison
        """
        super().__init__(video_path, model_loader)
        self.comparison_data = comparison_data or {}
    
    def extract(self):
        """Extract comparative features"""
        # Extract features from current video segments
        segments = self._segment_video()
        segment_features = []
        
        for segment in segments:
            features = self._extract_segment_features(segment)
            segment_features.append(features)
        
        # Analyze variance within video
        intra_video_variance = self._calculate_intra_variance(segment_features)
        
        # Identify patterns
        patterns = self._identify_patterns(segment_features)
        
        # Detect outliers
        outliers = self._detect_outliers(segment_features)
        
        # Cross-video comparison (if comparison data available)
        cross_video_metrics = {}
        if self.comparison_data:
            cross_video_metrics = self._compare_with_others(segment_features)
        
        return {
            "total_segments": len(segments),
            "intra_video_variance": float(intra_video_variance),
            "common_patterns": patterns,
            "outlier_segments": outliers,
            "segment_consistency": self._calculate_consistency(segment_features),
            "behavioral_stability": self._calculate_stability(segment_features),
            "cross_video_similarity": cross_video_metrics.get("similarity", 0.0),
            "relative_performance": cross_video_metrics.get("performance_rank", "N/A"),
            "unique_behaviors": self._identify_unique_behaviors(segment_features),
            "detailed_segments": segment_features[:20],
            "pattern_timeline": self._create_pattern_timeline(patterns, segments)
        }
    
    def _segment_video(self):
        """Segment video into analysis chunks"""
        segment_duration = 30.0  # 30 second segments
        segments = []
        
        current_segment = {
            "start_time": 0,
            "frames": []
        }
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=5):
            current_segment["frames"].append({
                "frame_num": frame_num,
                "timestamp": timestamp,
                "frame": frame
            })
            
            # Check if segment is complete
            if timestamp - current_segment["start_time"] >= segment_duration:
                current_segment["end_time"] = timestamp
                segments.append(current_segment)
                
                current_segment = {
                    "start_time": timestamp,
                    "frames": []
                }
        
        # Add final segment
        if current_segment["frames"]:
            current_segment["end_time"] = self.duration
            segments.append(current_segment)
        
        return segments
    
    def _extract_segment_features(self, segment):
        """Extract features from a video segment"""
        frames = segment["frames"]
        
        if not frames:
            return {}
        
        # Calculate activity level
        activity_levels = []
        for i in range(1, len(frames)):
            diff = np.mean(np.abs(
                frames[i]["frame"].astype(float) - 
                frames[i-1]["frame"].astype(float)
            ))
            activity_levels.append(diff)
        
        avg_activity = np.mean(activity_levels) if activity_levels else 0.0
        
        # Calculate interaction density
        high_activity_frames = sum(1 for a in activity_levels if a > 20)
        interaction_density = high_activity_frames / len(activity_levels) if activity_levels else 0.0
        
        # Calculate color variance (UI changes)
        color_variances = []
        for frame_data in frames:
            frame = frame_data["frame"]
            color_var = np.var(frame)
            color_variances.append(color_var)
        
        avg_color_variance = np.mean(color_variances) if color_variances else 0.0
        
        return {
            "start_time": segment["start_time"],
            "end_time": segment["end_time"],
            "average_activity": float(avg_activity),
            "interaction_density": float(interaction_density),
            "color_variance": float(avg_color_variance),
            "num_frames": len(frames)
        }
    
    def _calculate_intra_variance(self, segment_features):
        """Calculate variance across segments within video"""
        if len(segment_features) < 2:
            return 0.0
        
        activities = [s["average_activity"] for s in segment_features]
        densities = [s["interaction_density"] for s in segment_features]
        
        activity_var = np.var(activities)
        density_var = np.var(densities)
        
        # Combined variance
        return (activity_var + density_var) / 2
    
    def _identify_patterns(self, segment_features):
        """Identify common patterns in segments"""
        patterns = []
        
        # Pattern 1: Consistent high activity
        high_activity_segments = [s for s in segment_features if s["average_activity"] > 30]
        if len(high_activity_segments) >= len(segment_features) * 0.6:
            patterns.append({
                "type": "consistent_high_activity",
                "frequency": len(high_activity_segments) / len(segment_features)
            })
        
        # Pattern 2: Increasing activity trend
        activities = [s["average_activity"] for s in segment_features]
        if len(activities) >= 3:
            trend = np.polyfit(range(len(activities)), activities, 1)[0]
            if trend > 1:
                patterns.append({
                    "type": "increasing_activity",
                    "slope": float(trend)
                })
            elif trend < -1:
                patterns.append({
                    "type": "decreasing_activity",
                    "slope": float(trend)
                })
        
        # Pattern 3: Periodic bursts
        if self._has_periodic_pattern(activities):
            patterns.append({
                "type": "periodic_bursts",
                "frequency": "detected"
            })
        
        return patterns
    
    def _detect_outliers(self, segment_features):
        """Detect outlier segments"""
        outliers = []
        
        activities = [s["average_activity"] for s in segment_features]
        mean_activity = np.mean(activities)
        std_activity = np.std(activities)
        
        for i, segment in enumerate(segment_features):
            # Z-score method
            z_score = abs((segment["average_activity"] - mean_activity) / std_activity) if std_activity > 0 else 0
            
            if z_score > 2:  # 2 standard deviations
                outliers.append({
                    "segment_index": i,
                    "start_time": segment["start_time"],
                    "end_time": segment["end_time"],
                    "z_score": float(z_score),
                    "reason": "unusual_activity_level"
                })
        
        return outliers
    
    def _calculate_consistency(self, segment_features):
        """Calculate behavioral consistency score (0-1)"""
        if len(segment_features) < 2:
            return 1.0
        
        activities = [s["average_activity"] for s in segment_features]
        densities = [s["interaction_density"] for s in segment_features]
        
        # Low variance = high consistency
        activity_consistency = 1 - min(np.std(activities) / (np.mean(activities) + 1), 1.0)
        density_consistency = 1 - min(np.std(densities) / (np.mean(densities) + 1), 1.0)
        
        return float((activity_consistency + density_consistency) / 2)
    
    def _calculate_stability(self, segment_features):
        """Calculate behavioral stability (resistance to change)"""
        if len(segment_features) < 3:
            return 0.5
        
        # Calculate rate of change between consecutive segments
        changes = []
        for i in range(1, len(segment_features)):
            change = abs(
                segment_features[i]["average_activity"] - 
                segment_features[i-1]["average_activity"]
            )
            changes.append(change)
        
        avg_change = np.mean(changes)
        
        # Low change = high stability
        stability = 1 - min(avg_change / 50, 1.0)
        
        return float(stability)
    
    def _compare_with_others(self, segment_features):
        """Compare with other videos"""
        if not self.comparison_data:
            return {}
        
        # Calculate similarity with each comparison video
        similarities = []
        
        for video_id, other_features in self.comparison_data.items():
            similarity = self._calculate_similarity(segment_features, other_features)
            similarities.append(similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Rank performance (simplified)
        performance_rank = "average"
        if avg_similarity > 0.8:
            performance_rank = "similar"
        elif avg_similarity < 0.4:
            performance_rank = "unique"
        
        return {
            "similarity": float(avg_similarity),
            "performance_rank": performance_rank,
            "num_comparisons": len(similarities)
        }
    
    def _calculate_similarity(self, features1, features2):
        """Calculate similarity between two feature sets"""
        # Simplified similarity based on activity patterns
        if not features1 or not features2:
            return 0.0
        
        activities1 = [f["average_activity"] for f in features1]
        activities2 = [f.get("average_activity", 0) for f in features2]
        
        # Normalize lengths
        min_len = min(len(activities1), len(activities2))
        activities1 = activities1[:min_len]
        activities2 = activities2[:min_len]
        
        # Calculate correlation
        if len(activities1) > 1:
            correlation = np.corrcoef(activities1, activities2)[0, 1]
            return abs(correlation)
        
        return 0.5
    
    def _identify_unique_behaviors(self, segment_features):
        """Identify unique behaviors not seen in comparison data"""
        unique = []
        
        # Check for extreme values
        activities = [s["average_activity"] for s in segment_features]
        
        if max(activities) > 100:
            unique.append("extremely_high_activity")
        
        if min(activities) < 5:
            unique.append("extremely_low_activity")
        
        # Check for unique patterns
        if self._has_unique_pattern(activities):
            unique.append("unique_interaction_pattern")
        
        return unique
    
    def _has_periodic_pattern(self, values):
        """Check for periodic patterns in values"""
        if len(values) < 6:
            return False
        
        # Simple peak detection
        peaks = 0
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks += 1
        
        return peaks >= 3
    
    def _has_unique_pattern(self, values):
        """Check for unique patterns"""
        # Simplified: check for unusual distribution
        if len(values) < 3:
            return False
        
        std = np.std(values)
        mean = np.mean(values)
        
        # High coefficient of variation suggests unique pattern
        cv = std / mean if mean > 0 else 0
        
        return cv > 1.0
    
    def _create_pattern_timeline(self, patterns, segments):
        """Create timeline of pattern occurrences"""
        timeline = []
        
        for pattern in patterns:
            timeline.append({
                "pattern_type": pattern["type"],
                "detected_at": "throughout" if pattern.get("frequency", 0) > 0.5 else "sporadic"
            })
        
        return timeline[:20]
