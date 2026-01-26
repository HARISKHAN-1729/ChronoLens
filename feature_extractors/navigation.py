"""
Navigation feature extractor
Tracks page/screen changes, navigation paths, and efficiency
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class NavigationExtractor(FeatureExtractor):
    """Extract navigation and page transition features"""
    
    def extract(self):
        """Extract all navigation features"""
        page_changes = []
        navigation_path = []
        backtrack_count = 0
        
        prev_frame = None
        prev_page_hash = None
        page_visit_count = defaultdict(int)
        current_page_id = 0
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=3):
            if prev_frame is not None:
                # Detect page/screen change
                page_hash = self._compute_page_hash(frame)
                
                if prev_page_hash is not None:
                    similarity = self._hash_similarity(page_hash, prev_page_hash)
                    
                    # Significant change indicates navigation
                    if similarity < 0.7:
                        current_page_id += 1
                        page_changes.append({
                            "timestamp": timestamp,
                            "frame": frame_num,
                            "from_page": current_page_id - 1,
                            "to_page": current_page_id,
                            "transition_type": self._classify_transition(frame, prev_frame)
                        })
                        
                        navigation_path.append(current_page_id)
                        page_visit_count[page_hash] += 1
                        
                        # Check if returning to previously visited page (backtrack)
                        if page_visit_count[page_hash] > 1:
                            backtrack_count += 1
                
                prev_page_hash = page_hash
            else:
                prev_page_hash = self._compute_page_hash(frame)
                navigation_path.append(0)
            
            prev_frame = frame.copy()
        
        # Calculate navigation efficiency
        unique_pages = len(set(navigation_path))
        total_navigations = len(page_changes)
        efficiency = unique_pages / total_navigations if total_navigations > 0 else 1.0
        
        # Calculate path complexity
        path_complexity = self._calculate_path_complexity(navigation_path)
        
        return {
            "total_page_changes": total_navigations,
            "unique_pages_visited": unique_pages,
            "backtrack_count": backtrack_count,
            "navigation_efficiency": float(efficiency),
            "path_complexity": float(path_complexity),
            "average_time_per_page": float(self.duration / unique_pages) if unique_pages > 0 else 0.0,
            "navigation_rate_per_minute": float(total_navigations / (self.duration / 60)) if self.duration > 0 else 0.0,
            "backtrack_ratio": float(backtrack_count / total_navigations) if total_navigations > 0 else 0.0,
            "navigation_path": navigation_path[:100],
            "detailed_page_changes": page_changes[:50],
            "transition_types": self._get_transition_distribution(page_changes),
            "most_visited_pages": self._get_most_visited(page_visit_count)
        }
    
    def _compute_page_hash(self, frame):
        """Compute perceptual hash of page/screen"""
        # Resize to small size for hashing
        small = cv2.resize(frame, (32, 32))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        
        # Compute DCT-based hash
        dct = cv2.dct(np.float32(gray))
        dct_low = dct[:8, :8]
        
        # Create hash from DCT coefficients
        median = np.median(dct_low)
        hash_val = (dct_low > median).flatten()
        
        return hash_val.tobytes()
    
    def _hash_similarity(self, hash1, hash2):
        """Calculate similarity between two hashes"""
        if hash1 == hash2:
            return 1.0
        
        # Hamming distance
        arr1 = np.frombuffer(hash1, dtype=np.uint8)
        arr2 = np.frombuffer(hash2, dtype=np.uint8)
        
        # Unpack bits
        bits1 = np.unpackbits(arr1)
        bits2 = np.unpackbits(arr2)
        
        # Calculate similarity
        matches = np.sum(bits1 == bits2)
        total = len(bits1)
        
        return matches / total if total > 0 else 0.0
    
    def _classify_transition(self, frame, prev_frame):
        """Classify type of page transition"""
        # Analyze transition pattern
        diff = cv2.absdiff(frame, prev_frame)
        mean_diff = np.mean(diff)
        
        # Check for fade/slide patterns
        if mean_diff > 100:
            return "instant"
        elif mean_diff > 50:
            return "fade"
        else:
            return "slide"
    
    def _calculate_path_complexity(self, path):
        """Calculate complexity of navigation path"""
        if len(path) < 2:
            return 0.0
        
        # Count direction changes
        changes = 0
        for i in range(2, len(path)):
            if path[i] != path[i-1]:
                changes += 1
        
        # Normalize by path length
        return changes / len(path) if len(path) > 0 else 0.0
    
    def _get_transition_distribution(self, page_changes):
        """Get distribution of transition types"""
        distribution = defaultdict(int)
        
        for change in page_changes:
            distribution[change["transition_type"]] += 1
        
        return dict(distribution)
    
    def _get_most_visited(self, page_visit_count):
        """Get most frequently visited pages"""
        sorted_pages = sorted(page_visit_count.items(), key=lambda x: x[1], reverse=True)
        return [{"page_hash": str(k), "visits": v} for k, v in sorted_pages[:5]]
