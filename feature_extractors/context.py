"""
Context feature extractor
Detects device type, platform, screen resolution, browser/app context
"""
import cv2
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor

class ContextExtractor(FeatureExtractor):
    """Extract contextual information about the environment"""
    
    def extract(self):
        """Extract all context features"""
        # Analyze first few frames for context
        sample_frames = []
        
        for frame_num, timestamp, frame in self.iterate_frames(skip_frames=30):
            sample_frames.append(frame)
            if len(sample_frames) >= 10:
                break
        
        if not sample_frames:
            return self._empty_result()
        
        # Detect device type and platform
        device_info = self._detect_device_type(sample_frames[0])
        platform_info = self._detect_platform(sample_frames[0])
        browser_info = self._detect_browser(sample_frames[0])
        
        # Screen resolution
        resolution = {
            "width": self.width,
            "height": self.height,
            "aspect_ratio": float(self.width / self.height) if self.height > 0 else 0.0
        }
        
        # Detect application context
        app_context = self._detect_application_context(sample_frames)
        
        # Detect UI framework/style
        ui_framework = self._detect_ui_framework(sample_frames[0])
        
        # Lighting and video quality
        quality_metrics = self._analyze_video_quality(sample_frames)
        
        return {
            "device_type": device_info["type"],
            "device_confidence": float(device_info["confidence"]),
            "platform": platform_info["platform"],
            "platform_confidence": float(platform_info["confidence"]),
            "browser_or_app": browser_info["name"],
            "browser_confidence": float(browser_info["confidence"]),
            "screen_resolution": resolution,
            "application_context": app_context,
            "ui_framework": ui_framework,
            "video_quality": quality_metrics["quality"],
            "average_brightness": float(quality_metrics["brightness"]),
            "average_contrast": float(quality_metrics["contrast"]),
            "color_depth": quality_metrics["color_depth"],
            "recording_type": self._detect_recording_type(sample_frames[0])
        }
    
    def _detect_device_type(self, frame):
        """Detect device type from screen characteristics"""
        aspect_ratio = self.width / self.height if self.height > 0 else 0
        
        # Common aspect ratios
        if 0.55 < aspect_ratio < 0.65:  # ~9:16
            return {"type": "mobile_portrait", "confidence": 0.8}
        elif 1.7 < aspect_ratio < 1.9:  # ~16:9
            if self.width >= 1920:
                return {"type": "desktop", "confidence": 0.9}
            else:
                return {"type": "laptop", "confidence": 0.7}
        elif 1.3 < aspect_ratio < 1.4:  # ~4:3
            return {"type": "tablet", "confidence": 0.7}
        elif 1.5 < aspect_ratio < 1.7:  # ~16:10
            return {"type": "laptop", "confidence": 0.8}
        else:
            return {"type": "unknown", "confidence": 0.3}
    
    def _detect_platform(self, frame):
        """Detect operating system/platform"""
        # Use OCR to look for platform indicators
        ocr_reader = self.model_loader.get_ocr_reader()
        results = ocr_reader.readtext(frame)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        # Look for platform-specific UI elements
        if "windows" in text_content or self._has_windows_ui(frame):
            return {"platform": "Windows", "confidence": 0.8}
        elif "mac" in text_content or self._has_macos_ui(frame):
            return {"platform": "macOS", "confidence": 0.8}
        elif "android" in text_content or self._has_android_ui(frame):
            return {"platform": "Android", "confidence": 0.8}
        elif "ios" in text_content or self._has_ios_ui(frame):
            return {"platform": "iOS", "confidence": 0.8}
        elif self._has_linux_ui(frame):
            return {"platform": "Linux", "confidence": 0.7}
        else:
            return {"platform": "Unknown", "confidence": 0.3}
    
    def _detect_browser(self, frame):
        """Detect browser or application"""
        ocr_reader = self.model_loader.get_ocr_reader()
        
        # Focus on top bar area
        top_bar = frame[:100, :]
        results = ocr_reader.readtext(top_bar)
        
        text_content = " ".join([text for _, text, _ in results]).lower()
        
        browsers = {
            "chrome": ["chrome", "google"],
            "firefox": ["firefox", "mozilla"],
            "safari": ["safari"],
            "edge": ["edge", "microsoft edge"],
            "brave": ["brave"]
        }
        
        for browser, keywords in browsers.items():
            if any(keyword in text_content for keyword in keywords):
                return {"name": browser.capitalize(), "confidence": 0.8}
        
        # Check for web indicators
        if "http" in text_content or "www" in text_content:
            return {"name": "Web Browser", "confidence": 0.6}
        
        return {"name": "Desktop Application", "confidence": 0.5}
    
    def _detect_application_context(self, frames):
        """Detect what application/website is being used"""
        # Analyze color scheme and layout
        avg_frame = np.mean(frames, axis=0).astype(np.uint8)
        
        # Check for common application patterns
        if self._has_video_player_ui(avg_frame):
            return "video_streaming"
        elif self._has_social_media_ui(avg_frame):
            return "social_media"
        elif self._has_productivity_ui(avg_frame):
            return "productivity"
        elif self._has_ecommerce_ui(avg_frame):
            return "ecommerce"
        else:
            return "general"
    
    def _detect_ui_framework(self, frame):
        """Detect UI framework or design system"""
        # Analyze color palette
        colors = self._extract_dominant_colors(frame)
        
        # Material Design (Google)
        if self._is_material_design(colors, frame):
            return "Material Design"
        
        # Fluent Design (Microsoft)
        elif self._is_fluent_design(colors, frame):
            return "Fluent Design"
        
        # iOS Human Interface
        elif self._is_ios_design(frame):
            return "iOS HIG"
        
        else:
            return "Custom/Unknown"
    
    def _analyze_video_quality(self, frames):
        """Analyze video quality metrics"""
        brightness_values = []
        contrast_values = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness_values.append(np.mean(gray))
            contrast_values.append(np.std(gray))
        
        avg_brightness = np.mean(brightness_values)
        avg_contrast = np.mean(contrast_values)
        
        # Determine quality
        if avg_brightness > 100 and avg_contrast > 40:
            quality = "high"
        elif avg_brightness > 60 and avg_contrast > 25:
            quality = "medium"
        else:
            quality = "low"
        
        return {
            "quality": quality,
            "brightness": avg_brightness,
            "contrast": avg_contrast,
            "color_depth": "24-bit"  # Assume standard
        }
    
    def _detect_recording_type(self, frame):
        """Detect if screen recording or camera recording"""
        # Screen recordings typically have sharp edges and UI elements
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        edge_density = np.sum(edges > 0) / (frame.shape[0] * frame.shape[1])
        
        if edge_density > 0.1:
            return "screen_recording"
        else:
            return "camera_recording"
    
    def _has_windows_ui(self, frame):
        """Check for Windows UI elements"""
        # Look for taskbar at bottom
        bottom_bar = frame[-50:, :]
        avg_color = np.mean(bottom_bar, axis=(0, 1))
        
        # Windows taskbar is typically dark
        return avg_color[0] < 50 and avg_color[1] < 50 and avg_color[2] < 50
    
    def _has_macos_ui(self, frame):
        """Check for macOS UI elements"""
        # Look for menu bar at top
        top_bar = frame[:30, :]
        avg_color = np.mean(top_bar, axis=(0, 1))
        
        # macOS menu bar is typically light gray
        return 200 < avg_color[0] < 240 and 200 < avg_color[1] < 240 and 200 < avg_color[2] < 240
    
    def _has_android_ui(self, frame):
        """Check for Android UI elements"""
        # Look for navigation bar at bottom
        bottom_bar = frame[-100:, :]
        return np.mean(bottom_bar) < 30
    
    def _has_ios_ui(self, frame):
        """Check for iOS UI elements"""
        # Look for rounded corners and specific spacing
        return False  # Simplified
    
    def _has_linux_ui(self, frame):
        """Check for Linux UI elements"""
        return False  # Simplified
    
    def _has_video_player_ui(self, frame):
        """Check for video player interface"""
        # Look for play button, progress bar
        h, w = frame.shape[:2]
        center_region = frame[int(h*0.4):int(h*0.6), int(w*0.4):int(w*0.6)]
        return np.mean(center_region) < 50
    
    def _has_social_media_ui(self, frame):
        """Check for social media interface"""
        return False  # Simplified
    
    def _has_productivity_ui(self, frame):
        """Check for productivity app interface"""
        return False  # Simplified
    
    def _has_ecommerce_ui(self, frame):
        """Check for ecommerce interface"""
        return False  # Simplified
    
    def _extract_dominant_colors(self, frame):
        """Extract dominant colors from frame"""
        pixels = frame.reshape(-1, 3)
        pixels = np.float32(pixels)
        
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        k = 5
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        return centers
    
    def _is_material_design(self, colors, frame):
        """Check if Material Design"""
        return False  # Simplified
    
    def _is_fluent_design(self, colors, frame):
        """Check if Fluent Design"""
        return False  # Simplified
    
    def _is_ios_design(self, frame):
        """Check if iOS design"""
        return False  # Simplified
    
    def _empty_result(self):
        """Return empty result"""
        return {
            "device_type": "unknown",
            "device_confidence": 0.0,
            "platform": "Unknown",
            "platform_confidence": 0.0,
            "browser_or_app": "Unknown",
            "browser_confidence": 0.0,
            "screen_resolution": {"width": 0, "height": 0, "aspect_ratio": 0.0},
            "application_context": "unknown",
            "ui_framework": "Unknown",
            "video_quality": "unknown",
            "average_brightness": 0.0,
            "average_contrast": 0.0,
            "color_depth": "unknown",
            "recording_type": "unknown"
        }
