"""
Audio/Verbal feature extractor
Uses Whisper for transcription, sentiment analysis, hesitation detection
"""
import numpy as np
from collections import defaultdict
from .base_extractor import FeatureExtractor
from config import AUDIO_CONFIG

class AudioVerbalExtractor(FeatureExtractor):
    """Extract audio and verbal communication features"""
    
    def extract(self):
        """Extract all audio/verbal features"""
        # Extract audio from video
        audio_path = self._extract_audio()
        
        if audio_path is None:
            return self._empty_result()
        
        # Transcribe using Whisper
        transcription = self._transcribe_audio(audio_path)
        
        if not transcription:
            return self._empty_result()
        
        # Analyze transcription
        hesitations = self._detect_hesitations(transcription)
        sentiment = self._analyze_sentiment(transcription)
        speech_rate = self._calculate_speech_rate(transcription)
        pauses = self._detect_pauses(transcription)
        
        return {
            "transcription": transcription.get("text", ""),
            "total_words": len(transcription.get("text", "").split()),
            "speech_rate_wpm": float(speech_rate),
            "total_hesitations": len(hesitations),
            "hesitation_rate": float(len(hesitations) / (self.duration / 60)) if self.duration > 0 else 0.0,
            "average_sentiment": float(sentiment["average"]),
            "sentiment_variance": float(sentiment["variance"]),
            "positive_ratio": float(sentiment["positive_ratio"]),
            "negative_ratio": float(sentiment["negative_ratio"]),
            "total_pauses": len(pauses),
            "average_pause_duration": float(np.mean([p["duration"] for p in pauses])) if pauses else 0.0,
            "detailed_hesitations": hesitations[:50],
            "detailed_pauses": pauses[:50],
            "sentiment_timeline": sentiment["timeline"][:50],
            "segments": transcription.get("segments", [])[:20]
        }
    
    def _extract_audio(self):
        """Extract audio track from video"""
        import subprocess
        import os
        from config import TEMP_DIR
        
        audio_path = TEMP_DIR / f"audio_{id(self)}.wav"
        
        try:
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg", "-i", str(self.video_path),
                "-vn", "-acodec", "pcm_s16le",
                "-ar", str(AUDIO_CONFIG["sample_rate"]),
                "-ac", "1", "-y", str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=60)
            
            if result.returncode == 0 and audio_path.exists():
                return audio_path
        except Exception as e:
            print(f"Audio extraction failed: {e}")
        
        return None
    
    def _transcribe_audio(self, audio_path):
        """Transcribe audio using Whisper"""
        try:
            import whisper
            
            model = self.model_loader.get_whisper_model()
            result = model.transcribe(str(audio_path), word_timestamps=True)
            
            return result
        except Exception as e:
            print(f"Transcription failed: {e}")
            return None
    
    def _detect_hesitations(self, transcription):
        """Detect hesitation words and patterns"""
        hesitations = []
        hesitation_words = AUDIO_CONFIG["hesitation_words"]
        
        text = transcription.get("text", "").lower()
        segments = transcription.get("segments", [])
        
        for segment in segments:
            words = segment.get("words", [])
            for word_info in words:
                word = word_info.get("word", "").strip().lower()
                
                if word in hesitation_words:
                    hesitations.append({
                        "word": word,
                        "timestamp": word_info.get("start", 0),
                        "duration": word_info.get("end", 0) - word_info.get("start", 0)
                    })
        
        return hesitations
    
    def _analyze_sentiment(self, transcription):
        """Analyze sentiment of speech"""
        try:
            from transformers import pipeline
            
            sentiment_analyzer = pipeline("sentiment-analysis", 
                                         model="distilbert-base-uncased-finetuned-sst-2-english")
            
            segments = transcription.get("segments", [])
            sentiments = []
            timeline = []
            
            for segment in segments:
                text = segment.get("text", "").strip()
                if text:
                    result = sentiment_analyzer(text[:512])[0]  # Limit text length
                    
                    score = result["score"] if result["label"] == "POSITIVE" else -result["score"]
                    sentiments.append(score)
                    
                    timeline.append({
                        "timestamp": segment.get("start", 0),
                        "sentiment": result["label"],
                        "score": float(result["score"])
                    })
            
            if sentiments:
                positive_count = sum(1 for s in sentiments if s > 0)
                negative_count = sum(1 for s in sentiments if s < 0)
                total = len(sentiments)
                
                return {
                    "average": np.mean(sentiments),
                    "variance": np.var(sentiments),
                    "positive_ratio": positive_count / total if total > 0 else 0.0,
                    "negative_ratio": negative_count / total if total > 0 else 0.0,
                    "timeline": timeline
                }
        except Exception as e:
            print(f"Sentiment analysis failed: {e}")
        
        return {
            "average": 0.0,
            "variance": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "timeline": []
        }
    
    def _calculate_speech_rate(self, transcription):
        """Calculate words per minute"""
        text = transcription.get("text", "")
        words = text.split()
        
        if self.duration > 0:
            return (len(words) / self.duration) * 60
        
        return 0.0
    
    def _detect_pauses(self, transcription):
        """Detect pauses in speech"""
        pauses = []
        segments = transcription.get("segments", [])
        
        for i in range(1, len(segments)):
            prev_end = segments[i-1].get("end", 0)
            curr_start = segments[i].get("start", 0)
            
            pause_duration = curr_start - prev_end
            
            if pause_duration > 0.5:  # Significant pause
                pauses.append({
                    "start_time": prev_end,
                    "end_time": curr_start,
                    "duration": pause_duration
                })
        
        return pauses
    
    def _empty_result(self):
        """Return empty result when audio processing fails"""
        return {
            "transcription": "",
            "total_words": 0,
            "speech_rate_wpm": 0.0,
            "total_hesitations": 0,
            "hesitation_rate": 0.0,
            "average_sentiment": 0.0,
            "sentiment_variance": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "total_pauses": 0,
            "average_pause_duration": 0.0,
            "detailed_hesitations": [],
            "detailed_pauses": [],
            "sentiment_timeline": [],
            "segments": []
        }
