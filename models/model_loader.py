"""
Model loader for all open-source AI models
"""
import os
import logging
from pathlib import Path
import torch
import whisper
from ultralytics import YOLO
from transformers import pipeline
import mediapipe as mp
import easyocr
from sentence_transformers import SentenceTransformer
from config import MODELS, MODELS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelLoader:
    """Loads and manages all open-source AI models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Model instances
        self.whisper_model = None
        self.yolo_model = None
        self.sentiment_pipeline = None
        self.pose_detector = None
        self.face_detection = None
        self.face_mesh = None
        self.hands_detector = None
        self.ocr_reader = None
        self.embedding_model = None
        
    def load_all_models(self):
        """Load all models at startup"""
        logger.info("Loading all AI models...")
        
        try:
            self.load_whisper()
            self.load_yolo()
            self.load_sentiment()
            self.load_mediapipe()
            self.load_ocr()
            self.load_embeddings()
            logger.info("All models loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def load_whisper(self):
        """Load Whisper for speech-to-text"""
        logger.info("Loading Whisper model...")
        model_name = MODELS["whisper"]["model_name"]
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        logger.info(f"Whisper model '{model_name}' loaded")
    
    def load_yolo(self):
        """Load YOLO for object/UI detection"""
        logger.info("Loading YOLO model...")
        model_path = MODELS["yolo"]["model_path"]
        self.yolo_model = YOLO(model_path)
        logger.info(f"YOLO model loaded from {model_path}")
    
    def load_sentiment(self):
        """Load sentiment analysis model"""
        logger.info("Loading sentiment analysis model...")
        model_name = MODELS["sentiment"]["model_name"]
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=0 if self.device == "cuda" else -1
        )
        logger.info("Sentiment model loaded")
    
    def load_mediapipe(self):
        """Load MediaPipe models for pose, face, and hands"""
        logger.info("Loading MediaPipe models...")
        
        # Pose detection
        self.pose_detector = mp.solutions.pose.Pose(
            model_complexity=MODELS["pose_estimation"]["model_complexity"],
            min_detection_confidence=MODELS["pose_estimation"]["min_detection_confidence"],
            min_tracking_confidence=MODELS["pose_estimation"]["min_tracking_confidence"]
        )
        
        # Face detection
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        
        # Face mesh for detailed facial features
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=2,
            min_detection_confidence=0.5
        )
        
        # Hand detection
        self.hands_detector = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        
        logger.info("MediaPipe models loaded")
    
    def load_ocr(self):
        """Load EasyOCR for text recognition"""
        logger.info("Loading OCR model...")
        languages = MODELS["ocr"]["languages"]
        gpu = MODELS["ocr"]["gpu"] and self.device == "cuda"
        self.ocr_reader = easyocr.Reader(languages, gpu=gpu)
        logger.info("OCR model loaded")
    
    def load_embeddings(self):
        """Load sentence transformer for semantic similarity"""
        logger.info("Loading embedding model...")
        model_name = MODELS["embedding"]["model_name"]
        self.embedding_model = SentenceTransformer(model_name)
        if self.device == "cuda":
            self.embedding_model = self.embedding_model.to(self.device)
        logger.info("Embedding model loaded")
    
    def get_model(self, model_name):
        """Get a specific model by name"""
        model_map = {
            "whisper": self.whisper_model,
            "yolo": self.yolo_model,
            "sentiment": self.sentiment_pipeline,
            "pose": self.pose_detector,
            "face_detection": self.face_detection,
            "face_mesh": self.face_mesh,
            "hands": self.hands_detector,
            "ocr": self.ocr_reader,
            "embedding": self.embedding_model
        }
        
        model = model_map.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' not found or not loaded")
        return model
    
    def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up models...")
        
        if self.pose_detector:
            self.pose_detector.close()
        if self.face_detection:
            self.face_detection.close()
        if self.face_mesh:
            self.face_mesh.close()
        if self.hands_detector:
            self.hands_detector.close()
        
        logger.info("Cleanup complete")
