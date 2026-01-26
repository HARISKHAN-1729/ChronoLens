"""
HCI Multi-Video Analysis Tool - Main Application
Interactive Streamlit interface for video analysis
"""
import streamlit as st
import sys
from pathlib import Path
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import UI_CONFIG
from models.model_loader import ModelLoader
from utils.video_processor import get_video_files, validate_video, get_video_info
from utils.visualization import generate_heatmap, generate_timeline, generate_comparison_chart
from utils.json_exporter import export_results
from similarity.segment_matcher import SegmentMatcher

# Import all feature extractors
from feature_extractors.temporal import TemporalFeatureExtractor
from feature_extractors.spatial import SpatialFeatureExtractor
from feature_extractors.interaction_type import InteractionTypeExtractor
from feature_extractors.error_recovery import ErrorRecoveryExtractor
from feature_extractors.navigation import NavigationExtractor
from feature_extractors.content_interaction import ContentInteractionExtractor
from feature_extractors.audio_verbal import AudioVerbalExtractor
from feature_extractors.facial_body import FacialBodyExtractor
from feature_extractors.ui_state import UIStateExtractor
from feature_extractors.context import ContextExtractor
from feature_extractors.cognitive_load import CognitiveLoadExtractor
from feature_extractors.success_outcome import SuccessOutcomeExtractor
from feature_extractors.emotional_affective import EmotionalAffectiveExtractor
from feature_extractors.collaboration import CollaborationExtractor
from feature_extractors.comparative import ComparativeExtractor
from feature_extractors.technical_performance import TechnicalPerformanceExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature extractor mapping
FEATURE_EXTRACTORS = {
    "temporal": ("Temporal Features", TemporalFeatureExtractor),
    "spatial": ("Spatial/Visual Features", SpatialFeatureExtractor),
    "interaction_type": ("Interaction Type", InteractionTypeExtractor),
    "error_recovery": ("Error and Recovery", ErrorRecoveryExtractor),
    "navigation": ("Navigation", NavigationExtractor),
    "content_interaction": ("Content Interaction", ContentInteractionExtractor),
    "audio_verbal": ("Audio/Verbal", AudioVerbalExtractor),
    "facial_body": ("Facial/Body Language", FacialBodyExtractor),
    "ui_state": ("UI State", UIStateExtractor),
    "context": ("Context", ContextExtractor),
    "cognitive_load": ("Cognitive Load", CognitiveLoadExtractor),
    "success_outcome": ("Success/Outcome", SuccessOutcomeExtractor),
    "emotional_affective": ("Emotional/Affective", EmotionalAffectiveExtractor),
    "collaboration": ("Collaboration", CollaborationExtractor),
    "comparative": ("Comparative", ComparativeExtractor),
    "technical_performance": ("Technical/Performance", TechnicalPerformanceExtractor)
}

# Page configuration
st.set_page_config(
    page_title=UI_CONFIG["page_title"],
    page_icon=UI_CONFIG["page_icon"],
    layout=UI_CONFIG["layout"]
)

def main():
    st.title("🎥 HCI Multi-Video Analysis Tool")
    st.markdown("**Comprehensive video analysis using open-source AI models**")
    st.markdown("---")
    
    # Initialize session state
    if 'model_loader' not in st.session_state:
        st.session_state.model_loader = None
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'similar_segments' not in st.session_state:
        st.session_state.similar_segments = None
    
    # Sidebar: Configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Feature selection
        st.subheader("Feature Selection")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Select All"):
                for key in FEATURE_EXTRACTORS.keys():
                    st.session_state[f"feature_{key}"] = True
        with col2:
            if st.button("Deselect All"):
                for key in FEATURE_EXTRACTORS.keys():
                    st.session_state[f"feature_{key}"] = False
        
        st.markdown("---")
        
        selected_features = {}
        for key, (name, _) in FEATURE_EXTRACTORS.items():
            default_value = st.session_state.get(f"feature_{key}", True)
            selected_features[key] = st.checkbox(
                name,
                value=default_value,
                key=f"feature_{key}"
            )
        
        st.markdown("---")
        
        # Similarity detection
        st.subheader("Similarity Detection")
        enable_similarity = st.checkbox("Enable Similarity Detection", value=True)
        similarity_threshold = st.slider(
            "Similarity Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.75,
            step=0.05,
            help="Higher values = more strict matching"
        )
        
        st.markdown("---")
        st.info("💡 **Tip:** Deselect unused features to speed up processing")
    
    # Main content area
    tab1, tab2 = st.tabs(["📁 Input & Processing", "📊 Results"])
    
    with tab1:
        st.header("Video Input")
        
        # Input method selection
        input_method = st.radio(
            "Select input method:",
            ["Folder Path", "File Upload"]
        )
        
        video_files = []
        
        if input_method == "Folder Path":
            folder_path = st.text_input(
                "Enter folder path containing videos:",
                placeholder="/path/to/videos"
            )
            
            if folder_path:
                try:
                    video_files = get_video_files(folder_path)
                    if video_files:
                        st.success(f"✅ Found {len(video_files)} video(s)")
                        with st.expander("View detected videos"):
                            for vf in video_files:
                                st.text(f"• {vf.name}")
                    else:
                        st.warning("No video files found in folder")
                except Exception as e:
                    st.error(f"Error: {e}")
        
        else:  # File Upload
            uploaded_files = st.file_uploader(
                "Upload video files",
                type=['mp4', 'avi', 'mov', 'mkv', 'webm'],
                accept_multiple_files=True
            )
            
            if uploaded_files:
                # Save uploaded files temporarily
                temp_dir = Path("temp/uploads")
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                video_files = []
                for uploaded_file in uploaded_files:
                    temp_path = temp_dir / uploaded_file.name
                    with open(temp_path, 'wb') as f:
                        f.write(uploaded_file.getbuffer())
                    video_files.append(temp_path)
                
                st.success(f"✅ Uploaded {len(video_files)} video(s)")
        
        st.markdown("---")
        
        # Processing section
        if video_files:
            st.header("Processing")
            
            # Count selected features
            num_selected = sum(selected_features.values())
            
            if num_selected == 0:
                st.warning("⚠️ Please select at least one feature to extract")
            else:
                st.info(f"Will extract {num_selected} feature categor{'y' if num_selected == 1 else 'ies'}")
                
                if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
                    process_videos(
                        video_files,
                        selected_features,
                        enable_similarity,
                        similarity_threshold
                    )
    
    with tab2:
        display_results()

def process_videos(video_files, selected_features, enable_similarity, similarity_threshold):
    """Process all videos with selected features"""
    
    # Load models
    with st.spinner("🔄 Loading AI models..."):
        try:
            if st.session_state.model_loader is None:
                st.session_state.model_loader = ModelLoader()
                st.session_state.model_loader.load_all_models()
            st.success("✅ Models loaded successfully!")
        except Exception as e:
            st.error(f"Error loading models: {e}")
            logger.error(f"Model loading error: {e}", exc_info=True)
            return
    
    model_loader = st.session_state.model_loader
    
    # Process each video
    all_results = {}
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for idx, video_path in enumerate(video_files):
        status_text.text(f"Processing {idx + 1}/{len(video_files)}: {video_path.name}")
        
        # Validate video
        is_valid, message = validate_video(video_path)
        if not is_valid:
            st.warning(f"⚠️ Skipping {video_path.name}: {message}")
            continue
        
        video_results = {}
        
        # Extract selected features
        with st.expander(f"📹 {video_path.name}", expanded=False):
            feature_progress = st.progress(0)
            
            selected_list = [(k, v) for k, v in selected_features.items() if v]
            
            for feat_idx, (feature_key, enabled) in enumerate(selected_list):
                if not enabled:
                    continue
                
                feature_name, extractor_class = FEATURE_EXTRACTORS[feature_key]
                
                try:
                    st.text(f"  ⏳ Extracting {feature_name}...")
                    
                    with extractor_class(video_path, model_loader) as extractor:
                        features = extractor.extract()
                        video_results[feature_key] = features
                    
                    st.text(f"  ✅ {feature_name} complete")
                    
                except Exception as e:
                    st.error(f"  ❌ Error in {feature_name}: {str(e)[:100]}")
                    logger.error(f"Feature extraction error ({feature_name}): {e}", exc_info=True)
                    video_results[feature_key] = None
                
                feature_progress.progress((feat_idx + 1) / len(selected_list))
        
        all_results[str(video_path)] = video_results
        progress_bar.progress((idx + 1) / len(video_files))
    
    status_text.text("✅ Video processing complete!")
    
    # Similarity detection
    similar_segments = []
    if enable_similarity and len(all_results) > 1:
        with st.spinner("🔍 Detecting similar segments..."):
            try:
                matcher = SegmentMatcher(all_results, similarity_threshold)
                selected_categories = [k for k, v in selected_features.items() if v]
                similar_segments = matcher.find_all_similarities(selected_categories)
                st.success(f"✅ Found {len(similar_segments)} similar segment groups")
            except Exception as e:
                st.error(f"Error in similarity detection: {e}")
                logger.error(f"Similarity detection error: {e}", exc_info=True)
    
    # Store results in session state
    st.session_state.results = all_results
    st.session_state.similar_segments = similar_segments
    
    # Export JSON
    try:
        output_path = Path("output") / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        export_results(
            all_results,
            similar_segments,
            output_path,
            metadata={
                "selected_features": [k for k, v in selected_features.items() if v],
                "similarity_detection_enabled": enable_similarity,
                "similarity_threshold": similarity_threshold
            }
        )
        st.success(f"📁 Results exported to: {output_path}")
    except Exception as e:
        st.error(f"Error exporting results: {e}")

def display_results():
    """Display analysis results"""
    
    if st.session_state.results is None:
        st.info("👆 Process videos in the 'Input & Processing' tab to see results here")
        return
    
    results = st.session_state.results
    similar_segments = st.session_state.similar_segments or []
    
    # Create result tabs
    result_tabs = st.tabs([
        "📈 Summary",
        "📹 Per-Video Results",
        "🔗 Similarities",
        "📊 Visualizations",
        "💾 Export"
    ])
    
    with result_tabs[0]:  # Summary
        display_summary(results, similar_segments)
    
    with result_tabs[1]:  # Per-Video
        display_per_video(results)
    
    with result_tabs[2]:  # Similarities
        display_similarities(similar_segments)
    
    with result_tabs[3]:  # Visualizations
        display_visualizations(results)
    
    with result_tabs[4]:  # Export
        display_export(results, similar_segments)

def display_summary(results, similar_segments):
    """Display summary statistics"""
    st.header("Summary Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Videos", len(results))
    
    with col2:
        total_features = sum(len(v) for v in results.values())
        st.metric("Total Features Extracted", total_features)
    
    with col3:
        st.metric("Similar Segments Found", len(similar_segments))
    
    st.markdown("---")
    
    # Feature availability matrix
    st.subheader("Feature Extraction Status")
    
    feature_matrix = []
    for video_path, features in results.items():
        video_name = Path(video_path).name
        row = {"Video": video_name}
        
        for feature_key, (feature_name, _) in FEATURE_EXTRACTORS.items():
            row[feature_name] = "✅" if feature_key in features and features[feature_key] else "❌"
        
        feature_matrix.append(row)
    
    st.dataframe(feature_matrix, use_container_width=True)

def display_per_video(results):
    """Display per-video results"""
    st.header("Per-Video Analysis")
    
    video_names = [Path(vp).name for vp in results.keys()]
    selected_video = st.selectbox("Select video:", video_names)
    
    if selected_video:
        # Find the video path
        video_path = None
        for vp in results.keys():
            if Path(vp).name == selected_video:
                video_path = vp
                break
        
        if video_path:
            features = results[video_path]
            
            st.subheader(f"Results for: {selected_video}")
            
            for feature_key, feature_data in features.items():
                if feature_data is None:
                    continue
                
                feature_name, _ = FEATURE_EXTRACTORS[feature_key]
                
                with st.expander(f"📊 {feature_name}", expanded=False):
                    st.json(feature_data)

def display_similarities(similar_segments):
    """Display similar segments"""
    st.header("Cross-Video Similarities")
    
    if not similar_segments:
        st.info("No similar segments found. Try lowering the similarity threshold.")
        return
    
    st.write(f"Found **{len(similar_segments)}** similar segment groups")
    
    for i, segment in enumerate(similar_segments):
        with st.expander(
            f"Segment {i+1}: {segment['similarity_type']} (Score: {segment['similarity_score']:.3f})",
            expanded=False
        ):
            st.json(segment)

def display_visualizations(results):
    """Display visualizations"""
    st.header("Visualizations")
    
    # Heatmap visualization
    st.subheader("Spatial Heatmaps")
    
    for video_path, features in results.items():
        video_name = Path(video_path).name
        
        if "spatial" in features and features["spatial"]:
            heatmap_data = features["spatial"].get("heatmap_data")
            
            if heatmap_data:
                st.write(f"**{video_name}**")
                fig = generate_heatmap(heatmap_data, title=f"Interaction Heatmap - {video_name}")
                st.pyplot(fig)
    
    # Comparison charts
    st.subheader("Cross-Video Comparisons")
    
    # Collect metrics for comparison
    metric_options = []
    for features in results.values():
        for category, category_data in features.items():
            if isinstance(category_data, dict):
                for key in category_data.keys():
                    if isinstance(category_data[key], (int, float)):
                        metric_key = f"{category}.{key}"
                        if metric_key not in metric_options:
                            metric_options.append(metric_key)
    
    if metric_options:
        selected_metric = st.selectbox("Select metric to compare:", metric_options)
        
        if selected_metric:
            category, key = selected_metric.split(".", 1)
            
            video_names = []
            values = []
            
            for video_path, features in results.items():
                if category in features and features[category]:
                    value = features[category].get(key)
                    if value is not None:
                        video_names.append(Path(video_path).name)
                        values.append(value)
            
            if values:
                fig = generate_comparison_chart(video_names, values, selected_metric)
                st.plotly_chart(fig, use_container_width=True)

def display_export(results, similar_segments):
    """Display export options"""
    st.header("Export Results")
    
    # JSON export
    st.subheader("Download JSON")
    
    output_data = {
        "analysis_metadata": {
            "timestamp": datetime.now().isoformat(),
            "tool_version": "1.0.0",
            "total_videos": len(results)
        },
        "videos": results,
        "similar_segments": similar_segments
    }
    
    json_str = json.dumps(output_data, indent=2)
    
    st.download_button(
        label="📥 Download Complete Results (JSON)",
        data=json_str,
        file_name=f"hci_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json",
        use_container_width=True
    )
    
    st.markdown("---")
    
    # Summary report
    st.subheader("Generate Summary Report")
    
    if st.button("📄 Generate Text Report"):
        report = generate_text_report(results, similar_segments)
        st.download_button(
            label="📥 Download Summary Report (TXT)",
            data=report,
            file_name=f"hci_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

def generate_text_report(results, similar_segments):
    """Generate a text summary report"""
    lines = []
    lines.append("HCI VIDEO ANALYSIS SUMMARY REPORT")
    lines.append("=" * 50)
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total Videos Analyzed: {len(results)}")
    lines.append("")
    
    for video_path, features in results.items():
        lines.append(f"Video: {Path(video_path).name}")
        lines.append("-" * 50)
        
        for feature_key, feature_data in features.items():
            if feature_data:
                feature_name, _ = FEATURE_EXTRACTORS[feature_key]
                lines.append(f"  {feature_name}:")
                
                if isinstance(feature_data, dict):
                    for key, value in feature_data.items():
                        if isinstance(value, (int, float)):
                            lines.append(f"    - {key}: {value}")
        
        lines.append("")
    
    lines.append("SIMILAR SEGMENTS")
    lines.append("=" * 50)
    lines.append(f"Total similar segments found: {len(similar_segments)}")
    
    return "\n".join(lines)

if __name__ == "__main__":
    main()
