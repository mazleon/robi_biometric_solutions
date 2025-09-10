"""
Streamlit Frontend for Robi Face Verification System
Integrated with our FastAPI backend endpoints
"""

import streamlit as st
import requests
from PIL import Image
from typing import Dict, Any, Optional
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json

# Configure Streamlit page
st.set_page_config(
    page_title="Robi Face Verification System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .result-card {
        background: #f8f9fa;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .similarity-high {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .similarity-medium {
        background: linear-gradient(135deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .similarity-low {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        font-weight: bold;
    }
    
    .upload-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
    }
    
    .stats-container {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
    }
    
    .verified-success {
        background: linear-gradient(135deg, #28a745, #20c997);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
    
    .verified-failed {
        background: linear-gradient(135deg, #dc3545, #e83e8c);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Configuration - Updated to match our API structure
API_BASE_URL = "http://127.0.0.1:8000/api/v1"

def get_similarity_class(score: float) -> str:
    """Get CSS class based on similarity score"""
    if score >= 0.8:
        return "similarity-high"
    elif score >= 0.6:
        return "similarity-medium"
    else:
        return "similarity-low"

def get_confidence_emoji(score: float) -> str:
    """Get emoji based on confidence level"""
    if score >= 0.9:
        return "🎯"
    elif score >= 0.8:
        return "✅"
    elif score >= 0.7:
        return "👍"
    elif score >= 0.6:
        return "⚠️"
    else:
        return "❌"

def format_timing_metrics(process_time_ms: float) -> None:
    """Display timing metrics (without FPS)."""
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>⚡ Process Time</h4>
            <h2>{process_time_ms:.1f}ms</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        performance = "Excellent" if process_time_ms < 500 else "Good" if process_time_ms < 1000 else "Slow"
        emoji = "🚀" if performance == "Excellent" else "⚡" if performance == "Good" else "📈"
        st.markdown(f"""
        <div class="metric-card">
            <h4>{emoji} Performance</h4>
            <h2>{performance}</h2>
        </div>
        """, unsafe_allow_html=True)

def enroll_user_api(image_file, user_id: str, name: Optional[str] = None, metadata: Optional[str] = None) -> Dict[str, Any]:
    """Enroll a user using our API"""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        data = {"user_id": user_id}
        if name:
            data["name"] = name
        if metadata:
            data["metadata"] = metadata
        
        response = requests.post(f"{API_BASE_URL}/enroll", files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Enrollment failed: {str(e)}")
        return {}

def verify_face_api(image_file, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Verify a face using our API"""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        data = {}
        if threshold is not None:
            data["threshold"] = threshold
        
        response = requests.post(f"{API_BASE_URL}/verify", files=files, data=data, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Verification failed: {str(e)}")
        return {}

def detect_faces_api(image_file) -> Dict[str, Any]:
    """Detect faces using our API"""
    try:
        files = {"file": (image_file.name, image_file.getvalue(), image_file.type)}
        
        response = requests.post(f"{API_BASE_URL}/detect", files=files, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Face detection failed: {str(e)}")
        return {}

@st.cache_data(ttl=300)
def get_users_api() -> Dict[str, Any]:
    """Get all users using our API"""
    try:
        response = requests.get(f"{API_BASE_URL}/users", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get users: {str(e)}")
        return {}

def delete_user_api(user_id: str) -> Dict[str, Any]:
    """Delete a user using our API"""
    try:
        response = requests.delete(f"{API_BASE_URL}/users/{user_id}", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to delete user: {str(e)}")
        return {}

@st.cache_data(ttl=60)
def get_system_stats_api() -> Dict[str, Any]:
    """Get system statistics using our API"""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get system stats: {str(e)}")
        return {}

def get_health_api() -> Dict[str, Any]:
    """Get health status using our API"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get health status: {str(e)}")
        return {}

def display_verification_results(results: Dict[str, Any], query_image: Image.Image) -> None:
    """Display verification results"""
    if not results or not results.get('success'):
        st.error("Verification failed or returned no results.")
        return

    # Display timing metrics
    st.subheader("⏱️ Performance Metrics")
    format_timing_metrics(results.get('process_time_ms', 0))

    # Display verification result
    verified = results.get('verified', False)
    confidence = results.get('confidence', 0.0)
    threshold = results.get('threshold', 0.35)
    # Coerce to floats to avoid formatting errors when values are None or non-numeric
    try:
        confidence = float(confidence) if confidence is not None else 0.0
    except Exception:
        confidence = 0.0
    try:
        threshold = float(threshold) if threshold is not None else 0.35
    except Exception:
        threshold = 0.35
    user_id = results.get('user_id')
    name = results.get('name')

    st.subheader("🎯 Verification Result")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(query_image, caption="Query Image", width=300)
    
    with col2:
        if verified:
            st.markdown(f"""
            <div class="verified-success">
                <h2>✅ VERIFIED</h2>
                <h3>User: {user_id}</h3>
                <h4>Name: {name or 'N/A'}</h4>
                <p>Confidence: {confidence:.4f}</p>
                <p>Threshold: {threshold:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verified-failed">
                <h2>❌ NOT VERIFIED</h2>
                <p>No matching face found</p>
                <p>Best confidence: {confidence:.4f}</p>
                <p>Required threshold: {threshold:.4f}</p>
            </div>
            """, unsafe_allow_html=True)

    # Display candidates if available
    candidates = results.get('candidates', [])
    if candidates:
        st.subheader("📊 Similar Faces Found")
        
        # Create similarity chart
        df_candidates = pd.DataFrame([
            {
                'User ID': candidate['user_id'],
                'Name': candidate.get('name', 'N/A'),
                'Similarity': float(candidate.get('similarity') or 0.0)
            }
            for candidate in candidates
        ])
        
        fig = px.bar(
            df_candidates,
            x='User ID',
            y='Similarity',
            color='Similarity',
            color_continuous_scale='RdYlGn',
            title="Similarity Scores for Candidate Matches",
            hover_data=['Name']
        )
        fig.add_hline(y=threshold, line_dash="dash", line_color="red", 
                     annotation_text=f"Threshold ({threshold:.3f})")
        fig.update_layout(height=400)
        st.plotly_chart(fig, width='stretch')
        
        # Display detailed candidate information
        for i, candidate in enumerate(candidates):
            similarity = float(candidate.get('similarity') or 0.0)
            similarity_class = get_similarity_class(similarity)
            confidence_emoji = get_confidence_emoji(similarity)
            with st.expander(f"{confidence_emoji} Candidate {i+1}: {candidate['user_id']} (Similarity: {similarity:.4f})"):
                st.markdown(f"""
                <div class="{similarity_class}">
                    <h4>Similarity: {similarity:.4f}</h4>
                </div>
                <div class="result-card">
                    <p><strong>User ID:</strong> {candidate['user_id']}</p>
                    <p><strong>Name:</strong> {candidate.get('name', 'N/A')}</p>
                    <p><strong>Above Threshold:</strong> {'✅' if similarity >= threshold else '❌'}</p>
                </div>
                """, unsafe_allow_html=True)
                try:
                    img_resp = requests.get(f"{API_BASE_URL}/users/{candidate['user_id']}/image", timeout=5)
                    if img_resp.status_code == 200:
                        img = Image.open(BytesIO(img_resp.content))
                        st.image(img, caption=f"Enrolled Image for {candidate['user_id']}", width=300)
                except Exception:
                    st.warning("Could not load candidate image.")

def display_detection_results(results: Dict[str, Any], query_image: Image.Image) -> None:
    """Display face detection results"""
    if not results or not results.get('success'):
        st.error("Face detection failed.")
        return

    # Display timing metrics
    st.subheader("⏱️ Performance Metrics")
    format_timing_metrics(results.get('process_time_ms', 0))

    # Display detection results
    face_count = results.get('face_count', 0)
    faces = results.get('faces', [])

    st.subheader(f"👥 Detected {face_count} Face(s)")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(query_image, caption="Query Image", width=300)
    
    with col2:
        if face_count > 0:
            st.success(f"Successfully detected {face_count} face(s)")
            
            # Display face information
            for i, face in enumerate(faces):
                with st.expander(f"Face {i+1} Details"):
                    bbox = face.get('bbox', [])
                    area = face.get('area', 0)
                    confidence = face.get('confidence', 0)
                    
                    st.markdown(f"""
                    <div class="result-card">
                        <h4>Face {i+1}</h4>
                        <p><strong>Bounding Box:</strong> {bbox}</p>
                        <p><strong>Area:</strong> {area:.0f} pixels</p>
                        <p><strong>Detection Confidence:</strong> {confidence:.4f}</p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No faces detected in the image")

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🔍 Robi Face Verification System</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎛️ Control Panel")
        
        # Get system health
        with st.spinner("Checking system health..."):
            health = get_health_api()
        
        if health and health.get('status') == 'healthy':
            st.success("✅ System Online")
        else:
            st.error("❌ System Offline")
        
        # Get system stats
        with st.spinner("Loading system stats..."):
            stats = get_system_stats_api()
        
        if stats and stats.get('success'):
            db_stats = stats.get('database_stats', {})
            vector_stats = stats.get('vector_store_stats', {})
            
            st.markdown(f"""
            <div class="stats-container">
                <h3>📊 System Status</h3>
                <p><strong>Total Users:</strong> {db_stats.get('total_users', 0)}</p>
                <p><strong>Vector Embeddings:</strong> {vector_stats.get('total_vectors', 0)}</p>
                <p><strong>Index Status:</strong> {'✅' if vector_stats.get('index_trained') else '❌'}</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Verification settings
        st.markdown("### 🔧 Verification Settings")
        # Use backend-configured threshold as default, fallback to 0.35
        default_threshold = 0.35
        try:
            default_threshold = float(stats.get('system_info', {}).get('cosine_threshold', default_threshold))
        except Exception:
            pass
        similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, default_threshold, 0.01)
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("Face verification system using InsightFace and FAISS for fast similarity search.")
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔍 Verify Face", "📤 Enroll User", "👁️ Detect Faces", "👥 Manage Users", "📊 System Stats"])
    
    with tab1:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("## 🔍 Face Verification")
        st.markdown("Upload an image to verify if the person is enrolled in the system")
        st.markdown('</div>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose an image file for verification",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload a clear image with a visible face",
            key="verify_tab"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Image to Verify", width=400)
            
            if st.button("🔍 Verify Face", type="primary", width='stretch'):
                with st.spinner("🔄 Processing verification..."):
                    uploaded_file.seek(0)
                    results = verify_face_api(uploaded_file, similarity_threshold)
                    
                    if results:
                        display_verification_results(results, image)
    
    with tab2:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("## 📤 User Enrollment")
        st.markdown("Enroll a new user by uploading their face image")
        st.markdown('</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            user_id = st.text_input(
                "User ID *",
                placeholder="Enter unique user ID",
                help="Required: Unique identifier for the user"
            )
        
        with col2:
            name = st.text_input(
                "Full Name",
                placeholder="Enter user's full name",
                help="Optional: Display name for the user"
            )
        
        metadata = st.text_area(
            "Metadata (JSON)",
            placeholder='{"department": "IT", "role": "developer"}',
            help="Optional: Additional user information in JSON format"
        )
        
        upload_file = st.file_uploader(
            "Choose face image for enrollment",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload a clear image with a visible face",
            key="enroll_tab"
        )
        
        if upload_file is not None:
            image = Image.open(upload_file)
            st.image(image, caption="Image to Enroll", width=400)
            
            if st.button("📤 Enroll User", type="primary", width='stretch'):
                if not user_id:
                    st.error("User ID is required for enrollment")
                else:
                    with st.spinner("🔄 Processing enrollment..."):
                        upload_file.seek(0)
                        results = enroll_user_api(upload_file, user_id, name, metadata)
                        
                        if results and results.get('success'):
                            st.success("✅ User enrolled successfully!")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown(f"""
                                <div class="result-card">
                                    <h4>📋 Enrollment Details</h4>
                                    <p><strong>User ID:</strong> {results.get('user_id')}</p>
                                    <p><strong>Face Count:</strong> {results.get('face_count', 0)}</p>
                                    <p><strong>Embedding ID:</strong> {results.get('embedding_id', 'N/A')}</p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            with col2:
                                st.markdown("### ⏱️ Enrollment Performance")
                                format_timing_metrics(results.get('process_time_ms', 0))
                        else:
                            st.error("❌ Enrollment failed")
    
    with tab3:
        st.markdown('<div class="upload-section">', unsafe_allow_html=True)
        st.markdown("## 👁️ Face Detection")
        st.markdown("Detect and analyze faces in an image without verification")
        st.markdown('</div>', unsafe_allow_html=True)
        
        detect_file = st.file_uploader(
            "Choose an image file for face detection",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff', 'webp'],
            help="Upload an image to detect faces",
            key="detect_tab"
        )
        
        if detect_file is not None:
            image = Image.open(detect_file)
            st.image(image, caption="Image for Face Detection", width=400)
            
            if st.button("👁️ Detect Faces", type="primary", width='stretch'):
                with st.spinner("🔄 Detecting faces..."):
                    detect_file.seek(0)
                    results = detect_faces_api(detect_file)
                    
                    if results:
                        display_detection_results(results, image)
    
    with tab4:
        st.markdown("## 👥 User Management")
        
        # Get users
        with st.spinner("Loading users..."):
            users_data = get_users_api()
        
        if users_data and users_data.get('success'):
            users = users_data.get('users', [])
            total_count = users_data.get('total_count', 0)
            
            st.markdown(f"### Total Users: {total_count}")
            
            if users:
                # Pagination settings
                users_per_page = 10
                total_pages = (len(users) + users_per_page - 1) // users_per_page
                
                # Initialize session state for pagination
                if 'current_page' not in st.session_state:
                    st.session_state.current_page = 1
                
                # Page navigation
                col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
                
                with col1:
                    if st.button("⏮️ First", disabled=st.session_state.current_page == 1):
                        st.session_state.current_page = 1
                        st.rerun()
                
                with col2:
                    if st.button("◀️ Previous", disabled=st.session_state.current_page == 1):
                        st.session_state.current_page -= 1
                        st.rerun()
                
                with col3:
                    st.markdown(f"<div style='text-align: center; padding: 8px;'><strong>Page {st.session_state.current_page} of {total_pages}</strong></div>", unsafe_allow_html=True)
                
                with col4:
                    if st.button("▶️ Next", disabled=st.session_state.current_page == total_pages):
                        st.session_state.current_page += 1
                        st.rerun()
                
                with col5:
                    if st.button("⏭️ Last", disabled=st.session_state.current_page == total_pages):
                        st.session_state.current_page = total_pages
                        st.rerun()
                
                # Calculate pagination
                start_idx = (st.session_state.current_page - 1) * users_per_page
                end_idx = min(start_idx + users_per_page, len(users))
                current_users = users[start_idx:end_idx]
                
                # Display users in a table
                df_users = pd.DataFrame([
                    {
                        'User ID': user['user_id'],
                        'Name': user.get('name', 'N/A'),
                        'Created': user.get('created_at', 'N/A'),
                        'Updated': user.get('updated_at', 'N/A')
                    }
                    for user in current_users
                ])
                
                st.dataframe(df_users, width='stretch')
                
                st.markdown(f"Showing {start_idx + 1}-{end_idx} of {total_count} users")
                
                # User deletion section with pagination
                st.markdown("### 🗑️ Delete User")
                
                for user in current_users:
                    col1, col2, col3 = st.columns([3, 3, 2])
                    with col1:
                        st.markdown(f"**{user['user_id']}**")
                    with col2:
                        st.markdown(f"Person: {user.get('name', 'N/A')}")
                    with col3:
                        if st.button(f"Delete {user['user_id']}", key=f"delete_{user['user_id']}", type="secondary"):
                            with st.spinner(f"Deleting user {user['user_id']}..."):
                                delete_result = delete_user_api(user['user_id'])
                                if delete_result and delete_result.get('success'):
                                    st.success(f"✅ User {user['user_id']} deleted successfully")
                                    st.cache_data.clear()
                                    # Reset to first page if current page becomes empty
                                    remaining_users = len(users) - 1
                                    max_pages = (remaining_users + users_per_page - 1) // users_per_page if remaining_users > 0 else 1
                                    if st.session_state.current_page > max_pages:
                                        st.session_state.current_page = max_pages
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to delete user")
            else:
                st.info("No users enrolled in the system")
        else:
            st.error("Failed to load users")
    
    with tab5:
        st.markdown("## 📊 System Statistics")
        
        # Refresh button
        if st.button("🔄 Refresh Stats", type="secondary"):
            st.rerun()
        
        with st.spinner("Loading system statistics..."):
            stats = get_system_stats_api()
        
        if stats and stats.get('success'):
            db_stats = stats.get('database_stats', {})
            vector_stats = stats.get('vector_store_stats', {})
            system_info = stats.get('system_info', {})
            
            # Database stats
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"""
                <div class="result-card">
                    <h4>🗄️ Database Statistics</h4>
                    <p><strong>Total Users:</strong> {db_stats.get('total_users', 0)}</p>
                    <p><strong>Database Path:</strong> {db_stats.get('database_path', 'N/A')}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="result-card">
                    <h4>🔍 Vector Store Statistics</h4>
                    <p><strong>Total Vectors:</strong> {vector_stats.get('total_vectors', 0)}</p>
                    <p><strong>Index Trained:</strong> {'✅' if vector_stats.get('index_trained') else '❌'}</p>
                    <p><strong>Dimension:</strong> {vector_stats.get('dimension', 0)}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # System info
            st.markdown(f"""
            <div class="result-card">
                <h4>⚙️ System Information</h4>
                <p><strong>Uptime:</strong> {system_info.get('uptime_seconds', 0):.0f} seconds</p>
                <p><strong>Cosine Threshold:</strong> {system_info.get('cosine_threshold', 0)}</p>
                <p><strong>Detection Size:</strong> {system_info.get('detection_size', 'N/A')}</p>
                <p><strong>Max File Size:</strong> {system_info.get('max_file_size', 0):,} bytes</p>
                <p><strong>Allowed Extensions:</strong> .jpg, .jpeg, .png, .bmp, .tiff, .webp</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Performance metrics chart
            process_time = stats.get('process_time_ms', 0)
            st.markdown("### ⏱️ API Performance")
            format_timing_metrics(process_time)
            
        else:
            st.error("Failed to load system statistics")

if __name__ == "__main__":
    main()
