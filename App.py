import streamlit as st
import cv2
import dlib
import numpy as np
import time
import plotly.graph_objects as go
from collections import deque
from inference_sdk import InferenceHTTPClient
from PIL import Image

class DriverMonitoringApp:
    LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
    MOUTH_IDX = [60, 61, 62, 63, 64, 65, 66, 67]

    def __init__(self):
        self.setup_streamlit()
        self.initialize_detection_systems()
        self.setup_state_variables()

    def setup_streamlit(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Driver Monitoring System",
            page_icon="🚗",
            layout="centered"
        )
        st.title("Driver Monitoring System")
        
        # Center the layout using custom CSS
        st.markdown(""" 
            <style>
                .block-container {
                    padding-top: 2rem;
                    padding-bottom: 2rem;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    flex-direction: column;
                }
                .css-18e3th9 {
                    width: 50%;
                }
            </style>
        """, unsafe_allow_html=True)

    def initialize_detection_systems(self):
        """Initialize detection models and variables"""
        # Roboflow setup
        self.roboflow_client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key="YOUR_API"
        )
        self.model_id = "YOUR_MODEL"

        # Drowsiness detection setup
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        
        # Parameters
        self.ear_threshold = 0.2
        self.mar_threshold = 0.6
        self.ear_values = deque(maxlen=50)
        self.mar_values = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)

    def setup_state_variables(self):
        """Initialize session state variables"""
        if 'start_monitoring' not in st.session_state:
            st.session_state.start_monitoring = False
        if 'current_ear' not in st.session_state:
            st.session_state.current_ear = 0.0
        if 'driver_state' not in st.session_state:
            st.session_state.driver_state = "Normal"
        if 'alert_message' not in st.session_state:
            st.session_state.alert_message = ""

    def get_eye_aspect_ratio(self, eye):
        """Calculate eye aspect ratio"""
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def get_mouth_aspect_ratio(self, mouth):
        """Calculate mouth aspect ratio"""
        A = np.linalg.norm(mouth[2] - mouth[10])
        B = np.linalg.norm(mouth[4] - mouth[8])
        C = np.linalg.norm(mouth[0] - mouth[6])
        return (A + B) / (2.0 * C)

    def detect_phone(self, frame):
        """Detect phone usage using Roboflow model"""
        # Convert frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Convert frame to RGB and save temporarily
        rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
        temp_path = "temp_frame.jpg"
        cv2.imwrite(temp_path, rgb_frame)
        
        # Perform inference
        result = self.roboflow_client.infer(temp_path, model_id=self.model_id)
        
        if 'predictions' in result and result['predictions']:
            return True, "⚠️ Phone Usage Detected"
        return False, "Normal"

    def update_ear_mar_plot(self):
        """Create and update EAR and MAR plot using Plotly"""
        fig = go.Figure()
        
        # Add EAR values line
        fig.add_trace(go.Scatter(
            y=list(self.ear_values),
            mode='lines+markers',
            name='EAR',
            line=dict(color='blue', width=2)
        ))

        # Add MAR values line
        fig.add_trace(go.Scatter(
            y=list(self.mar_values),
            mode='lines+markers',
            name='MAR',
            line=dict(color='green', width=2)
        ))
        
        # Add threshold lines
        fig.add_trace(go.Scatter(
            y=[self.ear_threshold] * len(self.ear_values),
            mode='lines',
            name='EAR Threshold',
            line=dict(color='red', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            y=[self.mar_threshold] * len(self.mar_values),
            mode='lines',
            name='MAR Threshold',
            line=dict(color='orange', width=2, dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Real-time EAR and MAR',
            xaxis_title='Time',
            yaxis_title='Value',
            yaxis_range=[0, 1],
            height=400,
            margin=dict(l=0, r=0, t=40, b=0)
        )
        
        return fig

    def process_frame(self, frame):
        """Process a single frame for all detections"""
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        # Initialize status
        ear = 0.0
        mar = 0.0
        driver_state = "Normal"
        alert_message = ""
        
        # Phone detection
        phone_detected, phone_message = self.detect_phone(frame)
        if phone_detected:
            driver_state = "Distracted"
            alert_message = phone_message
        
        # Face detection and drowsiness monitoring
        for face in faces:
            landmarks = self.predictor(gray, face)
            
            # Get eye landmarks
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) 
                                for n in self.LEFT_EYE_IDX])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) 
                                 for n in self.RIGHT_EYE_IDX])
            
            # Get mouth landmarks
            mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) 
                              for n in self.MOUTH_IDX])
            
            # Calculate EAR and MAR
            ear = (self.get_eye_aspect_ratio(left_eye) + 
                  self.get_eye_aspect_ratio(right_eye)) / 2.0
            mar = self.get_mouth_aspect_ratio(mouth)
            
            # Update EAR and MAR values
            self.ear_values.append(ear)
            self.mar_values.append(mar)
            self.timestamps.append(time.time())
            
            # Check for drowsiness
            if ear < self.ear_threshold:
                driver_state = "Drowsy"
                alert_message = "⚠️ Drowsiness Detected"
        
        return frame, ear, mar, driver_state, alert_message

    def toggle_monitoring(self):
        """Toggle the monitoring state"""
        st.session_state.start_monitoring = not st.session_state.start_monitoring

    def create_layout(self):
        """Create the initial layout with placeholders"""
        col1, col2 = st.columns([2, 1])
        
        with col1:
            video_placeholder = st.empty()
            plot_placeholder = st.empty()
        
        with col2:
            st.subheader("Driver Status")
            ear_placeholder = st.empty()
            mar_placeholder = st.empty()
            state_placeholder = st.empty()
            alert_placeholder = st.empty()
            
            st.button(
                "Start Monitoring" if not st.session_state.start_monitoring else "Stop Monitoring",
                on_click=self.toggle_monitoring,
                key="monitoring_button"
            )
        
        return video_placeholder, plot_placeholder, ear_placeholder, mar_placeholder, state_placeholder, alert_placeholder

    def run(self):
        """Main app loop"""
        # Create initial layout and get placeholders
        video_placeholder, plot_placeholder, ear_placeholder, mar_placeholder, state_placeholder, alert_placeholder = self.create_layout()
        
        # Main monitoring loop
        cap = cv2.VideoCapture(0)
        
        try:
            while True:
                if st.session_state.start_monitoring:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to access webcam")
                        break
                    
                    # Process frame
                    frame, ear, mar, driver_state, alert_message = self.process_frame(frame)
                    
                    # Resize the frame to a smaller size
                    frame_resized = cv2.resize(frame, (640, 480))
                    
                    # Update video feed
                    video_placeholder.image(
                        cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_container_width=True
                    )
                    
                    # Update EAR and MAR
                    ear_placeholder.metric("EAR", f"{ear:.2f}")
                    mar_placeholder.metric("MAR", f"{mar:.2f}")
                    state_placeholder.metric("Driver State", driver_state)
                    alert_placeholder.write(alert_message)
                    
                    # Update the plot
                    plot_placeholder.plotly_chart(
                        self.update_ear_mar_plot(),
                        use_container_width=True,
                        key="ear_mar_plot"
                    )
                    
                    time.sleep(0.1)
                else:
                    time.sleep(0.1)
        finally:
            cap.release()

if __name__ == "__main__":
    app = DriverMonitoringApp()
    app.run()