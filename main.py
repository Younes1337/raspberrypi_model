import streamlit as st
import cv2
import dlib
import numpy as np
import time
import plotly.graph_objects as go
from collections import deque
from inference_sdk import InferenceHTTPClient
from PIL import Image
import os
import pywhatkit as kit
import json
import asyncio

class DriverMonitoringApp:
    LEFT_EYE_IDX = [36, 37, 38, 39, 40, 41]
    RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
    MOUTH_IDX = [60, 61, 62, 63, 64, 65, 66, 67]

    MODELS = {
        "v6 (95.9% Precision, 96.3% Recall)": "dmd-tfiw0/6",
        "v5 (98.3% Precision, 95.1% Recall)": "dmd-tfiw0/5",
        "v3 (97.8% Precision, 93.2% Recall)": "dmd-tfiw0/3"
    }

    def __init__(self):
        self.initialize_session_state()
        self.setup_streamlit()
        self.initialize_detection_systems()
        self.setup_state_variables()
        self.load_configurations()

    def initialize_session_state(self):
        if 'start_monitoring' not in st.session_state:
            st.session_state.start_monitoring = False
        if 'alert_sent' not in st.session_state:
            st.session_state.alert_sent = False

    def setup_streamlit(self):
        st.set_page_config(page_title="Driver Monitoring System", page_icon="🚗", layout="wide")
        st.title("🚗 Driver Monitoring System")
        st.markdown("### Monitor driver behavior in real-time to ensure safety on the road.")
        st.sidebar.markdown("## Controls")
        st.sidebar.button("Start Monitoring" if not st.session_state.start_monitoring else "Stop Monitoring", on_click=self.toggle_monitoring, key="monitoring_button")
        self.setup_configuration_options()

    def setup_configuration_options(self):
        st.sidebar.markdown("## Configuration Options")
        self.ear_threshold = st.sidebar.slider("EAR Threshold", min_value=0.1, max_value=0.3, value=0.2)
        self.mar_threshold = st.sidebar.slider("MAR Threshold", min_value=0.5, max_value=0.7, value=0.6)
        self.drowsy_time_threshold = st.sidebar.slider("Drowsy Time Threshold (seconds)", min_value=1, max_value=10, value=5)
        self.model_choice = st.sidebar.selectbox("Select Model", list(self.MODELS.keys()))
        st.sidebar.button("Save Configurations", on_click=self.save_configurations)

    def load_configurations(self):
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, "r") as file:
                config = json.load(file)
                self.ear_threshold = config.get("ear_threshold", 0.2)
                self.mar_threshold = config.get("mar_threshold", 0.6)
                self.drowsy_time_threshold = config.get("drowsy_time_threshold", 5)
                self.model_choice = config.get("model_choice", list(self.MODELS.keys())[0])
        else:
            self.ear_threshold = 0.2
            self.mar_threshold = 0.6
            self.drowsy_time_threshold = 5
            self.model_choice = list(self.MODELS.keys())[0]

    def save_configurations(self):
        config = {
            "ear_threshold": self.ear_threshold,
            "mar_threshold": self.mar_threshold,
            "drowsy_time_threshold": self.drowsy_time_threshold,
            "model_choice": self.model_choice
        }
        with open("config.json", "w") as file:
            json.dump(config, file)

    def initialize_detection_systems(self):
        self.roboflow_client = InferenceHTTPClient(api_url="https://detect.roboflow.com", api_key="TOJjLSsaTb0Nk4Heiwnf")
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        self.ear_values = deque(maxlen=50)
        self.mar_values = deque(maxlen=50)
        self.timestamps = deque(maxlen=50)
        self.blink_timestamps = deque(maxlen=100)
        self.drowsy_start_time = None

    def setup_state_variables(self):
        initial_state = {
            "current_ear": 0.0,
            "current_mar": 0.0,
            "driver_state": "Normal",
            "alert_message": "",
            "class": "",
            "confidence": 0.0,
            "class_counts": {
                "DangerousDriving": 0,
                "Distracted": 0,
                "Drinking": 0,
                "SafeDriving": 0,
                "SleepyDriving": 0,
                "Yawn": 0
            }
        }
        for key, value in initial_state.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def get_eye_aspect_ratio(self, eye):
        A, B, C = np.linalg.norm(eye[1] - eye[5]), np.linalg.norm(eye[2] - eye[4]), np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C)

    def get_mouth_aspect_ratio(self, mouth):
        A = np.linalg.norm(mouth[1] - mouth[7])  # Vertical distance (top to bottom)
        B = np.linalg.norm(mouth[2] - mouth[6])  # Vertical distance
        C = np.linalg.norm(mouth[3] - mouth[5])  # Vertical distance
        D = np.linalg.norm(mouth[0] - mouth[4])  # Horizontal distance (corner to corner)

        mar = (A + B + C) / (2.0 * D)  # MAR formula
        return mar

    def detect_phone(self, frame):
        model_id = self.MODELS[self.model_choice]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        cv2.imwrite("temp_frame.jpg", rgb_frame)
        result = self.roboflow_client.infer("temp_frame.jpg", model_id=model_id)
        if 'predictions' in result and result['predictions']:
            prediction = result['predictions'][0]
            class_name = prediction['class']
            confidence = prediction['confidence'] * 100  # Convert to percentage
            if class_name in st.session_state["class_counts"]:
                st.session_state["class_counts"][class_name] += 1
            self.log_event(class_name, confidence)
            return True, class_name, confidence
        return False, "Normal", 0.0

    def update_plots(self):
        ear_fig = go.Figure()
        mar_fig = go.Figure()
        
        ear_fig.add_trace(go.Scatter(y=list(self.ear_values), mode='lines+markers', name='EAR', line=dict(color='blue', width=2)))
        ear_fig.add_trace(go.Scatter(y=[self.ear_threshold] * len(self.ear_values), mode='lines', name='EAR Threshold', line=dict(color='red', width=2, dash='dash')))
        ear_fig.update_layout(title='EAR Over Time', xaxis_title='Time', yaxis_title='EAR', yaxis_range=[0, 1], height=400)
        
        mar_fig.add_trace(go.Scatter(y=list(self.mar_values), mode='lines+markers', name='MAR', line=dict(color='green', width=2)))
        mar_fig.add_trace(go.Scatter(y=[self.mar_threshold] * len(self.mar_values), mode='lines', name='MAR Threshold', line=dict(color='orange', width=2, dash='dash')))
        mar_fig.update_layout(title='MAR Over Time', xaxis_title='Time', yaxis_title='MAR', yaxis_range=[0, 1], height=400)
        
        return ear_fig, mar_fig

    def update_class_counts_plot(self):
        class_counts = st.session_state["class_counts"]
        class_names = list(class_counts.keys())
        counts = list(class_counts.values())

        class_counts_fig = go.Figure()
        class_counts_fig.add_trace(go.Bar(x=class_names, y=counts, marker_color='blue'))
        class_counts_fig.update_layout(title='Class Counts', xaxis_title='Class', yaxis_title='Count', height=400)

        return class_counts_fig

    def calculate_focus_rate(self):
        safe_driving_count = st.session_state["class_counts"]["SafeDriving"]
        total_detections = sum(st.session_state["class_counts"].values())
        if total_detections == 0:
            return 0.0  # Avoid division by zero
        focus_rate = safe_driving_count / total_detections
        return focus_rate

    def update_focus_rate_plot(self):
        focus_rate = self.calculate_focus_rate()
        focus_rate_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=focus_rate * 100,  # Convert to percentage
            title={'text': "Driver Focus Rate (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green" if focus_rate > 80 else "red"},
                'steps': [
                    {'range': [0, 50], 'color': "red"},
                    {'range': [50, 80], 'color': "yellow"},
                    {'range': [80, 100], 'color': "green"}
                ],
            }
        ))
        return focus_rate_fig

    def calculate_blink_rate(self):
        current_time = time.time()
        self.blink_timestamps.append(current_time)
        # Remove blinks older than 60 seconds
        while self.blink_timestamps and current_time - self.blink_timestamps[0] > 60:
            self.blink_timestamps.popleft()
        blink_rate = len(self.blink_timestamps)  # Blinks per minute
        return blink_rate

    def update_blink_rate_plot(self):
        blink_rate = self.calculate_blink_rate()
        blink_rate_fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=blink_rate,
            title={'text': "Blink Rate (blinks per minute)"},
            gauge={
                'axis': {'range': [0, 60]},
                'bar': {'color': "green" if blink_rate > 10 else "red"},
                'steps': [
                    {'range': [0, 10], 'color': "red"},
                    {'range': [10, 20], 'color': "yellow"},
                    {'range': [20, 60], 'color': "green"}
                ],
            }
        ))
        return blink_rate_fig

    def log_event(self, class_name, confidence):
        event = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "class": class_name,
            "confidence": confidence
        }
        with open("event_log.json", "a") as file:
            file.write(json.dumps(event) + "\n")

    async def send_whatsapp_alert(self, message):
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, kit.sendwhatmsg_instantly, "+2120625789782", message)
        print("WhatsApp message sent!")

    async def check_and_send_alert(self):
        if not st.session_state.alert_sent:
            await self.send_whatsapp_alert("Drowsiness Detected! Please take a break.")
            st.session_state.alert_sent = True

    def process_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        ear, mar, driver_state, alert_message = 0.0, 0.0, "Normal", ""
        phone_detected, class_name, confidence = self.detect_phone(frame)
        st.session_state["class"] = class_name
        st.session_state["confidence"] = confidence
        
        if not phone_detected and ear < self.ear_threshold:
            if self.drowsy_start_time is None:
                self.drowsy_start_time = time.time()
            elif time.time() - self.drowsy_start_time >= self.drowsy_time_threshold:
                driver_state = "Drowsy"
                alert_message = "⚠️ Drowsiness Detected"
                st.session_state["class_counts"]["SleepyDriving"] += 1
                asyncio.run(self.check_and_send_alert())
        else:
            self.drowsy_start_time = None
            st.session_state.alert_sent = False
            if phone_detected:
                if class_name == "SafeDriving":
                    driver_state, alert_message = "SafeDriving", f"✅ {class_name} Detected ({confidence:.2f}%)"
                else:
                    driver_state, alert_message = "Distracted", f"⚠️ {class_name} Detected ({confidence:.2f}%)"
        
        for face in faces:
            landmarks = self.predictor(gray, face)
            left_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in self.LEFT_EYE_IDX])
            right_eye = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in self.RIGHT_EYE_IDX])
            mouth = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in self.MOUTH_IDX])
            ear = (self.get_eye_aspect_ratio(left_eye) + self.get_eye_aspect_ratio(right_eye)) / 2.0
            mar = self.get_mouth_aspect_ratio(mouth)
            self.ear_values.append(ear)
            self.mar_values.append(mar)
            self.timestamps.append(time.time())
            if ear < self.ear_threshold:  # Blink detected
                self.blink_timestamps.append(time.time())
        return frame, ear, mar, driver_state, alert_message

    def toggle_monitoring(self):
        st.session_state.start_monitoring = not st.session_state.start_monitoring
        if not st.session_state.start_monitoring:
            self.save_configurations()

    def create_layout(self):
        col1, col2 = st.columns([2, 1])
        video_placeholder, ear_plot_placeholder, mar_plot_placeholder = col1.empty(), col1.empty(), col1.empty()
        ear_placeholder, mar_placeholder, state_placeholder, alert_placeholder, class_placeholder, confidence_placeholder, class_counts_placeholder, focus_rate_placeholder, blink_rate_placeholder = col2.empty(), col2.empty(), col2.empty(), col2.empty(), col2.empty(), col2.empty(), col2.empty(), col2.empty(), col2.empty()
        return video_placeholder, ear_plot_placeholder, mar_plot_placeholder, ear_placeholder, mar_placeholder, state_placeholder, alert_placeholder, class_placeholder, confidence_placeholder, class_counts_placeholder, focus_rate_placeholder, blink_rate_placeholder

    def run(self):
        video_placeholder, ear_plot_placeholder, mar_plot_placeholder, ear_placeholder, mar_placeholder, state_placeholder, alert_placeholder, class_placeholder, confidence_placeholder, class_counts_placeholder, focus_rate_placeholder, blink_rate_placeholder = self.create_layout()
        cap = cv2.VideoCapture(0)
        try:
            while st.session_state.start_monitoring:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access webcam")
                    break
                frame, ear, mar, driver_state, alert_message = self.process_frame(frame)
                video_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)  # Display the frame without resizing
                ear_fig, mar_fig = self.update_plots()
                class_counts_fig = self.update_class_counts_plot()
                focus_rate_fig = self.update_focus_rate_plot()
                blink_rate_fig = self.update_blink_rate_plot()
                ear_plot_placeholder.plotly_chart(ear_fig, use_container_width=True, key="ear_plot_" + str(time.time()))
                mar_plot_placeholder.plotly_chart(mar_fig, use_container_width=True, key="mar_plot_" + str(time.time()))
                class_counts_placeholder.plotly_chart(class_counts_fig, use_container_width=True, key="class_counts_plot_" + str(time.time()))
                focus_rate_placeholder.plotly_chart(focus_rate_fig, use_container_width=True, key="focus_rate_plot_" + str(time.time()))
                blink_rate_placeholder.plotly_chart(blink_rate_fig, use_container_width=True, key="blink_rate_plot_" + str(time.time()))

                ear_color = "red" if ear < self.ear_threshold else "green"
                mar_color = "red" if mar > self.mar_threshold else "green"

                ear_placeholder.markdown(
                    f"<div style='padding:10px;border-radius:5px;'>EAR Value: <span style='color:{ear_color};'>{ear:.3f}</span></div>", 
                    unsafe_allow_html=True
                )
                mar_placeholder.markdown(
                    f"<div style='padding:10px;border-radius:5px;'>MAR Value: <span style='color:{mar_color};'>{mar:.3f}</span></div>", 
                    unsafe_allow_html=True
                )

                state_placeholder.markdown(f"**Driver State:** {driver_state}")
                if alert_message:
                    alert_placeholder.error(alert_message)
                elif driver_state == "SafeDriving":
                    alert_placeholder.success(f"✅ {driver_state}")
                
                time.sleep(0.1)
        finally:
            cap.release()

if __name__ == "__main__":
    app = DriverMonitoringApp()
    app.run()