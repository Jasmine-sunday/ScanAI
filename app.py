import cv2
import time
import math as m
import mediapipe as mp


# Calculate distance
def findDistance(x1, y1, x2, y2):
    dist = m.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return dist


# Calculate angle.
def findAngle(x1, y1, x2, y2):
    theta = m.acos((y2 - y1) * (-y1) / (m.sqrt(
        (x2 - x1) ** 2 + (y2 - y1) ** 2) * y1))
    degree = int(180 / m.pi) * theta
    return degree


"""
Function to send alert. Use this function to send alert when bad posture detected.
Feel free to get creative and customize as per your convenience.
"""


def sendWarning(x):
    pass


# =============================CONSTANTS and INITIALIZATIONS=====================================#
# Initilize frame counters.
good_frames = 0
bad_frames = 0

# Font type.
font = cv2.FONT_HERSHEY_SIMPLEX

# Colors.
blue = (255, 127, 0)
red = (50, 50, 255)
green = (127, 255, 0)
dark_blue = (127, 20, 0)
light_green = (127, 233, 100)
yellow = (0, 255, 255)
pink = (255, 0, 255)

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
# ===============================================================================================#


if __name__ == "__main__":
    # For webcam input replace file name with 0.
    file_name = 'input.mp4'
    cap = cv2.VideoCapture(file_name)

    # Meta.
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # Video writer.
    video_output = cv2.VideoWriter('output.mp4', fourcc, fps, frame_size)

    display_scale = 0.5  # Adjust the scale (0.5 means half the size, increase or decrease as needed)

# Initialize cumulative posture times
cumulative_bad_time = 0
cumulative_good_time = 0

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Null.Frames")
        break

    h, w = image.shape[:2]

    # Convert image to RGB for processing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    keypoints = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    lm = keypoints.pose_landmarks
    lmPose = mp_pose.PoseLandmark

    # Reset good and bad posture frames count for the current frame
    bad_frames = 0
    good_frames = 0

    if lm:
        # Get coordinates of key landmarks for posture evaluation
        # Shoulders
        l_shldr_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)
        l_shldr_y = int(lm.landmark[lmPose.LEFT_SHOULDER].y * h)
        r_shldr_x = int(lm.landmark[lmPose.RIGHT_SHOULDER].x * w)
        r_shldr_y = int(lm.landmark[lmPose.RIGHT_SHOULDER].y * h)

        # Hips
        l_hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
        l_hip_y = int(lm.landmark[lmPose.LEFT_HIP].y * h)
        r_hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
        r_hip_y = int(lm.landmark[lmPose.RIGHT_HIP].y * h)

        # Knees
        l_knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
        l_knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)
        r_knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
        r_knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

        # Ankles
        l_ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
        l_ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)
        r_ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
        r_ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

        # Calculate distances and angles for various joints (e.g., torso and leg inclinations)
        torso_inclination = findAngle(l_hip_x, l_hip_y, l_shldr_x, l_shldr_y)
        knee_inclination = findAngle(l_hip_x, l_hip_y, l_knee_x, l_knee_y)

        # Draw landmarks and lines between them
        # Upper body
        cv2.circle(image, (l_shldr_x, l_shldr_y), 7, yellow, -1)
        cv2.circle(image, (r_shldr_x, r_shldr_y), 7, pink, -1)
        cv2.circle(image, (l_hip_x, l_hip_y), 7, yellow, -1)

        # Lower body
        cv2.circle(image, (l_knee_x, l_knee_y), 7, yellow, -1)
        cv2.circle(image, (r_knee_x, r_knee_y), 7, pink, -1)
        cv2.circle(image, (l_ankle_x, l_ankle_y), 7, yellow, -1)
        cv2.circle(image, (r_ankle_x, r_ankle_y), 7, pink, -1)

        # Lines connecting landmarks for full body
        cv2.line(image, (l_shldr_x, l_shldr_y), (l_hip_x, l_hip_y), green, 4)  # Torso
        cv2.line(image, (l_hip_x, l_hip_y), (l_knee_x, l_knee_y), green, 4)  # Left thigh
        cv2.line(image, (l_knee_x, l_knee_y), (l_ankle_x, l_ankle_y), green, 4)  # Left leg

        cv2.line(image, (r_shldr_x, r_shldr_y), (r_hip_x, r_hip_y), red, 4)  # Right torso
        cv2.line(image, (r_hip_x, r_hip_y), (r_knee_x, r_knee_y), red, 4)  # Right thigh
        cv2.line(image, (r_knee_x, r_knee_y), (r_ankle_x, r_ankle_y), red, 4)  # Right leg

        # Add text about angles and posture evaluation
        angle_text_string = f'Torso Inclination: {int(torso_inclination)}'
        cv2.putText(image, angle_text_string, (10, 30), font, 0.9, green if torso_inclination < 10 else red, 2)

        # Posture warnings and accumulate good/bad time
        if torso_inclination >= 10:
            bad_frames += 1  # Increment bad frames when bad posture detected
        else:
            good_frames += 1  # Increment good frames when good posture detected

        # Calculate time only for the current frame based on posture detection
        bad_time = (1 / fps) * bad_frames
        good_time = (1 / fps) * good_frames

        # Accumulate time only when specific posture detected
        if bad_frames > 0:
            cumulative_bad_time += bad_time
        if good_frames > 0:
            cumulative_good_time += good_time

        # Display cumulative posture times on the video
        cv2.putText(image, f'Cumulative Bad Posture Time: {cumulative_bad_time:.2f}s', (10, 70), font, 0.8, red, 2)
        cv2.putText(image, f'Cumulative Good Posture Time: {cumulative_good_time:.2f}s', (10, 110), font, 0.8, green, 2)

        if bad_time > 180:
            sendWarning()

    # Resize the frame to the specified scale for display
    display_image = cv2.resize(image, (int(w * display_scale), int(h * display_scale)))

    # Write and display the resized frame
    video_output.write(image)
    cv2.imshow('MediaPipe Pose - Full Body', display_image)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()