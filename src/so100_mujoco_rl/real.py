import queue
import threading
import time
from pathlib import Path

import click
import cv2
import gymnasium as gym
import stable_baselines3
from ultralytics import YOLO

# while not called directly, we need to import this so the environments are registered
import so100_mujoco_rl
from so100_mujoco_rl.arm_control import So100ArmController
from so100_mujoco_rl.envs.utils import JOINT_STEP_SCALE


# Shared queues
frame_queue = queue.Queue(maxsize=1)
detection_queue = queue.Queue(maxsize=1)
display_queue = queue.Queue(maxsize=1)


# Thread 1: Frame capture
def capture_frames(rotate: bool):
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    while True:
        time.sleep(0.001)
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Frame capture failed")
            continue
        # Rotate the frame 90 degrees if specified
        if rotate:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if ret:
            if not frame_queue.full():
                frame_queue.put(frame)
            else:
                print("Frame queue is full, skipping frame")


# Thread 2: Object detection
def object_detection(object_detection_model_path: str, device: str):
    prev_frame_time = 0
    model = YOLO(object_detection_model_path)
    cached_ob_center_x = 0.5
    cached_ob_center_y = 0.5
    tracking_id = None
    tracker_path = Path(__file__).parent / "envs" / "tracker.yaml"
    fps_alpha = 0.1  # Smoothing factor for FPS
    smoothed_fps = 0  # Initialize smoothed FPS
    while True:
        time.sleep(0.001)
        if not frame_queue.empty():
            current_frame_time = time.time()
            raw_fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
            prev_frame_time = current_frame_time
            # Apply high-pass filter to FPS
            smoothed_fps = fps_alpha * raw_fps + (1 - fps_alpha) * smoothed_fps
            fps = smoothed_fps

            frame = frame_queue.get()

            results = model.track(
                frame,
                persist=True,
                device=device,
                verbose=False,
                tracker=str(tracker_path),
                conf=0.25,
                iou=0.3,
            )

            obs_center_x_f = -1.0
            obs_center_y_f = -1.0

            for result in results:
                for box in result.boxes:
                    # if int(box.cls[0]) != 1:
                    #     continue
                    confidence = box.conf[0]
                    if confidence < 0.4:
                        continue
                    if tracking_id is None and box.id is not None and confidence > 0.5:
                        tracking_id = box.id[0]
                    if tracking_id is not None and box.id is not None and box.id[0] != tracking_id:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2
                    # need to flip the coordinates as the y axis is different for the
                    # offscreen renders that the policy was trained on
                    center_y = frame.shape[0] - center_y
                    center_x_f = center_x / frame.shape[1]
                    center_y_f = center_y / frame.shape[0]
                    width = x2 - x1
                    height = y2 - y1
                    width_f = width / frame.shape[1]
                    height_f = height / frame.shape[0]

                    obs_center_x_f = center_x_f
                    obs_center_y_f = center_y_f
                    cached_ob_center_x = center_x_f
                    cached_ob_center_y = center_y_f

            if obs_center_x_f == -1.0 and obs_center_y_f == -1.0:
                tracking_id = None

            for result in results:
                for box in result.boxes:
                    # if int(box.cls[0]) != 1:
                    #     continue
                    confidence = box.conf[0]
                    if confidence < 0.4:
                        continue
                    if int(box.cls[0]) != 0:
                        c = (255, 255, 0)
                    elif tracking_id is not None and box.id is not None and box.id[0] == tracking_id:
                        c = (0, 255, 0)
                    elif tracking_id is not None and box.id is not None and box.id[0] != tracking_id:
                        c = (255, 0, 0)
                    else:
                        c = (0, 0, 255)
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    label = box.cls[0]
                    label_text = f"{model.names[int(label)]} {confidence:.2f}"

                    # Draw bounding box
                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
                    # Put label text
                    frame = cv2.putText(
                        frame, label_text, (x1, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2
                    )

            # Put label text
            frame = cv2.putText(
                frame, f"x = {obs_center_x_f:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 2
            )
            frame = cv2.putText(
                frame, f"y = {obs_center_y_f:.2f}", (10, 88), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 2
            )
            frame = cv2.putText(
                frame, f"FPS: {fps:.2f}", (10, 68*2), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 0, 0), 2
            )

            # Send to both policy and display
            if not detection_queue.full():
                detection = [
                    cached_ob_center_x,
                    cached_ob_center_y
                ]
                detection_queue.put(detection)
            if not display_queue.full():
                display_queue.put(frame)


@click.command(name="look-at", help="Uses a trained YOLO model to detect objects in web camera feed")
@click.option('-r', '--rotate', is_flag=True, help="Rotate the web camera 90 degrees (default: False)")
@click.option('-s', '--source', default=0, type=int, help="Web camera source (default: 0)")
@click.option('-d', '--device', default='cpu', type=str, help="Device used to run YOLO model (default: 'cpu')")
@click.option('-omp', '--object-detection-model-path', required=True, type=str, help="Full path to the trained model for object detection (weights *.pt file)")
@click.option('-rp', '--robot-policy-path', required=True, type=str, help="Full path to the trained policy for moving arm (*.zip file)")
@click.option(
    '-a',
    '--algorithm',
    required=True,
    type=str,
    default='PPO',
    help="Stable Baseline3 algorithm used to train policy (eg; A2C, DDPG, DQN, PPO, SAC, TD3)"
)
@click.option(
    '-p',
    '--port',
    default=None,
    type=str,
    help="USB port for the robot"
)
@click.option('-e', '--environment', required=True, type=str, help="id of Gymnasium environment (eg; Env01-v1)")
def run_look_at(
        rotate: bool,
        source: int,
        device: str,
        object_detection_model_path: str,
        robot_policy_path: str,
        algorithm: str,
        port: str,
        environment: str
    ):

    click.echo("Running detection on images from web camera...")
    click.echo("Rotate: {}".format(rotate))
    click.echo("Source: {}".format(source))
    click.echo("Press 'q' to quit")

    algorithm_class = getattr(stable_baselines3, algorithm, None)
    env = gym.make(environment, render_mode='human')
    policy = algorithm_class.load(robot_policy_path, env=env)

    arm_controller = So100ArmController()
    src_folder = Path(__file__).parent.parent
    config_folder = src_folder / "configs"
    arm_controller.connect(port, calibration_dir=config_folder)
    time.sleep(0.2)
    arm_controller.update()
    joint_positions = arm_controller.joint_actual_positions

    # Launch threads
    threading.Thread(target=capture_frames, args=(rotate,), daemon=True).start()
    threading.Thread(target=object_detection, args=(object_detection_model_path, device,), daemon=True).start()

    while True:
        time.sleep(0.001)
        if not display_queue.empty():
            frame = display_queue.get()
            # frame = cv2.resize(frame, (frame.shape[1] // 2, frame.shape[0] // 2))

            # can only be done in the main thread
            cv2.imshow('Camera', frame)

            # Check for key presses
            key = cv2.waitKey(1)
            if key == ord('q'):  # Press 'q' to exit the loop
                break

        if not detection_queue.empty():
            # something in this block causes the python interpreter to crash when running
            # in another non-main thread
            detection = detection_queue.get()

            obs = [
                *joint_positions,
                detection[0] * 5.0,
                detection[1] * 5.0,
            ]

            # get the actions from the policy
            a, _ = policy.predict(obs)

            new_joint_positions = [
                joint_positions[i] + float(a[i]) * JOINT_STEP_SCALE for i in range(len(joint_positions))
            ]

            # Apply a high-pass filter to smooth the joint positions
            alpha = 0.2  # Smoothing factor (adjust as needed)
            smoothed_joint_positions = [
                alpha * new_joint_positions[i] + (1 - alpha) * joint_positions[i]
                for i in range(len(joint_positions))
            ]
            # print(f"old_joint_angles: {joint_positions}")
            # print(f"new_joint_angles: {new_joint_positions}")
            arm_controller.set_joint_set_positions(smoothed_joint_positions)
            arm_controller.set_positions()

            joint_positions = list(smoothed_joint_positions)

    cv2.destroyAllWindows()


@click.group()
def cli():
    pass


cli.add_command(run_look_at)


if __name__ == '__main__':
    cli()
