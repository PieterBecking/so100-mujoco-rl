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

    cam = cv2.VideoCapture(source)

    model = YOLO(object_detection_model_path)
    tracker_path = Path(__file__).parent / "envs" / "tracker.yaml"
    # id of the cube currently being tracked
    tracking_id = None

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

    cached_ob_center_x = 0.5
    cached_ob_center_y = 0.5

    prev_frame_time = 0

    while True:
        # Calculate FPS
        current_frame_time = time.time()
        fps = 1 / (current_frame_time - prev_frame_time) if prev_frame_time != 0 else 0
        prev_frame_time = current_frame_time

        ret, img = cam.read()

        if not ret:
            break

        # Rotate the frame 90 degrees if specified
        if rotate:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

        results = model.track(
            img,
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
                center_y = img.shape[0] - center_y
                center_x_f = center_x / img.shape[1]
                center_y_f = center_y / img.shape[0]
                width = x2 - x1
                height = y2 - y1
                width_f = width / img.shape[1]
                height_f = height / img.shape[0]

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
                img = cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
                # Put label text
                img = cv2.putText(
                    img, label_text, (x1, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2
                )
        
        # Put label text
        img = cv2.putText(
            img, f"x = {obs_center_x_f:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )
        img = cv2.putText(
            img, f"y = {obs_center_y_f:.2f}", (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2
        )

        img = cv2.putText(
            img, f"FPS: {fps:.2f}", (10, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        # print(f"FPS: {fps:.2f}")

        # didn't get good results when using the actual positions returned from the real
        # robot. Hence why the following is commented out
        # arm_controller.update()
        # joint_positions = arm_controller.joint_actual_positions
        obs = [
            *joint_positions,
            cached_ob_center_x,
            cached_ob_center_y * 1.0,
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

        # Display the frame with detections
        cv2.imshow('Camera', img)

        # Check for key presses
        key = cv2.waitKey(1)
        if key == ord('q'):  # Press 'q' to exit the loop
            break

    # Release the capture object
    cam.release()
    cv2.destroyAllWindows()


@click.group()
def cli():
    pass

cli.add_command(run_look_at)



if __name__ == '__main__':
    cli()
