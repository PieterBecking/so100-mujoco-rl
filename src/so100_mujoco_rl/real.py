import click
import cv2
from ultralytics import YOLO
from pathlib import Path

@click.command(name="look-at", help="Uses a trained YOLO model to detect objects in web camera feed")
@click.option('-r', '--rotate', is_flag=True, help="Rotate the web camera 90 degrees (default: False)")
@click.option('-s', '--source', default=0, type=int, help="Web camera source (default: 0)")
@click.option('-d', '--device', default='cpu', type=str, help="Device used to run YOLO model (default: 'cpu')")
@click.option('-omp', '--object-detection-model-path', required=True, type=str, help="Full path to the trained model for object detection (weights *.pt file)")
@click.option('-rp', '--robot-policy-path', required=True, type=str, help="Full path to the trained policy for moving arm (*.zip file)")
def run_look_at(
        rotate: bool,
        source: int,
        device: str,
        object_detection_model_path: str,
        robot_policy_path: str
    ):

    click.echo("Running detection on images from web camera...")
    click.echo("Rotate: {}".format(rotate))
    click.echo("Source: {}".format(source))
    click.echo("Press 'q' to quit")

    cam = cv2.VideoCapture(source)

    model = YOLO(object_detection_model_path)
    tracker_path = Path(__file__).parent / "envs" / "tracker.yaml"

    tracking_id = None

    while True:
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
                center_x_f = center_x / img.shape[1]
                center_y_f = center_y / img.shape[0]
                width = x2 - x1
                height = y2 - y1
                width_f = width / img.shape[1]
                height_f = height / img.shape[0]

                obs_center_x_f = center_x_f
                obs_center_y_f = center_y_f

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
