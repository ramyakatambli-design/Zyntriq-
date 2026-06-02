import cv2
import argparse
from ultralytics import YOLO

def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Crowd Management using YOLOv8 and OpenCV")
    parser.add_argument("--source", type=str, default="0", help="Video source (IP camera URL, file path, or '0' for webcam)")
    parser.add_argument("--threshold", type=int, default=5, help="Crowd count threshold for alert")
    parser.add_argument("--output", type=str, default=None, help="Path to save the output video")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model version (e.g., yolov8n.pt, yolov8s.pt)")
    parser.add_argument("--show", action="store_true", help="Display the video window (not recommended for headless environments)")
    args = parser.parse_args()

    # Load the YOLOv8 model
    model = YOLO(args.model)

    # Initialize video capture
    # Convert source to int if it's '0' or other digit
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source}.")
        return

    # Video writer setup if output is specified
    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    print(f"Starting crowd management on {source} with threshold {args.threshold}...")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Run YOLOv8 inference
            results = model(frame, verbose=False)

            # Count people
            person_count = 0
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if cls == 0:  # 'person' class
                        person_count += 1

            # Visualize detections
            annotated_frame = results[0].plot()

            # Add count and alert
            cv2.putText(annotated_frame, f"Count: {person_count}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if person_count > args.threshold:
                print(f"ALERT: Crowd threshold exceeded! Count: {person_count}")
                cv2.putText(annotated_frame, "CROWD ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # Write to output file if requested
            if writer:
                writer.write(annotated_frame)

            # Display the frame if requested
            if args.show:
                cv2.imshow("Crowd Management", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        print("Done.")

if __name__ == "__main__":
    main()
