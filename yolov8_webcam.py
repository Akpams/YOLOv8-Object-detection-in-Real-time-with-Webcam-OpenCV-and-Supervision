from ultralytics import YOLO
import cv2
import argparse
import numpy
# from supervision.tools.detections import Detections, BoxAnnotator
import supervision


def args_argument() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="yahaya work")
    parser.add_argument(
        "--webcam-resolution",
        default=[640, 480],
        nargs=2,
        type=int
    )
    return parser.parse_args()
def main():
    args = args_argument()
    frame_width, frame_height = args.webcam_resolution
    print(f"frame width: {frame_width}, frame height: {frame_height}")
    cap = cv2.VideoCapture(0)

    model = YOLO("yolov8n.pt")
    box_annotator = supervision.BoxAnnotator(thickness=2,
                                    text_thickness=2,
                                    text_scale=1
                                    )
    custom_class= model.model.names
    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"no frame detected!!! ", ret)
            break
        results = model(frame)
        xyxys=[]
        conf=[]
        class_ids=[]

        for result in results[0]:
            class_id=result.boxes.cls.cpu().numpy().astype(int)
            if class_id ==0:
                print(f"class id: {class_id}")

                xyxys.append(result.boxes.xyxy.cpu().numpy())
                conf.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))
        print(class_id)

        detections = supervision.Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )
        labels= [custom_class[class_id]
                for class_id
                in detections.class_id
                ]
        
        frame =box_annotator.annotate(frame, detections=detections, labels=labels)
        cv2.imshow("detect object", frame)
        if cv2.waitKey(1) & 0xff ==ord("q"):
            break
if __name__=="__main__":
    main()