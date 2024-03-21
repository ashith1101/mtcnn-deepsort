import cv2
from mtcnn.mtcnn import MTCNN
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

def main():
    # Initialize MTCNN for face detection
    mtcnn = MTCNN()

    # Initialize DeepSORT tracker
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT,
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE,
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE,
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, n_init=cfg_deep.DEEPSORT.N_INIT, nn_budget=cfg_deep.DEEPSORT.NN_BUDGET,
                        use_cuda=True)

    # Capture video
    cap = cv2.VideoCapture("00372.mp4")

    # Initialize unique ID counter
    next_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        try:
            # Detect faces using MTCNN
            faces = mtcnn.detect_faces(frame)
            
            bbox_xyxy = []
            confs = []
            oids = []
            
            for face in faces:
                x, y, w, h = face['box']
                conf = face['confidence']
                
                # Convert bounding box format to DeepSORT input format
                bbox_xyxy.append([x, y, x+w, y+h])
                confs.append([conf])
                
                # Assign a unique ID to each detected face
                oids.append(next_id)
                next_id += 1  # Increment unique ID counter
                
                # Draw bounding box around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Update DeepSORT tracker
            if bbox_xyxy:
                confs_flat = [conf for sublist in confs for conf in sublist]  # Flatten the list of lists
                outputs = deepsort.update(bbox_xyxy, confs_flat, oids, frame)
                for output in outputs:
                    bbox_tlwh = output[:4]
                    identity = output[-2]
                    object_id = output[-1]
                    
                    # Adjust bounding box size based on face area
                    x, y, w, h = bbox_tlwh
                    face_area = (w - x) * (h - y)
                    bbox_ratio = min(1.0, 1.3 * (face_area / ((w - x) * (h - y))))
                    w_new = int((w - x) * bbox_ratio)
                    h_new = int((h - y) * bbox_ratio)
                    x_new = max(0, int(x - (w_new - (w - x)) / 2))
                    y_new = max(0, int(y - (h_new - (h - y)) / 2))
                    bbox_tlwh = [x_new, y_new, x_new + w_new, y_new + h_new]
                    
                    # Draw tracked bounding box and ID
                    cv2.rectangle(frame, (bbox_tlwh[0], bbox_tlwh[1]), (bbox_tlwh[2], bbox_tlwh[3]), (0, 255, 0), 2)
                    cv2.putText(frame, str(identity), (bbox_tlwh[0], bbox_tlwh[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("User exited the program.")
                break

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            continue

    # Release capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
