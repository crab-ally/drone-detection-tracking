import os
import cv2
from ultralytics import YOLO

# ==========================================
# 0. Configuration & Parameters
# ==========================================
# 모드 설정: "image" (이미지 탐지) 또는 "video" (동영상 탐지 및 추적)
MODE = "video"

# 입력 파일 경로
INPUT_PATH = "sample_video.mp4"  # 또는 "sample_image.jpg"

# 결과 저장 설정
SAVE_RESULT = True
OUTPUT_DIR = "output_results"
OUTPUT_FILENAME = "result_output.mp4"  # 이미지는 "result_output.jpg"

# 모델 및 추론 설정
MODEL_NAME = "yolov8n.pt"  # 모델명 또는 커스텀 학습 가중치 경로 (예: 'best.pt')
CONF_THRES = 0.25         # Confidence threshold
TRACKER_CONFIG = "botsort.yaml"  # Ultralytics 지원 트래커: "botsort.yaml" 또는 "bytetrack.yaml"
DEVICE = "0"              # "cuda:0", "0", "cpu" 등 (GPU 사용 시 "0" 또는 "cuda", CPU만 쓸 경우 "cpu")


def process_image(model: YOLO, input_path: str, save_result: bool, output_dir: str, output_filename: str):
    """
    이미지 파일에 대해 YOLOv8n 객체 탐지를 수행하고 결과를 시각화/저장합니다.
    """
    if not os.path.exists(input_path):
        print(f"[Error] 입력 이미지 파일을 찾을 수 없습니다: {input_path}")
        return

    img = cv2.imread(input_path)
    if img is None:
        print(f"[Error] 이미지 파일 로드에 실패했습니다: {input_path}")
        return

    print(f"\n--- [IMAGE DETECTION START] ---")
    print(f"입력 파일: {input_path}")

    # YOLOv8 추론 수행
    results = model.predict(
        source=img,
        conf=CONF_THRES,
        device=DEVICE,
        verbose=False
    )

    result = results[0]
    boxes = result.boxes

    det_count = len(boxes) if boxes is not None else 0
    print(f"총 탐지된 객체 수: {det_count}")

    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            # 좌표, Confidence, Class 번호 및 이름 파싱
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            print(f"  - [{idx + 1}] Class: {cls_name} (ID: {cls_id}) | Conf: {conf:.2f} | BBox: {xyxy.tolist()}")

            # BBox 및 텍스트 시각화
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 결과 표시
    cv2.imshow("YOLOv8 Object Detection", img)
    print("Press 'q' or any key on the window to exit.")

    # 결과 저장
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(save_path, img)
        print(f"결과 이미지 저장 완료: {save_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model: YOLO, input_path: str, save_result: bool, output_dir: str, output_filename: str):
    """
    동영상 파일에 대해 YOLOv8n + BoT-SORT 객체 탐지 및 추적을 수행합니다.
    """
    if not os.path.exists(input_path):
        print(f"[Error] 입력 동영상 파일을 찾을 수 없습니다: {input_path}")
        return

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[Error] 비디오 스트림을 열 수 없습니다: {input_path}")
        return

    # 비디오 메타데이터 추출
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    writer = None
    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_filename)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
        print(f"결과 동영상 저장 위치: {save_path}")

    print(f"\n--- [VIDEO DETECTION & TRACKING START] ---")
    print(f"입력 파일: {input_path}")
    print(f"추적 트래커: {TRACKER_CONFIG}")

    frame_num = 0

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("동영상 재생 완료 또는 프레임 읽기 실패.")
                break

            frame_num += 1

            # Ultralytics 공식 추적 API 사용 (persist=True로 프레임 간 ID 유지)
            results = model.track(
                source=frame,
                conf=CONF_THRES,
                persist=True,
                tracker=TRACKER_CONFIG,
                device=DEVICE,
                verbose=False
            )

            result = results[0]
            boxes = result.boxes

            # 예외 처리: 바운딩 박스가 존재하지 않는 경우 대응
            if boxes is not None and len(boxes) > 0:
                # Track ID 존재 여부 확인 (추적 실패 시 None일 수 있음)
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(boxes)

                print(f"[Frame {frame_num:04d}] 탐지/추적 객체 수: {len(boxes)}")

                for box, track_id in zip(boxes, track_ids):
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names[cls_id]

                    id_str = f"ID:{track_id}" if track_id is not None else "ID: N/A"
                    print(f"  - [{id_str}] Class: {cls_name} | Conf: {conf:.2f} | BBox: {xyxy.tolist()}")

                    # BBox 및 정보 오버레이 시각화
                    x1, y1, x2, y2 = xyxy
                    color = (0, 255, 0) if track_id is not None else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{id_str} {cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print(f"[Frame {frame_num:04d}] 탐지된 객체 없음")

            # 프레임 번호 화면 표기
            cv2.putText(frame, f"Frame: {frame_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 결과 프레임 저장이 설정된 경우 기록
            if writer is not None:
                writer.write(frame)

            # 화면 출력
            cv2.imshow("YOLOv8 + BoT-SORT Tracking", frame)

            # 'q' 키 입력 시 즉시 종료
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n사용자에 의해 실행 중단 ('q' 키 감지).")
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
        print(f"동영상 처리 완료. 총 {frame_num} 프레임 처리됨.")


def main():
    # 1. 모델 단 1회 로드 (재사용)
    print(f"YOLO 모델 로딩 중 ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)

    # 2. 모드에 따라 실행
    if MODE == "image":
        process_image(
            model=model,
            input_path=INPUT_PATH,
            save_result=SAVE_RESULT,
            output_dir=OUTPUT_DIR,
            output_filename=OUTPUT_FILENAME
        )
    elif MODE == "video":
        process_video(
            model=model,
            input_path=INPUT_PATH,
            save_result=SAVE_RESULT,
            output_dir=OUTPUT_DIR,
            output_filename=OUTPUT_FILENAME
        )
    else:
        print(f"[Error] 지원하지 않는 MODE입니다: {MODE}. 'image' 또는 'video'를 사용하세요.")


if __name__ == "__main__":
    main()