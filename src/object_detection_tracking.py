import os
import cv2
from ultralytics import YOLO
from pathlib import Path

# ==========================================
# 0. Configuration & Parameters
# ==========================================
# 모드 설정: 
#   - "image": 단일 이미지 탐지
#   - "video": 동영상 파일 탐지 및 추적
#   - "sequence": 연속된 이미지 폴더 탐지 및 추적
MODE = "sequence"

BASE_DIR = Path(__file__).resolve().parent.parent

# [설정 1] 단일 이미지 모드용 경로
IMAGE_INPUT_DIR = BASE_DIR / "data" / "image_test"
IMAGE_FILENAME = "v1.jpg"
IMAGE_INPUT_PATH = IMAGE_INPUT_DIR / IMAGE_FILENAME

# [설정 2] 동영상 모드용 경로
VIDEO_INPUT_PATH = BASE_DIR / "data" / "sample_video.mp4"

# [설정 3] 연속 이미지 시퀀스 모드용 경로
SEQUENCE_INPUT_DIR = BASE_DIR / "data" / "sequence_frames"  # 연속 이미지가 들어있는 폴더 경로

# 결과 저장 설정 (이미지만 지원)
SAVE_RESULT = True
OUTPUT_DIR = BASE_DIR / "debug_images"

# 모델 및 추론 설정
MODEL_NAME = "yolov8n.pt"        # 모델명 또는 커스텀 학습 가중치 경로 (예: 'best.pt')
CONF_THRES = 0.25                # Confidence threshold
TRACKER_CONFIG = "botsort.yaml"  # 트래커 설정
DEVICE = "cpu"                   # "0" 또는 "cuda", "cpu"
PLAYBACK_FPS = 30                # sequence 모드 재생 속도 (FPS 기준 대기 시간 계산용)


def process_image(model: YOLO, input_path: str, save_result: bool, output_dir: str, output_filename: str):
    """
    단일 이미지 파일에 대해 YOLOv8n 객체 탐지를 수행하고 결과를 시각화/저장합니다.
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

    results = model.predict(source=img, conf=CONF_THRES, device=DEVICE, verbose=False)
    result = results[0]
    boxes = result.boxes

    det_count = len(boxes) if boxes is not None else 0
    print(f"총 탐지된 객체 수: {det_count}")

    if boxes is not None and len(boxes) > 0:
        for idx, box in enumerate(boxes):
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())
            cls_name = model.names[cls_id]

            print(f"  - [{idx + 1}] Class: {cls_name} (ID: {cls_id}) | Conf: {conf:.2f} | BBox: {xyxy.tolist()}")

            x1, y1, x2, y2 = xyxy
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("YOLOv8 Object Detection", img)
    print("Press 'q' or any key on the window to exit.")

    if save_result:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(save_path, img)
        print(f"결과 이미지 저장 완료: {save_path}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(model: YOLO, input_path: str):
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

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

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

            if boxes is not None and len(boxes) > 0:
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

                    x1, y1, x2, y2 = xyxy
                    color = (0, 255, 0) if track_id is not None else (0, 165, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                    label = f"{id_str} {cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print(f"[Frame {frame_num:04d}] 탐지된 객체 없음")

            cv2.putText(frame, f"Frame: {frame_num}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow("YOLOv8 + BoT-SORT Tracking (Video)", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("\n사용자에 의해 실행 중단 ('q' 키 감지).")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"동영상 처리 완료. 총 {frame_num} 프레임 처리됨.")


def process_image_sequence(model: YOLO, input_dir: Path):
    """
    연속된 이미지(JPG) 폴더를 순서대로 불러와 YOLOv8n + BoT-SORT 추적을 수행하고 연속 재생합니다.
    """
    if not os.path.exists(input_dir):
        print(f"[Error] 입력 이미지 디렉터리를 찾을 수 없습니다: {input_dir}")
        return

    # 1. 디렉터리 내 모든 jpg, png 파일 목록 수집 및 파일명 기준 정렬
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = sorted([
        f for f in os.listdir(input_dir) 
        if f.lower().endswith(valid_extensions)
    ])

    if len(image_files) == 0:
        print(f"[Error] 처리할 이미지 파일이 존재하지 않습니다: {input_dir}")
        return

    print(f"\n--- [IMAGE SEQUENCE DETECTION & TRACKING START] ---")
    print(f"입력 디렉터리: {input_dir}")
    print(f"총 프레임(이미지) 수: {len(image_files)}개")
    print(f"추적 트래커: {TRACKER_CONFIG}")

    # 비디오 재생 대기 시간 계산 (ms)
    delay_ms = int(1000 / PLAYBACK_FPS)

    frame_num = 0

    try:
        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"[Warning] 이미지 로드 실패, 건너뜁니다: {filename}")
                continue

            frame_num += 1
            height, width = frame.shape[:2]

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
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(boxes)

                print(f"[Frame {frame_num:04d} | {filename}] 탐지/추적 객체 수: {len(boxes)}")

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
                print(f"[Frame {frame_num:04d} | {filename}] 탐지된 객체 없음")

            # 파일명 및 프레임 번호 화면 표기
            cv2.putText(frame, f"Frame: {frame_num} ({filename})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 화면 출력 및 재생 속도 조절
            cv2.imshow("YOLOv8 + BoT-SORT Tracking (Image Sequence)", frame)

            # 'q' 키 입력 시 즉시 종료 (delay_ms 값으로 재생 속도 제어)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("\n사용자에 의해 실행 중단 ('q' 키 감지).")
                break

    finally:
        cv2.destroyAllWindows()
        print(f"이미지 시퀀스 처리 완료. 총 {frame_num}개 프레임 처리됨.")


def main():
    # 1. 모델 단 1회 로드 (재사용)
    print(f"YOLO 모델 로딩 중 ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)

    # 2. 모드에 따라 실행
    if MODE == "image":
        process_image(
            model=model,
            input_path=str(IMAGE_INPUT_PATH),
            save_result=SAVE_RESULT,
            output_dir=str(OUTPUT_DIR),
            output_filename=f"{IMAGE_INPUT_PATH.stem}_result{IMAGE_INPUT_PATH.suffix}"
        )
    elif MODE == "video":
        process_video(
            model=model,
            input_path=str(VIDEO_INPUT_PATH)
        )
    elif MODE == "sequence":
        process_image_sequence(
            model=model,
            input_dir=SEQUENCE_INPUT_DIR
        )
    else:
        print(f"[Error] 지원하지 않는 MODE입니다: {MODE}. 'image', 'video', 'sequence' 중 하나를 선택하세요.")


if __name__ == "__main__":
    main()