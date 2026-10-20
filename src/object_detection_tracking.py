import os
import json
import cv2
import yaml
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ==========================================
# 0. Configuration & Parameters
# ==========================================
# 모드 설정: 
#   - "image": 단일 이미지 탐지
#   - "sequence": 연속된 이미지 폴더 탐지 및 추적
MODE = "sequence"

BASE_DIR = Path(__file__).resolve().parent.parent

# [설정 1] 단일 이미지 모드용 경로
IMAGE_INPUT_DIR = BASE_DIR / "data" / "image_test"
IMAGE_FILENAME = "v1.jpg"
IMAGE_INPUT_PATH = IMAGE_INPUT_DIR / IMAGE_FILENAME

# [설정 3] 연속 이미지 시퀀스 모드용 경로
SEQUENCE_INPUT_DIR = BASE_DIR / "data" / "sequence-test" / "video01"

# 결과 저장 설정 (이미지만 지원)
SAVE_RESULT = True
OUTPUT_DIR = BASE_DIR / "debug_images"

# 모델 및 추론 설정
MODEL_NAME = "yolov8n.pt"               # 모델명 또는 커스텀 학습 가중치 경로 (예: 'best.pt')
CONF_THRES = 0.25                       # Confidence threshold
TRACKER_CONFIG = BASE_DIR / "config" / "botsort.yaml"         # 트래커 설정
DEVICE = "cpu"                          # "0" 또는 "cuda", "cpu"
PLAYBACK_FPS = 30                       # sequence 모드 재생 속도

# 개별 설정값 불러오기
with open(TRACKER_CONFIG, "r", encoding="utf-8") as f:
    tracker_config = yaml.safe_load(f)

track_buffer = tracker_config.get("track_buffer", 30)

# 이벤트 판정 및 트래커 버퍼 설정
APPEAR_STREAK_THRES = 6                 # 신규 출현 확정을 위한 최소 연속 탐지 횟수 (0.2초)


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


def process_image_sequence(model: YOLO, input_dir: Path):
    """
    연속된 이미지(JPG) 폴더를 순서대로 불러와 YOLOv8n + BoT-SORT 추적을 수행하고, 
    신규 출현 및 소실 이벤트 판정 및 JSON 로그를 출력합니다.
    """
    if not os.path.exists(input_dir):
        print(f"[Error] 입력 이미지 디렉터리를 찾을 수 없습니다: {input_dir}")
        return

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

    delay_ms = int(1000 / PLAYBACK_FPS)
    frame_num = 0

    # 트랙 상태 관리 딕셔너리 (Key: track_id, Value: dict)
    # 구조: { track_id: { 'hit_streak': int, 'missing_count': int, 'appeared_logged': bool, 'class_name': str, 'conf': float, 'bbox': list } }
    active_tracker_states = {}

    try:
        for filename in image_files:
            img_path = os.path.join(input_dir, filename)
            frame = cv2.imread(img_path)
            
            if frame is None:
                print(f"[Warning] 이미지 로드 실패, 건너뜁니다: {filename}")
                continue

            h, w = frame.shape[:2]
            scale = min(960 / w, 540 / h, 1.0)
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

            # 현재 프레임에서 관측된 트랙 ID 집합
            current_frame_ids = set()

            if boxes is not None and len(boxes) > 0:
                has_ids = boxes.id is not None
                track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(boxes)

                print(f"\n[Frame {frame_num:04d} | {filename}] 탐지/추적 객체 수: {len(boxes)}")

                for box, track_id in zip(boxes, track_ids):
                    if track_id is None:
                        continue  # ID가 부여되지 않은 박스는 추적 대상 제외

                    current_frame_ids.add(track_id)
                    xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                    conf = float(box.conf[0].cpu().numpy())
                    cls_id = int(box.cls[0].cpu().numpy())
                    cls_name = model.names[cls_id]

                    # 1. 트랙 상태 초기화 또는 갱신
                    if track_id not in active_tracker_states:
                        active_tracker_states[track_id] = {
                            "hit_streak": 1,
                            "missing_count": 0,
                            "appeared_logged": False,
                            "class_name": cls_name,
                            "conf": conf,
                            "bbox": xyxy
                        }
                    else:
                        state = active_tracker_states[track_id]
                        state["hit_streak"] += 1
                        state["missing_count"] = 0  # 재등장이므로 미탐지 카운트 리셋
                        state["class_name"] = cls_name
                        state["conf"] = conf
                        state["bbox"] = xyxy

                    state = active_tracker_states[track_id]

                    # 2. 신규 출현 이벤트 판정 (hit_streak가 임계값에 도달하고 아직 로그를 안 찍었을 때)
                    if state["hit_streak"] >= APPEAR_STREAK_THRES and not state["appeared_logged"]:
                        state["appeared_logged"] = True
                        event_data = {
                            "timestamp": datetime.now().isoformat(),
                            "frame_idx": frame_num,
                            "source_file": filename,
                            "event": "TARGET_APPEARED",
                            "target_info": {
                                "track_id": track_id,
                                "class_name": cls_name,
                                "confidence": conf,
                                "bbox": xyxy,
                                "stability_count": state["hit_streak"]
                            }
                        }
                        print(f"[EVENT JSON LOG]: {json.dumps(event_data, ensure_ascii=False)}")

                    # 터미널 상세 출력
                    print(f"  - [ID:{track_id}] Class: {cls_name} | Conf: {conf:.2f} | BBox: {xyxy}")

                    # BBox 및 정보 오버레이 시각화
                    x1, y1, x2, y2 = xyxy
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"ID:{track_id} {cls_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)
            else:
                print(f"[Frame {frame_num:04d} | {filename}] 탐지된 객체 없음")

            # 3. 소실 이벤트 판정 (이전에 존재했으나 현재 프레임에 없는 ID들의 missing_count 관리)
            expired_ids = []
            for track_id, state in active_tracker_states.items():
                if track_id not in current_frame_ids:
                    state["missing_count"] += 1
                    # missing_count가 track_buffer(30)에 도달하면 최종 소실로 확정
                    if state["missing_count"] >= track_buffer:
                        if state["appeared_logged"]:  # 출현했던 대상에 한해서만 소실 로그 출력
                            event_data = {
                                "timestamp": datetime.now().isoformat(),
                                "frame_idx": frame_num,
                                "source_file": filename,
                                "event": "TARGET_DISAPPEARED",
                                "target_info": {
                                    "track_id": track_id,
                                    "class_name": state["class_name"],
                                    "last_bbox": state["bbox"]
                                }
                            }
                            print(f"[EVENT JSON LOG]: {json.dumps(event_data, ensure_ascii=False)}")
                        expired_ids.append(track_id)

            # 완전히 만료된 트랙 ID 정리
            for track_id in expired_ids:
                del active_tracker_states[track_id]

            # 파일명 및 프레임 번호 화면 표기
            cv2.putText(frame, f"Frame: {frame_num} ({filename})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 화면 출력 및 재생 속도 조절
            display_frame = cv2.resize(
                frame,
                (int(w * scale), int(h * scale))
            ) if scale < 1.0 else frame
            
            window_name = "YOLOv8 + BoT-SORT Tracking & Events"
            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("\n사용자에 의해 실행 중단 ('q' 키 감지).")
                break

    finally:
        cv2.destroyAllWindows()
        print(f"이미지 시퀀스 처리 완료. 총 {frame_num}개 프레임 처리됨.")


def main():
    print(f"YOLO 모델 로딩 중 ({MODEL_NAME})...")
    model = YOLO(MODEL_NAME)

    if MODE == "image":
        process_image(
            model=model,
            input_path=str(IMAGE_INPUT_PATH),
            save_result=SAVE_RESULT,
            output_dir=str(OUTPUT_DIR),
            output_filename=f"{IMAGE_INPUT_PATH.stem}_result{IMAGE_INPUT_PATH.suffix}"
        )
    elif MODE == "sequence":
        process_image_sequence(
            model=model,
            input_dir=SEQUENCE_INPUT_DIR
        )
    else:
        print(f"[Error] 지원하지 않는 MODE입니다: {MODE}.")


if __name__ == "__main__":
    main()