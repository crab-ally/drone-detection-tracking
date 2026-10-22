import os
import json
import cv2
import yaml
import time
from pathlib import Path
from datetime import datetime
from ament_index_python.packages import get_package_share_directory

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from ultralytics import YOLO


class ObjectDetectionTrackingNode(Node):
    def __init__(self):
        super().__init__('object_detection_tracking_node')

        # Topic Publishers
        self.event_publisher = self.create_publisher(String, '/drone/tracker/events', 10)
        self.detection_publisher = self.create_publisher(String, '/drone/tracker/detections', 10)

        # ==========================================
        # 0. Configuration & Parameters (기존 설정 유지)
        # ==========================================
        try:
            package_share_dir = Path(get_package_share_directory('drone_vision_pkg'))
            self.main_config_path = package_share_dir / "config" / "config.yaml"
            self.tracker_config_path = package_share_dir / "config" / "botsort.yaml"
        except Exception:
            # 하드코딩
            base_dir = Path("/workspace/ros_ws/drone_tracker")
            self.main_config_path = base_dir / "config" / "config.yaml"
            self.tracker_config_path = base_dir / "config" / "botsort.yaml"

        # config.yaml 파싱
        with open(self.main_config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)["app_config"]

        # 설정값 및 경로 매핑
        self.mode = cfg.get("mode", "sequence")
        self.sequence_input_dir = Path(cfg.get("sequence_input_dir"))
        self.image_input_path = Path(cfg.get("image_input_path"))
        self.output_dir = Path(cfg.get("output_dir"))

        self.save_result = cfg.get("save_result", True)
        self.model_name = cfg.get("model_name", "yolov8n.pt")
        self.conf_thres = cfg.get("conf_thres", 0.25)
        self.device = cfg.get("device", "cpu")
        self.playback_fps = cfg.get("playback_fps", 30)
        self.appear_streak_thres = cfg.get("appear_streak_thres", 6)

        # BoT-SORT 트래커 설정 로드
        if os.path.exists(self.tracker_config_path):
            with open(self.tracker_config_path, "r", encoding="utf-8") as f:
                tracker_cfg = yaml.safe_load(f)
            self.track_buffer = tracker_cfg.get("track_buffer", 30)
        else:
            self.get_logger().warn(f"Tracker config not found, using default track_buffer=30")
            self.track_buffer = 30

        # 모델 로드
        self.get_logger().info(f"YOLO 모델 로딩 중 ({self.model_name})...")
        self.model = YOLO(self.model_name)

    def run_pipeline(self):
        if self.mode == "image":
            self.process_image(
                input_path=str(self.image_input_path),
                save_result=self.save_result,
                output_dir=str(self.output_dir),
                output_filename=f"{self.image_input_path.stem}_result{self.image_input_path.suffix}"
            )
        elif self.mode == "sequence":
            self.process_image_sequence(
                input_dir=self.sequence_input_dir
            )
        else:
            self.get_logger().error(f"지원하지 않는 MODE입니다: {self.mode}")

    def process_image(self, input_path: str, save_result: bool, output_dir: str, output_filename: str):
        if not os.path.exists(input_path):
            self.get_logger().error(f"입력 이미지 파일을 찾을 수 없습니다: {input_path}")
            return

        img = cv2.imread(input_path)
        if img is None:
            self.get_logger().error(f"이미지 파일 로드에 실패했습니다: {input_path}")
            return

        self.get_logger().info(f"--- [IMAGE DETECTION START] ---")
        results = self.model.predict(source=img, conf=self.conf_thres, device=self.device, verbose=False)
        result = results[0]
        boxes = result.boxes

        det_count = len(boxes) if boxes is not None else 0
        self.get_logger().info(f"총 탐지된 객체 수: {det_count}")

        if boxes is not None and len(boxes) > 0:
            for idx, box in enumerate(boxes):
                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_id]

                self.get_logger().info(f"  - [{idx + 1}] Class: {cls_name} (ID: {cls_id}) | Conf: {conf:.2f} | BBox: {xyxy.tolist()}")

                x1, y1, x2, y2 = xyxy
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{cls_name} {conf:.2f}"
                cv2.putText(img, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("YOLOv8 Object Detection", img)
        
        if save_result:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(save_path, img)
            self.get_logger().info(f"결과 이미지 저장 완료: {save_path}")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def process_image_sequence(self, input_dir: Path):
        if not os.path.exists(input_dir):
            self.get_logger().error(f"입력 이미지 디렉터리를 찾을 수 없습니다: {input_dir}")
            return

        valid_extensions = (".jpg", ".jpeg", ".png")
        image_files = sorted([
            f for f in os.listdir(input_dir) 
            if f.lower().endswith(valid_extensions)
        ])

        if len(image_files) == 0:
            self.get_logger().error(f"처리할 이미지 파일이 존재하지 않습니다: {input_dir}")
            return

        self.get_logger().info(f"--- [IMAGE SEQUENCE DETECTION & TRACKING START] ---")
        self.get_logger().info(f"입력 디렉터리: {input_dir} (총 {len(image_files)}개 프레임)")

        # 프레임 간격
        delay_ms = int(1000 / self.playback_fps)
        frame_num = 0
        active_tracker_states = {}

        # 성능 측정용 시작 시간 기록
        start_time = time.time()

        try:
            for filename in image_files:
                # ROS 2 스핀 처리 (콜백 및 시스템 이벤트 유지)
                rclpy.spin_once(self, timeout_sec=0.001)

                img_path = os.path.join(input_dir, filename)
                frame = cv2.imread(img_path)
                
                if frame is None:
                    self.get_logger().warn(f"이미지 로드 실패, 건너뜁니다: {filename}")
                    continue

                h, w = frame.shape[:2]
                scale = min(960 / w, 540 / h, 1.0)
                frame_num += 1

                # Ultralytics 추적 수행 (필요한 클래스만 지정)
                results = self.model.track(
                    source=frame,
                    conf=self.conf_thres,
                    persist=True,
                    tracker=str(self.tracker_config_path) if os.path.exists(self.tracker_config_path) else "/workspace/ros_ws/drone_tracker/config/botsort.yaml",
                    device=self.device,
                    verbose=False,
                    classes=[4, 14]
                )

                result = results[0]
                boxes = result.boxes
                current_frame_ids = set()

                frame_detections = []

                if boxes is not None and len(boxes) > 0:
                    has_ids = boxes.id is not None
                    track_ids = boxes.id.int().cpu().tolist() if has_ids else [None] * len(boxes)

                    for box, track_id in zip(boxes, track_ids):
                        if track_id is None:
                            continue

                        current_frame_ids.add(track_id)
                        xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
                        conf = float(box.conf[0].cpu().numpy())
                        cls_id = int(box.cls[0].cpu().numpy())
                        cls_name = self.model.names[cls_id]

                        x1, y1, x2, y2 = xyxy
                        cx = (x1 + x2) / 2
                        cy = (y1 + y2) / 2
                        width = x2 - x1
                        height = y2 - y1

                        det_info = {
                            "track_id": track_id,
                            "class_name": cls_name,
                            "confidence": round(conf, 2),
                            "bbox": xyxy
                        }
                        frame_detections.append(det_info)

                        # 트랙 상태 갱신
                        if track_id not in active_tracker_states:
                            active_tracker_states[track_id] = {
                                "hit_streak": 1,
                                "missing_count": 0,
                                "appeared_logged": False,
                                "appeared_at": None,
                                "class_name": cls_name,
                                "conf": conf,
                                "bbox": xyxy,
                                "center": {"x": cx, "y": cy},
                                "size": {"width": width, "height": height}
                            }
                        else:
                            state = active_tracker_states[track_id]
                            state["hit_streak"] += 1
                            state["missing_count"] = 0
                            state["class_name"] = cls_name
                            state["conf"] = conf
                            state["bbox"] = xyxy
                            state["center"] = {"x": cx, "y": cy}
                            state["size"] = {"width": width, "height": height}

                        state = active_tracker_states[track_id]

                        # 신규 출현 이벤트 판정 및 토픽 퍼블리시
                        if state["hit_streak"] >= self.appear_streak_thres and not state["appeared_logged"]:
                            state["appeared_logged"] = True
                            state["appeared_at"] = datetime.now().astimezone()
                            event_data = {
                                "timestamp": state["appeared_at"].isoformat(),
                                "event": "TARGET_APPEARED",
                                "target_info": {
                                    "track_id": track_id,
                                    "class_name": cls_name,
                                    "confidence": round(conf, 2),
                                    "coordinate_system": "image",
                                    "unit": "pixel",
                                    "bbox": xyxy,
                                    "center": {"x": cx, "y": cy},
                                    "size": {"width": width, "height": height},
                                    "stability_count": state["hit_streak"]
                                }
                            }
                            json_str = json.dumps(event_data, ensure_ascii=False, indent=2)
                            self.get_logger().info(f"[EVENT JSON LOG]:\n{json_str}")
                            
                            # ROS 2 토픽 발행
                            msg = String()
                            msg.data = json_str
                            self.event_publisher.publish(msg)

                        # 시각화 오버레이
                        color = (0, 255, 0)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                        label = f"ID:{track_id} {cls_name} {conf:.2f}"
                        cv2.putText(frame, label, (x1, max(y1 - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 3)

                # 탐지 결과를 ROS 2 토픽으로 발행
                if frame_detections:
                    det_msg_data = {
                        "frame_idx": frame_num,
                        "source_file": filename,
                        "detections": frame_detections
                    }
                    det_msg = String()
                    det_msg.data = json.dumps(det_msg_data, ensure_ascii=False)
                    self.detection_publisher.publish(det_msg)

                # 소실 이벤트 판정
                expired_ids = []
                for track_id, state in active_tracker_states.items():
                    if track_id not in current_frame_ids:
                        state["missing_count"] += 1
                        if state["missing_count"] >= self.track_buffer:
                            if state["appeared_logged"]:
                                disappeared_at = datetime.now().astimezone()
                                duration_sec = (disappeared_at - state["appeared_at"]).total_seconds()
                                event_data = {
                                    "timestamp": disappeared_at.isoformat(),
                                    "event": "TARGET_DISAPPEARED",
                                    "target_info": {
                                        "track_id": track_id,
                                        "class_name": state["class_name"],
                                        "appeared_at": state["appeared_at"].isoformat(),
                                        "duration_sec": round(duration_sec, 2),
                                        "coordinate_system": "image",
                                        "unit": "pixel",
                                        "last_bbox": state["bbox"],
                                        "last_center": state["center"],
                                        "last_size": state["size"],
                                        "missing_count": state["missing_count"]
                                    }
                                }
                                json_str = json.dumps(event_data, ensure_ascii=False, indent=2)
                                self.get_logger().info(f"[EVENT JSON LOG]:\n{json_str}")
                                
                                msg = String()
                                msg.data = json_str
                                self.event_publisher.publish(msg)
                            
                            expired_ids.append(track_id)

                for track_id in expired_ids:
                    del active_tracker_states[track_id]

                # 프레임 시각화 표시
                cv2.putText(frame, f"Frame: {frame_num} ({filename})", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                display_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1.0 else frame
                cv2.imshow("YOLOv8 + BoT-SORT ROS2 Tracking", display_frame)

                if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                    self.get_logger().info("사용자에 의해 실행 중단 ('q' 키 감지).")
                    break

        finally:
            # 성능 측정 종료 시간 기록 및 통계 산출
            end_time = time.time()
            elapsed_time = end_time - start_time
            avg_fps = (frame_num / elapsed_time) if elapsed_time > 0 else 0.0

            cv2.destroyAllWindows()
            self.get_logger().info(f"이미지 시퀀스 처리 완료. 총 {frame_num}개 프레임 처리됨.")
            self.get_logger().info(f"==================================================")
            self.get_logger().info(f"⏱️ 총 처리 시간: {elapsed_time:.2f} 초")
            self.get_logger().info(f"🚀 평균 처리 속도 (FPS): {avg_fps:.2f} FPS")
            self.get_logger().info(f"==================================================")


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionTrackingNode()
    
    try:
        node.run_pipeline()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()