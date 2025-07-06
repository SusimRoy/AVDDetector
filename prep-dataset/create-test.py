import os
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from insightface.app import FaceAnalysis
import cv2
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Extract faces from videos")
parser.add_argument('--video_dir', type=str, required=True, help='Directory containing video files')
parser.add_argument('--output_base_dir', type=str, required=True, help='Base directory to save extracted faces')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

video_dir = args.video_dir
output_base_dir = args.output_base_dir

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(96, 96))

def extract_faces(video_path, output_dir, json_path=None, app=None):

    cap = cv2.VideoCapture(video_path)
    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    face_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        faces = app.get(frame)
        if faces:
            for i, face in enumerate(faces):
                bbox = face.bbox.astype(int)

                x1, y1, x2, y2 = map(int, bbox)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

                face_crop = frame[y1:y2, x1:x2]

                # Save face crop as image
                output_path = os.path.join(output_dir, f"frame_{frame_idx:05d}.jpg")
                face_crop = cv2.resize(face_crop, (224, 224))
                cv2.imwrite(output_path, face_crop)
                face_count += 1

        frame_idx += 1

    cap.release()
    return face_count

def process_video(fname):
    if fname.endswith('.mp4'):
        video_path = os.path.join(video_dir, fname)
        video_name = os.path.splitext(fname)[0]
        output_dir = os.path.join(output_base_dir, video_name)
        extract_faces(video_path, output_dir, app=app)

if __name__ == "__main__":
    video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    video_files = sorted(video_files)
    max_workers = os.cpu_count()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_video, fname) for fname in video_files]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Processing videos"):
            pass