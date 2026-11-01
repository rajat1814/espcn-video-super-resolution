# Source videos (not included in the repository)

The experiments use 8 landscape, talking-head clips downloaded from Pexels (free licence; the raw clips
must not be redistributed, so they are not tracked by Git). To reproduce the results, download your own
clips, name them talk01.mp4 ... talk08.mp4, and place them in this folder.

| Clip | Resolution | FPS | Length | Role | Source URL |
|---|---|---|---|---|---|
| talk01 | 1920x1080 | 25 | 16.5 s | train | https://www.pexels.com/video/casual-businessman-looking-at-camera-and-talking-5941016/ |
| talk02 | 1920x1080 | 25 | 28.3 s | train | https://www.pexels.com/video/man-in-gray-sweater-talking-7262257/ |
| talk03 | 1920x1080 (cropped to 16:9) | 25 | 39.0 s | train | https://www.pexels.com/video/a-businesswoman-greeting-and-talking-7643308/ |
| talk04 | 1280x720 (cropped to 16:9) | 25 | 30.0 s | train | https://www.pexels.com/video/man-is-talking-whil-looking-at-camera-7643311/ |
| talk05 | 1280x720 (cropped to 16:9) | 25 | 19.6 s | train | https://www.pexels.com/video/man-talking-while-wearing-headset-7706881/ |
| talk06 | 1920x1080 | 25 | 11.3 s | train | https://www.pexels.com/video/an-elderly-man-talking-and-smiling-7999352/ |
| talk07 | 1920x1080 | 25 | 13.3 s | **test (held out)** | https://www.pexels.com/video/man-looking-at-the-camera-while-talking-8048255/ |
| talk08 | 1920x1080 | 25 | 14.1 s | **test (held out)** | https://www.pexels.com/video/woman-talking-in-front-of-a-camera-8814086/ |

Requirements: landscape 16:9, one person talking, indoor, static camera, at least about 8 seconds, 720p or higher.
Clips that were not exactly 16:9 were centre-cropped with FFmpeg before use.

Dataset build (from the repository root):
python make_datasets.py --videos data/videos --out data_v2 --frames-per-seq 60 --segments 4 --test-clips talk07 talk08 --jpeg-q 10 20 30 50 70 90 --h264-crf 24 30 36 42 45 --clean-out