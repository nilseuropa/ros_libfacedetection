## ROS Wrapper for libfacedetection



This is a ROS package utilizing `libfacedetection` - that is an open source library for CNN-based face detection. The CNN model has been converted to static variables in C source files. The source code does not depend on any other libraries. SIMD instructions are used to speed up the detection. You can enable AVX2 if you use Intel CPU or NEON for ARM.

### libfacedetection

Trained model can be found in the official library repository: https://github.com/ShiqiYu/libfacedetection

|                    | Time          | FPS           | Time         | FPS          |
| ------------------ | ------------- | ------------- | ------------ | ------------ |
|                    | Single-thread | Single-thread | Multi-thread | Multi-thread |
| cnn (CPU, 640x480) | 58.03 ms      | 17.23         | 13.85 ms     | 72.20        |
| cnn (CPU, 320x240) | 14.18 ms      | 70.51         | 3.38 ms      | 296.21       |
| cnn (CPU, 160x120) | 3.25 ms       | 308.15        | 0.82 ms      | 1226.56      |
| cnn (CPU, 128x96)  | 2.11 ms       | 474.38        | 0.52 ms      | 1929.60      |

- Minimal face size ~10x10 pixels
- Intel(R) Core(TM) i7-1065G7 CPU @ 1.3GHz

**Performance on WIDER Face dataset:** `AP_easy=0.852, AP_medium=0.823, AP_hard=0.646`

**Author:** Shiqi Yu, shiqi.yu@gmail.com

### ROS wrapper

The `face_detector_node` subscribes to `camera/image` and `camera/camera_info` topics and publishes `FaceObject` messages.

