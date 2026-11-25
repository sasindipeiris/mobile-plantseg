# LeafSeg (Android demo)

One screen: take or pick a leaf photo → RepViT lesion segmentation on the phone → lesions overlaid
(red = diagnosed disease, amber = other predicted diseases), disease name, diseased area, latency.

## Models

Copy the outputs of `plantseg/src/deploy/export.py` into `app/src/main/assets/` (one of each):

    <name>_fp32.tflite   GPU (Galaxy S25, runs in FP16)
    <name>_int8.tflite   CPU 4 threads (Galaxy M02)
    labels.txt

## Build and install

1. Android Studio → Open → this `android/` folder → let Gradle sync (the wrapper pins Gradle 8.7; set Gradle JDK to 17+).
2. Phone: Settings → About phone → Software information → tap _Build number_ 7× →
   Developer options → USB debugging on. Connect by USB and allow the computer.
3. Run ▶ (or `./gradlew installDebug`).

Use **GPU FP32** on the S25 and **CPU INT8** on the M02. Reported latency numbers come from
`plantseg/src/deploy/benchmark.sh`; the in-app time (mean of 5 runs) is for the demo.
