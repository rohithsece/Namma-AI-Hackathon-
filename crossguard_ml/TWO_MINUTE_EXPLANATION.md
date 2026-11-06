# CROSSGUARD AI — demonstration script

Road crossings near schools and busy community streets need affordable ways to observe pedestrians and vehicles. Continuous cloud video processing can require reliable connectivity and adds operating costs.

CROSSGUARD AI uses a lightweight pretrained object detector with ExecuTorch. Our target device is Raspberry Pi 4. It processes supplied video or images locally and detects people, bicycles, motorcycles, cars, buses and trucks. It saves annotated frames, category counts, and a report showing average inference time and approximate throughput.

For a fixed camera, we can also configure a pedestrian crossing zone and a vehicle approach zone. If both zones are occupied in the same frame, the system flags that moment for attention. This is an optional occupancy rule, not a prediction that a collision will happen.

One practical application is reviewing activity near a school gate. Staff can inspect periods with pedestrians and vehicles present together and use the observations when considering crossing supervision or road-layout improvements.

Our demonstration will show the organiser-provided input, at least two detection outputs, the per-frame counts and the actual Raspberry Pi timing report. Counts describe objects visible in each frame, not unique people or vehicles over the entire clip. Processing stays on the device; our package does not send images to a cloud inference service.

[Before presenting: replace the demonstration sentence with what you have actually completed. Quote measured Raspberry Pi timing only after the Pi run.]
