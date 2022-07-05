docs: https://pollen-robotics.github.io/reachy-2019-docs/docs/program-your-robot/control-the-arm/#arm-coordinate-system \
https://docs.pollen-robotics.com/sdk/first-moves/arm/ (seems to be wrong a lot more than the other link)
## Quickstart (Accenture Office)
1. Connect to Wifi hotspot that's linked with Reachy:
    ```yml
    Hotspot
    name: BOSMIFI1002
    pw: ACN!Access
    ```
1. SSH into Reachy:
    ```yml
    command: ssh pi@192.168.0.2
    pw: reachy
    ```
1. Running a Python script with real Reachy:
    ```bash
    cd zed-opencv-native/python/
    python3 zed_opencv_native.py "15618"
    ```
1. Running a Jupyter Notebok with real Reachy:
    ```bash
    # starting the Jupyter server
    $ jupyter notebook --ip 0.0.0.0
    # IP with token to access notebook from local machine without using VNC
    http://192.168.0.2:8888/?token=<very long token generated from server startup>
    ```

# Reachy Cameras
The [Reachy 2019 docs](https://pollen-robotics.github.io/reachy-2019-docs/docs/technical-specifications/head/) mentions that Reachy uses "2 [Raspberry Pi cameras](https://www.raspberrypi.com/documentation/accessories/camera.html) cameras associated with 2 optical lenses (one macro and one wide angle)":
- Sony IMX219 8-megapixel sensor (specs according to [seed studio.com](https://www.seeedstudio.com/Raspberry-Pi-High-Quality-Cam-p-4463.html)):
    - Resolution: 3280 × 2464 pixels
    - Sensor image area: 3.68 x 2.78 mm (4.6 mm diagonal)
    - Horizontal Field of View: 62.2°
    - Vertical Field of View: 48.8°
    - Focal Length: 3.04mm
- Sony IMX477 12.3 megapixel sensor (specs according to [kksb](https://kksb-cases.com/collections/raspberry-pi/products/imx477-160-12-3mp-camera-160-fov)):
    - Resolution: 12.3MP | 4056 x 3040p
    - Pixel Size: 1.55 µm (Horizontal) x 1.55 µm (Vertical)
    - CMOS Diagonal Length: 7.9mm
    - Back Focal Length (BFL): 5.52
    - Aperture (F): 2.2
    - Field of View (FoV): 160° (D) | 118° (H) | 87° 9V)

# Using Reachy Simulator
Online Simulator: https://pollen-robotics.github.io/reachy-simulator/
Offline Simulator with Unity (Github): https://github.com/pollen-robotics/reachy-unity-package

When using the Online Simulator, you have to first start up the simulator call in your code, then hit the "Connect" button in the online Sim (Webgl) window.

## Arm Positions
```python
# code to determine position
current_position = [m.present_position for m in reachy.right_arm.motors]
print(reachy.right_arm.forward_kinematics(joints_position=current_position))
```
### Right Arm
```bash
# Resting position:
[20.197999999999993, -9.076999999999998, -5.758, -87.077, -9.824, -24.22, 5.718, -30.938]
[[-0.02438356  0.00754791 -0.99967418  0.24214034]
 [ 0.02584408  0.99964205  0.00691729 -0.27471851]
 [ 0.99936856 -0.02566699 -0.02456991 -0.34972267]
 [ 0.          0.          0.          1.        ]]

# Resting position -1
[20.197999999999993, -9.076999999999998, -5.758, -87.077, -9.824, -24.22, 5.132, -30.938]
[[-0.0348856   0.00610565 -0.99937266  0.24188312]
 [ 0.02805175  0.99959332  0.00512778 -0.27470299]
 [ 0.99899754 -0.02785526 -0.03504269 -0.34923551]
 [ 0.          0.          0.          1.        ]]
```