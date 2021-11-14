import numpy as np
from reachy import Reachy, parts
from scipy.spatial.transform import Rotation as R
import time
import math

class RobotMode:
    REAL = 'real'
    SIM = 'sim'

def get_origin_position(part):
    name = part.name
    origin_position_map = {
        # left arm:
        'left_arm.shoulder_pitch': 20.637,
        'left_arm.shoulder_roll': 4.154,
        'left_arm.arm_yaw': -0.571,
        'left_arm.elbow_pitch': -89.714,
        'left_arm.hand.forearm_yaw': -3.079,
        'left_arm.hand.wrist_pitch': -21.582,
        'left_arm.hand.wrist_roll': 3.666,
        'left_arm.hand.gripper': 3.372,
        # right arm:
        'right_arm.shoulder_pitch': 13.253,
        'right_arm.shoulder_roll': -4.418,
        'right_arm.arm_yaw': -5.758,
        'right_arm.elbow_pitch': -85.143,
        'right_arm.hand.forearm_yaw': -4.839,
        'right_arm.hand.wrist_pitch': -20.088,
        'right_arm.hand.wrist_roll': 3.079,
        'right_arm.hand.gripper': -30.938
    }
    return origin_position_map[name]

class MovePartTo:
    def __init__(self, part, speed, end_position=None) -> None:
        self._part = part
        self.speed = speed
        self._start_position = self._part.present_position
        self._end_position = end_position if end_position else get_origin_position(self._part)

    def already_in_position(self, target, tolerance=0.01) -> bool:
        return math.isclose(self._part.present_position, target, abs_tol=tolerance)

    def calculate_total_duration(self):
        return round(abs(self._end_position - self._start_position)/self.speed)
    
    def move(self, target):
        if self.already_in_position(target):
            print(f"{self._part} already is already at {target}")
            return

        duration = self.calculate_total_duration()
        self._part.compliant = False
        print(f"moving {self._part} to {target}")
        self._part.goto(
            goal_position=target,  # in degrees
            duration=duration,  # in seconds
            wait=True
        )
    
    def execute(self):
        target = self._end_position
        if self.already_in_position(target):
            print(f"{self._part} already is already at {target}")
            return

        # move only if arm part is not already in position
        duration = self.calculate_total_duration()
        self._part.compliant = False
        print(f"moving {self._part} to {target}")
        self._part.goto(
            goal_position=target,  # in degrees
            duration=duration,  # in seconds
            wait=True
        )
    
    def undo(self):
        if not math.isclose(self._part.present_position, self._start_position, abs_tol=0.01):
            duration = self.calculate_total_duration()
            print(f"{self._part} going back to start position")
            self._part.compliant = False
            self._part.goto(
                goal_position=self._start_position,
                duration=duration,
                wait=True
            )

def move_to_origin(arm, mode=None):
    mode = mode if mode else RobotMode.SIM
    if mode == RobotMode.REAL:
        print(
            "ERROR: Currently not safe to execute on real robot. Nothing happen. "
            "Please switch to sim mode if you'd like to execute the movements."
        )
        return None
    
    for motor in arm.motors:
        target_pos = get_origin_position(motor)
        motor_movement = MovePartTo(part=motor, speed=1, end_position=target_pos)
        motor_movement.execute


def start_reachy(mode):
    if mode == RobotMode.REAL:
        try:
            # initialize real hardware
            return Reachy(
                right_arm=parts.RightArm(
                    io='/dev/ttyUSB*',
                    hand='force_gripper',
                ),
                left_arm=parts.LeftArm(
                io='/dev/ttyUSB*',
                hand='force_gripper')
            )
        except ValueError:
            print("No real robot hooked up, using robot simulator")
    else:
        robot = Reachy(
            right_arm=parts.RightArm(io='ws', hand='force_gripper'),
            left_arm=parts.LeftArm(io='ws', hand='force_gripper'),
        )
        # move into same starting position as hardware
        move_to_origin(robot.left_arm)
        move_to_origin(robot.right_arm)
        return robot


if __name__=="__main__":
    mode = RobotMode.REAL
    robot = start_reachy(mode)
    move_to_origin(robot.left_arm, mode=mode)
    LEFT_ARM = robot.left_arm
    RIGHT_ARM = robot.right_arm