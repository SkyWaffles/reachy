import numpy as np
from reachy import Reachy, parts
from scipy.spatial.transform import Rotation as R
import time
import math

def start_reachy():
    # initialize Reachy
    return Reachy(
        right_arm=parts.RightArm(
            io='/dev/ttyUSB*',
            hand='force_gripper',
        ),
        left_arm=parts.LeftArm(
        io='/dev/ttyUSB*',
        hand='force_gripper')
    )
reachy = start_reachy()
LEFT_ARM = reachy.left_arm
RIGHT_ARM = reachy.right_arm

def stiffen(arm):
    for m in arm.motors:
        m.compliant = False

def relax(arm):
    for motor in arm.motors:
        motor.compliant = True

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

    def calculate_total_duration(self):
        return round(abs(self._end_position - self._start_position)/self.speed)
    
    def execute(self):
        if not math.isclose(self._part.present_position, self._end_position, abs_tol=0.01):
            # move only if arm part is not already in position
            duration = self.calculate_total_duration()
            self._part.compliant = False
            print(f"moving {self._part}")
            self._part.goto(
                goal_position=self._end_position,  # in degrees
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

def execute_reset(arm, in_reverse=False):
    if 'left' in arm.name:
        reset_steps = [
            MovePartTo(
                part=arm.elbow_pitch,
                speed=15,
                end_position=-134
            ),
            MovePartTo(
                part=arm.hand.forearm_yaw,
                speed=15
            )
        ]

    if in_reverse:
        reverse = list(reversed(reset_steps))
        print(reverse)
        for step in reverse:
            step.undo()
    else:
        for step in reset_steps:
            step.execute()




current_position = [m.present_position for m in reachy.right_arm.motors]
print(current_position)
print(reachy.right_arm.forward_kinematics(joints_position=current_position))



reset_sequence = [
    {
        # lift forearm as high as it'll go
        'part': LEFT_ARM.elbow_pitch,
        'goal_position': -134,
        'degrees_per_second': 15
    },
    # # straighten wrist
    # {
    #     # rotate forearm to be parallel with body
    #     'part': LEFT_ARM.
    # }
    # # straighten arm all the way 
]
first_step = reset_sequence[0]
cur_pos = first_step['part'].present_position
speed = round(abs(cur_pos - first_step['goal_position'])/15)
print(speed)

reset_sequence[0]['part'].compliant = False
stiffen(arm=reset_sequence[0]['part'].name.split("_")[0])
for step in reset_sequence:
    cur_pos = step['part'].present_position
    speed = round(abs(cur_pos - step['goal_position'])/15)
    step['part'].goto(
        goal_position=first_step['goal_position'],  # in degrees
        duration=speed,  # in seconds
        wait=True,
    )

LEFT_ARM.elbow_pitch.goto(
    goal_position=-89.363,
    duration=3,
    wait=True
)
relax(arm=LEFT_ARM)



def store_current_position(motors):
    position_dict = {}
    for motor in motors:
        position_dict[motor.name] = motor.present_position
    return position_dict

reset_position_left_arm = store_current_position(LEFT_ARM.motors)
print(reset_position_left_arm)



stored = list(reset_position_left_arm.keys())[0]
print(stored)
print(type(stored))

# create a test motor for testing truthiness
fake_config = {
    'offset': 1,
    'orientation': 'direct'
}
compare = parts.motor.DynamixelMotor(root_part="left_arm", name="shoulder_pitch", luos_motor=None, config=fake_config)
stored == compare