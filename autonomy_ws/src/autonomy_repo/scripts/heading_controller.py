#!/usr/bin/env python3

import numpy as np
import rclpy 
from asl_tb3_lib.control import BaseHeadingController
from asl_tb3_lib.math_utils import wrap_angle
from asl_tb3_msgs.msg import TurtleBotControl, TurtleBotState

class HeadingController(BaseHeadingController):
    def __init__(self,node_name: str = "heading_controller"):
        super().__init__(node_name)
        self.kp = 2.0

    def compute_control_with_goal(self,
        state: TurtleBotState,
        goal: TurtleBotState
    ) -> TurtleBotControl:
        head_err = wrap_angle(goal.theta - state.theta)
        om = self.kp * head_err
        new_control = TurtleBotControl()
        new_control.omega = om
        return new_control

if __name__ == "__main__":
    rclpy.init()        # initialize ROS2 context (must run before any other rclpy call)
    node = HeadingController()  # instantiate the HeadingController node
    rclpy.spin(node)    # Use ROS2 built-in schedular for executing the node
    rclpy.shutdown()    # cleanly shutdown ROS2 context