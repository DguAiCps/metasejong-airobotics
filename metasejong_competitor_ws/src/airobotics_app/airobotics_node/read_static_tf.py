import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from tf2_msgs.msg import TFMessage  # <‑‑ The correct type for /tf_static>

TARGET_PARENT = "metasejong2025/map"      #  e.g. world frame
TARGET_CHILD  = "odom"    #  e.g. your sensor frame


class OneShotStaticTF(Node):
    def __init__(self) -> None:
        super().__init__("one_shot_static_tf")

        # /tf_static must be subscribed with **Transient‑Local + Reliable**
        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        # The topic type is tf2_msgs/TFMessage
        self.sub = self.create_subscription(
            TFMessage,
            "/tf_static",
            self._callback,
            qos,
        )

    def _callback(self, msg: TFMessage) -> None:
        """Scan the bundle for the desired transform and quit once found."""
        for tf in msg.transforms:  # each is a TransformStamped
            if tf.header.frame_id == TARGET_PARENT and tf.child_frame_id == TARGET_CHILD:
                tr = tf.transform
                self.get_logger().info(
                    f"{TARGET_PARENT} -> {TARGET_CHILD} | "
                    f"translation: ( {tr.translation.x:.3f}, {tr.translation.y:.3f}, {tr.translation.z:.3f} )  | "
                    f"rotation(q): ( {tr.rotation.x:.3f}, {tr.rotation.y:.3f}, {tr.rotation.z:.3f}, {tr.rotation.w:.3f} )"
                )
                rclpy.shutdown()  # we got what we came for
                break  # stop scanning once we've handled it


def main(args=None) -> None:
    rclpy.init(args=args)
    node = OneShotStaticTF()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
