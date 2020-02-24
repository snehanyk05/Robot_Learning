#!/usr/bin/env python

import rospy
from std_msgs.msg import String #String message 
from std_msgs.msg import Int8
from geometry_msgs.msg import Twist
import sys,tty,termios
msg = Twist()

class _Getch:
    def __call__(self):
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                ch = sys.stdin.read(3)
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            return ch
def get():
        inkey = _Getch()
        while (1):
                msg.linear.x = 0
                msg.angular.z = 0
                pub.publish(msg)

                k=inkey()
                if k!='':break
        if k=='\x1b[A':
                keys("up")
        elif k=='\x1b[B':
                keys("down")
        elif k=='\x1b[C':
                keys("right")
        elif k=='\x1b[D':
                keys("left")
        elif k=='end':
                exit()
        else:
                print ("not an arrow key!")


def keys(k):
                    
        linear_x = 0
        angular_z = 0
            
        rate = rospy.Rate(3)
        rospy.loginfo(str(k))
        
        if (k=="up"):
                linear_x = 0.3
                angular_z = 0
        elif (k=="down"):
                linear_x = -0.3
                angular_z = 0
        elif (k=="left"):
                linear_x = 0
                angular_z = 0.3
        elif (k=="right"):
                linear_x = 0
                angular_z = -0.3
                

        rospy.loginfo(k)
        msg.linear.x = linear_x
        msg.angular.z = angular_z
        pub.publish(msg)

        rate.sleep()


def main():
    global pub
    
    rospy.init_node('keypress',anonymous=True)
    
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)  
    


    print("Press arrow keys to move the robot or type \"end\" to exit.")
    while not rospy.is_shutdown():
                
                get()
    rospy.spin()

if __name__=='__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass