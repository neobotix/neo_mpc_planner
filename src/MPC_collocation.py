#!/usr/bin/env python3
from __future__ import division
import numpy as np
import rospy
from geometry_msgs.msg import Pose, Twist, PoseArray, Point, PolygonStamped
from nav_msgs.msg import Odometry, Path
from omgtools import *
from tf.transformations import euler_from_quaternion, quaternion_from_euler, quaternion_matrix
import time

class MPC():
	def __init__(self,):
		self.PubTwist = rospy.Publisher('/cmd_vel1', Twist, queue_size = 1)
		self.pub_marker = rospy.Publisher('/poses', Pose, queue_size=1)
		# self.data =  np.loadtxt('/home/pradheep/fab_ws/src/path_following_ctrl/paths/data1.txt', delimiter = ' ')
		self.data = []
		self.SubGoalPath = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.CBglobplan)
		self.vehicle = Holonomic3D(Plate(Rectangle(0.5, 1.), height=0.1), bounds={'vmax': 0.4, 'wmax': np.pi/3., 'wmin': -np.pi/3.})
		self.current_state = Pose()
		self.current_velocity = Twist()
		# create environment
		print('Using environment for known example')
		#for now fix environment to the one for which A*-path is known
		start = [0,0,0]
		goal = [1,0,0]
		self.vehicle.set_initial_conditions(start)
		self.vehicle.set_terminal_conditions(goal)
		self.pub_pred_velocity = Twist()
		self.environment = Environment(room={'shape': Cube(20.)})
		self.current_states = [0, 0, 0]
		self.current_time = 0
		self.update_time = 0.01
		self.sample_time = 0.01
		self.problem = Point2point(self.vehicle, self.environment, freeT=False)
		self.problem.init()
		self.deployer = Deployer(self.problem, self.sample_time, self.update_time)
		self.target_reached = False
		self.t00 = time.time()
		self.SubOdom = rospy.Subscriber('/odom', Odometry, self.CBodom)
		self.ind = 15
		self.waypoint_init = [0,0,0]
		self.problem.set_options({'hard_term_con': False, 'horizon_time': 3.})
		self.tolerance_x = 0.05
		self.tolerance_y = 0.05
		self.len_data = 0
		self.global_plan = 0
		self.last_control = [0,0,0]
		self.last_pose =  [0,0]
		self.PubPath = rospy.Publisher("Eval_path", Path, queue_size = 1)
		self.last_dist = 0
		self.time2 = []
		self.incr = 0

	def CBglobplan(self,planner_data):
		waypoints_pos_x = []
		waypoints_pos_y = []
		waypoints_pos_z = []
		no_pose = len(planner_data.poses)
		pose = Pose()
		for i in range (no_pose):
			waypoints_pos_x.append(planner_data.poses[i].pose.position.x)
			waypoints_pos_y.append(planner_data.poses[i].pose.position.y)
			pose.orientation.x = planner_data.poses[i].pose.orientation.x
			pose.orientation.y = planner_data.poses[i].pose.orientation.y
			pose.orientation.z = planner_data.poses[i].pose.orientation.z
			pose.orientation.w = planner_data.poses[i].pose.orientation.w
			quaternion = (pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w)
			theta = (euler_from_quaternion(quaternion))
			waypoints_pos_z.append(theta[2])
		if(self.global_plan == 0):
			rospy.loginfo("New goal recieved for MPC")
			self.data = np.transpose(np.asarray([waypoints_pos_x, waypoints_pos_y, waypoints_pos_z]))
			self.final_goal = [self.data[:,0][len(self.data[:,1])-1],self.data[:,1][len(self.data[:,1])-1],self.data[:,2][len(self.data[:,1])-1]]
			self.len_data = len(self.data[:,1])
			self.global_plan = 1
			self.Path = planner_data
			self.PubPath.publish(self.Path)

	def CBodom(self,data):
		self.current_state.position.x = data.pose.pose.position.x
		self.current_state.position.y = data.pose.pose.position.y
		self.current_state.position.z = data.pose.pose.position.z

		self.current_state.orientation.x = data.pose.pose.orientation.x
		self.current_state.orientation.y = data.pose.pose.orientation.y
		self.current_state.orientation.z = data.pose.pose.orientation.z
		self.current_state.orientation.w = data.pose.pose.orientation.w

		self.current_velocity.linear.x = data.twist.twist.linear.x
		self.current_velocity.linear.y = data.twist.twist.linear.y
		self.current_velocity.linear.z = data.twist.twist.linear.z
		self.current_velocity.angular.z = data.twist.twist.angular.z

		quaternion = (self.current_state.orientation.x,self.current_state.orientation.y,\
		self.current_state.orientation.z,self.current_state.orientation.w)
		theta = euler_from_quaternion(quaternion)
		self.theta_use = theta[2]

	def get_incr_ind(self, last_pose):
		curr_pos = np.array((self.current_state.position.x, self.current_state.position.y))
		last_pose = np.asarray(last_pose)
		dist_travelled = np.linalg.norm(curr_pos - last_pose)
		return dist_travelled 

	def update(self,):
		
		if(self.data == []):
			rospy.loginfo_throttle(10,"Did not recieve any global plan")
		else:
			self.current_states = [self.current_state.position.x, self.current_state.position.y, self.current_state.orientation.z]
			if(self.ind<len(self.data)):
				waypoint = [self.data[:,0][self.ind],self.data[:,1][self.ind],0]
			else:
				waypoint = [self.data[:,0][self.len_data-1],self.data[:,1][self.len_data-1],0]
				Flag1 = 1
			self.vehicle.set_initial_conditions(self.waypoint_init)
			self.vehicle.set_terminal_conditions(waypoint)
			t0 = time.time() - self.t00
			
			if (t0-self.current_time-self.update_time) >= 0.:
				self.current_time = t0
			trajectories = self.deployer.update(self.current_time, self.current_states)
			self.pub_pred_velocity.linear.x = trajectories['input'][0, :][0]
			self.pub_pred_velocity.linear.y = trajectories['input'][1, :][0]
			self.pub_pred_velocity.angular.z = 0
			self.PubTwist.publish(self.pub_pred_velocity)

			self.d_trav = self.get_incr_ind(self.last_pose)

			dist = self.d_trav+self.last_dist
			print(dist)
			if(dist >= 0.025):
				incr, rem = divmod(dist, 0.025)
				self.last_dist = rem
			else:
				self.last_dist = dist
				incr = 0

			self.incr += incr
			self.ind = self.ind + int(incr)

			self.last_pose = [self.current_state.position.x, self.current_state.position.y]

			if(incr>0):
				self.last_pose = [self.current_state.position.x, self.current_state.position.y]

			if(((self.data[:,0][-1]-0.05 <= self.current_state.position.x) and \
					(self.data[:,0][-1]+0.05 >= self.current_state.position.x)) and \
					((self.data[:,1][-1]-0.05 <= self.current_state.position.y) and \
					(self.data[:,1][-1]+0.05 >= self.current_state.position.y)) and Flag1 == 1):
				Flag2 = 1
				self.pub_pred_velocity.linear.x = 0
				self.pub_pred_velocity.linear.y = 0
				self.pub_pred_velocity.angular.z = 0
				self.PubTwist.publish(self.pub_pred_velocity)
				self.global_plan = 0
				self.ind = 23
				self.cost = 0
				self.data = []
				self.deployer.reset()

			self.waypoint_init = waypoint

def main():

	# create vehicle 
	rospy.init_node("Model_Predictive_Control_Collocation")
	rospy.loginfo("Starting the MPC-collocation node")

	ctrl = MPC()

	while not rospy.is_shutdown():
		ctrl.update()
	rospy.loginfo("Finishing")
if __name__ == '__main__':
	main()