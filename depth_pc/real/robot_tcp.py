import math
import socket
import time
from enum import Enum

import numpy as np


class TcpCommandObject(Enum):
    # AGV = 1
    JKROBOT = 2
    # GRASP = 3
    # ROBOTS = 3
#
class TcpCommandType(Enum):
    # ENABLE = 1
    # DISABLE = 2
    JOINTMOVE = 3
    ENDVELMOVE = 4
    # ENDRELMOVE = 5
    ENDMOVE = 6
    ENDMOVECIRCLE = 20
    ENDMOVEIMMEDIATE = 21

    # NAVLOCATION = 7
    # NAVSTATION = 8
    # FORWARD = 9
    # BACK = 10
    # LEFT = 11
    # RIGHT = 12

    IOCONTROL = 13
    SUCKSTART = 14
    SUCKSTOP = 19

    STOP = 15

    GETENDPOS = 16
    GETENDVEL = 17
    GETJOINTPOS = 18



class RobotTcp:
    def __init__(self, ip, port):
        # 通过设置标志位 是否通过打印信息调试
        self.debug = True
        self.ip = ip
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((ip, port))
        print("robot tcp 对象被创建与连接：\nip:{0}\nport:{1}".format(self.ip, self.port))

    def myprint(self, str):
        if self.debug:
            print(str)

    def stop_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.STOP, " ", 1)
        self.myprint(rec_data)

    # eg: joints_angle = [-90.0 (°),0.0,-90.0,0.0,90.0,90.0]
    def joints_move_jkrobot(self, joints_angle):
        data = self.list2str(joints_angle)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.JOINTMOVE, data, 0)
        self.myprint(rec_data)

    # eg: end_vel = [0.0 (cm/s),0.0,0.0,0.0,0.0,5.0 (°/s)]
    def end_vel_jkrobot(self, end_vel):
        data = self.list2str(end_vel)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDVELMOVE, data, 1)
        self.myprint(rec_data)

    # eg: end_abspos = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmove_jkrobot(self, end_abspos):
        data = self.list2str(end_abspos)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVE, data, 0)
        self.myprint(rec_data)

    # eg: p_end = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmovecircle_jkrobot(self, p_len, radius, p_transition, p_end, vel=0.1, acc=0.5, dec=0.5):
        circle_pos = []
        circle_pos.append(p_len)
        circle_pos.append(radius)
        circle_pos += p_transition + p_end
        circle_pos.append(vel)
        circle_pos.append(acc)
        circle_pos.append(dec)
        data = self.list2str(circle_pos)
        print(data)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVECIRCLE, data, 0)
        self.myprint(rec_data)

    # eg: end_pos = [? (m), ?, ?, ?, ?, ? (°)]
    def end_absmoveimmediate_jkrobot(self, end_pos):
        data = self.list2str(end_pos)
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.ENDMOVEIMMEDIATE, data, 1)
        self.myprint(rec_data)

    # eg: return [x (m), y, z, r (deg), p, y]
    def get_end_pos_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETENDPOS, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output

    # eg: return [vx (cm/s), vy, vz, vr (deg/s), vp, vy]
    def get_end_vel_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETENDVEL, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output

    # eg: return [j1 (deg), j2, j3, j4, j5, j6]
    def get_joint_pos_jkrobot(self):
        rec_data = self.send(TcpCommandObject.JKROBOT, TcpCommandType.GETJOINTPOS, " ", 1)
        # self.myprint(rec_data)
        datas = rec_data[1:-1].split(",")
        output = [float(data) for data in datas]
        return output

    def list2str(self, data):
        return str(data).replace("[", "").replace("]", "")

    # object: 控制对象 command: 控制命令 data: 控制数据 emergency: 是否紧急指令1 OR 0
    def send(self, object, command, data, emergency):
        msg = bytes(object.name + "#" + command.name + "#" + data + "#" + str(emergency), encoding='utf-8')
        self.sock.sendall(msg)
        data = self.sock.recv(1024)
        return data.decode()

    def close(self):
        self.sock.close()

    def __enter__(self):
        print("robot tcp 对象进入with：\nip:{0}\nport:{1}".format(self.ip, self.port))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        print("robot tcp 对象退出with：\nip:{0}\nport:{1}".format(self.ip, self.port))


    def __del__(self):
        self.close()
        print("robot tcp 对象被析构：\nip:{0}\nport:{1}".format(self.ip, self.port))



if __name__ == '__main__':

    robot_tcp = RobotTcp('172.16.11.132', 8001)
    robot_tcp.myprint("okk")
