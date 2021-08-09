#!/home/xny/anaconda3/envs/tf/bin/python3.7
# coding=utf-8
import math , time
import numpy as np

class Pid(object):
    def __init__(self, kp, ki, kd, exp, In, Out, T, Fwd, Out_Limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.exp = exp
        self.In = In
        self.Out = Out
        self.last_err = 0
        self.sum_err = 0
        self.T = T
        self.last_time = time.clock()
        self.Fwd = Fwd
        self.min_out = Out_Limit[0]
        self.max_out = Out_Limit[1]
    
    def Pid_Compute(self):
        if (time.clock() - self.last_time >= self.T):
            # 计算方向
            if (self.Fwd == True):
                err = self.exp[0] - self.In[0]
            else:
                err = self.In[0] - self.exp[0]
            self.sum_err += err
            self.Out[0] = self.kp * err + self.ki * self.sum_err + self.kd * (err - self.last_err) * self.T
            # 输出限制
            if self.Out[0] < self.min_out:
                self.Out[0] = self.min_out
            elif self.Out[0] > self.max_out:
                self.Out[0] = self.max_out
            
            self.last_err = err
            self.last_time = time.clock()
            return True
        else:
            return False
        
        def Set_Pid_Fwd(self,Fwd):
            self.Fwd = Fwd
            
        def Set_Pid_Out_Limit(Out_Limit):
            self.min_out = Out_Limit[0]
            self.max_out = Out_Limit[1]