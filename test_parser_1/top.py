# -*- coding: utf-8 -*-
from skidl import *
from sub1_2 import sub1_2
from sub1_1 import sub1_1

@subcircuit
def top():
    # Local nets
    N_3 = Net('N$3')
    N_6 = Net('N$6')

    # Hierarchical subcircuits
    sub1_2(N_3, N_6)
    sub1_1(N_3, N_6)
    return
