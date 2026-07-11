# -*- coding: utf-8 -*-
from skidl import *

@subcircuit
def sub1_1(N_3, N_6):
    # Local nets
    N_4 = Net('N$4')

    # Components
    C1 = Part('Device', 'C', value='0.001', ref='C1')
    C2 = Part('Device', 'C', value='0.002', ref='C2')
    R1 = Part('Device', 'R', value='0.001', ref='R1')
    R2 = Part('Device', 'R', value='0.002', ref='R2')


    # Connections
    N_3 += C1['1'], R1['1']
    N_4 += C1['2'], C2['1'], R1['2'], R2['1']
    N_6 += C2['2'], R2['2']
    return
