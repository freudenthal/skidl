# -*- coding: utf-8 -*-
from skidl import *

@subcircuit
def sub1_2(N_3, N_6):
    # Local nets
    N_9 = Net('N$9')

    # Components
    C3 = Part('Device', 'C', value='0.001', ref='C3')
    C4 = Part('Device', 'C', value='0.002', ref='C4')
    R3 = Part('Device', 'R', value='0.001', ref='R3')
    R4 = Part('Device', 'R', value='0.002', ref='R4')


    # Connections
    N_3 += C3['1'], R3['1']
    N_6 += C4['2'], R4['2']
    N_9 += C3['2'], C4['1'], R3['2'], R4['1']
    return
