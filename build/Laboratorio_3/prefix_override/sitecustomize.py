import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/giacomo/Desktop/scoperta_ws/install/Laboratorio_3'
