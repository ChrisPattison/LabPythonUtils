from detector import *

if __name__ == '__main__':
    if len(sys.argv) < 1:
        print('detector.py <lower voltage>, <upper voltage>')
    else:
        test_tube(lower_voltage = float(sys.argv[1]), upper_voltage = float(sys.argv[2]), points=10)