import pyautogui
import time
from mss import mss

start_x = 500
start_y = 400

bbox = (start_x, start_y, start_x + 300, start_y + 1)

cords_x = [0, 100, 200, 299]
def test_time():
    with mss() as sct:
        t1 = time.time()
        for i in range(100):
            img = sct.grab(bbox)
            pyautogui.click(86, 300)


        t2 = time.time()
        print(t2-t1)


def print_mouse_pos():
    while True:
        print(pyautogui.position())
        time.sleep(1)

def start():
    with mss() as sct:
        while True:
            img = sct.grab(bbox)
            for cord in cords_x:
                if img.pixel(cord, 0)[0] < 100:
                    pyautogui.click(start_x + cord, start_y)
                    break



time.sleep(5)
start()