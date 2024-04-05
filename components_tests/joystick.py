# import pygame
from pygame.locals import *
from pygame import event, display, joystick


def get_numControllers():
    return joystick.get_count()

display.init()
joystick.init()
print(f"{get_numControllers()} joystick connected")
js = joystick.Joystick(0)
while True:
    for e in event.get():
        if e.type == JOYAXISMOTION:
            ax0 = js.get_axis(0)
            ax1 = js.get_axis(1)
            ax2 = js.get_axis(2)
            ax3 = js.get_axis(3)
            ax4 = js.get_axis(4)
            ax5 = js.get_axis(5)
            print("---")
            print(f"axis 0: {ax0}")
            print(f"axis 1: {ax1}")
            print(f"axis 2: {ax2}")
            print(f"axis 3: {ax3}")
            print(f"axis 4: {ax4}")
            print(f"axis 5: {ax5}")
            print("---")
        elif e.type == JOYBUTTONDOWN:
            bt0 = js.get_button(0)
            bt1 = js.get_button(1)
            bt2 = js.get_button(2)
            bt3 = js.get_button(3)
            bt4 = js.get_button(4)
            bt5 = js.get_button(5)
            bt6 = js.get_button(6)
            bt7 = js.get_button(7)
            bt8 = js.get_button(8)
            bt9 = js.get_button(9)
            bt10 = js.get_button(10)
            bt11 = js.get_button(11)
            bt12 = js.get_button(12)
            print("---")
            print(f"button 0: {bt0}")
            print(f"button 1: {bt1}")
            print(f"button 2: {bt2}")
            print(f"button 3: {bt3}")
            print(f"button 4: {bt4}")
            print(f"button 5: {bt5}")
            print(f"button 6: {bt6}")
            print(f"button 7: {bt7}")
            print(f"button 8: {bt8}")
            print(f"button 9: {bt9}")
            print(f"button 10: {bt10}")
            print(f"button 11: {bt11}")
            print(f"button 12: {bt12}")
            print("---")
