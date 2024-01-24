import pygame
import sys
import os

#Initialize pygame
pygame.init()

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize only the joystick module
pygame.joystick.init()

# Check if any joysticks/controllers are available
if pygame.joystick.get_count() == 0:
    print("No controllers available")
    sys.exit(1)

# Get the first joystick (change index if needed)
joystick = pygame.joystick.Joystick(0)
joystick.init()

prev_button_states = [False] * joystick.get_numbuttons()

try:
    while True:
        pygame.event.pump()  # Process events

        # Get current button states
        button_states = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        # Check for button press and release events
        for button_index in range(len(button_states)):
            if button_states[button_index] != prev_button_states[button_index]:
                if button_states[button_index]:
                    print(f"Button {button_index} pressed")
                else:
                    print(f"Button {button_index} released")

        prev_button_states = button_states

except KeyboardInterrupt:
    pygame.quit()
    sys.exit()
