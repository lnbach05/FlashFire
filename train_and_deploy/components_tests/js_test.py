import pygame
import sys
import os

os.environ["SDL_VIDEODRIVER"] = "dummy"

# Initialize Pygame
pygame.init()

# Initialize the joystick (controller)
pygame.joystick.init()

# Get the first joystick (change index if needed)
joystick = pygame.joystick.Joystick(0)
joystick.init()

# Main loop
try:
    while True:
        pygame.event.pump()  # Process events

        # Get controller input values
        axes = [joystick.get_axis(i) for i in range(joystick.get_numaxes())]
        buttons = [joystick.get_button(i) for i in range(joystick.get_numbuttons())]

        # Display input values in the terminal
        print("Axes:", axes)
        print("Buttons:", buttons)

        # Wait briefly before checking input again (adjust as needed)
        pygame.time.delay(100)

except KeyboardInterrupt:
    pygame.quit()
    sys.exit()
