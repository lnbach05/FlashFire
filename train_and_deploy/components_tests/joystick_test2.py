import pygame

def main():
    pygame.init()
    pygame.joystick.init()

    # Check for available controllers
    if pygame.joystick.get_count() == 0:
        print("No controllers found.")
        return

    # Initialize the first controller
    controller = pygame.joystick.Joystick(0)
    controller.init()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYAXISMOTION:
                    # Handle joystick axis motion
                    print("Axis: {}, Value: {:.2f}".format(event.axis, event.value))

                elif event.type == pygame.JOYBUTTONDOWN:
                    # Handle button press
                    print("Button {} down".format(event.button))

                elif event.type == pygame.JOYBUTTONUP:
                    # Handle button release
                    print("Button {} up".format(event.button))

    except KeyboardInterrupt:
        pass

    pygame.quit()

if __name__ == "__main__":
    main()
