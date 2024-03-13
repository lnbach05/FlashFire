import pygame

pygame.init()
pygame.joystick.init()

if pygame.joystick.get_count() == 0:
    print("No joystick found.")
else:
    js = pygame.joystick.Joystick(0)
    js.init()

    print("Press buttons on your controller. Press Ctrl+C to exit.")

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    print("Button pressed:", event.button)
    except KeyboardInterrupt:
        pygame.quit()