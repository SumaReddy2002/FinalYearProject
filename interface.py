import pygame
import sys
import json

# Initialize Pygame
pygame.init()

# Set up some constants
WIDTH = 1000  # The width remains constant
BOX_SIZE = 300  # The size of the box
BOX_GAP = 0  # Gap between the boxes

# Load the box image and scale it to the desired size
box_image = pygame.image.load('./assets/room.jpg')
box_image = pygame.transform.scale(box_image, (BOX_SIZE, BOX_SIZE))

# List of device names
device_names = ['bulb', 'tube', 'fan', 'oven', 'ac', 'lamp', 'cooler', 'tv', 'fridge']

# Load the inner images and their alternatives, and scale them to the desired sizes
device_images = {
    name: (
        pygame.transform.scale(pygame.image.load(f'./assets/{name}.png'), size), 
        pygame.transform.scale(pygame.image.load(f'./assets/{name}'+"_on.png"), size)
    )
    for name, size in zip(device_names, [(30, 30), (100, 50), (110, 100), (80, 60), 
                                         (100, 30), (90, 140), (90, 100), (100, 100), (80, 120)])
}

# Specify the positions for the inner images within the box
inner_positions = [(20, 20), (100, 20), (185, -10), (110, 110), (100, 60), 
                   (-20, 110), (20, 140), (100, 180), (200, 120)]

# Get the number of boxes from a text file
with open("num_boxes.txt", "r") as f:
    num_boxes = int(f.read().strip())

# Get the devices for each box from separate text files
devices_in_boxes = []
for i in range(num_boxes):
    with open(f"box{i+1}.txt", "r") as f:
        devices_in_boxes.append([name.strip() for name in f.read().split(',') if name.strip() != 'nil'])

# The height is adjusted to accommodate the boxes
HEIGHT = (BOX_SIZE + BOX_GAP) * (num_boxes // 2 + num_boxes % 2) + BOX_GAP + 100

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))

# Load the background image
background = pygame.image.load('./assets/shadow.jpg')
background = pygame.transform.scale(background, (WIDTH, HEIGHT))
roof = pygame.image.load('./assets/roof.png')
roof = pygame.transform.scale(roof, (700, 400))

# Generate the box locations and their associated clicked states and device lists
boxes = []
for i in range(num_boxes):
    if i % 2 == 0:  # If it's an even-indexed box, place on the left
        x = 200  # some padding from the edge of the screen
    else:  # If it's an odd-indexed box, place on the right
        x = WIDTH - BOX_SIZE - 200  # some padding from the edge of the screen

    # calculate the y position: (BOX_SIZE + BOX_GAP) times the number of boxes already on this side
    y = 100 + (BOX_SIZE + BOX_GAP) * (i // 2)

    devices = devices_in_boxes[i]

    boxes.append((x, y, [False]*len(devices), devices))  # each box has its own clicked list and device list

# Game loop
running = True
while running:
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = pygame.mouse.get_pos()
            for box in boxes:
                
                for i, pos in enumerate(inner_positions[:len(box[3])]):
                    
                    if pygame.Rect(box[0] + pos[0], box[1] + pos[1], device_images[box[3][i]][0].get_width(), device_images[box[3][i]][0].get_height()).collidepoint(mouse_pos):
                        box[2][i] = not box[2][i]
                        with open('device_state.json', 'w') as f:
                            json.dump([{name: state for name, state in zip(box[3], box[2])} for box in boxes], f)
                    

    # Draw everything
    screen.blit(background, (0, 0))

    for box in boxes:
        screen.blit(box_image, (box[0], box[1]))
        for i, pos in enumerate(inner_positions[:len(box[3])]):
            img = device_images[box[3][i]][1] if box[2][i] else device_images[box[3][i]][0]
            screen.blit(img, (box[0] + pos[0], box[1] + pos[1]))

    screen.blit(roof, (150, -135))

    # Flip the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
