import random
import sys
import numpy as np

from action_state_spaces import *

# pygame.init()

# Game window
width, height = 400, 400
#window = pygame.display.set_mode((width, height))

#Colors
black = (0,0,0)
green = (0, 255, 0)
red = (255, 0, 0)

snake_size = 20
snake_body = [(100, 100), (80, 100), (60,100)] #Starting position of the snake
snake_speed = 20

food_position = (random.randrange(1, (width//snake_size)) * snake_size,
								 random.randrange(1, (height//snake_size)) * snake_size)
food_spawned = True

def reset_game():
    global snake_body, food_position, food_spawned

    # Reinicia a cobra no centro
    snake_body = [(100, 100), (80, 100), (60, 100)]
    
    # Gera nova posição para a comida
    food_position = spawn_new_food(snake_body, width, height)
    food_spawned = True

    state = get_state(snake_body, food_position, width, height, snake_size)
    state = np.reshape(state, [1, 8])  # reshape para o modelo

    return state, snake_body, food_position


# # Game loop
# clock = pygame.time.Clock()
# running = True
# while running:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             running = False

#     window.fill(black)

#     # Draw the snake
#     for block in snake_body:
#         pygame.draw.rect(window, green, pygame.Rect(block[0], block[1], snake_size, snake_size))

#     # Draw the food
#     pygame.draw.rect(window, red, pygame.Rect(food_position[0], food_position[1], snake_size, snake_size))

#     pygame.display.update()
#     clock.tick(snake_speed)

# pygame.quit()
# sys.exit()
