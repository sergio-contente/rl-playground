from game_setup import *

import numpy as np

def get_state(snake_body, food_position, width, height, snake_size):
	head_x, head_y = snake_body[0]

# Is the snake near the borders of the board?
	direction_up = head_y == 0
	direction_down = head_y == height - snake_size
	direction_left = head_x == 0
	direction_right = head_x == width - snake_size

# Return state vector: [head_x, head_y, food_x, food_y, is_neard_edge]
	state = [
		head_x, head_y,
		food_position[0], food_position[1],
		direction_up, direction_down, direction_left, direction_right
	]

	return state

def get_reward(snake_body, food_position, collision):
	if snake_body[0] == food_position:
		return 10 # eating food
	elif collision:
		return -10 # penalty for hitting the wall or itself
	else:
		dist = np.linalg.norm(np.array(snake_body[0]) - np.array(food_position))
		return -0.01 * dist  # quanto mais perto, melhor
	
def spawn_new_food(snake_body, width, height):
    import random
    while True:
        pos = (random.randint(0, width - 1), random.randint(0, height - 1))
        if pos not in snake_body:
            return pos


def step(action, snake_body, width, height, food_position, snake_size):
    directions_map = {
        0: (-1, 0),  # Left
        1: (1, 0),   # Right
        2: (0, -1),  # Up
        3: (0, 1)    # Down
    }

    dx, dy = directions_map[action]
    head_x, head_y = snake_body[0]
    new_head = (head_x + dx, head_y + dy)

    collision = False
    done = False

    # Verifica colisão ANTES de atualizar a snake
    if (new_head[0] < 0 or new_head[0] >= width or
        new_head[1] < 0 or new_head[1] >= height or
        new_head in snake_body):
        collision = True
        done = True
        reward = get_reward(snake_body, food_position, collision)
        return get_state(snake_body, food_position, width, height, snake_size), reward, done, food_position

    # Move a snake
    snake_body.insert(0, new_head)
    ate_food = new_head == food_position

    if not ate_food:
        snake_body.pop()  # remove a cauda

    reward = get_reward(snake_body, food_position, collision)

    if ate_food:
        food_position = spawn_new_food(snake_body, width, height)

    state = get_state(snake_body, food_position, width, height, snake_size)
    return state, reward, done, food_position
