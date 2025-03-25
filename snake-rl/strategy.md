# Game Overview
- Grid: the game is represented by a grid where the snake moves from one cell to another
- Movement: the snake moves in four directions
- Food collection: randomly placed food on the grid gives the snake a reson to move and grow
- Growing snake: each time the snake eats, it adds a new segment to its body - making navigation less easy

# State Space
- Snake position: the coordinates of every segment of the snake's body
- Food position: the location of the food on the grid
- Obstacles: the edges of the game grid and the snakes's own body (barriers)

# Action Space
- UP, DOWN, LEFT, RIGHT

# Reward
- Positive: everytime the snake eats food, it should receive a positive reward => encouragin the agent to seek food
- Negative: crashing into a wall or its own body resulting in a penalty
- Neutral: moving without eating nor crashing
