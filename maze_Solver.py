def maze_Runner(maze, x, y, exit, i):
    """
    This function solves and prints a maze of open paths (0's) and walls (1's), while also tracking movement through the maze.

    Parameters:
    maze (list): 2D array representing the maze
    x (int): X-coordinate of the current location
    y (int): Y-coordinate of the current location
    exit (tuple): Tuple representing the coordinates of the exit / goal
    i (int): Arbitrary number used to track movement throughout the maze
    """
    num_rows = len(maze)
    num_cols = len(maze[0])

    # Base case: Current location is on the exit
    maze[x][y] = i
    if (x, y) == exit:
        for row in maze:
            print(row)
        return True
    
    # Base Case # 2: No illegal moves
    if not (0 <= x < num_rows and 0 <= y < num_cols) or maze[x][y] == 1:
        return False

    # Recursive case: Move to the next location
    if x + 1 < num_rows and maze[x + 1][y] == 0:
        if maze_Runner(maze, x + 1, y, exit, i):
            return True
    elif x - 1 >= 0 and maze[x - 1][y] == 0:
        if maze_Runner(maze, x - 1, y, exit, i):
            return True
    elif y + 1 < num_cols and maze[x][y + 1] == 0:
        if maze_Runner(maze, x, y + 1, exit, i):
            return True
    elif y - 1 >= 0 and maze[x][y - 1] == 0:
        if maze_Runner(maze, x, y - 1, exit, i):
            return True

    # Base Case #3: If no valid moves, increment i and try to move to the highest adjacent number
    i += 1
    adj_positions = [
        (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)
    ]
    max_value = -1
    max_pos = None
    for nx, ny in adj_positions:
        if 0 <= nx < num_rows and 0 <= ny < num_cols:
            if maze[nx][ny] > max_value and maze[nx][ny] != 1:
                max_value = maze[nx][ny]
                max_pos = (nx, ny)  
    if max_pos:
        nx, ny = max_pos
        if maze_Runner(maze, nx, ny, exit, i):
            return True

    # If no valid moves, return False to backtrack
    return False


maze = [
    [0, 1, 0, 0, 0],
    [0, 1, 0, 1, 0],
    [0, 0, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 0]
]

starting_Point = (0, 0)
exit = (4, 4)
i = 2
maze_Runner(maze, starting_Point[0], starting_Point[1], exit, i)
