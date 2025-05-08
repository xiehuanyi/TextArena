# Connect Four Environment Documentation

## Overview

**Connect Four** is a two-player connection game where players alternately drop discs into a vertical grid. The objective is to connect four of one's own discs in a row—vertically, horizontally, or diagonally—before the opponent does. This text-based environment allows customization of grid size and supports both Open and Closed game modes.

## Action Space

- **Format:** Actions are strings in the format `[col x]`, where `x` is a valid column number (0 to `num_cols - 1`).
- **Examples:** 
    - `[col 3]`
    - `I will drop the disc into column 3: [col 3]` 
- **Notes:** 
    - Players are free to include additional text before or after the special tokens.
    - Make sure each action contains a valid column 
    - If multiple moves are provided, only the first will be considered


## Observation Space

### Observations

Players receive a series of messages exchanged during the game, including their own & the opponent's moves and the current board state. This information aids in making informed decisions about future moves or conceding the game.

**Reset Observation:**

In the first observation, each player receives a prompt detailing the initial board state and game specific instructions. For example:
```plaintext
[GAME] You are Player 0 in Connect Four.
Your disc symbol: X.
The game board has 6 rows and 7 columns.
Players take turns dropping their disc into one of the columns (0 to 6).
First to connect four discs vertically, horizontally, or diagonally wins.
On your turn, enter the column number in squared brackets to make your move.
For example: '[col 4]' or '[col 1]'.
```
If the game is in open mode, the board state is provided as well. For example:
```plaintext
The game board is visible to both players.
Current Board:
0 1 2 3 4 5 6
-------------
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
```

**Step Observation:**
After each subsequent step, players receive updates about moves made and the current board state. For example:
```plaintext
[Player 0] [col 0]
[GAME] Board state:
0 1 2 3 4 5 6
-------------
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
X . . . . . .
[Player 1] [col 1]
[GAME] Board state:
0 1 2 3 4 5 6
-------------
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
X O . . . . .

```


## Gameplay
- **Grid Size:** Customizable (`num_rows` x `num_cols`).
- **Players**: 2
- **Turns**: Players alternate making moves until one player wins or all fields are used (draw).
- **Move Format**: The moves are col, followed by the column number (e.g., `[col 0]`)
- **Disc Placement:** A disc falls to the lowest available space within the chosen column.
- **Winning Condition:** Connect four discs vertically, horizontally, or diagonally.
- **Draw Condition:** The game ends in a draw if the board is full without any player connecting four discs.

## Key Rules

1. **Move Mechanics**:
    - Players take turns dropping discs into a column using the `[col x]` format.
    - A valid move must correspond to an unfilled column in the range `[0, num_cols - 1]`.
    - The disc falls to the lowest available space in the chosen column.

2. **Game Termination**:
    - **Win**: The game ends immediately when a player connects four discs in a row vertically, horizontally, or diagonally.
    - **Draw**: The game ends in a draw if all columns are filled and no player has achieved four in a row.

3. **Invalid Moves**:
    - If a player attempts to place a disc in a full column, the move is considered invalid.
    - If a player provides an incorrectly formatted action (e.g., missing brackets or an invalid column number), the move is rejected.
    - Players must ensure their moves are properly formatted and legal to avoid penalties.

4. **Game Visibility**:
    - If `is_open` is `True`, the full board state is visible after each move.
    - If `is_open` is `False`, players only receive textual updates about moves made.

5. **Turn Structure**:
    - Players alternate turns sequentially, with Player 0 starting the game.
    - The game continues until a player wins or a draw is reached.




## Rewards

| Outcome          | Reward for Player  | Reward for Opponent |
|------------------|:------------------:|:-------------------:|
| **Win**          |       `+1`         |        `-1`         |
| **Lose**         |       `-1`         |        `+1`         |
| **Draw**         |        `0`         |         `0`         |
| **Invalid Move** |       `-1`         |         `0`         |

## Parameters

- `is_open` (`bool`): 
  - **Description:** Determines whether the game board is visible to both players.
  - **Impact:** 
    - `True:` Players can see the current state of the board.
    - `False:` Players receive only textual updates without seeing the board.

- `num_rows` (`int`): 
  - **Description:** Specifies the number of rows in the game board.
  - **Impact:** Affects the vertical size of the grid, influencing how discs stack.

- `num_cols` (`int`): 
  - **Description:** Specifies the number of columns in the game board.
  - **Impact:** Determines the horizontal size of the grid and available columns for disc placement.

## Variants

| Env-id                   | is_open | num_rows | num_cols |
|--------------------------|:-------:|:--------:|:--------:|
| `ConnectFour-v0`         | `True`  | `6`      | `7`      |
| `ConnectFour-v0-blind`   | `False` | `6`      | `7`      |
| `ConnectFour-v0-large`   | `True`  | `12`     | `15`     |


### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg