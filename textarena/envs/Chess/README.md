# Chess Environment Documentation

## Overview

**Chess** is a strategic two-player board game where each participant controls an army of 16 pieces: one King, one Queen, two Rooks, two Knights, two Bishops, and eight Pawns. The objective is to checkmate the opponent's King, rendering it unable to escape capture. Players take turns making moves in UCI (Universal Chess Interface) format, strategically advancing their pieces to control the board, attack the opponent's pieces, and defend their own.

## Action Space

- **Format:** Actions are strings representing a chess move in UCI notation, enclosed in square brackets.
- **Special Tokens:** 
    - **[<uci_move>]**: To make a chess move.
        - **Example:** `[e2e4]`
- **Example:** 
    - `"I advance my pawn to [e2e4]."`
    - `"[e7e5]"`
    - `"That's a brilliant move! [Ng1f3]"`
- **Notes:** 
    - Players can include additional text before or after the special tokens (move).
    - Make sure each action contains the special token.
    - If multiple moves are provided, only the first will be considred.

## Observation Space

### Observations

Players receive a series of messages exchanged during the game, including their own and the opponent's moves, current board state, and available valid moves. This information aids in making informed decisions about future moves or conceding the game.


**Reset Observation:**

On reset, each player receives a prompt detailing their assigned color, the initial board state, and instructions on how to interact within the game. For example:
```plaintext
[GAME]: Game started.
You are playing White in a game of Chess.
Make your move in UCI format enclosed in square brackets (e.g., [Move] e2e4).
You can also include additional text in your messages.
It's your turn. What is your move?
```
If the game is in open mode, the board state is provided as well. For example:
```plaintext
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . . . . .
. . . . . . . .
P P P P P P P P
R N B Q K B N R
```

**Step Observation:**
After each step, players receive updates about moves made and the current board state. For example:
```plaintext
Player 0: I advance my pawn to [Move] e2e4.
[GAME]: Player 0 (White) moves from e2 to e4.
```
If the game is in open mode, the board state is provided as well. For example:
```plaintext
[GAME]
r n b q k b n r
p p p p p p p p
. . . . . . . .
. . . . . . . .
. . . . P . . .
. . . . . . . .
P P P P . P P P
R N B Q K B N R
```

If the game has show_valid=True, the valid moves are also provided. For example:
```plaintext
[GAME] Valid moves: [g8h6], [g8f6], [b8c6], [b8a6], [h7h6], [g7g6], [f7f6], [e7e6], [d7d6], [c7c6], [b7b6], [a7a6], [h7h5], [g7g5], [f7f5], [e7e5], [d7d5], [c7c5], [b7b5], [a7a5]
```


## Gameplay
- **Players**: 2
- **Colors**: Player 0 plays as White, Player 1 plays as Black.
- **Turns**: Players alternate making moves, starting with White.
- **Move Format**: All moves must be in UCI notation enclosed in square brackets (e.g., `[e2e4]`).
- **Objective**: Checkmate the opponent's King to win the game.
- **Turn Limit**: When the turn limit is reached, the game ends in a draw.

## Key Rules
1. Move Mechanics:
    - Players take turns making moves in UCI format.
    - A valid move must follow standard chess rules and be legal based on the current board state.
    - Example: Moving a pawn from e2 to e4 is represented as `[e2e4]`.

2. Game Termination:
    - **Checkmate**: The game ends when a player's King is checkmated.
    - **Draw**: The game ends in a draw if the maximum number of turns is reached or by stalemate, insufficient material, threefold repetition, or the fifty-move rule.

3. Invalid Moves:
    - If a player makes an illegal move or provides an incorrectly formatted action, the game will terminate with a penalty.
    - Players are encouraged to ensure their moves are legal and correctly formatted to avoid unintended penalties.

## Rewards

| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`                |
| **Lose**         | `-1`              | `+1`                |
| **Draw**         |  `0`              |  `0`                |
| **Invalid Move** | `-1`              |  `0`                |


## Parameters

- `is_open` (`bool`):
    - **Description**: Determines whether both players can see the current board state.
    - **Impact**:
        - `True`: Both players receive the full board state after each move.
        - `False`: Players receive minimal information, enhancing the challenge of deducing the opponent's strategy.

- `max_turns` (`int`):
    - **Description**: Specifies the maximum number of turns allowed before the game ends automatically.
    - **Impact**: Limits the duration of the game, encouraging strategic and efficient play to achieve checkmate within the turn limit.

- `show_valid` (`int`):
    - **Description**: Determines whether players can see a list of valid moves.
    - **Impact**:
        - `True`: Players are provided with a list of all legal moves available to them on their turn.
        - `False`: Players must deduce valid moves based on the board state and previous actions.



## Variants

| Env-id                   | is_open  | max_turns | show_valid |
|--------------------------|:--------:|:---------:|:----------:|
| `Chess-v0`               | `True`   |   `100`   |   `True`   |
| `Chess-v0-long`          | `True`   |   `250`   |   `True`   |
| `Chess-v0-blind`         | `False`  |   `150`   |   `False`  |


### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg