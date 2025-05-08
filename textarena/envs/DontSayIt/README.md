# Don't Say It Environment Documentation

## Overview

**Don't Say It** is a two-player conversational game where each player is assigned a secret word. The objective is to subtly guide the conversation to make the other player say your secret word while avoiding mentioning theirs. The game encourages strategic communication and subtlety, making it both challenging and entertaining.

## Action Space

- **Format:** Actions are strings representing the player's messages.
- **Example:** `"Let's talk about something interesting."`, `"Do you enjoy traveling?"`
- **Notes:** Players can converse freely, but mentioning the opponent's secret word results in a loss.

## Observation Space

### Observations

Players receive messages in the form of a conversation history, including their own secret word in the initial prompt.

**Reset Observation:**

On reset, each player receives a prompt containing their secret word and game instructions. For example:
```plaintext
[GAME]: You are Player {player_id} in 'Don't Say It'.
Your secret word is: '{secret_word}'.
Your goal is to get the other player to say your secret word before you say theirs.
You can converse freely, but try to be subtle to avoid mentioning your own secret word.
On your turn, simply type your message.
The game lasts for {max_turns} turns in total.
```

**Step Observation:**
After each step, the players receive the latest message from the other player. For example:
```plaintext
[Player 0]: How was your weekend?
[Player 1]: It was great! I went hiking in the mountains.
```

## Gameplay
- **Players**: 2
- **Turns**: Players alternate sending messages to each other.
- **Secret Words**: Each player has a unique secret word assigned at the start of the game.
- **Objective**: Make the opponent say your secret word while avoiding saying theirs.
- **Turn Limit**: The game can be configured with a maximum number of turns, after which it ends in a draw if no secret words are mentioned.

## Key Rules
1. Secret Words:
    - Each player is assigned a secret word at the start.
    - Players must avoid saying the opponent's secret word, while trying to make them say theirs.

2. Valid Moves:
    - Players send messages as strings.
    - Messages can contain any text but should aim to indirectly lead the opponent to say the secret word.

3. Winning Conditions:
    - **Win**: If a player successfully makes the opponent say their secret word.
    - **Loss**: If a player accidentally says the opponents secret word.
    - **Draw**: If the maximum number of turns is reached without any secret word being mentioned.

4. Game Termination:
    - The game ends immediately upon a win or loss.
    - If the turn limit is reached, the game ends in a draw.

## Rewards

| Outcome          | Reward for Player | Reward for Opponent |
|------------------|:-----------------:|:-------------------:|
| **Win**          | `+1`              | `-1`               |
| **Lose**         | `-1`              | `+1`               |
| **Draw**         | `0`               | `0`                |


## Parameters

- `hardcore` (`bool`):
    - **Description**: Determines the complexity of the secret words.
    - **Impact**:
        - `True`: Uses a full English word set, including complex words.
        - `False`: Uses a simplified English word set for easier gameplay.

- `max_turns` (`int`):
    - **Description**: Specifies the maximum number of turns allowed before the game ends in a draw.
    - **Impact**: Limits the duration of the game, promoting quicker resolutions.



## Variants

| Env-id                   | hardcore | max_turns |
|--------------------------|:--------:|:---------:|
| `DontSayIt-v0`           | `False`  |    `20`   |
| `DontSayIt-v0-hardcore`  | `True`   |    `30`   |
| `DontSayIt-v0-unlimited` | `False`  |   `None`  |


### Contact
If you have questions or face issues with this specific environment, please reach out directly to Guertlerlo@cfar.a-star.edu.sg