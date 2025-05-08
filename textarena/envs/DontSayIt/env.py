import nltk, random 
from nltk import pos_tag
from nltk.corpus import words
from typing import Any, Dict, Optional, Tuple, List

import textarena as ta
import os
import json

nltk.download("words")
nltk.download("averaged_perceptron_tagger_eng")


class DontSayItEnv(ta.Env):
    """Environment for Don't Say It game"""

    def __init__(self, hardcore: Optional[bool] = False, max_turns: Optional[int] = None):
        """
        Initialize the 'Don't Say It' game environment.

        Args:
            hardcore (bool): If True, use the full English word set; otherwise, use a simplified word set.
            max_turns (int): Maximum number of turns before the game ends in a draw.
        """
        # Load the word list
        self._load_word_list(hardcore=hardcore)
        self.max_turns = max_turns
        self.game_idx = 1
        self.step_idx = 1
        self.history = []

    @property
    def terminal_render_keys(self):
        return ["target_words"]

    def _load_word_list(self, hardcore: bool = False) -> None:
        """
        Load the word list based on the 'hardcore' parameter.

        Args:
            hardcore (bool): Determines whether to load the full or simplified word list.
        """
        # Get word list
        if hardcore:
            word_list = words.words("en")
        else:
            word_list = words.words("en-basic")

        # Filter words based on POS tags
        self.word_list = [
            word for word in word_list if pos_tag([word])[0][1] in ["NN"]
        ]

    def reset(self, num_players: int, seed: Optional[int]=None):
        """ Reset the 'Don't Say It' game to its initial state """
        # Initialize game state variables
        self.state = ta.State(num_players=num_players, min_players=2, max_players=2, max_turns=self.max_turns)

        # Assign secret words to players
        target_words = {0: random.choice(self.word_list), 1: random.choice(self.word_list)}
        while target_words[0] == target_words[1]:
            target_words[1] = random.choice(self.word_list)

        self.state.reset(
            seed=seed,
            game_state={"target_words": target_words},
            player_prompt_function=self._generate_player_prompt
        )

    def _generate_player_prompt(self, player_id: int, game_state: Dict[int, Any]) -> str:
        """ Generate the initial prompt for a player, providing them with their secret word and instructions """
        prompt = (
            f"You are playing 'Don't Say It'. You are Player {player_id}\n"
            f"Your secret word is: '{game_state['target_words'][player_id]}'.\n"
            "Your goal is to get the other player to say your secret word before you say theirs.\n"
            "You can converse freely, but try to be subtle to avoid making it obvious.\n"
            "On your turn, simply type your message.\n"
        )
        if self.state.max_turns:
            prompt += f"The game lasts for {self.state.max_turns} turns in total.\n"
        if os.getenv("LFE", 'true').lower() == 'true':
            if player_id == 1 and self.game_idx != 1:
                finished_his = []
                for idx, i in enumerate(self.history[::-1]):
                    if '<game_over>' in i:
                        if idx == 0:
                            finished_his = self.history.copy()
                        else:
                            finished_his = self.history[:-idx]
                        break
                prompt += f"Below is the history game trajectories: \n<|history|>{json.dumps(finished_his)}<|history|>\n"
        if len(prompt) > 30000:
            prompt = prompt[:30000]
        return prompt


    def step(self, action: str) -> Tuple[bool, ta.Info]:
        """ Process the player's action """
        player_id = self.state.current_player_id

        # update the observations and log the action
        self.state.add_observation(from_id=player_id, to_id=-1, message=action)
        if self.step_idx == 1:
            his = (f"<game_begin>Game {self.game_idx}:"
                    f"player {player_id} took action: {action}.\n")
        else:
            his = f"player {player_id} took action: {action}.\n"
        # Check if the action mentions the opponent's secret word
        if self.state.game_state["target_words"][1 - player_id].lower() in action.lower():
            his += f"\n{player_id} wins!<game_over>\n"
            self.game_idx += 1
            self.step_idx = 1
            reason=f"Player {player_id} mentioned the opponent's secret word."
            self.state.set_winners(player_ids=[1-player_id], reason=reason)
        else:
            self.step_idx += 1
        self.history.append(his)
        return self.state.step()

