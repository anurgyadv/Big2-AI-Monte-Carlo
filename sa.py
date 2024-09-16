# Big Two Simulation Notebook

# Import necessary libraries
import random
import itertools
import math
from collections import defaultdict
from typing import List, Optional
from tqdm import tqdm  # For progress bars

# Define the Card class
class Card:
    SUITS = {'D': 0, 'C': 1, 'H': 2, 'S': 3}  # Diamonds, Clubs, Hearts, Spades
    RANKS = {'3': 0, '4': 1, '5': 2, '6': 3, '7': 4, '8': 5, '9': 6,
             'T': 7, 'J': 8, 'Q': 9, 'K': 10, 'A': 11, '2': 12}  # Ordered ranks

    def __init__(self, rank: str, suit: str):
        self.rank = rank
        self.suit = suit
        self.value = self.get_value()

    def get_value(self) -> int:
        # Value for comparison, combining rank and suit
        return Card.RANKS[self.rank] * 4 + Card.SUITS[self.suit]

    def __repr__(self):
        return f"{self.rank}{self.suit}"

# Define the Deck class
class Deck:
    def __init__(self):
        self.cards = [Card(rank, suit) for rank in Card.RANKS.keys() for suit in Card.SUITS.keys()]
        random.shuffle(self.cards)

    def deal(self, num_players: int) -> List[List[Card]]:
        return [sorted(self.cards[i::num_players], key=lambda card: card.value) for i in range(num_players)]

# Define the Player class
class Player:
    def __init__(self, player_num: int, hand: List[Card]):
        self.player_num = player_num
        self.hand = hand
        self.hand_size = len(hand)
        self.passed = False  # Indicates if the player has passed in the current round

    def remove_cards(self, cards: List[Card]):
        for card in cards:
            self.hand.remove(card)
        self.hand_size = len(self.hand)

# Define the Trick class
class Trick:
    def __init__(self, player_num: int, cards: List[Card]):
        self.player_num = player_num
        self.cards = cards

# Define the GameState class
class GameState:
    def __init__(self, players: List[Player], current_player_index: int, last_trick: Optional[Trick], starting_player_index: int):
        self.players = players
        self.current_player_index = current_player_index
        self.last_trick = last_trick  # The last trick played
        self.starting_player_index = starting_player_index  # The player who started the current round

# Base class for AI players
class AIPlayerBase:
    def __init__(self, player_num: int, hand: List[Card]):
        self.player_num = player_num
        self.hand = hand

    def choose_move(self, game_state: GameState) -> List[Card]:
        raise NotImplementedError("Must implement choose_move method.")

# Random AI player
class RandomAI(AIPlayerBase):
    def choose_move(self, game_state: GameState) -> List[Card]:
        valid_moves = self.get_valid_moves(game_state)
        if valid_moves:
            return random.choice(valid_moves)
        else:
            return []

    def get_valid_moves(self, game_state: GameState) -> List[List[Card]]:
        # Generate all valid moves (simplified to singles for RandomAI)
        valid_moves = []
        for card in self.hand:
            if self.is_valid_play([card], game_state.last_trick):
                valid_moves.append([card])
        return valid_moves

    def is_valid_play(self, play: List[Card], last_trick: Optional[Trick]) -> bool:
        if not last_trick or not last_trick.cards:
            return True  # Any play is valid if there's no last trick
        if len(play) != len(last_trick.cards):
            return False  # Must match the number of cards
        return play[0].value > last_trick.cards[0].value  # Simplified comparison

# Algorithm AI player (placeholder for MCTS implementation)
class AlgorithmAI(AIPlayerBase):
    def choose_move(self, game_state: GameState) -> List[Card]:
        # Implement MCTS logic here
        valid_moves = self.get_valid_moves(game_state)
        if valid_moves:
            # Placeholder: choose the move with the highest card
            return max(valid_moves, key=lambda move: max(card.value for card in move))
        else:
            return []

    def get_valid_moves(self, game_state: GameState) -> List[List[Card]]:
        # Generate all valid moves (implement full logic here)
        valid_moves = []

        # Generate all possible singles
        singles = [[card] for card in self.hand if self.is_valid_play([card], game_state.last_trick)]
        valid_moves.extend(singles)

        # Implement logic to generate pairs, triples, and five-card hands if needed

        return valid_moves

    def is_valid_play(self, play: List[Card], last_trick: Optional[Trick]) -> bool:
        if not last_trick or not last_trick.cards:
            # Must play 3D if starting and have it
            if self.has_3D() and not any(card.rank == '3' and card.suit == 'D' for card in play):
                return False
            return True  # Any play is valid if there's no last trick
        if len(play) != len(last_trick.cards):
            return False  # Must match the number of cards
        return play[0].value > last_trick.cards[0].value  # Simplified comparison

    def has_3D(self) -> bool:
        return any(card.rank == '3' and card.suit == 'D' for card in self.hand)

# Big Two game class
class Big2Game:
    def __init__(self, players: List[AIPlayerBase]):
        self.players = players
        self.num_players = len(players)
        self.current_player_index = self.find_starting_player()
        self.last_trick = None
        self.starting_player_index = self.current_player_index
        self.passes_in_row = 0

    def find_starting_player(self) -> int:
        for i, player in enumerate(self.players):
            for card in player.hand:
                if card.rank == '3' and card.suit == 'D':
                    return i
        return 0  # Default to player 0 if not found (should not happen)

    def play_game(self) -> int:
        while not self.is_game_over():
            current_player = self.players[self.current_player_index]
            game_state = GameState(self.players, self.current_player_index, self.last_trick, self.starting_player_index)
            move = current_player.choose_move(game_state)

            if move:
                # Play the move
                current_player.remove_cards(move)
                self.last_trick = Trick(current_player.player_num, move)
                self.starting_player_index = self.current_player_index
                self.passes_in_row = 0
            else:
                # Pass
                self.passes_in_row += 1
                if self.passes_in_row >= self.num_players - 1:
                    # Reset the last trick if all other players passed
                    self.last_trick = None
                    self.passes_in_row = 0
                    self.starting_player_index = (self.current_player_index + 1) % self.num_players

            # Check if the current player has won
            if current_player.hand_size == 0:
                return current_player.player_num  # Return the winner's player number

            # Move to the next player
            self.current_player_index = (self.current_player_index + 1) % self.num_players

        return -1  # Should not reach here

    def is_game_over(self) -> bool:
        return any(player.hand_size == 0 for player in self.players)

# Simulation function
def simulate_games(num_games: int):
    ai_wins = {'AlgorithmAI': 0, 'RandomAI': 0}
    for _ in tqdm(range(num_games), desc="Simulating Games"):
        # Initialize deck and deal cards
        deck = Deck()
        hands = deck.deal(4)

        # Create players
        players = []
        for i in range(4):
            if i == 0:
                # First player uses AlgorithmAI
                players.append(AlgorithmAI(i, hands[i]))
            else:
                players.append(RandomAI(i, hands[i]))

        # Start the game
        game = Big2Game(players)
        winner_num = game.play_game()

        # Record the winner
        winner_type = type(players[winner_num]).__name__
        if winner_type == 'AlgorithmAI':
            ai_wins['AlgorithmAI'] += 1
        else:
            ai_wins['RandomAI'] += 1

    # Calculate win rates
    total_games = sum(ai_wins.values())
    for ai_type, wins in ai_wins.items():
        win_rate = (wins / total_games) * 100
        print(f"{ai_type} won {wins} out of {total_games} games ({win_rate:.2f}%)")

# Run the simulation
simulate_games(100000)
