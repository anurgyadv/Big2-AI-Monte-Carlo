import json
import time
import random
import math
import itertools
from typing import List, Optional, Tuple
from classes import *

class MCTSNode:
    def __init__(self, state: MatchState, parent=None, move: Optional[List[str]] = None):
        self.state = state
        self.parent = parent
        self.move = move
        self.children = []
        self.visits = 0
        self.value = 0

    def add_child(self, child_state: MatchState, move: List[str]):
        child = MCTSNode(child_state, parent=self, move=move)
        self.children.append(child)
        return child

    def update(self, result: float):
        self.visits += 1
        self.value += result


class Algorithm:
    def getAction(self, state: MatchState):
        start_time = time.time()
        print(f"Player {state.myPlayerNum} - Turn start")
        try:
            # Load data from previous turn
            if state.myData:
                data = json.loads(state.myData)
                move_history = data.get('move_history', {})
                info_sets = data.get('info_sets', {})
                opponent_model = data.get('opponent_model', {})
            else:
                move_history, info_sets, opponent_model = {}, {}, {}

            # Log current state
            print(f"My hand: {state.myHand}")
            print(f"To beat: {state.toBeat.cards if state.toBeat else 'None'}")

            # Analyze game history
            # print("Analyzing game history...")
            # self.analyze_game_history(state.matchHistory)
            # print("Game history analysis complete.")

            # Determine if it's the first turn and we have 3D
            if self.is_first_turn(state) and '3D' in state.myHand:
                print("First turn of the game and we have 3D.")
                three_d = self.convert_hand(['3D'])[0]
                action = self.perform_mcts(state, move_history, info_sets, opponent_model, include_cards=[three_d])
            else:
                # Perform MCTS and choose the best action
                # print("Starting MCTS...")
                action = self.perform_mcts(state, move_history, info_sets, opponent_model)
                # print("MCTS complete.")

            # Update strategies
            # print("Updating strategies...")
            move_history, info_sets, opponent_model = self.update_strategies(action, state, move_history, info_sets, opponent_model)
            # print("Strategy update complete.")

            # Prepare data for the next turn
            myData = json.dumps({
                'last_action': action,
                'move_history': move_history,
                'info_sets': info_sets,
                'opponent_model': opponent_model
            })

            if not action:
                print("No valid moves found, passing turn.")
                return [], myData

            print(f"Chosen action: {action}")
            print(f"Player {state.myPlayerNum} - Turn end")
            print(f"Turn took {time.time() - start_time:.3f} seconds")
            return action, myData
        except Exception as e:
            print("An error occurred in getAction: {}".format(str(e)))
            print("Error details: {}".format(str(e)))
            print("State at time of error:")
            print("  My hand: {}".format(state.myHand))
            print("  To beat: {}".format(state.toBeat.cards if state.toBeat else 'None'))
            print("  My player number: {}".format(state.myPlayerNum))
            print("  Number of players: {}".format(len(state.players)))
            print("  Current game state: {}".format(state.matchHistory[-1].finished))
            return [], json.dumps({})  # Return empty move and data in case of error

    def is_first_turn(self, state: MatchState) -> bool:
        current_game = state.matchHistory[-1]
        if not current_game.gameHistory or not any(current_game.gameHistory):
            return True
        return False

    def convert_hand(self, hand: List[str]) -> List[int]:
        values = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15}
        suits = {'D': 0, 'C': 1, 'H': 2, 'S': 3}
        result = []
        for card in hand:
            if len(card) != 2:
                print(f"Warning: Invalid card format: {card}")
                continue
            value, suit = card[0], card[1]
            if value not in values or suit not in suits:
                print(f"Warning: Unknown card: {card}")
                continue
            result.append((values[value] - 3) * 4 + suits[suit])
        return result
    
    def convert_back_hand(self, hand: List[int]) -> List[str]:
        values = '34567890JQKA2'
        suits = 'DCHS'
        return [f"{'T' if values[card // 4] == '0' else values[card // 4]}{suits[card % 4]}" for card in hand]

    def is_valid_play(self, combo: List[int], to_beat: List[int]) -> bool:
        if len(combo) != len(to_beat):
            return False

        combo_type = self.get_hand_type(combo)
        to_beat_type = self.get_hand_type(to_beat)

        # Define the strength of hand types
        hand_strength = {
            "Single": 1,
            "Pair": 2,
            "Triple": 3,
            "Straight": 4,
            "Flush": 5,
            "FullHouse": 6,
            "FourOfAKind": 7,
            "StraightFlush": 8
        }

        # Special rule: a stronger hand type beats a weaker one
        if hand_strength.get(combo_type, 0) > hand_strength.get(to_beat_type, 0):
            return True
        elif hand_strength.get(combo_type, 0) < hand_strength.get(to_beat_type, 0):
            return False

        # If they are the same type, compare the strength within the type
        return self.is_stronger(combo, to_beat)

    def get_hand_type(self, hand: List[int]) -> str:
        if len(hand) == 1:
            return "Single"
        elif len(hand) == 2:
            return "Pair" if hand[0] // 4 == hand[1] // 4 else "Invalid"
        elif len(hand) == 3:
            return "Triple" if hand[0] // 4 == hand[1] // 4 == hand[2] // 4 else "Invalid"
        elif len(hand) == 5:
            if self.is_straight(hand) and self.is_flush(hand):
                return "StraightFlush"
            elif self.is_four_of_a_kind(hand):
                return "FourOfAKind"
            elif self.is_full_house(hand):
                return "FullHouse"
            elif self.is_flush(hand):
                return "Flush"
            elif self.is_straight(hand):
                return "Straight"
        return "Invalid"

    def is_straight(self, hand: List[int]) -> bool:
        values = sorted(set(card // 4 for card in hand))
        return len(values) == 5 and values[-1] - values[0] == 4

    def is_flush(self, hand: List[int]) -> bool:
        return len(set(card % 4 for card in hand)) == 1

    def is_four_of_a_kind(self, hand: List[int]) -> bool:
        values = [card // 4 for card in hand]
        return any(values.count(value) == 4 for value in set(values))

    def is_full_house(self, hand: List[int]) -> bool:
        values = [card // 4 for card in hand]
        return len(set(values)) == 2 and (values.count(values[0]) in [2, 3])

    def is_stronger(self, combo: List[int], to_beat: List[int]) -> bool:
        combo_type = self.get_hand_type(combo)
        to_beat_type = self.get_hand_type(to_beat)

        if combo_type != to_beat_type:
            # Hands must be of the same type to compare
            return False

        if combo_type == "Flush":
            # Compare flushes by rank only
            return self.compare_flushes(combo, to_beat)
        elif combo_type == "Straight":
            # Compare highest cards
            return max(combo) > max(to_beat)
        elif combo_type == "StraightFlush":
            # Compare highest cards
            return max(combo) > max(to_beat)
        elif combo_type == "FullHouse":
            # Compare triplet values
            combo_triple = self.get_triple_value(combo)
            to_beat_triple = self.get_triple_value(to_beat)
            if combo_triple != to_beat_triple:
                return combo_triple > to_beat_triple
            # Compare pair values
            combo_pair = self.get_pair_value(combo)
            to_beat_pair = self.get_pair_value(to_beat)
            return combo_pair > to_beat_pair
        elif combo_type == "FourOfAKind":
            # Compare quads
            combo_quad = self.get_quad_value(combo)
            to_beat_quad = self.get_quad_value(to_beat)
            if combo_quad != to_beat_quad:
                return combo_quad > to_beat_quad
            # Compare kickers
            combo_kicker = self.get_kicker(combo, combo_quad)
            to_beat_kicker = self.get_kicker(to_beat, to_beat_quad)
            return combo_kicker > to_beat_kicker
        elif combo_type in ["Triple", "Pair", "Single"]:
            # Compare highest cards
            return max(combo) > max(to_beat)
        else:
            return False

    def compare_flushes(self, combo: List[int], to_beat: List[int]) -> bool:
        # Sort the flushes by rank in descending order
        combo_sorted = sorted(combo, key=lambda c: (c // 4), reverse=True)
        to_beat_sorted = sorted(to_beat, key=lambda c: (c // 4), reverse=True)
        
        for c_card, t_card in zip(combo_sorted, to_beat_sorted):
            c_rank = c_card // 4
            t_rank = t_card // 4
            if c_rank > t_rank:
                return True
            elif c_rank < t_rank:
                return False
        # All ranks are the same, flushes are equal
        return False
    
    def get_kicker(self, hand: List[int], quad_value: int) -> int:
        return max(card for card in hand if (card // 4) != (quad_value // 4))
    
    def get_pair_value(self, hand: List[int]) -> int:
        values = [card // 4 for card in hand]
        value_counts = {}
        for val in values:
            value_counts[val] = value_counts.get(val, 0) + 1
        # Find the value with a count of 2
        pair_value = max([val for val, count in value_counts.items() if count == 2])
        # Return the highest card of the pair (includes suit)
        pair_cards = [card for card in hand if card // 4 == pair_value]
        return max(pair_cards)


    def get_triple_value(self, hand: List[int]) -> int:
        values = [card // 4 for card in hand]
        value_counts = {}
        for val in values:
            value_counts[val] = value_counts.get(val, 0) + 1
        # Find the value with a count of 3
        triple_value = max([val for val, count in value_counts.items() if count == 3])
        # Return the highest card of the triplet (includes suit)
        triple_cards = [card for card in hand if card // 4 == triple_value]
        return max(triple_cards)


    def get_quad_value(self, hand: List[int]) -> int:
        values = [card // 4 for card in hand]
        value_counts = {}
        for val in values:
            value_counts[val] = value_counts.get(val, 0) + 1
        # Find the value with a count of 4
        quad_value = max([val for val, count in value_counts.items() if count == 4])
        # Return the highest card of the quad (includes suit)
        quad_cards = [card for card in hand if card // 4 == quad_value]
        return max(quad_cards)
        
    def analyze_game_history(self, match_history):
        current_game = match_history[-1]
        print(f"Current game finished: {current_game.finished}")
        print(f"Current game winner: {current_game.winnerPlayerNum}")

        for round_index, round in enumerate(current_game.gameHistory):
            print(f"Round {round_index + 1}:")
            for trick in round:
                print(f"  Player {trick.playerNum} played: {trick.cards}")

    def perform_mcts(self, state: MatchState, move_history, info_sets, opponent_model, include_cards: List[int]=None):
        # print("Initializing MCTS...")
        root = MCTSNode(state)
        end_time = time.time() + 0.8  # Reduced slightly to ensure we don't exceed time limit
        iterations = 0
        while time.time() < end_time:
            try:
                node = self.select(root)
                child = self.expand(node)
                result = self.simulate(child)
                self.backpropagate(child, result)
                iterations += 1
            except Exception as e:
                print(f"Error in MCTS iteration: {str(e)}")
                break
        # print(f"MCTS completed {iterations} iterations.")
        possible_moves = self.get_possible_moves(root.state, include_cards)
        if not possible_moves:
            return []  # Pass if no valid moves found
        best_move = self.choose_strategic_move(possible_moves, state, move_history, info_sets, opponent_model)
        return best_move


    def select(self, node: MCTSNode):
        # print("    Select: Starting")
        while node.children:
            if not all(child.visits > 0 for child in node.children):
                # print("    Select: Expanding non-visited child")
                return self.expand(node)
            # print("    Select: Selecting best child")
            node = self.ucb_select(node)
        # print("    Select: Returning node")
        return node

    def expand(self, node: MCTSNode):
        # print("    Expand: Starting")
        possible_moves = self.get_possible_moves(node.state)
        # print(f"    Expand: Found {len(possible_moves)} possible moves")
        for move in possible_moves:
            if not any(child.move == move for child in node.children):
                # print("    Expand: Creating new child node")
                new_state = self.apply_move(node.state, move)
                return node.add_child(new_state, move)
        # print("    Expand: No new moves to expand, returning node")
        return node

    def simulate(self, node: MCTSNode):
        # print("    Simulate: Starting")
        state = self.deep_copy_state(node.state)
        depth = 0
        while not self.is_terminal(state):
            possible_moves = self.get_possible_moves(state)
            if not possible_moves:
                # print(f"    Simulate: No moves available at depth {depth}, passing turn")
                state = self.pass_turn(state)
            else:
                # print(f"    Simulate: Choosing move at depth {depth}")
                move = self.choose_simulation_move(possible_moves)
                state = self.apply_move(state, move)
            depth += 1
            if depth > 100:  # Safeguard against infinite loops
                # print("    Simulate: Maximum depth reached, terminating simulation")
                break
        result = self.get_result(state)
        # print(f"    Simulate: Finished with result {result}")
        return result
    
    def choose_simulation_move(self, possible_moves):
    # Prioritize longer combinations
        longest_moves = [move for move in possible_moves if len(move) == max(len(m) for m in possible_moves)]
        return random.choice(longest_moves)

    def backpropagate(self, node: MCTSNode, result: float):
        # print("    Backpropagate: Starting")
        while node:
            node.update(result)
            node = node.parent
        # print("    Backpropagate: Finished")

    def ucb_select(self, node: MCTSNode):
        log_n_vertex = math.log(node.visits)
        return max(node.children, key=lambda c: self.calculate_ucb(c, log_n_vertex))

    def calculate_ucb(self, child: MCTSNode, log_n_vertex: float) -> float:
        if child.visits == 0:
            return float('inf')
        return (child.value / child.visits) + math.sqrt(2 * log_n_vertex / child.visits)

    def get_possible_moves(self, state: MatchState, include_cards: List[int]=None) -> List[List[str]]:
        hand = self.convert_hand(state.myHand)
        possible_moves = []
        
        all_moves = (self.get_four_of_a_kinds(hand) +
                    self.get_full_houses(hand) +
                    self.get_flushes(hand) +
                    self.get_straights(hand) +
                    self.get_triples(hand) +
                    self.get_pairs(hand) +
                    [[card] for card in hand])

        if state.toBeat is None or not state.toBeat.cards:
            possible_moves = all_moves
        else:
            to_beat = self.convert_hand(state.toBeat.cards)
            possible_moves = [move for move in all_moves if self.is_valid_play(move, to_beat)]

        if include_cards:
            include_cards_set = set(include_cards)
            possible_moves = [move for move in possible_moves if include_cards_set.issubset(set(move))]

        return [self.convert_back_hand(move) for move in possible_moves]


    def get_all_valid_combinations(self, hand: List[int]) -> List[List[int]]:
        combinations = []
        combinations.extend([[card] for card in hand])
        combinations.extend([combo for combo in self.get_pairs(hand)])
        combinations.extend([combo for combo in self.get_triples(hand)])
        combinations.extend([combo for combo in self.get_five_card_combos(hand)])
        return combinations

    def get_pairs(self, hand: List[int]) -> List[List[int]]:
        return [list(combo) for combo in set(tuple(sorted([c1, c2])) 
                for i, c1 in enumerate(hand) 
                for c2 in hand[i+1:] 
                if c1 // 4 == c2 // 4)]
    
    def get_straights(self, hand: List[int]) -> List[List[int]]:
        straights = []
        values = sorted(set(card // 4 for card in hand))
        for i in range(len(values) - 4):
            if values[i+4] - values[i] == 4:
                straight = [card for card in hand if values[i] <= card // 4 <= values[i+4]]
                straights.extend(combo for combo in itertools.combinations(straight, 5) 
                                if len(set(card // 4 for card in combo)) == 5)
        return [list(straight) for straight in straights]

    def get_flushes(self, hand: List[int]) -> List[List[int]]:
        flushes = []
        for suit in range(4):
            suited_cards = [card for card in hand if card % 4 == suit]
            if len(suited_cards) >= 5:
                flushes.extend(itertools.combinations(suited_cards, 5))  # Generate all 5-card flushes
        return [list(flush) for flush in flushes]

    def get_full_houses(self, hand: List[int]) -> List[List[int]]:
        triples = self.get_triples(hand)
        pairs = self.get_pairs(hand)
        full_houses = []
        for triple in triples:
            for pair in pairs:
                if triple[0] // 4 != pair[0] // 4:
                    full_houses.append(triple + pair)
        return full_houses

    def get_four_of_a_kinds(self, hand: List[int]) -> List[List[int]]:
        four_of_a_kinds = []
        values = [card // 4 for card in hand]
        for value in set(values):
            if values.count(value) == 4:
                four = [card for card in hand if card // 4 == value]
                kickers = [card for card in hand if card // 4 != value]
                four_of_a_kinds.extend([four + [kicker] for kicker in kickers])
        return four_of_a_kinds

    def get_triples(self, hand: List[int]) -> List[List[int]]:
        return [list(combo) for combo in set(tuple(sorted([c1, c2, c3])) 
                for i, c1 in enumerate(hand) 
                for j, c2 in enumerate(hand[i+1:], i+1) 
                for c3 in hand[j+1:] 
                if c1 // 4 == c2 // 4 == c3 // 4)]

    def get_five_card_combos(self, hand: List[int]) -> List[List[int]]:
        combos = []
        for combo in self.combinations(hand, 5):
            if is_straight(combo) or is_flush(combo) or is_full_house(combo) or is_four_of_a_kind(combo):
                combos.append(list(combo))
        return combos

    def combinations(self, iterable, r):
        pool = tuple(iterable)
        n = len(pool)
        if r > n:
            return
        indices = list(range(r))
        yield tuple(pool[i] for i in indices)
        while True:
            for i in reversed(range(r)):
                if indices[i] != i + n - r:
                    break
            else:
                return
            indices[i] += 1
            for j in range(i+1, r):
                indices[j] = indices[j-1] + 1
            yield tuple(pool[i] for i in indices)

    def apply_move(self, state: MatchState, move: List[str]) -> MatchState:
        new_state = self.deep_copy_state(state)
        new_state.myHand = [card for card in new_state.myHand if card not in move]
        new_state.toBeat = Trick(new_state.myPlayerNum, move)
        new_state.players[new_state.myPlayerNum].handSize -= len(move)
        return new_state

    def pass_turn(self, state: MatchState) -> MatchState:
        new_state = self.deep_copy_state(state)
        new_state.toBeat = Trick(new_state.myPlayerNum, [])
        return new_state

    def is_terminal(self, state: MatchState) -> bool:
        return any(player.handSize == 0 for player in state.players)

    def get_result(self, state: MatchState) -> float:
        return 1.0 if state.players[state.myPlayerNum].handSize == 0 else 0.0

    def deep_copy_state(self, state: MatchState) -> MatchState:
        return MatchState(
            state.myPlayerNum,
            [Player(p.points, p.handSize) for p in state.players],
            state.myHand.copy(),
            Trick(state.toBeat.playerNum, state.toBeat.cards.copy()) if state.toBeat else None,
            [GameHistory(gh.finished, gh.winnerPlayerNum, [[Trick(t.playerNum, t.cards.copy()) for t in round] for round in gh.gameHistory]) for gh in state.matchHistory],
            state.myData
        )

    def update_strategies(self, action: List[str], state: MatchState, move_history: dict, info_sets: dict, opponent_model: dict) -> Tuple[dict, dict, dict]:
        # Update move history
        move_key = ','.join(sorted(action))
        move_history[move_key] = move_history.get(move_key, 0) + 1

        # Update info sets
        info_set_key = self.get_info_set_key(state)
        if info_set_key not in info_sets:
            info_sets[info_set_key] = []
        info_sets[info_set_key].append(action)

        # Update opponent model
        for player in state.players:
            if player.handSize == 0:
                opponent_key = f"player_{state.players.index(player)}"
                opponent_model[opponent_key] = opponent_model.get(opponent_key, []) + [state.matchHistory[-1]]

        return move_history, info_sets, opponent_model

    def get_info_set_key(self, state: MatchState) -> str:
        return f"hand_size_{len(state.myHand)}_to_beat_{state.toBeat.cards if state.toBeat else 'None'}"


    def get_move_key(self, move: List[str]) -> str:
        return ','.join(sorted(move))

    def get_card_value(self, card: str) -> int:
        values = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15}
        return values[card[0]]

    def calculate_hand_strength(self, hand: List[str]) -> float:
        if not hand:
            return 0.0  # Return 0 strength for empty hand
        
        numeric_hand = self.convert_hand(hand)  # Use self.convert_hand instead of global convert_hand
        strength = sum(card // 4 for card in numeric_hand) / len(hand)
            
        # Bonus for pairs, triples, etc.
        value_counts = {}
        for card in numeric_hand:
            value = card // 4
            value_counts[value] = value_counts.get(value, 0) + 1
        
        for count in value_counts.values():
            if count == 2:
                strength += 0.5
            elif count == 3:
                strength += 1
            elif count == 4:
                strength += 2
        
        # Bonus for potential straights and flushes
        if len(set(card % 4 for card in numeric_hand)) == 1:
            strength += 1
        if len(set(card // 4 for card in numeric_hand)) == len(hand) and max(numeric_hand) - min(numeric_hand) < 20:
            strength += 1
        
        return strength

    def analyze_opponents(self, state: MatchState, opponent_model: dict):
        for i, player in enumerate(state.players):
            if i != state.myPlayerNum:
                print(f"Analyzing opponent {i}:")
                print(f"  Hand size: {player.handSize}")
                print(f"  Points: {player.points}")
                
                opponent_key = f"player_{i}"
                if opponent_key in opponent_model:
                    games_played = len(opponent_model[opponent_key])
                    games_won = sum(1 for game in opponent_model[opponent_key] if game.winnerPlayerNum == i)
                    win_rate = games_won / games_played if games_played > 0 else 0
                    print(f"  Win rate: {win_rate:.2f}")
                    
                    avg_hand_size = sum(player.handSize for game in opponent_model[opponent_key] for player in game.players if game.players.index(player) == i) / games_played
                    print(f"  Average final hand size: {avg_hand_size:.2f}")
                else:
                    print("  No data available for this opponent yet.")

    def choose_strategic_move(self, possible_moves: List[List[str]], state: MatchState, move_history: dict, info_sets: dict, opponent_model: dict) -> List[str]:
        if not possible_moves:
            return []  # Pass if no moves available
        
        hand_type_priority = {
            "StraightFlush": 8,
            "FourOfAKind": 7,
            "FullHouse": 6,
            "Flush": 5,
            "Straight": 4,
            "Triple": 3,
            "Pair": 2,
            "Single": 1
        }

        def get_move_priority(move: List[str]) -> Tuple[int, float, int]:
            hand_type = self.get_hand_type(self.convert_hand(move))
            hand_strength = self.calculate_hand_strength(move)
            return (
                hand_type_priority.get(hand_type, 0),  # First, by hand type strength
                -hand_strength,  # Second, lower strength is better (negative to sort in descending order)
                len(move)  # Third, by move size (more cards = better)
            )

        best_move = max(possible_moves, key=get_move_priority)
        
        # Additional strategic considerations
        next_player = (state.myPlayerNum + 1) % len(state.players)
        if state.players[next_player].handSize <= 3:
            # If next player is close to winning, prefer longer moves
            longer_moves = [move for move in possible_moves if len(move) > len(best_move)]
            if longer_moves:
                best_move = max(longer_moves, key=get_move_priority)
        
        return best_move


    def get_move_key(self, move: List[str]) -> str:
        return ','.join(sorted(move))

    def calculate_hand_strength(self, hand: List[str]) -> float:
        if not hand:
            return 0.0  # Return 0 strength for empty hand
        
        numeric_hand = self.convert_hand(hand)
        strength = sum(card // 4 for card in numeric_hand) / len(hand)
        
        # Bonus for pairs, triples, etc.
        value_counts = {}
        for card in numeric_hand:
            value = card // 4
            value_counts[value] = value_counts.get(value, 0) + 1
        
        for count in value_counts.values():
            if count == 2:
                strength += 0.5
            elif count == 3:
                strength += 1
            elif count == 4:
                strength += 2
        
        # Bonus for potential straights and flushes
        if len(set(card % 4 for card in numeric_hand)) == 1:
            strength += 1
        if len(set(card // 4 for card in numeric_hand)) == len(hand) and max(numeric_hand) - min(numeric_hand) < 20:
            strength += 1
        
        return strength

    def convert_hand(self, hand: List[str]) -> List[int]:
        values = {'3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14, '2': 15}
        suits = {'D': 0, 'C': 1, 'H': 2, 'S': 3}
        result = []
        for card in hand:
            if len(card) != 2:
                print(f"Warning: Invalid card format: {card}")
                continue
            value, suit = card[0], card[1]
            if value not in values or suit not in suits:
                print(f"Warning: Unknown card: {card}")
                continue
            result.append((values[value] - 3) * 4 + suits[suit])
        return result
    
    def update_info_sets(self, move: List[str], state: MatchState, info_sets: dict):
        info_set_key = self.get_info_set_key(state)
        if info_set_key not in info_sets:
            info_sets[info_set_key] = []
        
        info_sets[info_set_key].append({
            'move': move,
            'hand_size': len(state.myHand),
            'to_beat': state.toBeat.cards if state.toBeat else None,
            'player_hand_sizes': [player.handSize for player in state.players]
        })

    def learn_from_info_sets(self, info_sets: dict):
        for key, moves in info_sets.items():
            if len(moves) > 1:
                print(f"Analysis for info set: {key}")
                move_counts = {}
                for move_info in moves:
                    move_key = self.get_move_key(move_info['move'])
                    move_counts[move_key] = move_counts.get(move_key, 0) + 1
                
                total_moves = sum(move_counts.values())
                for move, count in move_counts.items():
                    print(f"  Move {move}: played {count} times ({count/total_moves:.2f})")

    def end_game_analysis(self, state: MatchState, move_history: dict, info_sets: dict, opponent_model: dict):
        print("End of game analysis:")
        
        # Analyze final standings
        final_standings = sorted(enumerate(state.players), key=lambda x: x[1].handSize)
        for rank, (player_num, player) in enumerate(final_standings, 1):
            print(f"Rank {rank}: Player {player_num} (Hand size: {player.handSize})")
        
        # Analyze most successful moves
        print("\nMost successful moves:")
        sorted_moves = sorted(move_history.items(), key=lambda x: x[1], reverse=True)
        for move, count in sorted_moves[:5]:
            print(f"  {move}: played {count} times")
        
        # Learn from info sets
        print("\nLearning from info sets:")
        self.learn_from_info_sets(info_sets)
        
        # Update opponent model
        winner = final_standings[0][0]
        opponent_key = f"player_{winner}"
        if opponent_key not in opponent_model:
            opponent_model[opponent_key] = []
        opponent_model[opponent_key].append(state.matchHistory[-1])
        
        print(f"\nUpdated opponent model for winner (Player {winner})")

#