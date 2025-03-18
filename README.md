# Big2 AI Monte Carlo Bot 🃏

## What's This?

This is my AI bot for playing the card game Big2 (also known as Deuces or Dai Di), built using Monte Carlo Tree Search (MCTS). The bot competes in the Big2 AI Competition by making strategic decisions about which cards to play.

## How It Works

The bot uses MCTS to simulate possible game outcomes and choose the optimal move. Here's the workflow:

```
Initialize MCTS → Select Node → Expand Node → Simulate Game → Backpropagate Results → Choose Best Move
     |                 |             |              |                  |                 |
     v                 v             v              v                  v                 v
Setup Game State   UCB Selection   Create     Random Playouts     Update Values      Strategic
& Parameters       for Balance     Child        to End Game       Up the Tree       Card Selection
```

### Key Components

1. **MCTSNode**: Represents a node in the Monte Carlo search tree, tracking game state, visits, and value.

2. **Algorithm**:
   - `getAction`: Main entry point that decides the best move
   - `perform_mcts`: Runs the Monte Carlo Tree Search within time constraints
   - `get_possible_moves`: Identifies all valid plays based on the current state
   - `choose_strategic_move`: Makes the final decision using strategic considerations

3. **Card Combinations**:
   - Singles, Pairs, Triples
   - Straights, Flushes
   - Full Houses
   - Four of a Kind
   - Straight Flushes

## Strategy Highlights

- **Time Management**: Allocates 0.8 seconds per move to ensure decisions are made within tournament time limits
- **Opponent Modeling**: Tracks opponent tendencies to inform future decisions
- **Hand Evaluation**: Prioritizes stronger hand types and preserves stronger cards when possible
- **End-game Tactics**: Adjusts strategy when players are close to winning

## Leaderboard

I also created a leaderboard dashboard to track my bot's performance in the competition: [Big2 AI Monte Carlo Leaderboard](https://github.com/anurgyadv/Big2-Competition-Leaderboard.git)
The leaderboard automatically:
- Pulls game logs from the competition server
- Processes results and statistics
- Visualizes performance data
- Updates every 20 minutes

## Game Rules

Big2 is played with a standard 52-card deck. The card ranking is:
- 2 (highest)
- A, K, Q, J, 10, 9, 8, 7, 6, 5, 4, 3 (lowest)

The suits are ranked (from lowest to highest): Diamonds, Clubs, Hearts, Spades.

The goal is to be the first player to play all your cards by playing valid combinations that beat the previous play.

## What I Learned

Building this bot taught me:
- Monte Carlo Tree Search implementation
- Strategic decision-making algorithms
- Game state simulation
- Probability-based AI
- Card game combinatorics
- Time-constrained algorithm optimization

[Screenshot of the bot in action will go here]
