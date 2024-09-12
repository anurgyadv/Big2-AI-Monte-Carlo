from submission import Algorithm 
from classes import MatchState, Player, Trick, GameHistory
import time
import logging
from datetime import datetime

# Setup logging to file
logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def log_message(message):
    print(message)
    logging.info(message)

def run_tests():
    ai = Algorithm()

    def test_case(name, state, expected_action):
        log_message(f"\nRunning test case: {name}")
        start_time = time.time()
        try:
            # Get the action from the AI
            action, _ = ai.getAction(state)
            end_time = time.time()
            
            # Compare the action with the expected output
            log_message(f"Action returned: {action}")
            log_message(f"Time taken: {end_time - start_time:.2f} seconds")
            
            assert action == expected_action, f"Test case '{name}' failed. Expected {expected_action}, got {action}"
            log_message("Test case passed!")
        except Exception as e:
            # Log the error and continue
            log_message(f"An error occurred in test case '{name}': {str(e)}")
    
    # Define your test cases
    test_cases = [
    #     {
    #         "name": "Play lowest single card",
    #         "state": MatchState(
    #             myPlayerNum=0,
    #             players=[Player(0, 13) for _ in range(4)],
    #             myHand=['3D', '5H', '7S', '9C', 'JD', 'KH', '2S'],
    #             toBeat=None,
    #             matchHistory=[GameHistory(False, -1, [])],
    #             myData=""
    #         ),
    #         "expected_action": ['3D']
    #     },
        {
            "name": "Beat opponent's pair",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['4D', '4H', '7S', '9C', 'JD', 'KH', '2S'],
                toBeat=Trick(1, ['3D', '3H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['4D', '4H']
        },
        {
            "name": "Pass when can't beat",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['4D', '5H', '7S', '9C', 'JD', 'KH', 'AS'],
                toBeat=Trick(1, ['2D', '2H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": []
        },
        {
            "name": "Play a straight",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['3D', '4H', '5S', '6C', '7D', 'JD', 'KH', '2S'],
                toBeat=None,
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['3D', '4H', '5S', '6C', '7D']
        },
        {
            "name": "Flush beats straight",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['3D', '5D', '7D', '9D', 'JD', 'KH', '2S'],
                toBeat=Trick(1, ['4H', '5S', '6C', '7D', '8D']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['3D', '5D', '7D', '9D', 'JD']
        },
        {
            "name": "Play higher pair to beat opponent's pair",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['5D', '5S', '8C', '8S', 'JD', 'KH', '2S'],
                toBeat=Trick(1, ['3D', '3S']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['5D', '5S']
        },
        {
            "name": "Four-of-a-kind beats full house",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['6D', '6S', '6H', '6C', 'JD', 'KH', '2S'],
                toBeat=Trick(1, ['5D', '5S', '5H', '8C', '8S']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['6D', '6S', '6H', '6C', 'JD']
        },
        {
            "name": "Straight flush beats flush",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['3D', '4D', '5D', '6D', '7D', 'KH', '2S'],
                toBeat=Trick(1, ['2H', '5H', '7H', '8H', '9H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['3D', '4D', '5D', '6D', '7D']
        },
        {
            "name": "Full house beats triple",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['4D', '4S', '4H', '9C', '9D', 'KH', '2S'],
                toBeat=Trick(1, ['3D', '3S', '3H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['4D', '4S', '4H', '9C', '9D']
        },
        {
            "name": "Four-of-a-kind beats flush",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['2H', '2D', '2C', '2S', '3D', '9H', '8D'],
                toBeat=Trick(1, ['TH', '9H', '6H', '5H', '3H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['2H', '2D', '2C', '2S', '3D']
        },
        {
            "name": "Ace-high straight beats King-high straight",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['AS', 'KC', 'QH', 'JD', 'TH', '3H', '2S'],
                toBeat=Trick(1, ['KD', 'QH', 'JC', 'TD', '9H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['AS', 'KC', 'QH', 'JD', 'TH']
        },
        {
            "name": "Higher full house beats lower full house",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['KH', 'KD', 'KC', '9S', '9D'],
                toBeat=Trick(1, ['QH', 'QD', 'QC', 'JS', 'JD']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['KH', 'KD', 'KC', '9S', '9D']
        },
        {
            "name": "Higher four-of-a-kind beats lower four-of-a-kind",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['8H', '8D', '8C', '8S', '3D'],
                toBeat=Trick(1, ['6C', '6D', '6H', '6S', '4D']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['8H', '8D', '8C', '8S', '3D']
        },
        {
            "name": "Higher straight flush beats lower straight flush",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['9S', '8S', '7S', '6S', '5S', 'KH', '2D'],
                toBeat=Trick(1, ['8H', '7H', '6H', '5H', '4H']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['9S', '8S', '7S', '6S', '5S']
        },
        {
            "name": "Flush beats straight",
            "state": MatchState(
                myPlayerNum=0,
                players=[Player(0, 13) for _ in range(4)],
                myHand=['AH', 'QH', 'JH', '9H', '3H'],
                toBeat=Trick(1, ['AD', 'JD', '8D', '7D', '6D']),
                matchHistory=[GameHistory(False, -1, [])],
                myData=""
            ),
            "expected_action": ['AH', 'QH', 'JH', '9H', '3H']
        }
    ]

    # Run each test case and log the results
    for case in test_cases:
        try:
            test_case(case["name"], case["state"], case["expected_action"])
        except Exception as e:
            log_message(f"Test case '{case['name']}' encountered an error: {e}")

    log_message("\nAll test cases completed!")

if __name__ == "__main__":
    run_tests()
