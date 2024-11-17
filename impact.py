from typing import List

class Impact:
    def __init__(self) -> None:
        pass

    @staticmethod
    def card_impact(mana_value: int, turn: int = 0) -> float:
        if mana_value == 0:
            return 0
        if mana_value == 1:
            return mana_value + 1/0.95173
        elif mana_value == 2:
            return mana_value + 1/0.8241
        elif mana_value == 3:
            return mana_value + 1/0.63784
        elif mana_value == 4:
            return mana_value + 1/0.44051
        elif mana_value == 5:
            return mana_value + 1/0.27278
        elif mana_value == 6:
            return mana_value + 1/0.15229
        elif mana_value == 7:
            return mana_value + 1/0.07698
        return mana_value+1.5
    
    @staticmethod
    def turn_impact(cards : List[int], turn: int = 0) ->  float:
        """Given a turn computes the impact in a game

        Args:
            turn (List[int]): cards played in a turn

        Returns:
            float: impact
        """
        return sum([Impact.card_impact(card, turn) for card in cards])

    @staticmethod
    def sequence_impact(sequence: List[List[int]]) -> float:
        """Given a sequence computes the expected impact in a game

        Args:
            sequence (List[List[int]]): for each turn what cards have been played

        Returns:
            float: expected impact
        """
        return sum([Impact.turn_impact(cards, turn) for turn, cards in enumerate(sequence)])
