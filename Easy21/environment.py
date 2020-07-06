import random
from collections import namedtuple


Card = namedtuple('card', ('value', 'color'))
State = namedtuple('state', ('first_dealer_card_value', 'player_sum'))


class Dealer:
    def __init__(self):
        self.sum = 0
    
    def take_action(self):
        if self.sum >= 17:
            return 'stick'
        
        else:
            return 'hit'
    
    def busted(self):
        return self.sum < 1 or self.sum > 21


class Player:
    def __init__(self):
        self.sum = 0
    
    def busted(self):
        return self.sum < 1 or self.sum > 21

        
class Easy21Env:    
    def __init__(self, print_game=True):
        self.print_game = print_game

        self.dealer = Dealer()
        self.player = Player()

        self.action_space_size = 2
    
    @staticmethod
    def draw_card():
        value = random.randint(1, 10)
        color = random.sample(['black', 'black', 'black'], k=1)[0]
        
        return Card(value, color)
    
    def reset(self):
        self.print_info('__starting new game__')

        self.dealer.sum = self.draw_card().value
        self.player.sum = self.draw_card().value

        self.first_dealer_card_value = self.dealer.sum
        
        initial_state = State(self.dealer.sum, self.player.sum)
        return initial_state
        
    def step(self, action):
        if action == 'stick':
            self.print_info('player sticked')
            self.print_info('__dealer turn__')

            dealer_action = self.dealer.take_action()
            
            while dealer_action == 'hit':
                new_card = self.draw_card()
                
                self.print_info('dealer hit: ' + str(new_card))
                
                sum_multiplier = 1 if new_card.color == 'black' else -1
                self.dealer.sum += new_card.value * sum_multiplier
                
                if self.dealer.busted():
                    self.print_info('dealer busted')
    
                    reward = 1
                    break

                dealer_action = self.dealer.take_action()
                
                self.print_info('dealer: ' + str(self.dealer.sum))
            
            if dealer_action == 'stick':
                self.print_info('dealer sticked')

                if self.dealer.sum > self.player.sum:
                    reward = -1
                
                if self.dealer.sum == self.player.sum:
                    reward = 0
                
                if self.dealer.sum < self.player.sum:
                    reward = 1
            
            next_state = 'terminal'
            return reward, next_state
        
        if action == 'hit':
            new_card = self.draw_card()

            self.print_info('player hit: ' + str(new_card))
            
            sum_multiplier = 1 if new_card.color == 'black' else -1
            self.player.sum += new_card.value * sum_multiplier

            self.print_info('player: ' + str(self.player.sum))
            
            if self.player.busted():
                self.print_info('player busted')
    
                reward = -1
                next_state = 'terminal'
            
            else:
                reward = 0
                next_state = State(self.first_dealer_card_value, self.player.sum)
            
            return reward, next_state

    def print_info(self, info):
        if self.print_game:
            print(info)
