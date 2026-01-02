'''InferenceEngine Module
Maintains the world (state) and implements a histogram filter based on a color sensor'''

from tkinter import *
import random
import time
import datetime

class inferenceEngine():
    '''TBA :D'''

    def __init__(self):
        self.prob = {} # dict to stroe probabilites
        '''TTTBoard()
        creates a TTTBoard in starting position'''
        self.board = {} # dict to store player positions
        self.seqMoves = '' # String to store sequence of moves for logging
        for row in range(3):
            for column in range(3):
                coords = (row,column)
                self.board[coords] = None
        self.agent = [self.agent_0,self.agent_1,self.agent_2,self.agent_3,self.agent_4] 
        self.currentPlayer = 0
        self.endgame = None # replace with string when game ends

    def get_piece(self,coords):
        '''TTTBoard.get_piece(coords) -> int
        returns the piece at coords'''
        return self.board[coords]

    def get_endgame(self):
        '''TTTBoard.get_endgame() -> None or str
        returns endgame state'''
        return self.endgame

    def get_seqMoves(self):
        '''TTTBoard.get_seqMoves() -> str
        returns seqMoves string'''
        return self.seqMoves

    def get_player(self):
        '''TTTBoard.get_player() -> int
        returns the current player'''
        return self.currentTicTacToeEngine.py (Page 1 of 5)Player

    def next_player(self):
        '''TTTBoard.next_player()
        advances to next player'''
        self.currentPlayer = 1 - self.currentPlayer

    def get_legal_moves(self):
        '''TTTBoard.get_legal_moves() -> list
        returns a list of the current player's legal moves'''
        moves = []
        if self.endgame is not None:
            return moves
        for row in range(3):
            for column in range(3):
                coords = (row,column)
                # if space is empty
                if self.board[coords] is None:
                    moves.append(coords) # add to list
        return moves

    def try_move(self,coords):
        '''TTTBoard.try_move(coords)
        places the current player's piece in the given square
        if the square is empty
        Also goes to other player's turn'''
        if self.board[coords] is not None:
            return False # move not valid
        self.board[coords] = self.currentPlayer
        self.seqMoves += str(coords[0]) + str(coords[1])
        self.check_endgame() # check if game over
        if self.endgame is None:
            self.next_player() # next player's turn
        return True

    def evaluate_coordinate(self,coords):
        '''TTTBoard.evaluate_coordinate(coords) -> int
        returns the value of the (row,column) tuple coords'''
        coordValues= ((20,10,20),
                      (10,30,10),
                      (20,10,20))
        row,column = coords # unpack coords
        return coordValues[row][column]

    def agent_0(self):
        '''TTTBoard.agent_0() -> tuple
        returns a random move out of a list legal moves'''
        legalMoves = self.get_legal_moves()
        if len(legalMoves) == 0: # if no moves
            return None
        return random.choice(legalMoves)

    def coords_two_in_line(self,player):
        '''TTTBoard.coords_two_in_line(player) -> tuple
        Returns coordinates of an empty square if there are 2 other squares
        in line for the player'''
        units = self.get_units()
        for i in range(8):
            unit = units[i]
            if self.two_in_line(unit,player):
                if i < 6 and i % 2 == 0:
                    row = i//2
                    col = unit.index(None)
                    return(row,col)
                elif i < 6 and i % 2 == 1:
                    col = (i-1)//2
                    row = unit.index(None)
                    return(row,col)
                else:
                    if i == 6:
                        row = unit.index(None)
                        return(row,row)
                    else:
                        row = unit.index(None)
                        return(row,2-row)
        return None
    
    def agent_1(self):
        '''TTTBoard.agent_1() -> tuple
        returns a move which results in a win or blocks a win
        if no such move exists
        returns a random move out of a list legal moves'''
        legalMoves = self.get_legal_moves()
        units = self.get_units()
        player = self.currentPlayer
        coords = self.coords_two_in_line(player)
        if coords is not None:
            return coords
        coords = self.coords_two_in_line(1-player)
        if coords is not None:
            return coords
        return random.choice(legalMoves)

    def agent_2(self,player):
        '''TTTBoard.agent_2(player) -> int
        returns the value of the state, E(n)=X(n)-O(n) 
        100 for 3 in line, 10 for 2 in line, 1 for 1 in line'''
        units = self.get_units()
        value = 0
        for unit in units:
            if self.three_in_line(unit,player):
                value += 100
            elif self.two_in_line(unit,player):
                value += 10
            elif self.one_in_line(unit,player):
                value += 1
            elif self.three_in_line(unit,1-player):
                value -= 100
            elif self.two_in_line(unit,1-player):
                value -= 10
            elif self.one_in_line(unit,1-player):
                value -= 1
            else:
                value += 0    
        return value

    def agent_3(self,player):
        '''TTTBoard.agent_3(player) -> int
        returns the value of the state, E(n)=X(n)-O(n)
        1 for every win path'''
        playerWinPaths = 0
        oppWinPaths = 0
        units = self.get_units()
        for unit in units:
            if self.three_in_line(unit,player):
                playerWinPaths += 1
            elif self.two_in_line(unit,player):
                playerWinPaths += 1
            elif self.one_in_line(unit,player):
                playerWinPaths += 1
            elif self.none_in_line(unit,player):
                playerWinPaths += 1
        for unit in units:
            if self.three_in_line(unit,1-player):
                oppWinPaths += 1
            elif self.two_in_line(unit,1-player):
                oppWinPaths += 1
            elif self.one_in_line(unit,1-player):
                oppWinPaths += 1
            elif self.none_in_line(unit,1-player):
                oppWinPaths += 1
        return playerWinPaths-oppWinPaths

    def agent_4(self,player):
        '''TTTBoard.agent_4(player) -> int
        returns the utility of the state'''
        units = self.get_units()
        value = 0
        for unit in units:
            if self.three_in_line(unit,player):
                value += 1
            elif self.three_in_line(unit,1-player):
                value -= 1
        return value
        

    def computer_turn(self,hId,minimax,depth=4):
        '''TTTBoard.computer_turn()
        pick a good move'''
        if minimax:
            value,move = self.minimax(hId,depth)
        else:
            move = self.agent[hId]()
        self.try_move(move)

    def branch(self,move):
        '''TTTBoard.branch(move) -> TTTBoard
        returns a copy of self
        has the copy make the given move'''
        newBoard = TTTBoard()
        # copy self
        newBoard.board = dict(self.board)
        newBoard.currentPlayer = self.get_player()
        newBoard.endgame = self.get_endgame()
        newBoard.seqMoves = self.get_seqMoves()
        newBoard.try_move(move) # make move
        return newBoard

    def minimax(self,hId,depth=0):
        '''TTTBoard.minimax(depth) -> value,coords
        performs the minimax algorithm with depth
        returns the value of the move and best move'''
        legalMoves = self.get_legal_moves()
        if len(legalMoves) == 0: # if no moves
            return 0,None
        bestMoveValue = -999 # initialize best move tracking variable
        for move in legalMoves:
            player = self.currentPlayer
            newBoard = self.branch(move)
            moveValue = newBoard.agent[hId](player)
            if depth > 1: # if want to look more than 1 move ahead
                # create new board with this move
                oppValue,oppMove = newBoard.minimax(hId,depth-1)
                moveValue -= oppValue # subtract opponent's move value
            if moveValue > bestMoveValue: # if better move
                bestMoves = [move] # start new list
                bestMoveValue = moveValue # update tracking variables
            elif moveValue == bestMoveValue: # just as good
                bestMoves.append(move) # add to list
        # pick a "best" move at random
        return bestMoveValue, random.choice(bestMoves)

    def get_units(self):
        '''TTTBoard.get_units() -> list
        returns a list of rows,columns, and diagonals with player values'''
        # set up units
        units = []
        d1List = [] # list for main diagonal
        d2List = [] # list for counter diagonal
        for row in range(3):
            rList = [] # list for rows
            cList = [] # list for cols
            for col in range(3):
                rList.append(self.board[(row,col)])
                cList.append(self.board[(col,row)])
                if row == col:
                    d1List.append(self.board[(row,col)])
                if row+col==2:
                    d2List.append(self.board[(row,col)])
            units.append(rList)
            units.append(cList)
        units.append(d1List)
        units.append(d2List)
        return units # units has player values

    def three_in_line(self,unit,player):
        '''TTTBoard.three_in_line(unit) -> bool
        returns True if player got three in unit
        otherwise False'''
        trueList = [[player,player,player]]
        if unit in trueList:
            return True
        return False

    def two_in_line(self,unit,player):
        '''TTTBoard.two_in_line(unit) -> bool
        returns True if player got two in unit
        otherwise False'''
        trueList = [[player,player,None], [None, player, player], [player, None, player]]
        if unit in trueList:
            return True
        return False

    def one_in_line(self,unit,player):
        '''TTTBoard.one_in_line(unit) -> bool
        returns True if player got one in unit
        otherwise False'''
        trueList = [[player,None,None],[None,player,None],[None,None,player]]
        if unit in trueList:
            return True
        return False

    def none_in_line(self,unit,player):
        '''TTTBoard.none_in_line(unit) -> bool
        returns True if unit empty
        otherwise False'''
        trueList = [[None,None,None]]
        if unit in trueList:
            return True
        return False

    def check_endgame(self):
        '''TTTBoard.check_endgame()
        checks if game is over
        updates endgameMessage if over'''
        units = self.get_units()
        for unit in units:
            if self.three_in_line(unit,self.currentPlayer):
                self.endgame = self.currentPlayer
        if self.endgame == None and len(self.get_legal_moves()) == 0:
            self.endgame = 'draw'
