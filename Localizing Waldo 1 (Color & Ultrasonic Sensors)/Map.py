
REDACTED
































'''ParticleFilter Module
Implements a Particle Filter'''

from tkinter import *
import random
import time
import datetime


class Particle():
    '''Represents a Particle'''

    def __init__(self, pID, mapCoords, theta, phi, offsetTT, offsetUS, offsetCS):
        '''Particle()
        initializes a particle
        pID: Particle ID
        mapCoords: (xMap,yMap) of the robot in map coordinate system
        theta: robot heading in map coordinate system (wrt x axis in map coords) 
        phi: heading of US/CS wrt to robot coordinate system
        offsetTT: distance of TT from Robot origin
        offsetUS: distance of US from TT rotation axis
        offsettCS: distance of CS from TT rotation axis
        '''
        self.pID = pID
        self.mapCoords = mapCoords
        self.theta = theta
        self.phi = phi
        self.offsetTT = offsetTT
        self.offsetUS = offsetUS
        self.offsetCS = offsetCS
        self.weight = 1

    def get_pID(self):
        return self.pID

    def get_coords(self):
        return (self.x,self.y)

    def get_theta(self):
        return self.theta

    def get_phi(self):
        return self.phi

    def get_CS_coords(self):
        return (


class ParticleFilter():
    '''Implements a Particle Filter'''

    def __init__(self):
        '''ParticleFilter()
        initializes a particle filter by generating particles uniformly on the map'''
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
