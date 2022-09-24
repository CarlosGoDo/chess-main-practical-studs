#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy

import chess
import numpy as np
import sys
import queue
from typing import List

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game

    Methods:
    --------
    startGame(pos:stup) -> None
        Promotes a pawn that has reached the other side to another, or the same, piece

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.pathToTarget = []
        self.currentStateW = self.chess.boardSim.currentStateW;
        self.depthMax = 8;
        self.checkMate = False

    def hacer_movimiento(self, standard_current_state, standard_next_state):
        start = [e for e in standard_current_state if e not in standard_next_state]
        to = [e for e in standard_next_state if e not in standard_current_state]
        start, to = start[0][0:2], to[0][0:2]
        aichess.chess.moveSim(start, to)

    def getCurrentState(self):

        return self.myCurrentStateW

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False
    def nei_corrector(self, nei,listVisitedStates):
        """
        En esta función observaremos si el nei o estado futuro del tablero al que vamos tiene algún tipo de error
        como poner dos fichas en la misma posición o transformar una torre en rey.
        """

        if nei not in listVisitedStates:
            if nei[0][2] != nei[1][2]: #En este caso tenemos un estado del tablero futuro donde las 2 fichas són iguales
                if (nei[0][0] != nei[1][0]) and (nei[0][1] != nei[1][1]):#Aquí comprobamos que las fichas no se superpongan en la misma posición del tablero.
                    return True
        return False

    def isCheckMate(self, mystate):

        # Your Code

        listCheckMate_king= [[0,3,6], [1,3,6], [1,4,6], [1,5,6], [0,5,6]] #casos donde es checkMate con el rey

        #listCheckMate_Tower =[0,4] #casos donde es checkMate con la torre

        for pieces in mystate:
            if pieces[2] == 6: #checkMate rey
                if pieces in listCheckMate_king:
                    print("CheckMate del rey: ",mystate )
                    return mystate

            elif pieces[2] == 2: #checkMate Torre
                #hacemos un bucle observando si tenemos alguna pieza enmedio de la torre y el rey negro.

                if pieces[0] == 0 or pieces[1] == 4:#tenemos a la torre y rey en la misma fil o col
                    for oth_pieces in mystate:
                        if oth_pieces != pieces:#no queremos comparar a la misma pieza.
                            if pieces[0] == 0 and oth_pieces[0] == 0: #torre el rey negro y otra pieza en la misma fil
                                if pieces[1] > 4 and (oth_pieces[1]>4 and oth_pieces[1]<pieces[1]):
                                    #En este caso tenemos una pieza en medio de la torre y el rey
                                    return False
                                elif pieces[1] < 4 and (oth_pieces[1]<4 and oth_pieces[1]>pieces[1]):
                                    # En este caso tenemos una pieza en medio de la torre y el rey
                                    return False
                                else:
                                    print("CheckMate de la torre: ", mystate)
                                    return mystate
                            elif pieces[1] == 4 and oth_pieces[1] == 4:  #torre el rey negro y otra pieza la misma col

                                if pieces[0] > oth_pieces[0]:
                                    #tenemos una ficha entre la torre y el rey negro
                                    return False
                                else:
                                    print("CheckMate de la torre: ", mystate)
                                    return mystate

            #otros elif con otras fichas
        return False
        

    def DepthFirstSearch(self, currentState, depth):
        # Your Code here
        check = self.isCheckMate(currentState)

        if check:
            return check

        if currentState not in self.listVisitedStates:
            self.listVisitedStates.append(currentState)
            tupla = (copy.deepcopy(self.chess), currentState)

            for nei in self.getListNextStatesW(currentState):
                self.chess = tupla[0]
                self.currentStateW = tupla[1]
                #print("El estado actual es: ", currentState)
                #print("El self es         : ", self.currentStateW)
                if self.nei_corrector(nei,self.listVisitedStates):#comprobamos que nei sea un estado deseado.
                    #print("nos movemos a: ",nei)
                    #print("nos movemos a: ", nei[0][1])
                    self.hacer_movimiento(self.currentStateW, nei)
                    pth = self.DepthFirstSearch(nei,depth+1)
                    if pth:
                        return [currentState] + pth

        return False
 
    def BreadthFirstSearch(self, currentState):

        # Your Code here


        return 0


    def BestFirstSearch(self, currentState):
            
        # Your Code here
                
        return 0
                
    def AStarSearch(self, currentState):
            
        # Your Code here

        return 0



def translate(s):
    """
    Translates traditional board coordinates of chess into list indices
    """

    try:
        row = int(s[0])
        col = s[1]
        if row < 1 or row > 8:
            print(s[0] + "is not in the range from 1 - 8")
            return None
        if col < 'a' or col > 'h':
            print(s[1] + "is not in the range from a - h")
            return None
        dict = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        return (8 - row, dict[col])
    except:
        print(s + "is not in the format '[number][letter]'")
        return None


if __name__ == "__main__":
    #   if len(sys.argv) < 2:
    #       sys.exit(usage())

    # intiialize board
    TA = np.zeros((8, 8))
    # white pieces
    # TA[0][0] = 2
    # TA[2][4] = 6
    # # black pieces
    # TA[0][4] = 12

    TA[7][0] = 2
    TA[7][4] = 6
    TA[0][4] = 12

    # initialise board
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    currentState = aichess.chess.board.currentStateW.copy()

    print("printing board")
    aichess.chess.boardSim.print_board()

    # get list of next states for current state
    print("current State", currentState)

    # it uses board to get them... careful 
    aichess.getListNextStatesW(currentState)
    #   aichess.getListNextStatesW([[7,4,2],[7,4,6]])
    print("list next states ", aichess.listNextStates)

    # starting from current state find the end state (check mate) - recursive function
    # aichess.chess.boardSim.listVisitedStates = []
    # find the shortest path, initial depth 0
    depth = 0
    #aichess.BreadthFirstSearch(currentState)
    lista = aichess.DepthFirstSearch(currentState, depth)

    if lista:
        print("encontrado")
        print("Conjunto de movimientos: ", lista)
    else:
        print("no hay solucion")


    # MovesToMake = ['1e','2e','2e','3e','3e','4d','4d','3c']

    # for k in range(int(len(MovesToMake)/2)):

    #     print("k: ",k)

    #     print("start: ",MovesToMake[2*k])
    #     print("to: ",MovesToMake[2*k+1])

    #     start = translate(MovesToMake[2*k])
    #     to = translate(MovesToMake[2*k+1])

    #     print("start: ",start)
    #     print("to: ",to)

    #     aichess.chess.moveSim(start, to)

    # aichess.chess.boardSim.print_board()
    print("#Move sequence...  ", aichess.pathToTarget)
    #print("#Visited sequence...  ", aichess.listVisitedStates)
    print("#Current State...  ", aichess.chess.board.currentStateW)
