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
from queue import PriorityQueue
from typing import List
from random import randrange

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
        self.listVisitedStates = {}
        self.listaEstadosVisitados = []
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

    def invert_state(self,currentState):
        """
        Dado un estado donde el rey esta en la primera posción como por ejemplo [[7, 4, 6], [7, 0, 2]], la funcion
        invertira este estado y lo devolvera en forma de string [[7, 0, 2], [7, 4, 6]]
        """
        nei = copy.copy(currentState)
        if nei[0][2] == 6:
            aux = nei[0]
            nei[0] = nei[1]
            nei[1] = aux
        return str(nei)
    def nei_corrector(self, nei):
        """
        En esta función observaremos si el nei o estado futuro del tablero al que vamos tiene algún tipo de error
        como poner dos fichas en la misma posición o transformar una torre en rey.
        """


        if nei[0][2] != nei[1][2]: #En este caso tenemos un estado del tablero futuro donde las 2 fichas són iguales
            if (nei[0][0] != nei[1][0]) and (nei[0][1] != nei[1][1]):#Aquí comprobamos que las fichas no se superpongan en la misma posición del tablero.
                return True
        return False

    def isCheckMate(self, mystate):

        # Your Code
        listCheckMateStates = [[[0, 0, 2], [2, 4, 6]], [[0, 1, 2], [2, 4, 6]], [[0, 2, 2], [2, 4, 6]],
                               [[0, 6, 2], [2, 4, 6]], [[0, 7, 2], [2, 4, 6]]]

        if mystate in listCheckMateStates:
            print("is check Mate!")
            return True

        return False

    def DepthFirstSearch(self, currentState, depth):
        # Your Code here
        # check = self.isCheckMate(currentState)
        print("estamos en la profundidad", depth, "la maxima profunidad es", self.depthMax)
        if self.isCheckMate(currentState):
            return currentState

        if depth < 8:
            strState = str(currentState)
            if strState not in self.listVisitedStates or self.listVisitedStates[strState] > depth:
                self.listVisitedStates[strState] = depth
                for nei in self.getListNextStatesW(currentState):

                    if self.nei_corrector(nei):  # comprobamos que nei sea un estado deseado.
                        # print("seguimos en un vecino de",currentState)
                        # print("estamos en el estado: ",currentState," y vamos al estado: ",nei)
                        self.hacer_movimiento(currentState, nei)
                        pth = self.DepthFirstSearch(nei, depth + 1)
                        if pth:
                            return [currentState] + pth
                        else:
                            self.hacer_movimiento(nei, currentState)

        return False

    def BreadthFirstSearch(self, currentState):

        # Your Code here
        q = queue.Queue()
        q.put({'estado_act': currentState, 'chess': self.chess, 'pth': [currentState]})

        while not q.empty():
            current = q.get()
            self.listaEstadosVisitados.append(current['estado_act'])
            self.chess = current['chess']

            if current['estado_act'] == [[0, 0, 2], [2, 4, 6]] or current['estado_act'] == [[2, 4, 6], [0, 0, 2]]:
                return current['pth']

            for nei in self.getListNextStatesW(current['estado_act']):
                if nei not in self.listaEstadosVisitados and self.nei_corrector(nei):
                    self.hacer_movimiento(current['estado_act'], nei)
                    q.put({'estado_act': nei, 'chess': copy.deepcopy(self.chess),'pth': current['pth']+[nei]})
                    self.hacer_movimiento(nei, current['estado_act'])
                    self.listaEstadosVisitados.append(nei)

        return False

    def func_heuristic(self,estado1, estado2 ):
        nei1 = copy.copy(estado1)
        nei2 = copy.copy(estado2)
        if nei1[0][2]==6:
            aux = nei1[0]
            nei1[0] = nei1[1]
            nei1[1]=aux

        if nei2[0][2]==6:
            aux = nei2[0]
            nei2[0] = nei2[1]
            nei2[1]=aux

        dist1 = abs((nei2[0][0] - nei1[0][0])) + abs((nei2[0][1] - nei1[0][1]))
        dist2 =abs((nei2[1][0] - nei1[1][0])) + abs((nei2[1][1] - nei1[1][1]))
        return dist1 + dist2
    def BestFirstSearch(self, currentState):
        # Your Code here
        return 0


    def AStarSearch(self, currentState):
            
        # Your Code here
        objetivo = [[0, 0, 2], [2, 4, 6]]
        opened = PriorityQueue()
        closed = []
        path = []
        heu = self.func_heuristic(currentState,objetivo)
        opened.put([heu,0,{'estado_act': currentState, 'chess': self.chess, 'father': None}])

        identificador = 0
        while not opened.empty():
            actual_state = opened.get()
            self.chess = actual_state[2]['chess']
            if actual_state[2]['estado_act'] == [[0, 0, 2], [2, 4, 6]] or actual_state[2]['estado_act'] == [[2, 4, 6], [0, 0, 2]]:
                closed.append(actual_state[2]['estado_act'])
                path.append([actual_state[2]['estado_act'], actual_state[2]['father']])
                return path
            for nei in self.getListNextStatesW(actual_state[2]['estado_act']):

                if self.nei_corrector(nei) and nei not in closed:
                    self.hacer_movimiento(actual_state[2]['estado_act'], nei)
                    heu = self.func_heuristic(nei,objetivo) + 1

                    opened.put([heu,identificador,{'estado_act': nei, 'chess': copy.deepcopy(self.chess), 'father': actual_state[2]['estado_act']}])
                identificador+=1
                self.hacer_movimiento( nei, actual_state[2]['estado_act'])
            closed.append(actual_state[2]['estado_act'])
            path.append([actual_state[2]['estado_act'], actual_state[2]['father']])
        return False
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
    #lista2 = aichess.BreadthFirstSearch(currentState)
    #lista2 = aichess.DepthFirstSearch(currentState, depth)

    """
            #################################### DFS ####################################
    """
    dfs = aichess.DepthFirstSearch(currentState, depth)
    if (dfs):
        print("encontrado")
        print(dfs)
    else:
        print("no hay solucion")

    """
            #################################### BFS ####################################
    """
    bfs = aichess.BreadthFirstSearch(currentState)
    if(bfs):
        print("encontrado")
        print(bfs)
    else:
        print("no hay solucion")

    """
          #################################### AStar ####################################
    """
    path = []
    astar = aichess.AStarSearch(currentState)
    aux = astar[len(astar) - 1]
    if astar:
        while aux[1] != None:

            for e in astar:
                if e[0] == aux[1]:
                    path.append(aux[0])
                    aux = e
                    pass
                else:
                    pass

        print("encontrado")
        path.append(astar[0][0])
        path.reverse()
        print("Conjunto de movimientos: ", path)
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
