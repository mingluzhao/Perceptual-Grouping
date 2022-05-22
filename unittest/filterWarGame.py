import pandas as pd
import numpy as np
import pygame
import random

from pygame.locals import *
from sys import exit


#固定参数
radius = 25
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
GOLD = (255, 215, 0)
GREEN = (0, 255, 0)
colorList = [GOLD, BLUE, RED]


class PrintPreface:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self):

        fontPreface = pygame.font.SysFont('microsoft Yahei', 30)
        preface = fontPreface.render('Game Introduction', True, BLACK)
        self.screen.blit(preface, (self.length / 2 - 80, self.width / 5))
        preface2 = fontPreface.render('This area to put some guidance for subjects', True, BLACK)
        self.screen.blit(preface2, (self.length / 2 - 170, self.width / 5 + 30))
        # preface3 = fontPreface.render('This area is reserved for subjects to play the game', True, BLACK)
        # self.screen.blit(preface3, (self.length / 2 - 200, 2 * self.width / 3))
        return self.screen


class DrawCubes:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, mapSize, isEven, cubeWidth, isBoundary):
        pygame.draw.line(self.screen, BLACK, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2, self.width / 2 - cubeWidth),
                         (self.length / 2 + (mapSize - isEven) * cubeWidth + cubeWidth / 2, self.width / 2 - cubeWidth))
        pygame.draw.line(self.screen, BLACK, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2, self.width / 2),
                         (self.length / 2 + (mapSize - isEven) * cubeWidth + cubeWidth / 2, self.width / 2))
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        # 以下是用直线画法，更加美观，暂时保留在这里
        # for i in range((mapSize+1)*2-isEven):
        #     pygame.draw.line(screen, BLACK, (length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth*i, width / 2 - cubeWidth),
        #                      (length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth*i, width / 2))
        for i in range((mapSize + 1) * 2 - isEven - 1):
            pygame.draw.rect(self.screen, BLACK, (
            self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * i, self.width / 2 - cubeWidth, cubeWidth,
            cubeWidth), 1)

        #
        # for i in range((mapSize + 1) * 2 - isEven - 1):
        #     positionAnnotation = fontAnnotations.render(str(i - mapSize + isEven * (i >= mapSize)), False, BLACK)
        #     self.screen.blit(positionAnnotation,
        #                 (self.length / 2 - mapSize * cubeWidth + cubeWidth * i, self.width / 2 - 1.5 * cubeWidth))


        if isBoundary:
            for i in range(10):
                pygame.draw.line(self.screen, BLACK, (self.length / 2, self.width / 2 - cubeWidth + i/10*cubeWidth), (self.length / 2, self.width / 2 - cubeWidth + (i+1)/10*cubeWidth - cubeWidth/20), 2)
        return self.screen


class DrawBases:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, mapSize, cubeWidth, isSymmetrical, isEven, baseLocationRandom, soldiersA, soldiersB):
        pygame.draw.circle(self.screen, BLUE, (int(self.length / 2 - mapSize * cubeWidth), int(self.width / 2 - cubeWidth / 2)),
                           radius)
        fontBase = pygame.font.SysFont('microsoft Yahei', 30)
        baseA = fontBase.render('A', True, WHITE)
        self.screen.blit(baseA, (int(self.length / 2 - mapSize * cubeWidth) - 8, int(self.width / 2 - cubeWidth / 2) - 8))
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        initialSoldiersA = fontAnnotations.render('initial soldiers A: ' + str(int(soldiersA)), False, BLUE)
        self.screen.blit(initialSoldiersA, (self.length / 2 - mapSize * cubeWidth - 1.5 * cubeWidth, self.width / 2))
        initialSoldiersB = fontAnnotations.render('initial soldiers B: ' + str(int(soldiersB)), False, RED)
        self.screen.blit(initialSoldiersB, (self.length / 2 + (mapSize - isEven) * cubeWidth - 1.5 * cubeWidth, self.width / 2))

        if isSymmetrical:
            pygame.draw.circle(self.screen, RED, (
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - isEven) * cubeWidth),
                int(self.width / 2 - cubeWidth / 2)),
                               radius)
            baseB = fontBase.render('B', True, WHITE)
            self.screen.blit(baseB, (int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - isEven) * cubeWidth) - 8,
                                int(self.width / 2 - cubeWidth / 2) - 8))
        else:
            pygame.draw.circle(self.screen, RED, (
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - baseLocationRandom - isEven) * cubeWidth),
                int(self.width / 2 - cubeWidth / 2)), radius)
            baseB = fontBase.render('B', True, WHITE)
            self.screen.blit(baseB, (
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - baseLocationRandom - isEven) * cubeWidth) - 8,
                int(self.width / 2 - cubeWidth / 2) - 8))
        return self.screen


class DrawWarField:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, warField, mapSize, cubeWidth, isEven, typingPositionA, typingPositionB):
        for i in range(mapSize * 2 - isEven - 1):
            pygame.draw.rect(self.screen, colorList[warField[i]], (
            self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i+1)+2, self.width / 2 - cubeWidth+2, cubeWidth-4,
            cubeWidth-4), 0)
        for i in range(mapSize * 2 - isEven - 1):
            if typingPositionA == i:
                pygame.draw.circle(self.screen, GREEN, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)),
                                                   int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)), radius)
            else:
                pygame.draw.circle(self.screen, BLUE,
                                   (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)),int(self.width / 2 - cubeWidth / 2 + 2 * cubeWidth)), radius)
            if typingPositionB == i:
                pygame.draw.circle(self.screen, GREEN, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)),
                                    int(self.width / 2 - cubeWidth / 2 + 3*cubeWidth)), radius)
            else:
                pygame.draw.circle(self.screen, RED, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)),
                                                       int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)), radius)


class DrawPolicyA:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, policyA, mapSize, cubeWidth ,isEven):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        for i in range((mapSize * 2 - isEven - 1)):
            policyFont = fontAnnotations.render(str(policyA[i]), True, WHITE)
            self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)-8),
                                          int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)-8))


class DrawPolicyB:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, policyB, mapSize, cubeWidth ,isEven):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        for i in range((mapSize * 2 - isEven - 1)):
            policyFont = fontAnnotations.render(str(policyB[i]), True, WHITE)
            self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)-8),
                                          int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)-8))


class DrawRemainingSoldiers:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, remainingSoldiersA, remainingSoldiersB, mapSize, cubeWidth, isEven):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 30)
        for i in range((mapSize * 2 - isEven - 1)):
            fontRemainingSoldiers = fontAnnotations.render(str(max(remainingSoldiersA[i], remainingSoldiersB[i])), True, GREEN)
            self.screen.blit(fontRemainingSoldiers, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i+1)+2,
                                                     self.width / 2 - cubeWidth+2))
        return self.screen


class DrawLastRun:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, formerPolicyA, formerPolicyB, mapSize, cubeWidth ,isEven):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)

        for i in range((len(formerPolicyB))):
            policyFont = fontAnnotations.render('Last Run:', True, BLACK)
            self.screen.blit(policyFont,(int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (len(formerPolicyB) / 2) - 8),
                              int(self.width / 2 - cubeWidth / 2 + 3.8 * cubeWidth) - 8))
            policyFont = fontAnnotations.render(str(formerPolicyA[i]), True, BLUE)
            self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)-8),
                                          int(self.width / 2 - cubeWidth / 2 + 4.3 * cubeWidth)-8))
            policyFont = fontAnnotations.render(str(formerPolicyB[i]), True, RED)
            self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i + 1)-8),
                                          int(self.width / 2 - cubeWidth / 2 + 4.8 * cubeWidth)-8))


class Warnings:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, warningA, warningB):
        fontPreface = pygame.font.SysFont('microsoft Yahei', 30)
        if warningA == 1:
            warningForA = fontPreface.render('Policy of player A exceeds the limit', True, BLACK)
            self.screen.blit(warningForA, (self.length / 2 - 150, 2.5 * self.width / 3))
        if warningB == 1:
            warningForB = fontPreface.render('Policy of player B exceeds the limit', True, BLACK)
            self.screen.blit(warningForB, (self.length / 2 - 150, 2.5 * self.width / 3 + 50))

        return self.screen




def generatePositionListA(length, width, mapSize, isEven, cubeWidth):
    positionList = []
    for i in range(mapSize * 2 - isEven - 1):
        positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)), int(width / 2 - cubeWidth / 2 + 2*cubeWidth)))
    return  positionList

def generatePositionListB(length, width, mapSize, isEven, cubeWidth):
    positionList = []
    for i in range(mapSize * 2 - isEven - 1):
        positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)), int(width / 2 - cubeWidth / 2 + 3*cubeWidth)))
    return positionList

def detectPosition(pos, positionList):
    limit = radius
    for i in range(len(positionList)):
        if abs(positionList[i][0]-pos[0]) <= limit and abs(positionList[i][1]-pos[1]) <= limit:
            return i
    return -1

def transformKeyToInt(key):
    return key - 48

def transformTempListToInt(tempList):
    number = '0'
    for i in range(len(tempList)):
        number = number + str(tempList[i])
    return int(number)

def judgeResult(policyA, policyB, remainingSoldiersA, remainingSoldiersB):
    warField = [0 for i in range(len(policyA))]
    for i in range(len(warField)):
        if policyA[i] + remainingSoldiersA[i] >= policyB[i] + remainingSoldiersB[i]:
            warField[i] = 1
        elif policyA[i] + remainingSoldiersA[i] < policyB[i] + remainingSoldiersB[i]:
            warField[i] = 2
    return warField

def checkPolicy(policyA, policyB, soldiersA, soldiersB):
    warningA = 0
    warningB = 0
    if sum(policyA) > soldiersA:
        warningA = 1
    if sum(policyA) <= soldiersA:
        warningA = 0
    if sum(policyB) > soldiersB:
        warningB = 1
    if sum(policyB) <= soldiersB:
        warningB = 0

    return [warningA, warningB]

def calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB):
    remainingSoldiersNewA = [0 for i in range(len(policyA))]
    remainingSoldiersNewB = [0 for i in range(len(policyA))]
    for i in range(len(policyA)):
        remainingSoldiersNewA[i] = int((max(0, policyA[i] + remainingSoldiersA[i] - policyB[i] - remainingSoldiersB[i])) * (1 - (i + 1) / (len(policyA) + 1)))
        remainingSoldiersNewB[i] = int((max(0, policyB[i] + remainingSoldiersB[i] - policyA[i] - remainingSoldiersA[i])) * ((i + 1) / (len(policyA) + 1)))
    return [remainingSoldiersNewA, remainingSoldiersNewB]

def  calculateSoldiers(warField):
    soldierFromBase = 20
    soldierFromWarField = 10
    soldiersA = 0
    soldiersB = 0
    for i in range(len(warField)):
        soldiersA = soldiersA + (warField[i] == 1) * (1 - (i + 1) / (len(warField) + 1)) * soldierFromWarField
        soldiersB = soldiersB + (warField[i] == 2) * ((i + 1) / (len(warField) + 1)) * soldierFromWarField
    soldiersA += soldierFromBase
    soldiersB += soldierFromBase
    return [soldiersA, soldiersB]
