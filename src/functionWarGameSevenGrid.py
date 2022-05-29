import pandas as pd
import numpy as np
# import pygame
import random
import math
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
DEEPGREEN = (0, 128, 0)
colorList = [WHITE, BLUE, RED]
colorListRemainingSoldiers = [BLACK, BLUE, RED]
backgroundList = [GOLD, BLUE, RED]
backgroundColor = (255, 250, 240)


def transformToTriangle(x, y, radius):
    return [(x, y-radius), (x+0.87*radius, y+0.5*radius), (x-0.87*radius, y+0.5*radius)]

def transformToSquare(x, y, radius):
    return [(x, y-radius), (x+radius, y), (x, y+radius), (x-radius, y)]


class PrintPreface:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, winReward):

        fontPreface = pygame.font.SysFont('Times New Roman', 30)
        # preface = fontPreface.render('WIN:' + str(int(winReward)), True, BLACK)
        # self.screen.blit(preface, (10, 10))
        # preface2 = fontPreface.render('LOSS:0', True, BLACK)
        # self.screen.blit(preface2, (10, 35))
        # preface3 = fontPreface.render('DRAW:Soldiers*0.1', True, BLACK)
        # self.screen.blit(preface3, (10, 60))
        preface3 = fontPreface.render('Reward: Points*0.3', True, BLACK)
        self.screen.blit(preface3, (10, 10))

        # showSoldierFromField = fontPreface.render('soldiers gained per field: ' + str(int(soldierFromWarField)),
        #                                               True, BLACK)
        # self.screen.blit(showSoldierFromField, (self.length / 2 - 150, self.width / 5 + 10))
        return self.screen


class DrawSoldierFromBase:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 30
        self.thickness = 1

    def __call__(self, soldierFromBaseA, soldierFromBaseB, cubeWidth, mapSize, isEven):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        showSoldierFromBaseA = fontAnnotations.render('+' + str(int(soldierFromBaseA)), True, BLACK)
        self.screen.blit(showSoldierFromBaseA,
                         (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2+10, self.width / 2 - 1.6 * cubeWidth - self.devWidth))
        pygame.draw.circle(self.screen, BLACK, (int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2+25),
                                                int(self.width / 2 - 1.6 * cubeWidth - self.devWidth+10)), 21, self.thickness)
        showSoldierFromBaseB = fontAnnotations.render('+' + str(int(soldierFromBaseB)), True, BLACK)
        pygame.draw.polygon(self.screen, BLACK, transformToSquare(int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * ((mapSize + 1) * 2 - isEven - 2)+30),
                                                int(self.width / 2 - 1.6 * cubeWidth - self.devWidth+10), 21),
                           self.thickness)
        self.screen.blit(showSoldierFromBaseB,
                         (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * ((mapSize + 1) * 2 - isEven - 2)+15, self.width / 2 - 1.6 * cubeWidth - self.devWidth))


class DrawCubes:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 30

    def __call__(self, mapSize, isEven, cubeWidth, isBoundary, dev):
        pygame.draw.line(self.screen, BLACK, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2, self.width / 2 - cubeWidth - self.devWidth),
                         (self.length / 2 + (mapSize - isEven) * cubeWidth + cubeWidth / 2, self.width / 2 - cubeWidth - self.devWidth))
        pygame.draw.line(self.screen, BLACK, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2, self.width / 2 - self.devWidth),
                         (self.length / 2 + (mapSize - isEven) * cubeWidth + cubeWidth / 2, self.width / 2 - self.devWidth))
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        # 以下是用直线画法，更加美观，暂时保留在这里
        # for i in range((mapSize+1)*2-isEven):
        #     pygame.draw.line(screen, BLACK, (length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth*i, width / 2 - cubeWidth),
        #                      (length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth*i, width / 2))
        for i in range((mapSize + 1) * 2 - isEven - 1):
            pygame.draw.rect(self.screen, BLACK, (
            self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * i, self.width / 2 - cubeWidth - self.devWidth, cubeWidth,
            cubeWidth), 1)
        return self.screen


class DrawBases:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 30
        self.thickness = 2

    def __call__(self, mapSize, cubeWidth, isSymmetrical, isEven, baseLocationRandom, soldiersA, soldiersB):
        pygame.draw.circle(self.screen, BLACK, (int(self.length / 2 - mapSize * cubeWidth), int(self.width / 2 - cubeWidth / 2) - self.devWidth),
                           radius, self.thickness)
        fontBase = pygame.font.SysFont('Times New Roman', 25)
        baseA = fontBase.render('A', True, BLACK)
        self.screen.blit(baseA, (int(self.length / 2 - mapSize * cubeWidth) - 9, int(self.width / 2 - cubeWidth / 2) - 14 - self.devWidth))
        fontAnnotations = pygame.font.SysFont('Times New Roman', 25)

        if isSymmetrical:
            pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - isEven) * cubeWidth),
                int(self.width / 2 - cubeWidth / 2) - self.devWidth, radius),
                               self.thickness)
            baseB = fontBase.render('B', True, BLACK)
            self.screen.blit(baseB, (int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - isEven) * cubeWidth) - 6,
                                int(self.width / 2 - cubeWidth / 2) - 14 - self.devWidth))
        else:
            pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - isEven) * cubeWidth),
                int(self.width / 2 - cubeWidth / 2) - self.devWidth, radius),
                                self.thickness)
            baseB = fontBase.render('B', True, BLACK)
            self.screen.blit(baseB, (
                int(self.length / 2 - mapSize * cubeWidth + (2 * mapSize - baseLocationRandom - isEven) * cubeWidth) - 8,
                int(self.width / 2 - cubeWidth / 2) - 8 - self.devWidth))
        return self.screen


def generateBackgroundColor(mapSize, isEven, isBoundary, dev, assumedBoundary, normalColor):
    temp = [2 for i in range(mapSize * 2 - isEven - 1)]
    if isBoundary:
        if isEven:
            for i in range(assumedBoundary+dev):
                temp[i] = 1

        else:
            for i in range(dev+assumedBoundary):
                temp[i] = 1
            temp[dev+assumedBoundary] = 0

        return temp
    else:
        temp = [normalColor for i in range(mapSize * 2 - isEven - 1)]
        return temp


class DrawWarField:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 30
        self.thickness = 1
        self.downward = 20

    def __call__(self, warField, mapSize, cubeWidth, isEven, typingPositionA, typingPositionB, isSymmetrical, baseLocationRandom, isTypingA, isTypingB, mapColor):
        if isSymmetrical:
            # backgroundColor = generateBackgroundColor(mapSize, isEven, isBoundary, dev, assumedBoundary, self.normalColor)
            for i in range(mapSize * 2 - isEven - 1):
                pygame.draw.rect(self.screen, backgroundList[mapColor[i]], (
                self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i+1)+2, self.width / 2 - cubeWidth+2 - self.devWidth, cubeWidth-4,
                cubeWidth-4), 0)
            for i in range(mapSize * 2 - isEven + 1):
                if typingPositionA == i:
                    pygame.draw.circle(self.screen, GREEN, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)),
                                                       int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)+self.downward), radius)
                else:
                    pygame.draw.circle(self.screen, BLACK,
                                       (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)),int(self.width / 2 - cubeWidth / 2 + 2 * cubeWidth)+self.downward), radius, self.thickness)
                if typingPositionB == i:
                    pygame.draw.polygon(self.screen, GREEN, transformToSquare(int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)),
                                        int(self.width / 2 - cubeWidth / 2 + 3*cubeWidth)+self.downward, radius), 0)
                else:
                    pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                        int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)),
                        int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)+self.downward, radius), self.thickness)
            if not isTypingA:
                # pygame.draw.polygon(self.screen, BLACK,
                #                    transformToTriangle(int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (mapSize * 2 - isEven)),
                #                     int(self.width / 2 - cubeWidth / 2 + 2 * cubeWidth), radius-6), self.thickness)
                fontAnnotations = pygame.font.SysFont('Times New Roman', 30)
                yesA = fontAnnotations.render('√', True, BLACK)
                self.screen.blit(yesA, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (mapSize * 2 - isEven+1))-20,
                                 int(self.width / 2 - cubeWidth / 2 + 2 * cubeWidth)-15+self.downward))
            if not isTypingB:

                fontAnnotations = pygame.font.SysFont('Times New Roman', 30)
                yesB = fontAnnotations.render('√', True, BLACK)
                self.screen.blit(yesB, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (mapSize * 2 - isEven+1))-20,
                                    int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)-15+self.downward))

        else:
            j = 0
            for i in range(mapSize * 2 - isEven - 1):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                pygame.draw.rect(self.screen, colorList[warField[i]], (
                self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (j+1)+2, self.width / 2 - cubeWidth+2, cubeWidth-4,
                cubeWidth-4), 0)
                j = j + 1
            j = 0
            for i in range(mapSize * 2 - isEven - 1):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                if typingPositionA == i:
                    pygame.draw.circle(self.screen, GREEN, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j+1)),
                                                       int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)), radius)
                else:
                    pygame.draw.circle(self.screen, BLACK,
                                       (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j + 1)),int(self.width / 2 - cubeWidth / 2 + 2 * cubeWidth)), radius, 2)
                if typingPositionB == i:
                    pygame.draw.circle(self.screen, GREEN, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j+1)),
                                        int(self.width / 2 - cubeWidth / 2 + 3*cubeWidth)), radius)
                else:
                    pygame.draw.circle(self.screen, BLACK, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j + 1)),
                                                           int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)), radius, 2)
                j = j + 1


class DrawPolicyA:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, policyA, mapSize, cubeWidth ,isEven, isSymmetrical, baseLocationRandom):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        if isSymmetrical:
            for i in range((mapSize * 2 - isEven + 1)):
                policyFont = fontAnnotations.render(str(policyA[i]), True, BLACK)
                self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)-6),
                                              int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)-10))
        else:
            j = 0
            for i in range((mapSize * 2 - isEven + 1)):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                policyFont = fontAnnotations.render(str(policyA[i]), True, BLACK)
                self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j)-6),
                                              int(self.width / 2 - cubeWidth / 2 + 2*cubeWidth)-10))
                j = j + 1


class DrawPolicyB:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, policyB, mapSize, cubeWidth ,isEven, isSymmetrical, baseLocationRandom):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        if isSymmetrical:
            for i in range((mapSize * 2 - isEven + 1)):
                policyFont = fontAnnotations.render(str(policyB[i]), True, BLACK)
                self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (i)-6),
                                              int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)-10))
        else:
            j = 0
            for i in range((mapSize * 2 - isEven + 1)):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                policyFont = fontAnnotations.render(str(policyB[i]), True, BLACK)
                self.screen.blit(policyFont, (int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (j)-6),
                                              int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth)-10))
                j = j + 1


class DrawRemainingSoldiers:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 15
        self.thickness = 1

    def __call__(self, remainingSoldiersA, remainingSoldiersB, mapSize, cubeWidth, isEven, isSymmetrical, baseLocationRandom, warField):
        # fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        # if isSymmetrical:
        #     for i in range((mapSize * 2 - isEven - 1)):
        #         fontRemainingSoldiersA = fontAnnotations.render(str(remainingSoldiersA[i]), True, GREEN)
        #         self.screen.blit(fontRemainingSoldiersA, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i+1)+2,
        #                                                  self.width / 2 - cubeWidth))
        #         fontRemainingSoldiersB = fontAnnotations.render(str(remainingSoldiersB[i]), True, GREEN)
        #         self.screen.blit(fontRemainingSoldiersB, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i+1) + cubeWidth - 13,
        #                                                  self.width / 2 - 20))
        # else:
        #     j = 0
        #     for i in range((mapSize * 2 - isEven - 1)):
        #         if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
        #             j = j + 1
        #         fontRemainingSoldiersA = fontAnnotations.render(str(remainingSoldiersA[i]), True, GREEN)
        #         self.screen.blit(fontRemainingSoldiersA, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (j+1)+2,
        #                                                  self.width / 2 - cubeWidth))
        #         fontRemainingSoldiersB = fontAnnotations.render(str(remainingSoldiersB[i]), True, GREEN)
        #         self.screen.blit(fontRemainingSoldiersB, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (j+1) + cubeWidth - 13,
        #                                                  self.width / 2 - 20))
        #         j = j + 1
        remainingSoldiers = [remainingSoldiersA[i] + remainingSoldiersB[i] for i in range(len(remainingSoldiersA))]
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        if isSymmetrical:
            for i in range((mapSize * 2 - isEven + 1)):
                fontSoldiersGained = fontAnnotations.render(str(remainingSoldiers[i]), True, BLACK)
                self.screen.blit(fontSoldiersGained, (
                self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth / 2 - 7,
                self.width / 2 + 20 - self.devWidth))
                if warField[i] == 1:
                    pygame.draw.circle(self.screen, BLACK, (
                int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth / 2 - 2),
                int(self.width / 2 + 20 - self.devWidth) + 10), 16, self.thickness)
                if warField[i] == 2:
                    pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                        self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth / 2 -2,
                        self.width / 2 + 20 - self.devWidth + 10, 16), self.thickness)
        else:
            j = 0
            for i in range((mapSize * 2 - isEven + 1)):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                fontSoldiersGained = fontAnnotations.render(str(remainingSoldiers[i]), True, colorListRemainingSoldiers[warField[i]])
                self.screen.blit(fontSoldiersGained, (
                self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (j + 1) + cubeWidth / 2 - 7,
                self.width / 2 + 20 - self.devWidth))
                j = j + 1
        fontTag = fontAnnotations.render('Garrison:', True, BLACK)
        self.screen.blit(fontTag, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 - 100,
                self.width / 2 + 20 - self.devWidth))
        fontTag2 = fontAnnotations.render('Gain:', True, BLACK)
        self.screen.blit(fontTag2, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 - 80,
                                   self.width / 2 - 1.6 * cubeWidth - 2*self.devWidth))
        return self.screen


class DrawSoldiersGained:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width
        self.devWidth = 30
        self.thickness = 1

    def __call__(self, warField, mapSize, cubeWidth, isEven, isSymmetrical, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB):
        fontAnnotations = pygame.font.SysFont('Times New Roman', 20)
        soldiersGained = [soldierFromWarFieldA*(warField[i]==1) + soldierFromWarFieldB*(warField[i]==2) for i in range(len(warField))]
        soldiersGained[0] = soldierFromBaseA
        soldiersGained[-1] = soldierFromBaseB
        if isSymmetrical:
            for i in range((mapSize * 2 - isEven + 1)):
                fontSoldiersGained = fontAnnotations.render('+'+str(soldiersGained[i]), True, BLACK)
                self.screen.blit(fontSoldiersGained, (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth/2 - 10,
                                                          self.width / 2 - 1.6 * cubeWidth - self.devWidth))
                if warField[i] == 1:
                    if i == 0:
                        pygame.draw.circle(self.screen, BLACK, (
                        int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth / 2),
                        int(self.width / 2 - 1.6 * cubeWidth - self.devWidth) + 10), 21, self.thickness)
                    else:
                        pygame.draw.circle(self.screen, BLACK, (int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth/2),
                                                          int(self.width / 2 - 1.6 * cubeWidth - self.devWidth)+10), 16, self.thickness)
                if warField[i] == 2:
                    if i == mapSize * 2 - isEven:
                        pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                            int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (
                                i) + cubeWidth / 2),
                            int(self.width / 2 - 1.6 * cubeWidth - self.devWidth) + 10, 21), self.thickness)
                    else:
                        pygame.draw.polygon(self.screen, BLACK, transformToSquare(
                        int(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i) + cubeWidth / 2),
                        int(self.width / 2 - 1.6 * cubeWidth - self.devWidth) + 10, 16), self.thickness)

        return self.screen


class DrawLastRun:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

        self.devWidth = 30
        self.thickness = 10
        self.downward = 20

    def __call__(self, warLocation, mapSize, cubeWidth):

        if warLocation > 0:
            # pygame.draw.polygon(self.screen, BLACK, transformToTriangle(
            #     int(self.length / 2 - mapSize * cubeWidth + cubeWidth * (warLocation)),
            #     int(self.width / 2 - cubeWidth / 2 + 3 * cubeWidth) - 90, 15), self.thickness)
            pygame.draw.rect(self.screen, BLACK, (
                self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * warLocation,
                self.width / 2 - cubeWidth - self.devWidth, cubeWidth,
                cubeWidth), self.thickness)


class Warnings:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, warningA, warningB):
        fontPreface = pygame.font.SysFont('microsoft Yahei', 30)
        if warningA == 1:
            warningForA = fontPreface.render('Policy of player A exceeds the limit !', True, BLACK)
            self.screen.blit(warningForA, (self.length / 2 - 150, 0.8 * self.width / 3 - 50))
        if warningB == 1:
            warningForB = fontPreface.render('Policy of player B exceeds the limit !', True, BLACK)
            self.screen.blit(warningForB, (self.length / 2 - 150, 0.8 * self.width / 3  - 20))

        return self.screen


class DrawPeaceButton:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, color, cubeWidth):
        pygame.draw.circle(self.screen, colorList[color], (int(self.length - 1.5 * cubeWidth),
                                                int(self.width - 1.5 * cubeWidth)), radius)
        fontPreface = pygame.font.SysFont('microsoft Yahei', 20)
        peace = fontPreface.render('Peace', True, DEEPGREEN)
        self.screen.blit(peace, (self.length - 2 * cubeWidth + 12, self.width - 2 * cubeWidth + 24))


def generatePositionListA(length, width, mapSize, isEven, cubeWidth, isSymmetrical, baseLocationRandom):
    positionList = []
    if isSymmetrical:
        for i in range(mapSize * 2 - isEven - 1):
            positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)), int(width / 2 - cubeWidth / 2 + 2*cubeWidth)))
    else:
        j = 0
        for i in range(mapSize * 2 - isEven - 1):
            if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                j = j + 1
            positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (j+1)), int(width / 2 - cubeWidth / 2 + 2*cubeWidth)))
            j = j + 1

    return positionList


def generatePositionListB(length, width, mapSize, isEven, cubeWidth, isSymmetrical, baseLocationRandom):
    positionList = []
    if isSymmetrical:
        for i in range(mapSize * 2 - isEven - 1):
            positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (i+1)), int(width / 2 - cubeWidth / 2 + 3*cubeWidth)))
    else:
        j = 0
        for i in range(mapSize * 2 - isEven - 1):
            if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                j = j + 1
            positionList.append((int(length / 2 - mapSize * cubeWidth + cubeWidth * (j+1)), int(width / 2 - cubeWidth / 2 + 3*cubeWidth)))
            j = j + 1

    return positionList


def detectPosition(pos, positionList):
    limit = radius
    for i in range(len(positionList)):
        if abs(positionList[i][0]-pos[0]) <= limit and abs(positionList[i][1]-pos[1]) <= limit:
            return i
    return -1


def transformKeyToIntA(key):
    return key - 48


def transformKeyToIntB(key):
    return key - 256


def transformTempListToInt(tempList):
    if len(tempList) >0:
        number = '0'
        for i in range(len(tempList)):
            number = number + str(tempList[i])
        return int(number)
    else: return 0


def judgeResult(remainingSoldiersA, remainingSoldiersB, warField):
    warFieldNew = [0 for i in range(len(warField))]
    for i in range(len(warField)):
        if remainingSoldiersA[i] > 0:
            warFieldNew[i] = 1
        elif remainingSoldiersB[i] > 0:
            warFieldNew[i] = 2
        elif remainingSoldiersB[i] == 0 and remainingSoldiersA[i] == 0:
            warFieldNew[i] = warField[i]
    return warFieldNew


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


def transformPolicyToSoldierMove(policy):
    policy = list(policy)
    soldierMove = [0 for i in range(len(policy))]
    if policy.count(1) >0:
        index = policy.index(1)

        for i in range(len(policy)):
            if i < index:
                soldierMove[i] = 1
            if i == index:
                soldierMove[i] = 0
            if i > index:
                soldierMove[i] = -1
        return soldierMove
    else:
        return soldierMove

def calculateWinner(soldiersA, soldiersB):
    ratio = soldiersA/(soldiersA+soldiersB)
    coin = random.uniform(0,1)
    # print(coin)
    if coin < ratio:
        return 1
    else:
        return 2

def calculateLostRate(soldiersA, soldiersB):
    factorA = 0.02
    factorB = 1.1
    if soldiersA < 1:
        return [1, 0]
    if soldiersB < 1:
        return [0, 1]
    if soldiersB >= soldiersA:
        lostRateA = factorA * math.exp(factorB*(soldiersB / soldiersA - 1))
        lostRateB = factorA * math.exp(factorB*(1 - soldiersB / soldiersA))
    else:
        lostRateA = factorA * math.exp(factorB*(1 - soldiersA / soldiersB))
        lostRateB = factorA * math.exp(factorB*(soldiersA / soldiersB - 1))
    return [min(lostRateA,1), min(lostRateB,1)]

def simulateWarProcess(soldiersA, soldiersB):
    lostLimit = 0.8
    initialSoldiersA = soldiersA
    initialSoldiersB = soldiersB
    battleRound = 3
    winner = 1
    winnerList = []
    # for i in range(battleRound):
    #     lostRate = calculateLostRate(soldiersA, soldiersB)
    #     winnerList.append(calculateWinner(soldiersA, soldiersB))
    #     lamdaA = max(round(soldiersA*lostRate[0]), 1)
    #     lamdaB = max(round(soldiersB*lostRate[1]), 1)
    #     lostSoldierA = np.random.poisson(lamdaA, 1)
    #     lostSoldierB = np.random.poisson(lamdaB, 1)
    #     soldiersA -= lostSoldierA[0]
    #     soldiersB -= lostSoldierB[0]
    #     soldiersA = max(soldiersA, 0)
    #     soldiersB = max(soldiersB, 0)
    #     if soldiersB == 0:
    #         # print('B empty')
    #         return soldiersA, soldiersB, 1
    #     elif soldiersA == 0:
    #         # print('A empty')
    #         return soldiersA, soldiersB, 2
    #
    # if winnerList.count(1) > winnerList.count(2):
    #     winner = 1
    # else:
    #     winner = 2
    while True:
        lostRate = calculateLostRate(soldiersA, soldiersB)
        lamdaA = max(round(soldiersA*lostRate[0]), 1)
        lamdaB = max(round(soldiersB*lostRate[1]), 1)
        lostSoldierA = np.random.poisson(lamdaA, 1)
        lostSoldierB = np.random.poisson(lamdaB, 1)
        soldiersA -= lostSoldierA[0]
        soldiersB -= lostSoldierB[0]
        soldiersA = max(soldiersA, 0)
        soldiersB = max(soldiersB, 0)
        # print('A:')
        # print(initialSoldiersA - soldiersA)
        # print('B:')
        # print(initialSoldiersB - soldiersB)
        if soldiersB == 0:
            # print('B empty')
            return soldiersA, soldiersB, 1
        elif soldiersA == 0:
            # print('A empty')
            return soldiersA, soldiersB, 2
        if soldiersA/initialSoldiersA < lostLimit and soldiersB/initialSoldiersB >= lostLimit:
            return soldiersA, soldiersB, 2
        if soldiersA/initialSoldiersA >= lostLimit and soldiersB/initialSoldiersB < lostLimit:
            return soldiersA, soldiersB, 1
        if soldiersA / initialSoldiersA < lostLimit and soldiersB / initialSoldiersB < lostLimit:
            return soldiersA, soldiersB, calculateWinner(soldiersA, soldiersB)


def calculateRemainingSoldiers(policyA, policyB, remainingSoldiersA, remainingSoldiersB, warField, soldierFromWarFieldA, soldierFromWarFieldB, soldierFromBaseA, soldierFromBaseB):
    policyA = list(policyA)
    policyB = list(policyB)
    winBonus = 0
    warLocation = 0
    warLost = 0

    remainingSoldiersNewA = [0 for i in range(len(policyA))]
    remainingSoldiersNewB = [0 for i in range(len(policyA))]
    soldierMoveA = transformPolicyToSoldierMove(policyA)
    soldierMoveB = transformPolicyToSoldierMove(policyB)
    for i in range(len(policyA)):
        remainingSoldiersNewA[i+soldierMoveA[i]] += remainingSoldiersA[i]
        remainingSoldiersNewB[i+soldierMoveB[i]] += remainingSoldiersB[i]

    if policyA.count(1) >0 and policyB.count(1) >0 and policyA.index(1) > policyB.index(1):
        warLocation = policyA.count(1)
        tempSoldiersA, tempSoldiersB, winner = simulateWarProcess(remainingSoldiersNewA[policyA.index(1)], remainingSoldiersNewB[policyB.index(1)])

        if winner == 1:
            remainingSoldiersNewA[policyA.index(1)] = tempSoldiersA
            remainingSoldiersNewB[policyB.index(1)+1] += tempSoldiersB
            remainingSoldiersNewB[policyB.index(1)] = 0
        if winner == 2:
            remainingSoldiersNewB[policyB.index(1)] = tempSoldiersB
            remainingSoldiersNewA[policyA.index(1)-1] += tempSoldiersA
            remainingSoldiersNewA[policyA.index(1)] = 0

    remainingSoldiersNew2A = [0 for i in range(len(policyA))]
    remainingSoldiersNew2B = [0 for i in range(len(policyA))]
    remainingSoldiersNew2A[0] = remainingSoldiersNewA[0]
    remainingSoldiersNew2B[-1] = remainingSoldiersNewB[-1]

    for i in range(len(policyA)):
        if remainingSoldiersNewA[i] > 0 and remainingSoldiersNewB[i] > 0:
            warLocation = i

            tempSoldiersA, tempSoldiersB, winner = simulateWarProcess(remainingSoldiersNewA[i],
                                                                      remainingSoldiersNewB[i])

            if winner == 1:
                remainingSoldiersNewA[i] = tempSoldiersA
                remainingSoldiersNewB[i + 1] += tempSoldiersB
                remainingSoldiersNewB[i] = 0
            if winner == 2:
                remainingSoldiersNewB[i] = tempSoldiersB
                remainingSoldiersNewA[i - 1] += tempSoldiersA
                remainingSoldiersNewA[i] = 0


    for i in range(len(policyA)-2):
        remainingSoldiersNew2A[i+1] = math.ceil((max(0, remainingSoldiersNewA[i+1] - remainingSoldiersNewB[i+1])))
        remainingSoldiersNew2B[i+1] = math.ceil((max(0, remainingSoldiersNewB[i+1] - remainingSoldiersNewA[i+1])))

    warField = list(warField)
    warFieldNew = judgeResult(remainingSoldiersNew2A, remainingSoldiersNew2B, warField)
    boundaryOld = warField.count(1)
    boundaryNew = warFieldNew.count(1)
    if boundaryOld < boundaryNew:
        remainingSoldiersNew2A[boundaryNew-1] += winBonus
    if boundaryNew < boundaryOld:
        remainingSoldiersNew2B[boundaryNew-1] += winBonus


    for i in range(len(warFieldNew)-2):
        if warFieldNew[i+1] == 1:
            remainingSoldiersNew2A[0] += soldierFromWarFieldA
        if warFieldNew[i+1] == 2:
            remainingSoldiersNew2B[-1] += soldierFromWarFieldB
    remainingSoldiersNew2A[0] += soldierFromBaseA
    remainingSoldiersNew2B[-1] += soldierFromBaseB


    for i in range(len(policyA)):
        remainingSoldiersNew2A[i] = math.ceil((remainingSoldiersNew2A[i]) * (1 - (1 / (len(policyA) - i))))
        remainingSoldiersNew2B[i] = math.ceil((remainingSoldiersNew2B[i]) * (1 - (1 / (len(policyA) - 8 + i))))


    return remainingSoldiersNew2A, remainingSoldiersNew2B, warFieldNew, warLocation


# def calculateSoldiers(warField, soldierFromBase, soldiersA, soldiersB, baseLocationRandom):
#     soldierFromWarField = 10
#     soldiersGained = [0 for i in range(len(warField))]
#     for i in range(len(warField)):
#         soldiersA = soldiersA + int((warField[i] == 1) * (1 - (i + 1) / (len(warField) + 1)) * soldierFromWarField)
#         soldiersB = soldiersB + int((warField[i] == 2) * ((i + 1) / (len(warField) + 1)) * soldierFromWarField)
#         soldiersGained[i] = max(int((warField[i] == 1) * (1 - (i + 1) / (len(warField) + 1)) * soldierFromWarField),
#                                 int((warField[i] == 2) * ((i + 1) / (len(warField) + 1)) * soldierFromWarField))
#     soldiersA += soldierFromBase
#     soldiersB += soldierFromBase
#     return [soldiersA, soldiersB, soldiersGained]


# def calculateSoldiers(warField, soldierFromBaseA, soldierFromBaseB, soldiersA, soldiersB, baseLocationRandom, soldierFromWarField):
#     soldiersGained = [0 for i in range(len(warField))]
#     for i in range(len(warField)):
#         soldiersA = soldiersA + int((warField[i] == 1) * soldierFromWarField)
#         soldiersB = soldiersB + int((warField[i] == 2) * soldierFromWarField)
#         soldiersGained[i] = max(int((warField[i] == 1) * soldierFromWarField),
#                                 int((warField[i] == 2) * soldierFromWarField))
#     soldiersA += soldierFromBaseA
#     soldiersB += soldierFromBaseB
#     return [soldiersA, soldiersB, soldiersGained]


def isPeace(pos, length, width, cubeWidth):
    limit = radius
    buttonPos = [length - 1.5 * cubeWidth, width - 1.5 * cubeWidth]
    if abs(buttonPos[0] - pos[0]) <= limit and abs(buttonPos[1] - pos[1]) <= limit:
            return 1
    return -1


class CheckAutoPeace:
    print("Checking for 7 grids situation")

    def __init__(self, peaceEndTurn):
        self.count = 0
        self.peaceEndTurn = peaceEndTurn

    def __call__(self, policyA, policyB, warField):
        warField = list(warField)
        boundary = warField.count(1)
        if sum(policyA[boundary-1:]) == 0 and sum(policyB[:boundary+1]) == 0 and warField.count(0)==0:
            self.count += 1
        else:
            self.count = 0

        if self.count == self.peaceEndTurn:
            self.count = 0
            return 1
        else:
            return 0

class DrawBackGround:

    def __init__(self, screen, length, width, mapSize, cubeWidth, isBoundary):
        self.screen = screen
        self.length = length
        self.width = width
        self.cutSize = 17
        self.devSize = 9
        self.devWidth = 30

        # self.river = pygame.image.load('river.png')
        # self.mountain = pygame.image.load('mountain.png')
        self.circular = pygame.image.load('circular2.png')
        self.square = pygame.image.load('square2.png')
        self.diamond = pygame.image.load('diamond2.png')
        self.triangle = pygame.image.load('triangle2.png')
        self.circular = pygame.transform.scale(self.circular, (cubeWidth - self.cutSize, cubeWidth - self.cutSize))
        self.square = pygame.transform.scale(self.square, (cubeWidth - self.cutSize, cubeWidth - self.cutSize))
        self.diamond = pygame.transform.scale(self.diamond, (cubeWidth - self.cutSize, cubeWidth - self.cutSize))
        self.triangle = pygame.transform.scale(self.triangle, (cubeWidth - self.cutSize, cubeWidth - self.cutSize))
        self.shapeList = [self.circular, self.square, self.diamond, self.triangle]
        random.shuffle(self.shapeList)
        if isBoundary:
            self.otherList = [round((i + 2 - 0.1) / 2) for i in range((mapSize - 1) * 2)]
            random.shuffle(self.otherList)
        else:
            if mapSize == 3:
                self.otherList = [1, 2, 1, 2]
            if mapSize == 4:
                self.otherList = [1, 2, 3, 1, 2, 3]
        self.AsymmetricalList = [random.randint(0,3) for i in range(mapSize * 2 - 1)]

    def __call__(self, mapSize, cubeWidth, isSymmetrical, isEven, baseLocationRandom, isBoundary, assumedBoundary, dev):
        # self.river = pygame.transform.scale(self.river, (cubeWidth-5, cubeWidth-5))
        # self.mountain = pygame.transform.scale(self.mountain, (cubeWidth-5, cubeWidth-5))
        j = 0
        if isSymmetrical:
            if isBoundary:
                for i in range(mapSize * 2 - isEven - 1):
                    if i ==  assumedBoundary + dev:
                        self.screen.blit(self.shapeList[0],
                                         (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i + 1) + self.devSize,
                                          self.width / 2 - cubeWidth + self.devSize - self.devWidth))
                    else:
                        self.screen.blit(self.shapeList[self.otherList[j]], (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i + 1) + self.devSize,
                        self.width / 2 - cubeWidth + self.devSize - self.devWidth))
                        j = j + 1
            else:
                for i in range(mapSize * 2 - isEven - 1):
                    if i ==  assumedBoundary + dev:
                        self.screen.blit(self.shapeList[1],
                                         (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i + 1) + self.devSize,
                                          self.width / 2 - cubeWidth + self.devSize - self.devWidth))
                    else:
                        self.screen.blit(self.shapeList[self.otherList[j]], (self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (i + 1) + self.devSize,
                        self.width / 2 - cubeWidth + self.devSize - self.devWidth))
                        j = j + 1

        else:
            j = 0
            for i in range(mapSize * 2 - isEven - 1):
                if i == mapSize * 2 - isEven - 1 - baseLocationRandom:
                    j = j + 1
                self.screen.blit(self.shapeList[self.AsymmetricalList[i]],(self.length / 2 - mapSize * cubeWidth - cubeWidth / 2 + cubeWidth * (j + 1) + self.devSize,self.width / 2 - cubeWidth + self.devSize - self.devWidth))
                j = j + 1
        return self.otherList


class DrawResults:

    def __init__(self, screen, length, width, compulsoryEndTurn, maxTrial, soldierPoint, fieldPoint, factor):
        self.screen = screen
        self.length = length
        self.width = width
        self.compulsoryEndTurn = compulsoryEndTurn
        self.maxTrial = maxTrial
        self.rewardA = 0
        self.rewardB = 0
        self.trial = 0
        self.soldierPoint = soldierPoint
        self.fieldPoint = fieldPoint
        self.flag = 0
        self.factor = factor
        self.testTrial = 2

    def __call__(self, step, warField, isAutoPeace, remainingSoldiersA, remainingSoldiersB, trial):

        if trial > self.trial:
            self.trial = trial
            self.flag = 0
        if self.trial > 1 and self.flag == 0:
            self.rewardA += round(sum(remainingSoldiersA) * self.factor)
            self.rewardB += round(sum(remainingSoldiersB) * self.factor)
            self.flag = 1
        if trial < self.maxTrial:
            fontPreface = pygame.font.SysFont('Times New Roman', 30)
            if step == self.compulsoryEndTurn:
                result = fontPreface.render('Result: Compulsory End', True, BLACK)
                self.screen.blit(result, (500, 20))
                reward = fontPreface.render('reward '+'A:'+str(round(sum(remainingSoldiersA)*self.factor+(warField.count(1)-1)*self.fieldPoint*self.factor))+' B:'+str(round(sum(remainingSoldiersB)*self.factor+(warField.count(2)-1)*self.fieldPoint*self.factor)), True, BLACK)
                self.screen.blit(reward, (500, 50))
            elif isAutoPeace:
                result = fontPreface.render('Result: Auto Peace', True, BLACK)
                self.screen.blit(result, (500, 20))
                reward = fontPreface.render('reward '+'A:'+str(round(sum(remainingSoldiersA)*self.factor+(warField.count(1)-1)*self.fieldPoint*self.factor))+' B:'+str(round(sum(remainingSoldiersB)*self.factor+(warField.count(2)-1)*self.fieldPoint*self.factor)), True, BLACK)
                self.screen.blit(reward, (500, 50))
            elif warField.count(1) == len(warField)-1:
                result = fontPreface.render('Result: A dominates', True, BLACK)
                self.screen.blit(result, (500, 20))
                reward = fontPreface.render('reward '+'A:'+str(round((sum(remainingSoldiersB)+sum(remainingSoldiersA))*self.factor+(warField.count(1)-1)*self.fieldPoint*self.factor))+' B:'+str(0), True, BLACK)
                self.screen.blit(reward, (500, 50))
            elif warField.count(2) == len(warField)-1:
                result = fontPreface.render('Result: B dominates', True, BLACK)
                self.screen.blit(result, (500, 20))
                reward = fontPreface.render('reward '+'A:'+str(0)+' B:'+str(round((sum(remainingSoldiersB)+sum(remainingSoldiersA))*self.factor+(warField.count(2)-1)*self.fieldPoint*self.factor)), True, BLACK)
                self.screen.blit(reward, (500, 50))
            if self.trial == self.testTrial - 1:
                guide = fontPreface.render('{} Test trials are done!'.format(self.testTrial), True, RED)
            else:
                guide = fontPreface.render('Take a rest~', True, BLACK)
            self.screen.blit(guide, (500, 80))
        if trial == self.maxTrial:
            self.screen.fill(backgroundColor)
            fontPreface2 = pygame.font.SysFont('Times New Roman', 40)
            result = fontPreface2.render('Final Reward: ', True, BLACK)
            self.screen.blit(result, (300, 200))
            reward = fontPreface2.render('A:' + str(round(self.rewardA/self.maxTrial)) + ' B:' + str(
                round(self.rewardB/self.maxTrial)), True, BLACK)
            self.screen.blit(reward, (300, 240))
        return self.screen


class DrawRound:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, step):
        fontPreface = pygame.font.SysFont('Times New Roman', 25)
        round = fontPreface.render('round: ' + str(step+1), True, BLACK)
        self.screen.blit(round, (self.length / 2 + 250, self.width / 2 + 200))
        return self.screen


class DrawTime:

    def __init__(self, screen, length, width):
        self.screen = screen
        self.length = length
        self.width = width

    def __call__(self, maxTime, time):
        remainingTime = maxTime*1000 - time
        remainingTime = max(remainingTime, 0)
        fontPreface = pygame.font.SysFont('Times New Roman', 25)
        displayTime = fontPreface.render('time: ' + str(int(remainingTime/1000)), True, BLACK)
        self.screen.blit(displayTime, (self.length / 2 + 250, self.width / 2 + 250))
        return self.screen


class DrawPoint:

    def __init__(self, screen, length, width, soldierPoint, fieldPoint):
        self.screen = screen
        self.length = length
        self.width = width
        self.soldierPoint = soldierPoint
        self.fieldPoint = fieldPoint

    def __call__(self, remainingSoldiersA, remainingSoldiersB, warField):
        pointA = sum(remainingSoldiersA)*self.soldierPoint+(warField.count(1)-1)*self.fieldPoint
        pointB = sum(remainingSoldiersB)*self.soldierPoint+(warField.count(2)-1)*self.fieldPoint
        fontPreface = pygame.font.SysFont('Times New Roman', 25)
        displayA = fontPreface.render('A: ' + str(int(pointA)), True, BLACK)
        self.screen.blit(displayA, (self.length / 2 - 370, self.width / 2 + 200))
        displayB = fontPreface.render('B: ' + str(int(pointB)), True, BLACK)
        self.screen.blit(displayB, (self.length / 2 - 370, self.width / 2 + 250))
        return pointA, pointB


def transformIntToTempList(integer):
    tempList = []
    while(integer > 0):
        tempList.append(integer%10)
        integer = int(integer / 10)
    tempList.reverse()
    return tempList


def transformTypingPositionToPolicy(typingPosition, mapSize, isEven):
    policy = [0 for i in range(mapSize * 2 - isEven + 1)]
    if typingPosition>=0:
        policy[typingPosition] = 1
    return policy

def calculateAttackIntentionA(warField, policyA):
    if policyA.count(1) == 0:
        return warField.index(1)/2
    else:
        return policyA.index(1)

def calculateAttackIntentionB(warField, policyB):
    if policyB.count(1) == 0:
        return 3 - warField.index(1)/2
    else:
        return 7 - policyB.index(1)

def calculateNumOfWar(warField, policyA, policyB):
    if (policyA.count(1) > 0 and policyA.index(1) >= warField.index(2)) or (
            policyB.count(1) > 0 and policyB.index(1) <= (warField.index(2)-1)):
        return 1
    else:
        return 0