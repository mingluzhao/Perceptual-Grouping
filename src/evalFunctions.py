import numpy as np

def calcAgentsReward(traj):
    rewardIDinTraj = 2
    agentsTrajReward = np.sum([timeStepInfo[rewardIDinTraj] for timeStepInfo in traj], axis=0)
    return agentsTrajReward

def calcAgentsActionsMean(traj):
    actionIDinTraj = 1
    agent1ActionMean = np.mean([np.argmax(timeStepInfo[actionIDinTraj][0]) for timeStepInfo in traj])
    agent2ActionMean = np.mean([np.argmax(timeStepInfo[actionIDinTraj][1]) + 1 for timeStepInfo in traj])
    # account for agent2 actual action = [0, a0, a1, a2, a3, a4, a5, a6]
    return agent1ActionMean, agent2ActionMean