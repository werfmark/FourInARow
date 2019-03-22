import numpy as np
from random import randint
from sklearn.neural_network import MLPRegressor
from time import gmtime, strftime
import csv

boardFieldWidth = 7
boardFieldHeight = 6

class Board(object):
    def __init__(self):
        self.FieldWidth = boardFieldWidth
        self.FieldHeight = boardFieldHeight
        self.Field = np.zeros([self.FieldHeight, self.FieldWidth], dtype=int)
        self.FirstActivePlayer = 1
        self.ActivePlayer = self.FirstActivePlayer
        self.Sequence = [None] * self.FieldWidth * self.FieldHeight
        self.Ply = 0
        self.State = [None] * self.FieldWidth * self.FieldHeight


board = Board()

def makeMove(column, board):
    col = (board.Field[:, column])
    row = board.FieldHeight - np.sum(np.abs(col)) - 1
    board.Field[row, column] = board.ActivePlayer
    board.Sequence[board.Ply] = (row, column)
    board.State[board.Ply] = board.Field
    board.Ply += 1

def legalMoves(board):
    legalColumns = [1] * board.FieldWidth
    for i in range(0, board.FieldWidth):
        if(np.sum(np.abs(board.Field[:, i])) >= board.FieldHeight):
            legalColumns[i] = 0
    return legalColumns

for _ in range(0,6):
    makeMove(3, board)
    print(board.Field)


def contains4InARow(array):
    if ",1,1,1,1" in "," + ",".join(str(i) for i in array):
        return 1
    elif ",-1,-1,-1,-1" in "," + ",".join(str(i) for i in array):
        return -1
    else:
        return 0

def gameEnded(board):
    if np.sum(legalMoves(board)) == 0:
        return True
    row = board.Sequence[board.Ply - 1][0]
    col = board.Sequence[board.Ply - 1][1]
    rowToTest = board.Field[row,:]
    if (contains4InARow(rowToTest)):
        return True
    colToTest = board.Field[:,col]
    if (contains4InARow(colToTest)):
        return True
    diagonalToTest1 = np.diagonal(board.Field, col - row)
    if (contains4InARow(diagonalToTest1)):
        return True
    diagonalToTest2 = np.diagonal(np.fliplr(board.Field), board.FieldWidth - 1 - col - row)
    if (contains4InARow(diagonalToTest2)):
        return True




gameEnded(board)

makeMove(0, board)

# Game:
# initiate board
# set player names
# while not gameEnded:
# makeMoves, switch active player
# set winner

class Game(object):
    def __init__(self):
        self.board = Board()
        self.playerA = None
        self.playerB = None
        self.winner = None

def strategyBotv1(game):
    # get legalMoves
    lMoves = legalMoves(game.board)
    # determine move
    lMoveIndices = [i for i, x in enumerate(lMoves) if x == 1]
    move = lMoveIndices[randint(0, len(lMoveIndices) - 1)]
    # return move
    return move

def strategyBotv0(game):
    # get legalMoves
    lMoves = legalMoves(game.board)
    # determine move
    lMoveIndices = [i for i, x in enumerate(lMoves) if x == 1]
    move = lMoveIndices[randint(0, len(lMoveIndices) - 1)]
    # return move
    return move


def simulateGame(namePlayerA, namePlayerB):
    game = Game()
    game.playerA = namePlayerA
    game.playerB = namePlayerB
    gEnded = False
    while not gEnded:

        proposedMove = None
        if game.board.ActivePlayer == game.board.FirstActivePlayer:
            proposedMove = strategyBotv1(game)
        if game.board.ActivePlayer != game.board.FirstActivePlayer:
            proposedMove = strategyBotv0(game)

        makeMove(proposedMove, game.board)
        gEnded = gameEnded(game.board)
        if gEnded:
            game.winner = game.board.ActivePlayer
            if np.sum(legalMoves(game.board)) == 0:
                game.winner = 0
            #print("Winner: ", game.winner)
        game.board.ActivePlayer = game.board.ActivePlayer * -1
        #print(game.board.Ply)
        #print(game.board.Field)
        #print("")
    return game


game1 = simulateGame("rbot1", "rbot2")
print(game1.winner)

def runGames(num, windif):
    gamesPlayed = 0
    gameList = [None] * num
    while gamesPlayed < num:
        tempGame = simulateGame("rbot1", "rbot2")
        #print(tempGame.winner)
        gameList[gamesPlayed] = tempGame
        gamesPlayed += 1
    return gameList

while True:
    numGames = 1000
    glist = runGames(numGames, None)
    stateMat = np.asmatrix(np.zeros([0, boardFieldHeight * boardFieldWidth + 1], dtype=int))
    for i in range(0, len(glist)):
        print(i)
        states = glist[i].board.State
        gameResult = (glist[i].winner / 2) + 0.5 # werkt alleen als we uitgaan van speler1!
        for j in range(0, len(states)):
            #print(states[j])
            if(states[j] is not None):
                #print(states[j].shape)
                #flat = np.transpose(np.asmatrix(states[j].flatten()))
                flat = np.append(np.asmatrix(states[j].flatten()), np.transpose(np.asmatrix(gameResult)), axis=1)
                #print(flat)
                stateMat = np.append(stateMat, flat, axis=0)
                #print(state.flatten())


    timeNow = strftime("%Y-%m-%d %H-%M-%S", gmtime())
    np.savetxt("stateMat_" + timeNow + ".csv", stateMat, delimiter=",")

########

X = stateMat[:,:-1]
y = stateMat[:,-1]
mlp = MLPRegressor(hidden_layer_sizes=(100, ), solver='sgd', max_iter=100, activation='logistic', random_state=1, learning_rate_init=0.01, batch_size=X.shape[0])

mlp.fit(X, y)
pred1 = mlp.predict(X)