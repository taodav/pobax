from random import random
from numpy import random
from random import seed
from random import randint
from datetime import datetime
from datetime import timedelta
import math
import numpy
import time
import xlsxwriter

#http://datagenetics.com/blog/december32011/index.html

# ----Boat Generation----    
def ShipGenerator(board_horz,board_vert,Ship):
    Valid = 0
    #print("Generating " + Ship[0] + "...")
    while Valid == 0:
        #--Pick a random spot--
        C1 = (randint(0,9))
        C2 = (randint(0,9))

        #C1 = 9  #row
        #C2 = 4  #column
        
        if board_horz[C1][C2] == 0:
            
            #-----Check Ships Left and Right-----
            Left_Right = numpy.argwhere(board_horz[C1]>0).transpose()
            Left = numpy.argwhere(Left_Right[0]<C2)
            Right = numpy.argwhere(Left_Right[0]>C2)
            Space_Left_Right = numpy.squeeze(Left_Right)
            Space_Left = numpy.squeeze(Left)
            Space_Right = numpy.squeeze(Right)
            #print(Space_Left_Right)
            #print(Space_Left.size)
            #print(Space_Right.size)
            if Space_Left.size == 1:
                if Space_Left_Right.size == 1:
                    Closest_Left = Space_Left_Right
                else:
                    Closest_Left = Space_Left_Right[Space_Left]
            elif Space_Left.size == 0:
                Closest_Left = -1
            else:
                Closest_Left = Space_Left_Right[Space_Left[Space_Left.size - 1]]
            #print(Closest_Left)
            
            if Space_Right.size == 1:
                if Space_Left_Right.size == 1:
                    Closest_Right = Space_Left_Right
                else:
                    Closest_Right = Space_Left_Right[Space_Right]
            elif Space_Right.size == 0:
                Closest_Right = 10
            else:
                Closest_Right = Space_Left_Right[Space_Right[0]]
            #print(Closest_Right)
            
            Horz_Room = [Closest_Right - (Closest_Left + 1),C2 - (Closest_Left + 1),Closest_Right - C2 - 1]
            #print(Horz_Room)
            
            #-----Check Ships Above and Below-----
            Up_Down = numpy.argwhere(board_vert[C2]>0).transpose()
            Up = numpy.argwhere(Up_Down[0]<C1)
            Down = numpy.argwhere(Up_Down[0]>C1)
            Space_Up_Down = numpy.squeeze(Up_Down)
            Space_Up = numpy.squeeze(Up)
            Space_Down = numpy.squeeze(Down)
            #print(Space_Up_Down)
            #print(Space_Up.size)
            #print(Space_Down.size)
            if Space_Up.size == 1:
                if Space_Up_Down.size == 1:
                    Closest_Up = Space_Up_Down
                else:
                    Closest_Up = Space_Up_Down[Space_Up]
            elif Space_Up.size == 0:
                Closest_Up = -1
            else:
                Closest_Up = Space_Up_Down[Space_Up[Space_Up.size - 1]]
            #print(Closest_Up)
            
            if Space_Down.size == 1:
                if Space_Up_Down.size == 1:
                    Closest_Down = Space_Up_Down
                else:
                    Closest_Down = Space_Up_Down[Space_Down]
            elif Space_Down.size == 0:
                Closest_Down = 10
            else:
                Closest_Down = Space_Up_Down[Space_Down[0]]
            #print(Closest_Down)
            
            Vert_Room = [Closest_Down - (Closest_Up + 1),C1 - (Closest_Up + 1),Closest_Down - C1 - 1]
            #print(Vert_Room)
            
            if Horz_Room[0] >= Ship[1] or Vert_Room[0] >= Ship[1]:
                board_horz[C1][C2] = Ship[2]
                if Horz_Room[0] >= Ship[1] and Vert_Room[0] >= Ship[1]:
                    Direction = randint(1,2)
                elif Horz_Room[0] >= Ship[1]:
                    Direction = 1
                elif Vert_Room[0] >= Ship[1]:
                    Direction = 2
                
                Anchor_Array = numpy.array([1,2,3,4,5])
                Anchor_Ship = Anchor_Array[:Ship[1]]
                Anchor_Zeros = numpy.zeros((11-Ship[1]),dtype=int)
                Anchor_ArrayLeft = numpy.append(Anchor_Zeros,Anchor_Array[:Ship[1]-1])
                Anchor_ArrayRight = numpy.array([Ship[1],Ship[1],Ship[1],Ship[1],Ship[1],Ship[1],4,3,2,1])
                #print(Anchor_Array)

                if Direction == 1:
                    Anchor_Ship = Anchor_Ship[Anchor_ArrayLeft[9-Horz_Room[2]]:Anchor_ArrayRight[9-Horz_Room[1]]]
                    Anchor = Anchor_Ship[randint(0,Anchor_Ship.size-1)]
                    for x in range(Anchor):
                        board_horz[C1][C2-x] = Ship[2]
                    for x in range(Ship[1] - Anchor):
                        board_horz[C1][C2+(Ship[1] - Anchor - x)] = Ship[2]
                elif Direction == 2:
                    Anchor_Ship = Anchor_Ship[Anchor_ArrayLeft[9-Vert_Room[2]]:Anchor_ArrayRight[9-Vert_Room[1]]]
                    Anchor = Anchor_Ship[randint(0,Anchor_Ship.size-1)]
                    for x in range(Anchor):
                        board_horz[C1-x][C2] = Ship[2]
                    for x in range(Ship[1] - Anchor):
                        board_horz[C1+(Ship[1] - Anchor - x)][C2] = Ship[2]             
                    
                return board_horz,board_vert
    
# ----Random Guesser----
def RandomSearch(board_horz,turn_count):
    #print("\n")
    #print("Starting Search with Random Method")
    game = 1
    hit_count = 0
    
    while(game > 0):
    
        G1 = (randint(0,9)) #Row
        G2 = (randint(0,9)) #Column
        
        if board_horz[G1][G2] != 6 and board_horz[G1][G2] != 7:
            
            turn_count += 1

            if board_horz[G1][G2] != 0:
                board_horz[G1][G2] = 7
                #print("Hit!")
                hit_count += 1
                
            else:
                board_horz[G1][G2] = 6
            
            if hit_count == 17:
                #print("Game Complete")
                #print(turn_count)
                game = 0
                
    return board_horz,turn_count

# ----Hunt and Target----
def HuntTarget(board_horz,turn_count):
    #print("\n")
    #print("Starting Search with Hunt and Target Method")
    game = 1
    hit_count = 0
    target = "stop"
    Carrier_Hits = 0
    Carrier_Locations = [[]]
    Battleship_Hits = 0
    Battleship_Locations = [[]]
    Cruiser_Hits = 0
    Cruiser_Locations = [[]]
    Submarine_Hits = 0
    Submarine_Locations = [[]]
    Destroyer_Hits = 0
    Destroyer_Locations = [[]]
    
    while(game > 0):
        
        #time.sleep(2)
        #print(Legend_horz)
        #for x in board_horz:
        #    print(x)
        #print("\n")
            
        G1 = (randint(0,9)) #Row
        G2 = (randint(0,9)) #Column
        
        unsunk = 0
        G1_index = 0
        G2_index = 0
        
        for x in board_horz:
            for y in x:
                if y == 7:
                    unsunk = 1
                    G1 = G1_index
                    G2 = G2_index
                    break
                G2_index += 1
            if unsunk == 1:
                break
            G1_index += 1
            G2_index = 0
        
        if board_horz[G1][G2] != 6 and board_horz[G1][G2] != 8:
            
            if board_horz[G1][G2] != 7:
                turn_count += 1
 
            if board_horz[G1][G2] != 0 and board_horz[G1][G2] != 7:
                
                if board_horz[G1][G2] == 5:
                    Carrier_Hits += 1
                    Carrier_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 4:
                    Battleship_Hits += 1
                    Battleship_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 3:
                    Cruiser_Hits += 1
                    Cruiser_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 2:
                    Submarine_Hits += 1
                    Submarine_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 1:
                    Destroyer_Hits += 1
                    Destroyer_Locations = [[G1,G2]]
                    #print("Ship Hit")
                
                board_horz[G1][G2] = 7
                #print("Hit!")
                hit_count += 1
                
                target = "go"
            
            elif board_horz[G1][G2] == 7:
                target = "go"
            
            G1_Archive = G1
            G2_Archive = G2
            target_impossible1 = 0
            target_impossible2 = 0 
            target_impossible3 = 0
            target_impossible4 = 0
            
            while target == "go":
                    
                Directions = [1,2,3,4]
                Next_Coordinate = "Not Locked"
                    
                if G1 == 0:
                    Directions.remove(1)
                    target_impossible1 = 1
                if G1 == 9:
                    Directions.remove(2)        
                    target_impossible2 = 1
                if G2 == 0:
                    Directions.remove(3)
                    target_impossible3 = 1
                if G2 == 9:
                    Directions.remove(4)
                    target_impossible4 = 1
                
                #print(Directions)
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                target_direction = random.choice(Directions)
                #print("Direction Locked")
                #print(target_direction)
                #time.sleep(2)
                #print(Legend_horz)
                #for x in board_horz:
                #    print(x)
                #print("\n")
                
                #time.sleep(5)
              
                
                    #--UP--
                if target_direction == 1:
                    if board_horz[G1-1][G2] != 6 and board_horz[G1-1][G2] != 7 and board_horz[G1-1][G2] != 8:
                        G1 = G1 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible1 = 1
                        #print("up impossible")
                    #--DOWN--
                if target_direction == 2:
                    if board_horz[G1+1][G2] != 6 and board_horz[G1+1][G2] != 7 and board_horz[G1+1][G2] != 8:
                        G1 = G1 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible2 = 1
                        #print("down impossible")
                    #--LEFT--
                if target_direction == 3:  
                    if board_horz[G1][G2-1] != 6 and board_horz[G1][G2-1] != 7 and board_horz[G1][G2-1] != 8:
                        G2 = G2 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible3 = 1
                        #print("Left impossible")
                    #--RIGHT--
                if target_direction == 4:
                    if board_horz[G1][G2+1] != 6 and board_horz[G1][G2+1] != 7 and board_horz[G1][G2+1] != 8:
                        G2 = G2 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible4 = 1
                        #print("Right impossible")
                        
                if target_impossible1 == 1 and target_impossible2 == 1 and target_impossible3 == 1 and target_impossible4 == 1:
                    target = "stop"
                #print("Next G1= " + str(G1))
                #print("Next G2= " + str(G2))
                #time.sleep(1)
                
                if Next_Coordinate == "Locked":
                    
                    turn_count += 1
                    
                    if board_horz[G1][G2] == 5:
                        Carrier_Hits += 1
                        if Carrier_Hits == 1:
                            Carrier_Locations = [[G1,G2]] 
                        else:
                            Carrier_Locations = numpy.append(Carrier_Locations,[[G1,G2]],axis=0)
                        #print(Carrier_Locations)
                    elif board_horz[G1][G2] == 4:
                        Battleship_Hits += 1
                        if Battleship_Hits == 1:
                            Battleship_Locations = [[G1,G2]]
                        else:
                            Battleship_Locations = numpy.append(Battleship_Locations,[[G1,G2]],axis=0)
                        #print(Battleship_Locations)
                    elif board_horz[G1][G2] == 3:
                        Cruiser_Hits += 1
                        if Cruiser_Hits == 1:
                            Cruiser_Locations =[[G1,G2]]
                        else:
                            Cruiser_Locations = numpy.append(Cruiser_Locations,[[G1,G2]],axis=0)
                        #print(Cruiser_Locations)
                    elif board_horz[G1][G2] == 2:
                        Submarine_Hits += 1
                        if Submarine_Hits == 1:
                            Submarine_Locations = [[G1,G2]]
                        else:
                            Submarine_Locations = numpy.append(Submarine_Locations,[[G1,G2]],axis=0)
                        #print(Submarine_Locations)
                    elif board_horz[G1][G2] == 1:
                        Destroyer_Hits += 1
                        if Destroyer_Hits == 1: 
                            Destroyer_Locations = [[G1,G2]]
                        else:
                            Destroyer_Locations = numpy.append(Destroyer_Locations,[[G1,G2]],axis=0)
                        #print(Destroyer_Locations)
                    
                    if board_horz[G1][G2] != 0:
                        board_horz[G1][G2] = 7
                        G1_Archive = G1
                        G2_Archive = G2
                        #print("Hit!")
                        hit_count += 1
                        
                        #print(Legend_horz)
                        #for x in board_horz:
                        #    print(x)
                        #print("\n")
                        #time.sleep(1)
                        
                        if Carrier_Hits == 5:
                            for x in Carrier_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Carrier_Hits = 6
                            target = "stop"
                        if Battleship_Hits == 4:
                            for x in Battleship_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Battleship_Hits = 5
                            target = "stop"
                        if Cruiser_Hits == 3:
                            for x in Cruiser_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Cruiser_Hits = 4
                            target = "stop"
                        if Submarine_Hits == 3:
                            for x in Submarine_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Submarine_Hits = 4
                            target = "stop"
                        if Destroyer_Hits == 2:
                            for x in Destroyer_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Destroyer_Hits = 3
                            target = "stop"

                    else:
                        board_horz[G1][G2] = 6
                        G1 = G1_Archive
                        G2 = G2_Archive
                        #print("miss")
                        
                    target_impossible1 = 0
                    target_impossible2 = 0 
                    target_impossible3 = 0
                    target_impossible4 = 0
                
            else:
                board_horz[G1][G2] = 6
            
            if hit_count == 17:
                #print("Game Complete")
                #print(turn_count)
                game = 0
                
    return board_horz,turn_count

# ----Optimized Hunt and Target----
def HuntTargetOptimal(board_horz,turn_count):
    #print("\n")
    #print("Starting Search with Hunt and Target Method")
    game = 1
    hit_count = 0
    target = "stop"
    Carrier_Hits = 0
    Carrier_Locations = [[]]
    Battleship_Hits = 0
    Battleship_Locations = [[]]
    Cruiser_Hits = 0
    Cruiser_Locations = [[]]
    Submarine_Hits = 0
    Submarine_Locations = [[]]
    Destroyer_Hits = 0
    Destroyer_Locations = [[]]
    
    while(game > 0):
        
        #time.sleep(2)
        #print(Legend_horz)
        #for x in board_horz:
        #    print(x)
        #print("\n")
            
        G1 = (randint(0,9)) #Row
        G2 = (randint(0,9)) #Column
        
        unsunk = 0
        G1_index = 0
        G2_index = 0
        
        for x in board_horz:
            for y in x:
                if y == 7:
                    unsunk = 1
                    G1 = G1_index
                    G2 = G2_index
                    break
                G2_index += 1
            if unsunk == 1:
                break
            G1_index += 1
            G2_index = 0
        
        if board_horz[G1][G2] != 6 and board_horz[G1][G2] != 8:
            
            if board_horz[G1][G2] != 7:
                turn_count += 1
 
            if board_horz[G1][G2] != 0 and board_horz[G1][G2] != 7:
                
                if board_horz[G1][G2] == 5:
                    Carrier_Hits += 1
                    Carrier_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 4:
                    Battleship_Hits += 1
                    Battleship_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 3:
                    Cruiser_Hits += 1
                    Cruiser_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 2:
                    Submarine_Hits += 1
                    Submarine_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 1:
                    Destroyer_Hits += 1
                    Destroyer_Locations = [[G1,G2]]
                    #print("Ship Hit")
                
                board_horz[G1][G2] = 7
                #print("Hit!")
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                #time.sleep(3)
                hit_count += 1
                
                target = "go"
            
            elif board_horz[G1][G2] == 7:
                target = "go"
            
            G1_Archive = G1
            G2_Archive = G2
            G1_Initial = G1
            G2_Initial = G2
            target_archive = 5
            target_impossible1 = 0
            target_impossible2 = 0 
            target_impossible3 = 0
            target_impossible4 = 0
            Directions_init = [1,2,3,4]
                    
            if G1 == 0:
                Directions_init.remove(1)
                target_impossible1 = 1
            if G1 == 9:
                Directions_init.remove(2)        
                target_impossible2 = 1
            if G2 == 0:
                Directions_init.remove(3)
                target_impossible3 = 1
            if G2 == 9:
                Directions_init.remove(4)
                target_impossible4 = 1
            
            
            while target == "go":
                    
                Directions = [1,2,3,4]
                Next_Coordinate = "Not Locked"
                    
                if G1 == 0:
                    Directions.remove(1)
                    target_impossible1 = 1
                if G1 == 9:
                    Directions.remove(2)        
                    target_impossible2 = 1
                if G2 == 0:
                    Directions.remove(3)
                    target_impossible3 = 1
                if G2 == 9:
                    Directions.remove(4)
                    target_impossible4 = 1
                
                #print(Directions)
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                if target_archive in Directions:
                    target_direction = target_archive
                elif target_archive == 5:
                    target_direction = random.choice(Directions)
                else:
                    if target_archive >= 3:
                        target_direction = target_archive - 2
                    else:
                        target_direction = target_archive + 2
                    target_archive = 6
                    G1 = G1_Initial
                    G2 = G2_Initial
                        
                    if target_direction in Directions_init:
                        pass
                    else:
                        target_direction = random.choice(Directions_init)
                #print("Direction Locked")
                #print(target_direction)
                #time.sleep(2)
                #print(Legend_horz)
                #for x in board_horz:
                #    print(x)
                #print("\n")
                #time.sleep(2)
                #print("next")
                #print("\n")
              
                
                    #--UP--
                if target_direction == 1:
                    if board_horz[G1-1][G2] != 6 and board_horz[G1-1][G2] != 7 and board_horz[G1-1][G2] != 8:
                        G1 = G1 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible1 = 1
                        target_archive = 5
                        #print("up impossible")
                    #--DOWN--
                if target_direction == 2:
                    if board_horz[G1+1][G2] != 6 and board_horz[G1+1][G2] != 7 and board_horz[G1+1][G2] != 8:
                        G1 = G1 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible2 = 1
                        target_archive = 5
                        #print("down impossible")
                    #--LEFT--
                if target_direction == 3:  
                    if board_horz[G1][G2-1] != 6 and board_horz[G1][G2-1] != 7 and board_horz[G1][G2-1] != 8:
                        G2 = G2 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible3 = 1
                        target_archive = 5
                        #print("Left impossible")
                    #--RIGHT--
                if target_direction == 4:
                    if board_horz[G1][G2+1] != 6 and board_horz[G1][G2+1] != 7 and board_horz[G1][G2+1] != 8:
                        G2 = G2 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible4 = 1
                        target_archive = 5
                        #print("Right impossible")
                        
                if target_impossible1 == 1 and target_impossible2 == 1 and target_impossible3 == 1 and target_impossible4 == 1:
                    target = "stop"
                #print("Next G1= " + str(G1))
                #print("Next G2= " + str(G2))
                #time.sleep(1)
                
                if Next_Coordinate == "Locked":
                    
                    turn_count += 1
                    
                    if board_horz[G1][G2] == 5:
                        Carrier_Hits += 1
                        if Carrier_Hits == 1:
                            Carrier_Locations = [[G1,G2]] 
                        else:
                            Carrier_Locations = numpy.append(Carrier_Locations,[[G1,G2]],axis=0)
                        #print(Carrier_Locations)
                    elif board_horz[G1][G2] == 4:
                        Battleship_Hits += 1
                        if Battleship_Hits == 1:
                            Battleship_Locations = [[G1,G2]]
                        else:
                            Battleship_Locations = numpy.append(Battleship_Locations,[[G1,G2]],axis=0)
                        #print(Battleship_Locations)
                    elif board_horz[G1][G2] == 3:
                        Cruiser_Hits += 1
                        if Cruiser_Hits == 1:
                            Cruiser_Locations =[[G1,G2]]
                        else:
                            Cruiser_Locations = numpy.append(Cruiser_Locations,[[G1,G2]],axis=0)
                        #print(Cruiser_Locations)
                    elif board_horz[G1][G2] == 2:
                        Submarine_Hits += 1
                        if Submarine_Hits == 1:
                            Submarine_Locations = [[G1,G2]]
                        else:
                            Submarine_Locations = numpy.append(Submarine_Locations,[[G1,G2]],axis=0)
                        #print(Submarine_Locations)
                    elif board_horz[G1][G2] == 1:
                        Destroyer_Hits += 1
                        if Destroyer_Hits == 1: 
                            Destroyer_Locations = [[G1,G2]]
                        else:
                            Destroyer_Locations = numpy.append(Destroyer_Locations,[[G1,G2]],axis=0)
                        #print(Destroyer_Locations)
                    
                    if board_horz[G1][G2] != 0:
                        board_horz[G1][G2] = 7
                        G1_Archive = G1
                        G2_Archive = G2
                        #print("Hit!")
                        hit_count += 1
                        target_archive = target_direction
                        
                        #print(Legend_horz)
                        #for x in board_horz:
                        #    print(x)
                        #print("\n")
                        #time.sleep(1)
                        
                        if Carrier_Hits == 5:
                            for x in Carrier_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Carrier_Hits = 6
                            target = "stop"
                        if Battleship_Hits == 4:
                            for x in Battleship_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Battleship_Hits = 5
                            target = "stop"
                        if Cruiser_Hits == 3:
                            for x in Cruiser_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Cruiser_Hits = 4
                            target = "stop"
                        if Submarine_Hits == 3:
                            for x in Submarine_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Submarine_Hits = 4
                            target = "stop"
                        if Destroyer_Hits == 2:
                            for x in Destroyer_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Destroyer_Hits = 3
                            target = "stop"

                    else:
                        board_horz[G1][G2] = 6
                        G1 = G1_Archive
                        G2 = G2_Archive
                        target_archive = 6
                        #print("miss")
                        
                    target_impossible1 = 0
                    target_impossible2 = 0 
                    target_impossible3 = 0
                    target_impossible4 = 0
                
            else:
                board_horz[G1][G2] = 6
            
            if hit_count == 17:
                #print("Game Complete")
                #print(turn_count)
                game = 0
                
    return board_horz,turn_count

# ----Hunt and Target with Parity----
def HuntTargetParity(board_horz,turn_count):
    #print("\n")
    #print("Starting Search with Hunt and Target Method")
    game = 1
    hit_count = 0
    target = "stop"
    Carrier_Hits = 0
    Carrier_Locations = [[]]
    Battleship_Hits = 0
    Battleship_Locations = [[]]
    Cruiser_Hits = 0
    Cruiser_Locations = [[]]
    Submarine_Hits = 0
    Submarine_Locations = [[]]
    Destroyer_Hits = 0
    Destroyer_Locations = [[]]
    
    while(game > 0):
        
        #time.sleep(2)
        #print(Legend_horz)
        #for x in board_horz:
        #    print(x)
        #print("\n")
        
        G1 = (randint(0,9)) #Row
        if (G1 % 2) == 0:
            G2 = random.choice([0,2,4,6,8])
        else:
            G2 = random.choice([1,3,5,7,9]) #Column
            
        unsunk = 0
        G1_index = 0
        G2_index = 0
        
        for x in board_horz:
            for y in x:
                if y == 7:
                    unsunk = 1
                    G1 = G1_index
                    G2 = G2_index
                    break
                G2_index += 1
            if unsunk == 1:
                break
            G1_index += 1
            G2_index = 0
        
        if board_horz[G1][G2] != 6 and board_horz[G1][G2] != 8:
            
            if board_horz[G1][G2] != 7:
                turn_count += 1
 
            if board_horz[G1][G2] != 0 and board_horz[G1][G2] != 7:
                
                if board_horz[G1][G2] == 5:
                    Carrier_Hits += 1
                    Carrier_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 4:
                    Battleship_Hits += 1
                    Battleship_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 3:
                    Cruiser_Hits += 1
                    Cruiser_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 2:
                    Submarine_Hits += 1
                    Submarine_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 1:
                    Destroyer_Hits += 1
                    Destroyer_Locations = [[G1,G2]]
                    #print("Ship Hit")
                
                board_horz[G1][G2] = 7
                #print("Hit!")
                hit_count += 1
                
                target = "go"
            
            elif board_horz[G1][G2] == 7:
                target = "go"
            
            G1_Archive = G1
            G2_Archive = G2
            target_impossible1 = 0
            target_impossible2 = 0 
            target_impossible3 = 0
            target_impossible4 = 0
            
            while target == "go":
                    
                Directions = [1,2,3,4]
                Next_Coordinate = "Not Locked"
                    
                if G1 == 0:
                    Directions.remove(1)
                    target_impossible1 = 1
                if G1 == 9:
                    Directions.remove(2)        
                    target_impossible2 = 1
                if G2 == 0:
                    Directions.remove(3)
                    target_impossible3 = 1
                if G2 == 9:
                    Directions.remove(4)
                    target_impossible4 = 1
                
                #print(Directions)
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                target_direction = random.choice(Directions)
                #print("Direction Locked")
                #print(target_direction)
                #time.sleep(2)
                #print(Legend_horz)
                #for x in board_horz:
                #    print(x)
                #print("\n")
                
                #time.sleep(5)
              
                
                    #--UP--
                if target_direction == 1:
                    if board_horz[G1-1][G2] != 6 and board_horz[G1-1][G2] != 7 and board_horz[G1-1][G2] != 8:
                        G1 = G1 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible1 = 1
                        #print("up impossible")
                    #--DOWN--
                if target_direction == 2:
                    if board_horz[G1+1][G2] != 6 and board_horz[G1+1][G2] != 7 and board_horz[G1+1][G2] != 8:
                        G1 = G1 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible2 = 1
                        #print("down impossible")
                    #--LEFT--
                if target_direction == 3:  
                    if board_horz[G1][G2-1] != 6 and board_horz[G1][G2-1] != 7 and board_horz[G1][G2-1] != 8:
                        G2 = G2 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible3 = 1
                        #print("Left impossible")
                    #--RIGHT--
                if target_direction == 4:
                    if board_horz[G1][G2+1] != 6 and board_horz[G1][G2+1] != 7 and board_horz[G1][G2+1] != 8:
                        G2 = G2 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible4 = 1
                        #print("Right impossible")
                        
                if target_impossible1 == 1 and target_impossible2 == 1 and target_impossible3 == 1 and target_impossible4 == 1:
                    target = "stop"
                #print("Next G1= " + str(G1))
                #print("Next G2= " + str(G2))
                #time.sleep(1)
                
                if Next_Coordinate == "Locked":
                    
                    turn_count += 1
                    
                    if board_horz[G1][G2] == 5:
                        Carrier_Hits += 1
                        if Carrier_Hits == 1:
                            Carrier_Locations = [[G1,G2]] 
                        else:
                            Carrier_Locations = numpy.append(Carrier_Locations,[[G1,G2]],axis=0)
                        #print(Carrier_Locations)
                    elif board_horz[G1][G2] == 4:
                        Battleship_Hits += 1
                        if Battleship_Hits == 1:
                            Battleship_Locations = [[G1,G2]]
                        else:
                            Battleship_Locations = numpy.append(Battleship_Locations,[[G1,G2]],axis=0)
                        #print(Battleship_Locations)
                    elif board_horz[G1][G2] == 3:
                        Cruiser_Hits += 1
                        if Cruiser_Hits == 1:
                            Cruiser_Locations =[[G1,G2]]
                        else:
                            Cruiser_Locations = numpy.append(Cruiser_Locations,[[G1,G2]],axis=0)
                        #print(Cruiser_Locations)
                    elif board_horz[G1][G2] == 2:
                        Submarine_Hits += 1
                        if Submarine_Hits == 1:
                            Submarine_Locations = [[G1,G2]]
                        else:
                            Submarine_Locations = numpy.append(Submarine_Locations,[[G1,G2]],axis=0)
                        #print(Submarine_Locations)
                    elif board_horz[G1][G2] == 1:
                        Destroyer_Hits += 1
                        if Destroyer_Hits == 1: 
                            Destroyer_Locations = [[G1,G2]]
                        else:
                            Destroyer_Locations = numpy.append(Destroyer_Locations,[[G1,G2]],axis=0)
                        #print(Destroyer_Locations)
                    
                    if board_horz[G1][G2] != 0:
                        board_horz[G1][G2] = 7
                        G1_Archive = G1
                        G2_Archive = G2
                        #print("Hit!")
                        hit_count += 1
                        
                        #print(Legend_horz)
                        #for x in board_horz:
                        #    print(x)
                        #print("\n")
                        #time.sleep(1)
                        
                        if Carrier_Hits == 5:
                            for x in Carrier_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Carrier_Hits = 6
                            target = "stop"
                        if Battleship_Hits == 4:
                            for x in Battleship_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Battleship_Hits = 5
                            target = "stop"
                        if Cruiser_Hits == 3:
                            for x in Cruiser_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Cruiser_Hits = 4
                            target = "stop"
                        if Submarine_Hits == 3:
                            for x in Submarine_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Submarine_Hits = 4
                            target = "stop"
                        if Destroyer_Hits == 2:
                            for x in Destroyer_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Destroyer_Hits = 3
                            target = "stop"

                    else:
                        board_horz[G1][G2] = 6
                        G1 = G1_Archive
                        G2 = G2_Archive
                        #print("miss")
                        
                    target_impossible1 = 0
                    target_impossible2 = 0 
                    target_impossible3 = 0
                    target_impossible4 = 0
                
            else:
                board_horz[G1][G2] = 6
            
            if hit_count == 17:
                #print("Game Complete")
                #print(turn_count)
                game = 0
                
    return board_horz,turn_count

# ----Optimized Hunt and Target with Parity----
def HuntTargetOptimalParity(board_horz,turn_count):
    #print("\n")
    #print("Starting Search with Hunt and Target Method")
    game = 1
    hit_count = 0
    target = "stop"
    Carrier_Hits = 0
    Carrier_Locations = [[]]
    Battleship_Hits = 0
    Battleship_Locations = [[]]
    Cruiser_Hits = 0
    Cruiser_Locations = [[]]
    Submarine_Hits = 0
    Submarine_Locations = [[]]
    Destroyer_Hits = 0
    Destroyer_Locations = [[]]
    debug = 0
    
    while(game > 0):
        debug += 1
        if debug > 10000:
            game = 0
            turn_count = 1000
        #time.sleep(2)
        #print(Legend_horz)
        #for x in board_horz:
        #    print(x)
        #print("\n")
    
        G1 = (randint(0,9)) #Row
        if (G1 % 2) == 0:
            G2 = random.choice([0,2,4,6,8])
        else:
            G2 = random.choice([1,3,5,7,9]) #Column
        
        unsunk = 0
        G1_index = 0
        G2_index = 0
        
        for x in board_horz:
            for y in x:
                if y == 7:
                    unsunk = 1
                    G1 = G1_index
                    G2 = G2_index
                    break
                G2_index += 1
            if unsunk == 1:
                break
            G1_index += 1
            G2_index = 0
        
        if board_horz[G1][G2] != 6 and board_horz[G1][G2] != 8:
            if board_horz[G1][G2] != 7:
                turn_count += 1
 
            if board_horz[G1][G2] != 0 and board_horz[G1][G2] != 7:
                
                if board_horz[G1][G2] == 5:
                    Carrier_Hits += 1
                    Carrier_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 4:
                    Battleship_Hits += 1
                    Battleship_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 3:
                    Cruiser_Hits += 1
                    Cruiser_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 2:
                    Submarine_Hits += 1
                    Submarine_Locations = [[G1,G2]]
                    #print("Ship Hit")
                elif board_horz[G1][G2] == 1:
                    Destroyer_Hits += 1
                    Destroyer_Locations = [[G1,G2]]
                    #print("Ship Hit")
                
                board_horz[G1][G2] = 7
                #print("Hit!")
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                #time.sleep(3)
                hit_count += 1
                
                target = "go"
            
            elif board_horz[G1][G2] == 7:
                target = "go"
            
            G1_Archive = G1
            G2_Archive = G2
            G1_Initial = G1
            G2_Initial = G2
            target_archive = 5
            target_impossible1 = 0
            target_impossible2 = 0 
            target_impossible3 = 0
            target_impossible4 = 0
            Directions_init = [1,2,3,4]
                    
            if G1 == 0:
                Directions_init.remove(1)
                target_impossible1 = 1
            if G1 == 9:
                Directions_init.remove(2)        
                target_impossible2 = 1
            if G2 == 0:
                Directions_init.remove(3)
                target_impossible3 = 1
            if G2 == 9:
                Directions_init.remove(4)
                target_impossible4 = 1
            
            
            while target == "go":
                    
                Directions = [1,2,3,4]
                Next_Coordinate = "Not Locked"
                    
                if G1 == 0:
                    Directions.remove(1)
                    target_impossible1 = 1
                if G1 == 9:
                    Directions.remove(2)        
                    target_impossible2 = 1
                if G2 == 0:
                    Directions.remove(3)
                    target_impossible3 = 1
                if G2 == 9:
                    Directions.remove(4)
                    target_impossible4 = 1
                
                #print(Directions)
                #print("G1= " + str(G1))
                #print("G2= " + str(G2))
                if target_archive in Directions:
                    target_direction = target_archive
                elif target_archive == 5:
                    target_direction = random.choice(Directions)
                else:
                    if target_archive >= 3:
                        target_direction = target_archive - 2
                    else:
                        target_direction = target_archive + 2
                    target_archive = 6
                    G1 = G1_Initial
                    G2 = G2_Initial
                        
                    if target_direction in Directions_init:
                        pass
                    else:
                        target_direction = random.choice(Directions_init)
                #print("Direction Locked")
                #print(target_direction)
                #time.sleep(2)
                #print(Legend_horz)
                #for x in board_horz:
                #    print(x)
                #print("\n")
                #time.sleep(2)
                #print("next")
                #print("\n")
              
                
                    #--UP--
                if target_direction == 1:
                    if board_horz[G1-1][G2] != 6 and board_horz[G1-1][G2] != 7 and board_horz[G1-1][G2] != 8:
                        G1 = G1 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible1 = 1
                        target_archive = 5
                        #print("up impossible")
                    #--DOWN--
                if target_direction == 2:
                    if board_horz[G1+1][G2] != 6 and board_horz[G1+1][G2] != 7 and board_horz[G1+1][G2] != 8:
                        G1 = G1 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible2 = 1
                        target_archive = 5
                        #print("down impossible")
                    #--LEFT--
                if target_direction == 3:  
                    if board_horz[G1][G2-1] != 6 and board_horz[G1][G2-1] != 7 and board_horz[G1][G2-1] != 8:
                        G2 = G2 - 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible3 = 1
                        target_archive = 5
                        #print("Left impossible")
                    #--RIGHT--
                if target_direction == 4:
                    if board_horz[G1][G2+1] != 6 and board_horz[G1][G2+1] != 7 and board_horz[G1][G2+1] != 8:
                        G2 = G2 + 1
                        Next_Coordinate = "Locked"
                    else:
                        target_impossible4 = 1
                        target_archive = 5
                        #print("Right impossible")
                        
                if target_impossible1 == 1 and target_impossible2 == 1 and target_impossible3 == 1 and target_impossible4 == 1:
                    target = "stop"
                #print("Next G1= " + str(G1))
                #print("Next G2= " + str(G2))
                #time.sleep(1)
                
                if Next_Coordinate == "Locked":
                    
                    turn_count += 1
                    
                    if board_horz[G1][G2] == 5:
                        Carrier_Hits += 1
                        if Carrier_Hits == 1:
                            Carrier_Locations = [[G1,G2]] 
                        else:
                            Carrier_Locations = numpy.append(Carrier_Locations,[[G1,G2]],axis=0)
                        #print(Carrier_Locations)
                    elif board_horz[G1][G2] == 4:
                        Battleship_Hits += 1
                        if Battleship_Hits == 1:
                            Battleship_Locations = [[G1,G2]]
                        else:
                            Battleship_Locations = numpy.append(Battleship_Locations,[[G1,G2]],axis=0)
                        #print(Battleship_Locations)
                    elif board_horz[G1][G2] == 3:
                        Cruiser_Hits += 1
                        if Cruiser_Hits == 1:
                            Cruiser_Locations =[[G1,G2]]
                        else:
                            Cruiser_Locations = numpy.append(Cruiser_Locations,[[G1,G2]],axis=0)
                        #print(Cruiser_Locations)
                    elif board_horz[G1][G2] == 2:
                        Submarine_Hits += 1
                        if Submarine_Hits == 1:
                            Submarine_Locations = [[G1,G2]]
                        else:
                            Submarine_Locations = numpy.append(Submarine_Locations,[[G1,G2]],axis=0)
                        #print(Submarine_Locations)
                    elif board_horz[G1][G2] == 1:
                        Destroyer_Hits += 1
                        if Destroyer_Hits == 1: 
                            Destroyer_Locations = [[G1,G2]]
                        else:
                            Destroyer_Locations = numpy.append(Destroyer_Locations,[[G1,G2]],axis=0)
                        #print(Destroyer_Locations)
                    
                    if board_horz[G1][G2] != 0:
                        board_horz[G1][G2] = 7
                        G1_Archive = G1
                        G2_Archive = G2
                        #print("Hit!")
                        hit_count += 1
                        target_archive = target_direction
                        
                        #print(Legend_horz)
                        #for x in board_horz:
                        #    print(x)
                        #print("\n")
                        #time.sleep(1)
                        #print(board_horz)
                        if Carrier_Hits == 5:
                            for x in Carrier_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Carrier_Hits = 6
                            target = "stop"
                        if Battleship_Hits == 4:
                            for x in Battleship_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Battleship_Hits = 5
                            target = "stop"
                        if Cruiser_Hits == 3:
                            for x in Cruiser_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Cruiser_Hits = 4
                            target = "stop"
                        if Submarine_Hits == 3:
                            for x in Submarine_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Submarine_Hits = 4
                            target = "stop"
                        if Destroyer_Hits == 2:
                            for x in Destroyer_Locations:
                                #print(x)
                                coor1 = x[0]
                                coor2 = x[1]
                                board_horz[coor1][coor2] = 8
                                Destroyer_Hits = 3
                            target = "stop"

                    else:
                        #print(G1)
                        #print(G2)
                        board_horz[G1][G2] = 6
                        G1 = G1_Archive
                        G2 = G2_Archive
                        target_archive = 6
                        #print("set to 6, 1")
                        #print(board_horz)
                        
                    target_impossible1 = 0
                    target_impossible2 = 0 
                    target_impossible3 = 0
                    target_impossible4 = 0
                
            else:
                board_horz[G1][G2] = 6
            #    print("set to 6, 2")
            
            if hit_count == 17:
                #print("Game Complete")
                #time.sleep(0.01)
                #print(turn_count)
                game = 0
                
    return board_horz,turn_count

# ----Heatmap Generator----
def HeatmapGenerator(board_horz,Ship):
    
    board_horz_heat = numpy.array(
        [[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]]
        )
    
    goodx = 0
    goody = 0
    H1 = 0
    while H1 < 10:
        H2 = 0
        while H2 < 10:
            if board_horz[H1][H2] != 6 and board_horz[H1][H2] != 8:
                if 9 - H2 >= (Ship[1]-1):
                    for x in range(Ship[1]):
                        if board_horz[H1][H2+x] != 6 and board_horz[H1][H2+x] != 8:
                            goodx += 1
                        else:
                            goodx -= 100
                        if board_horz[H1][H2+x] == 7:
                            goodx += 10
                    if goodx > 0:    
                        for z in range(Ship[1]):
                            board_horz_heat[H1][H2+z] += 1
                    if goodx > 10:
                        for z in range(Ship[1]):
                            board_horz_heat[H1][H2+z] += 100
                if board_horz[H1][H2] == 7:
                    board_horz_heat[H1][H2] -= 10000
            H2 += 1
            goodx = 0
        H1 += 1
    
    H2 = 0
    while H2 < 10:
        H1 = 0
        while H1 < 10:
            if board_horz[H1][H2] != 6 and board_horz[H1][H2] != 8:
                if 9 - H1 >= (Ship[1]-1):
                    for x in range(Ship[1]):
                        if board_horz[H1+x][H2] != 6 and board_horz[H1+x][H2] != 8:
                            goody += 1
                        else:
                            goody -= 100
                        if board_horz[H1+x][H2] == 7:
                            goody += 10
                    if goody > 0:
                        for z in range(Ship[1]):
                            board_horz_heat[H1+z][H2] += 1
                    if goody > 10:
                        for z in range(Ship[1]):
                            board_horz_heat[H1+z][H2] += 100
                if board_horz[H1][H2] == 7:
                    board_horz_heat[H1][H2] -= 10000
            H1 += 1
            goody = 0
        H2 += 1
            
    return board_horz_heat

# ----Probability Guesser----
def ProbabilityGuessing(board_horz,turn_count):
    
    game = 1
    hit_count = 0
    Carrier_Hits = 0
    Carrier_Locations = [[]]
    Battleship_Hits = 0
    Battleship_Locations = [[]]
    Cruiser_Hits = 0
    Cruiser_Locations = [[]]
    Submarine_Hits = 0
    Submarine_Locations = [[]]
    Destroyer_Hits = 0
    Destroyer_Locations = [[]]
    
    while game > 0:
        #time.sleep(0.2)
        #print("Turn: \n")
        #print(turn_count)
        #print(board_horz)
        
        if Carrier_Hits != 6:
            Carrier_heatmap = HeatmapGenerator(board_horz,Carrier)
        else:
            Carrier_heatmap = numpy.array(
                    [[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]]
                    )
        if Battleship_Hits != 5:
            Battleship_heatmap = HeatmapGenerator(board_horz,Battleship)
        else:
            Battleship_heatmap = numpy.array(
                    [[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]]
                    )
        if Submarine_Hits != 4:
            Submarine_heatmap = HeatmapGenerator(board_horz,Submarine)
        else:
            Submarine_heatmap = numpy.array(
                    [[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]]
                    )
        if Cruiser_Hits != 4:
            Cruiser_heatmap = HeatmapGenerator(board_horz,Cruiser)
        else:
            Cruiser_heatmap = numpy.array(
                    [[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]]
                    )
        if Destroyer_Hits != 3:
            Destroyer_heatmap = HeatmapGenerator(board_horz,Destroyer)
        else:
            Destroyer_heatmap = numpy.array(
                    [[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]
                    ,[0,0,0,0,0,0,0,0,0,0]]
                    )
        
        Heatmap_ALL = Carrier_heatmap + Battleship_heatmap + Submarine_heatmap + Cruiser_heatmap + Destroyer_heatmap
        next_guess = numpy.where(Heatmap_ALL == numpy.amax(Heatmap_ALL))
        next_coordinates = list(zip(next_guess[0],next_guess[1]))
        First = (next_coordinates[randint(1,len(next_coordinates))-1])
        P1 = First[0]
        P2 = First[1]
        #print(P1)
        #print(P2)
        #print(Heatmap_ALL)
        
        if board_horz[P1][P2] > 0 and board_horz[P1][P2] < 6:
            
            turn_count += 1

            if board_horz[P1][P2] != 0 and board_horz[P1][P2] != 7:
                
                if board_horz[P1][P2] == 5:
                    Carrier_Hits += 1
                    if Carrier_Hits == 1:
                        Carrier_Locations = [[P1,P2]] 
                    else:
                        Carrier_Locations = numpy.append(Carrier_Locations,[[P1,P2]],axis=0)
                    #print(Carrier_Locations)
                elif board_horz[P1][P2] == 4:
                    Battleship_Hits += 1
                    if Battleship_Hits == 1:
                        Battleship_Locations = [[P1,P2]]
                    else:
                        Battleship_Locations = numpy.append(Battleship_Locations,[[P1,P2]],axis=0)
                    #print(Battleship_Locations)
                elif board_horz[P1][P2] == 3:
                    Cruiser_Hits += 1
                    if Cruiser_Hits == 1:
                        Cruiser_Locations =[[P1,P2]]
                    else:
                        Cruiser_Locations = numpy.append(Cruiser_Locations,[[P1,P2]],axis=0)
                    #print(Cruiser_Locations)
                elif board_horz[P1][P2] == 2:
                    Submarine_Hits += 1
                    if Submarine_Hits == 1:
                        Submarine_Locations = [[P1,P2]]
                    else:
                        Submarine_Locations = numpy.append(Submarine_Locations,[[P1,P2]],axis=0)
                    #print(Submarine_Locations)
                elif board_horz[P1][P2] == 1:
                    Destroyer_Hits += 1
                    if Destroyer_Hits == 1: 
                        Destroyer_Locations = [[P1,P2]]
                    else:
                        Destroyer_Locations = numpy.append(Destroyer_Locations,[[P1,P2]],axis=0)
                    #print(Destroyer_Locations)
                
                board_horz[P1][P2] = 7
                #print("Hit!")
                hit_count += 1
                
                if Carrier_Hits == 5:
                    for x in Carrier_Locations:
                        #print(x)
                        coor1 = x[0]
                        coor2 = x[1]
                        board_horz[coor1][coor2] = 8
                        Carrier_Hits = 6
                        board_horz[P1][P2] = 8
                if Battleship_Hits == 4:
                    for x in Battleship_Locations:
                        #print(x)
                        coor1 = x[0]
                        coor2 = x[1]
                        board_horz[coor1][coor2] = 8
                        Battleship_Hits = 5
                        board_horz[P1][P2] = 8
                if Cruiser_Hits == 3:
                    for x in Cruiser_Locations:
                        #print(x)
                        coor1 = x[0]
                        coor2 = x[1]
                        board_horz[coor1][coor2] = 8
                        Cruiser_Hits = 4
                        board_horz[P1][P2] = 8
                if Submarine_Hits == 3:
                    for x in Submarine_Locations:
                        #print(x)
                        coor1 = x[0]
                        coor2 = x[1]
                        board_horz[coor1][coor2] = 8
                        Submarine_Hits = 4
                        board_horz[P1][P2] = 8
                if Destroyer_Hits == 2:
                    for x in Destroyer_Locations:
                        #print(x)
                        coor1 = x[0]
                        coor2 = x[1]
                        board_horz[coor1][coor2] = 8
                        Destroyer_Hits = 3
                        board_horz[P1][P2] = 8
            
        else:
            board_horz[P1][P2] = 6
            #print("miss")
            turn_count += 1
        
        if hit_count == 17:
            #print("Game Complete")
            #print(turn_count)
            game = 0
        
    return board_horz,turn_count
    
# ----Game----
def Simulator(Method):
    turn_count = 0
    board_horz_init = numpy.array(
        [[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]]
        )
    board_vert_init = board_horz_init.transpose()
    
    # --start ship generation and initialize board--
    board_horz,board_vert = ShipGenerator(board_horz_init,board_vert_init,Carrier)
    ShipGenerator(board_horz,board_vert,Battleship)
    ShipGenerator(board_horz,board_vert,Submarine)
    ShipGenerator(board_horz,board_vert,Cruiser)
    ShipGenerator(board_horz,board_vert,Destroyer)
    
    #print("\n")
    #print("The game has generated these ships")
    #print("\n")
    #print(Legend_horz)
    #for x in board_horz:
    #    print(x)
        
    #need to do this to pass integers
    if Method == 1:
        board_horz,turn_count = RandomSearch(board_horz,turn_count)
    elif Method == 2:
        board_horz,turn_count = HuntTarget(board_horz,turn_count)
    elif Method == 3:
        board_horz,turn_count = HuntTargetOptimal(board_horz,turn_count)
    elif Method == 4:
        board_horz,turn_count = HuntTargetParity(board_horz,turn_count)
    elif Method == 5:
        board_horz,turn_count = HuntTargetOptimalParity(board_horz,turn_count)
    elif Method == 6:
        board_horz,turn_count = ProbabilityGuessing(board_horz,turn_count)
    else:
        print("Invalid Search Method or No Search Method Selected")
        return

    #print(Legend_horz)
    #for x in board_horz:
    #    print(x)

    return(turn_count)        


#----Base Values----
Legend_horz = numpy.array([0,1,2,3,4,5,6,7,8,9])
board_horz_init = numpy.array(
        [[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]
        ,[0,0,0,0,0,0,0,0,0,0]]
        )
board_vert_init = board_horz_init.transpose()

#boats = [name,length,value]
Carrier = ["Carrier",5,5]
Battleship = ["Battleship",4,4]
Cruiser = ["Cruiser",3,3]
Submarine = ["Submarine",3,2]
Destroyer = ["Destroyer",2,1]
search_list =["Random Search","Hunt and Target","Optimized Hunt and Target","Hunt and Target with Parity",
              "Optimized Hunt and Target with Parity","Probability Search"]
verify_method = "N"
verify_ite = "N"
verify_time = "N"

print("\nWelcome To Battleship!")
print("\n")

print("Generating Board...")
print("\n")
#print(Legend_horz)
for x in board_horz_init:
    print(x)
#for x in board_vert:
#    print(x)

while verify_method == "N":
    print("\n")
    print("1 - Random Search")
    print("2 - Hunt and Target")
    print("3 - Optimized Hunt and Target")
    print("4 - Hunt and Target with Parity")
    print("5 - Optimized Hunt and Target with Parity")
    print("6 - Probability Search")
    print("0 - Exit Simulator")
    method = int(input("Enter Desired Search Method:\n"))
    if method == 0:
        exit()
    print("The Simulator Will Run With The Following Method: " + search_list[method-1])
    verify_method = str(input("Is This Correct?  Y/N\n"))
    verify_method = verify_method.upper()

while verify_ite == "N":
    print("\n")
    ite = int(input("Enter Desired Iterations:\n"))
    if ite == 0:
        exit()
    print("The Simulator Will Run This Many Times: " + str(ite))
    verify_ite = str(input("Is This Correct?  Y/N\n"))
    verify_ite = verify_ite.upper()

total_time = [ite * 0.001620604,ite * 0.004848548,ite * 0.004476025,ite*0.006260266,ite*0.005884306,ite*0.18564876]  
  
while verify_time == "N":
    print("\n")
    print("The Simulation Will Take Around: " + str(timedelta(seconds=total_time[method-1])))
    verify_time = str(input("Continue?  Y/N\n"))
    verify_time = verify_time.upper()
    if verify_time == "N":
        exit()

workbook = xlsxwriter.Workbook(search_list[method-1]+' BattleShip Results.xlsx')
worksheet = workbook.add_worksheet()

threshold = 1000000
    
start_time = datetime.now()
for y in range(math.ceil(ite / threshold)):
    for x in range(ite//math.ceil(ite / threshold)):
        turn_count = Simulator(method)
        worksheet.write(x,y+1, turn_count)
        print("Completed " + str(y*threshold + (x+1)) + " Iterations")
end_time = datetime.now()
print("Simulation Length: " + str(end_time - start_time))
    
#save to file

#worksheet.write(0,0,"File Name")
worksheet.write(0,0,"Min Turns")
worksheet.write(2,0,"Max Turns")
worksheet.write(4,0,"Average Turns")
worksheet.write_formula(1,0,"=MIN(B1:K" + str(ite) + ")")
worksheet.write_formula(3,0,"=MAX(B1:K" + str(ite) + ")")
worksheet.write_formula(5,0,"=AVERAGE(B1:K" + str(ite) + ")")
print("Writing To File...")
workbook.close()
print("Done!")   
