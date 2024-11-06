import pygame, sys
from pygame.locals import *
import numpy as np
from keras.models import load_model
import cv2

# Initialize Pygame
pygame.init()

# Set up the window
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 600
BOUNDARY_INC=5

# Set up the colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED=(255,0,0)

IMAGESAVE=False
MODEL=load_model("bestmodel.h5")

LABELS={0:"Zero",1:"One",2:"Two",3:"Three",4:"Four",5:"Five",6:"Six",7:"Seven",8:"Eight",9:"Nine"}

FONT=pygame.font.Font("freesansbold.ttf", 18)
DISPLAYSURF=pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))

# Display instruction text at the bottom right corner
instruction_text = FONT.render("Press 'c' to clear the screen", True, WHITE)
instruction_text_rect = instruction_text.get_rect()
instruction_text_rect.bottomright = (WINDOW_WIDTH - 10, WINDOW_HEIGHT - 10) 
DISPLAYSURF.blit(instruction_text, instruction_text_rect)

pygame.display.set_caption("Handwritten Digit Recognizer")

iswriting = False

Number_X=[]
Number_Y=[]
img_cnt=0

PREDICT=True

while True:
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == MOUSEMOTION and iswriting:
            # Get the position of the mouse
            x, y = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (x, y), 4,0)
            Number_X.append(x)
            Number_Y.append(y)

        if event.type == MOUSEBUTTONDOWN:
            iswriting = True
        
        if event.type == MOUSEBUTTONUP:
            iswriting=False
            Number_X=sorted(Number_X)
            Number_Y=sorted(Number_Y)

            rect_min_x, rect_max_x = max(Number_X[0]-BOUNDARY_INC,0), min(Number_X[-1]+BOUNDARY_INC,WINDOW_WIDTH)
            rect_min_y, rect_max_y = max(Number_Y[0]-BOUNDARY_INC,0), min(Number_Y[-1]+BOUNDARY_INC,WINDOW_HEIGHT)

            Number_X=[]
            Number_Y=[]

            img_arr=np.array(pygame.PixelArray(DISPLAYSURF)[rect_min_x:rect_max_x,rect_min_y:rect_max_y]).T.astype(np.float32)
            if IMAGESAVE:
                cv2.imwrite("img.png",img_arr)
                img_cnt+=1

            if PREDICT:
                img_arr=cv2.resize(img_arr,(28,28))
                img_arr=np.pad(img_arr,(10,10),'constant',constant_values=0)
                img_arr=cv2.resize(img_arr,(28,28))/255
                
                label=str(LABELS[np.argmax(MODEL.predict(img_arr.reshape(1,28,28,1)))])
                text_surface=FONT.render(label,True,RED,WHITE)
                textRecObj=text_surface.get_rect()
                textRecObj.left, textRecObj.bottom=rect_min_x, rect_max_y

                DISPLAYSURF.blit(text_surface,textRecObj)
                pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

        if event.type == KEYDOWN:
            if event.unicode == "c":
                DISPLAYSURF.fill(BLACK)
                DISPLAYSURF.blit(instruction_text, instruction_text_rect)

    pygame.display.update()
