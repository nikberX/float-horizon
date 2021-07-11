from MathAndProjections import *
from tkinter import Tk,Canvas
import numpy as np
from math import e
#Screen height
SCR_H = 600
#Screen width
SCR_W = 600

root = Tk()

cv = Canvas(root, width=SCR_W, height=SCR_H, bg='black')

#Function to show
def func(x,z):
    return 1.5 * np.cos((x ** 2 + z ** 2) ** 0.5)
    #return np.sin(x)*np.sin(z)
    #a=(x*x+z*z)**(0.5)
    #return 2*np.cos(a)/(a+1)
    #return -1.5*(np.cos(1.75 * a)*e**(-a))+0.2*np.sin(x+np.pi)*np.cos(z+np.pi)
    #return -np.log(abs(x*x*(x*x-z*z))+0.15)
    #return np.cos(x)*np.cos(z)/(abs(x)+1)


#Camera class
class Camera:
    def __init__(self,cameraPosition):
        self.x = cameraPosition[0]
        self.y = cameraPosition[1]
        self.z = cameraPosition[2]

    def move(self,vec):
        self.x += vec[0]
        self.y += vec[1]
        self.z += vec[2]

    def getPos(self):
        return [self.x,self.y,self.z]

    def setPos(self,pos):
        self.x = pos[0]
        self.y = pos[1]
        self.z = pos[2]


#Pre loader class to improve performance
class PreLoader:
    def __init__(self, SCR_w, SCR_h, func, n, steps):

        #shows count of preloaded point arrays
        self.steps = steps

        self.w = SCR_w
        self.h = SCR_h

        self.f = func

        self.verticies = []

        #number of lines
        self.n = n

        #range of x values of func
        self.xmin = -4 * np.pi
        self.xmax = 4 * np.pi

        # oX steps
        self.dx = (self.xmax - self.xmin) / self.n

        #range of z values of func
        self.zmin = -4 * np.pi
        self.zmax = 4 * np.pi

        # oZ steps
        self.dz = (self.zmax - self.zmin) / self.n


        self.pointViews = np.zeros((self.n, self.n))

        self.upHorizon = np.zeros(self.w)
        self.downHorizon = np.repeat(self.h,self.w)

        self.m = self.n



        for z in np.arange(self.zmin,self.zmax,self.dz):
            for x in np.arange(self.xmin,self.xmax,self.dx):
                point = [x,self.f(x,z),z]
                self.verticies.append(point)

        self.verticies = np.array(self.verticies)

        self.pointList = []

    def generateFuturePoints(self):
        if (self.steps<300):
            cameraPos = [0,0,-5]
            #generating future camera positions and its view

            for step_index in range(self.steps):
                #angle of camera rotation
                cos = 0.99995
                sin = 0.00999983
                #rotated coords
                newY=cameraPos[1] * cos - cameraPos[2] * sin - cameraPos[1]
                newZ=cameraPos[1] * sin + cameraPos[2] * cos - cameraPos[2]
                cameraPos[0] = 0
                cameraPos[1] = newY + cameraPos[1]
                cameraPos[2] = newZ + cameraPos[2]
                #show progress to user
                print("Pre generating future points:", ((step_index)/self.steps)*100 ," % ")
                #generate view for position
                self.generateStep(cameraPos)


    def generateStep(self,cameraPos):

        pointsToAppend = []

        #float-horizon algorithm

        #math with function points to get screen coords
        self.M = self.verticies
        self.M, self.s = worldToViewer(self.M, [cameraPos[0], cameraPos[1], cameraPos[2]])
        self.M = parallel_proj(self.M, self.s)
        self.M = viewerPlaneToScreenIdealized(self.M, 10, self.w / 2, self.h / 2, 200, 200)
        self.M = np.reshape(self.M, (self.n, self.n, 2))

        self.up = np.zeros(self.w)
        self.down = np.repeat(self.h, self.w)


        def vis_state(point):
            ind = int(point[0])
            if point[1] < self.up[ind] and point[1] > self.down[ind]:
                return 0
            elif point[1] >= self.up[ind]:
                return 1
            elif point[1] <= self.down[ind]:
                return 2

        for i in range(self.m):

            self.pointViews[i][0] = vis_state(self.M[i][0])

            for j in range(1, self.n):

                self.pointViews[i][j] = vis_state(self.M[i][j])
                vis_temp = self.pointViews[i][j - 1]
                vis_point = self.pointViews[i][j]

                fir_p = self.M[i][j - 1]
                sec_p = self.M[i][j]


                case_1 = vis_temp == 1 and vis_point == 1
                case_2 = vis_temp == 2 and vis_point == 2

                if case_1 or case_2:

                    dx = (fir_p[0] - sec_p[0])
                    dy = (sec_p[1] - fir_p[1])

                    step_y = dy / dx
                    sec_p = np.copy(fir_p)

                    if case_1:
                        for k in range(int(dx) + 1):
                            ind = int(sec_p[0])
                            self.up[ind] = sec_p[1]
                            if i == 0:
                                self.down[ind] = sec_p[1]
                            sec_p[0] -= 1
                            sec_p[1] += step_y
                    else:
                        for k in range(int(dx) + 1):
                            ind = int(sec_p[0])
                            self.down[ind] = sec_p[1]
                            if i == 0:
                                self.up[ind] = sec_p[1]
                            sec_p[0] -= 1
                            sec_p[1] += step_y

                    if case_1:
                        fill = 'red'
                        pointsToAppend.append([fir_p[0], fir_p[1], sec_p[0], sec_p[1], fill])
                    else:
                        fill = 'green'
                        pointsToAppend.append([fir_p[0], fir_p[1], sec_p[0], sec_p[1], fill])

                else:

                    if vis_point > 0:
                        vis_temp, vis_point = vis_point, vis_temp
                        fir_p, sec_p = sec_p, fir_p

                    if vis_temp > 0:
                        if fir_p[0] == sec_p[0]:
                            if vis_temp == 1:
                                sec_p[1] = self.up[fir_p[0]]
                                self.up[fir_p[0]] = fir_p[1]
                            else:
                                sec_p[1] = self.down[fir_p[0]]
                                self.down[fir_p[0]] = fir_p[1]
                        else:
                            dx = (sec_p[0] - fir_p[0])
                            dy = (sec_p[1] - fir_p[1])
                            step_x = np.sign(dx)

                            step_y = dy / abs(dx)
                            sec_p = np.copy(fir_p)

                            if vis_temp == 1:
                                self.up[int(sec_p[0])] = sec_p[1]
                                sec_p[0] += step_x
                                sec_p[1] += step_y

                                ind = int(sec_p[0])
                                while sec_p[1] - 1.5 >= self.up[ind]:
                                    self.up[ind] = sec_p[1]
                                    sec_p[0] += step_x
                                    sec_p[1] += step_y
                                    ind = int(sec_p[0])
                            else:
                                self.down[int(sec_p[0])] = sec_p[1]
                                sec_p[0] += step_x
                                sec_p[1] += step_y

                                ind = int(sec_p[0])
                                while sec_p[1] + 1.5 <= self.down[ind]:
                                    self.down[ind] = sec_p[1]
                                    sec_p[0] += step_x
                                    sec_p[1] += step_y
                                    ind = int(sec_p[0])

                        fill = 'yellow'
                        pointsToAppend.append([fir_p[0], fir_p[1], sec_p[0], sec_p[1], fill])
        self.pointList.append([pointsToAppend])


    def getStep(self,step):
        return self.pointList[step]

#renderer class. Renders current camera view
class Renderer:
    def __init__(self,canvas,preloader):
        self.canvas = canvas
        self.canvas.pack()
        self.preloader = preloader

    def renderStep(self,step):
        points = self.preloader.getStep(step)
        self.canvas.delete('all')

        for item in points:
            for line in item:
                self.canvas.create_line(line[0],line[1],line[2],line[3],fill = line[4])




#handle user input
class EventHandler:
    def __init__(self,camera,renderer):
        self.camera = camera
        self.renderer = renderer
        self.step = 0

    def keyEvent(self, event):
        camP = self.camera.getPos()
        if event.char == 'w':
            self.renderer.renderStep(self.step)
            # 2 - speed of camera rotation
            self.step+=2
        elif event.char == 's':
            self.renderer.renderStep(self.step)
            # 2 - speed of camera rotation
            self.step-=2




def main():
    #creating camera obj
    camera = Camera([0,0,-5])
    #creating preloader obj
    preloader = PreLoader(SCR_W,SCR_H,func,45,200)
    preloader.generateFuturePoints()
    #create renderer class
    renderer = Renderer(cv,preloader)
    #create handler to process user input
    eventHandler = EventHandler(camera,renderer)

    #bind handler to app
    root.bind("<Key>",eventHandler.keyEvent)

    #start mainloop
    root.mainloop()

if __name__ == '__main__':
    main()
